from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re
import uvicorn


# ================== BASE PATH ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOXICITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "toxicity_classifier",
    "classifier.pkl"
)

# ================== NEGATION PREPROCESSOR ==================
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "shouldn't", "wouldn't", "couldn't"
}

def preprocess_with_negation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{token}_{tokens[i + 1]}")
            i += 2
        else:
            result.append(token)
            i += 1
    return " ".join(result)

# ================== SAFE PATTERNS WHITELIST ==================
SAFE_PATTERNS = [
    r"\bnot\s+(a\s+)?(bad|wrong|stupid|idiot|dumb|ugly|evil|racist|hateful|people|human|person)\b",
    r"\bdon'?t\s+(be|act|say)\b",
    r"\bnever\s+(said|meant|hurt|harm)\b",
    r"\bno\s+(hate|harm|offense)\b",
    r"\bnot\s+\w+ing\b",                      # "not joking", "not kidding"
    r"\byou\s+(are|guys?\s+are)\s+not\b",     # "you are not...", "you guys are not..."
    r"\bwe\s+are\s+not\b",
    r"\bthey\s+are\s+not\b",
    r"\bi\s+(will\s+not|won'?t|don'?t)\b",
]

def has_safe_pattern(text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in SAFE_PATTERNS)

# ================== LOAD MODEL ==================
print("🚀 Loading toxicity classifier...")

try:
    pipeline = joblib.load(TOXICITY_MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:", e)
    pipeline = None

# ================== FASTAPI APP ==================
app = FastAPI(title="Toxicity Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
         "http://localhost:5173",
         "https://springscircle.vercel.app"
          ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)
# ================== REQUEST BODY ==================
class RequestBody(BaseModel):
    text: str

# ================== HEALTH CHECK ==================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }

# ================== EXPLAINABILITY ==================
def extract_toxic_words(text: str, pipeline, top_n: int = 5):
    processed    = preprocess_with_negation(text)
    vectorizer   = pipeline.named_steps["tfidf"]
    model        = pipeline.named_steps["clf"]
    vocab        = vectorizer.vocabulary_
    coefficients = model.coef_[0]
    tokens       = processed.split()
    candidates   = []

    for i, token in enumerate(tokens):
        ngrams = [token]
        if i + 1 < len(tokens):
            ngrams.append(f"{token} {tokens[i + 1]}")
        for ngram in ngrams:
            idx = vocab.get(ngram)
            if idx is not None and coefficients[idx] > 0:
                candidates.append({
                    "word": ngram,
                    "weight": round(float(coefficients[idx]), 3)
                })

    seen, unique = set(), []
    for c in sorted(candidates, key=lambda x: x["weight"], reverse=True):
        if c["word"] not in seen:
            seen.add(c["word"])
            unique.append(c)

    return unique[:top_n]

# ================== PREDICTION ROUTE ==================
@app.post("/process/")
def process_text(body: RequestBody):
    if pipeline is None:
        return {"error": "Model not loaded"}

    text      = (body.text or "").strip()
    processed = preprocess_with_negation(text)

    pred_prob = pipeline.predict_proba([processed])[0][1]

    # Only flag if score is very high
    THRESHOLD = 0.80

    is_toxic = bool(pred_prob >= THRESHOLD)

    # Whitelist override: if a safe pattern is detected and score
    # is not extremely high (< 0.95), force non-toxic
    if is_toxic and pred_prob < 0.95 and has_safe_pattern(text):
        is_toxic = False

    toxic_words = extract_toxic_words(text, pipeline)

    return {
        "toxicity": is_toxic,
        "score":    round(float(pred_prob), 4),
        "bad_words": toxic_words,
        "original": text
    }

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )