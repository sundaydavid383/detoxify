# server.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import uvicorn

# ================== Load Model ==================
TOXICITY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "toxicity_classifier",
    "classifier.pkl"
)

print("🚀 Loading toxicity classifier...")

saved = joblib.load(TOXICITY_MODEL_PATH)

vectorizer = saved["vectorizer"]
model = saved["model"]

print("✅ Toxicity classifier loaded!")

# ================== FastAPI ==================
app = FastAPI(title="Toxicity Classifier API")

# ================== Request Schema ==================
class RequestBody(BaseModel):
    text: str

# ================== Toxic Word Extraction ==================
def extract_toxic_words(text, vectorizer, model, top_n=5):

    words = text.lower().split()

    coefficients = model.coef_[0]

    toxic_words = []

    for word in words:

        idx = vectorizer.vocabulary_.get(word)

        if idx is not None:

            weight = coefficients[idx]

            if weight > 0:

                toxic_words.append({
                    "word": word,
                    "weight": round(float(weight), 3)
                })

    toxic_words = sorted(
        toxic_words,
        key=lambda x: x["weight"],
        reverse=True
    )

    return toxic_words[:top_n]

# ================== Health Route ==================
@app.get("/")
def health():
    return {
        "status": "ok"
    }

# ================== Prediction Route ==================
@app.post("/process/")
def process_text(body: RequestBody):

    text = body.text.strip()

    text_vec = vectorizer.transform([text])

    pred_prob = model.predict_proba(text_vec)[0][1]

    is_toxic = bool(pred_prob >= 0.4)

    toxic_words = extract_toxic_words(
        text,
        vectorizer,
        model
    )

    return {
        "toxicity": is_toxic,
        "score": round(float(pred_prob), 4),
        "bad_words": toxic_words,
        "original": text
    }

# ================== Run Server ==================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )