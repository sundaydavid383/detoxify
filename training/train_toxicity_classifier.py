import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

print("🚀 Starting improved toxicity classifier training...\n")

# ================== 1. Load Dataset ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "jigsaw_train.csv")

df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} comments")

# ================== 2. Create Binary Label ==================
toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']
df['is_toxic'] = df[toxic_cols].max(axis=1).astype(int)

print(f"Toxic samples:     {df['is_toxic'].sum():,}")
print(f"Non-toxic samples: {len(df) - df['is_toxic'].sum():,}\n")

# ================== 3. Negation-Aware Preprocessing ==================
# This is the KEY fix: "not stupid" becomes "not_stupid" so the model
# treats it as a completely different token from "stupid" alone.

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "shouldn't", "wouldn't", "couldn't"
}

def preprocess_with_negation(text: str) -> str:
    """
    Joins negation words with the following word using an underscore.

    Examples:
        "you are not stupid"  → "you are not_stupid"
        "I will not hurt you" → "I will not_hurt you"
        "don't be an idiot"   → "don't_be an idiot"
    """
    if not isinstance(text, str):
        return ""

    # Lowercase & basic cleaning
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # keep apostrophes for contractions
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        # If this token is a negation word and there's a next word, merge them
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            merged = f"{token}_{tokens[i + 1]}"
            result.append(merged)
            i += 2  # skip the next word since we merged it
        else:
            result.append(token)
            i += 1

    return " ".join(result)

print("🔄 Preprocessing text with negation handling...")
df['processed_text'] = df['comment_text'].fillna("").apply(preprocess_with_negation)

# ================== 4. Split Data ==================
X = df['processed_text']
y = df['is_toxic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# ================== 5. Build Pipeline ==================
# Using bigrams (ngram_range=(1,2)) captures two-word phrases like
# "not_stupid" that the negation step created, as well as general
# two-word context like "kill you" vs "kill bacteria".
print("📚 Building TF-IDF + Logistic Regression pipeline...")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,          # more features for better coverage
        stop_words=None,             # DON'T remove stop words — "not", "no" matter!
        lowercase=True,
        min_df=2,
        ngram_range=(1, 2),          # unigrams AND bigrams
        sublinear_tf=True,           # log-scale TF to reduce extreme frequencies
        analyzer='word'
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight='balanced',     # handles class imbalance
        C=1.0,                       # regularisation strength
        solver='lbfgs',
        n_jobs=-1
    ))
])

# ================== 6. Train ==================
print("🧠 Training model...")
pipeline.fit(X_train, y_train)

# ================== 7. Evaluate ==================
print("\n📊 Evaluation Results:\n")
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Not Toxic', 'Toxic']))
print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ================== 8. Save ==================
save_dir = "models/toxicity_classifier"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "classifier.pkl")

joblib.dump(pipeline, save_path, compress=3)

print("\n✅ Model saved successfully!")
print(f"📍 Location: {save_path}")
print(f"📦 Size: {os.path.getsize(save_path)/1024/1024:.2f} MB")

# ================== 9. Explainability ==================
def extract_toxic_words(text: str, pipeline, top_n: int = 5) -> list[dict]:
    """
    Returns the words/bigrams in `text` that most contributed to a toxic
    prediction, based on TF-IDF weight × logistic regression coefficient.
    """
    processed = preprocess_with_negation(text)
    vectorizer = pipeline.named_steps["tfidf"]
    model      = pipeline.named_steps["clf"]

    feature_names  = vectorizer.get_feature_names_out()
    coefficients   = model.coef_[0]
    vocab          = vectorizer.vocabulary_

    tokens = processed.split()
    candidates = []

    # Check unigrams and bigrams
    for i, token in enumerate(tokens):
        for ngram in ([token] + ([f"{token} {tokens[i+1]}"] if i + 1 < len(tokens) else [])):
            idx = vocab.get(ngram)
            if idx is not None and coefficients[idx] > 0:
                candidates.append({"word": ngram, "weight": round(float(coefficients[idx]), 3)})

    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x["weight"], reverse=True):
        if c["word"] not in seen:
            seen.add(c["word"])
            unique.append(c)

    return unique[:top_n]

# ================== 10. Quick Test ==================
test_cases = [
    # Should NOT be toxic — the negation must be respected
    ("you are not stupid",       False),
    ("I will not hurt you",      False),
    ("don't be an idiot? no!",   False),
    ("this is not a bad idea",   False),
    # Should be toxic
    ("you are stupid and ugly",  True),
    ("I will kill you",          True),
    ("you idiot",                True),
    # Neutral
    ("Thank you for your help!", False),
    ("This is a great idea",     False),
]

THRESHOLD = 0.4

print("\n🔍 Quick Test Predictions:\n")
correct = 0
for text, expected in test_cases:
    processed  = preprocess_with_negation(text)
    pred_prob  = pipeline.predict_proba([processed])[0][1]
    is_toxic   = pred_prob >= THRESHOLD
    bad_words  = extract_toxic_words(text, pipeline)
    status     = "✅" if is_toxic == expected else "❌"

    correct += int(is_toxic == expected)

    print("=" * 60)
    print(f"{status} TEXT:      {text}")
    print(f"   PROCESSED: {processed}")
    print(f"   TOXIC:     {is_toxic}  (expected {expected})")
    print(f"   SCORE:     {pred_prob:.2%}")
    print(f"   BAD WORDS: {bad_words}")

print(f"\n🎯 Quick-test accuracy: {correct}/{len(test_cases)}")

# ================== 11. (Optional) Upgrade Path ==================
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 OPTIONAL FREE UPGRADE — HuggingFace Transformer
   If you want even better context understanding (handles
   sarcasm, complex negation, slang), run this once:

   pip install transformers torch

   Then replace pipeline.predict_proba([processed]) with:

   from transformers import pipeline as hf_pipeline
   classifier = hf_pipeline(
       "text-classification",
       model="martin-ha/toxic-comment-model"  # free, no API key
   )
   result = classifier("you are not stupid")
   # → [{'label': 'non-toxic', 'score': 0.99}]

   The model downloads once (~250 MB) and runs fully locally.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")