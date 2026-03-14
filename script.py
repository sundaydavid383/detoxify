# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import joblib
import torch
import os

# ================== CONFIG ==================
# ← Use your compressed fp16 folder here (after running the compress_model.py script)
MODEL_DIR = os.path.abspath("models/rewriter_model_fp16")   # ← CHANGE TO YOUR COMPRESSED FOLDER

# Toxicity classifier (this is small, ~few MB, no change needed)
TOXICITY_MODEL_PATH = "models/toxicity_classifier/classifier.pkl"

# ================== Load models at startup ==================
print("Loading toxicity classifier...")
toxicity_model = joblib.load(TOXICITY_MODEL_PATH)
print("Toxicity classifier loaded!")

print("Loading T5 rewriter (fp16 mode)...")
rewriter_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,           # ← Critical: ~50% less RAM
    low_cpu_mem_usage=True,              # Loads layer-by-layer, avoids memory spikes
    device_map="cpu"                     # Explicit CPU (Render has no GPU)
)

rewriter_tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
rewriter_model.eval()                    # Inference mode (no gradients)

print("Rewriter model loaded successfully!")

# ================== FastAPI setup ==================
app = FastAPI(title="Text Processing API")

class RequestBody(BaseModel):
    text: str

@app.post("/process/")
async def process_text(body: RequestBody):
    text = body.text.strip()

    # ---- Step 1: Toxicity check (very fast & low memory) ----
    pred_prob = toxicity_model.predict_proba([text])[0][1]
    is_toxic = pred_prob >= 0.4

    if not is_toxic:
        return {"toxicity": False, "original": text}

    # ---- Step 2: Rewrite if toxic ----
    prefix = "rewrite: "   # ← Make sure this matches exactly what you used during training!
    input_text = prefix + text

    inputs = rewriter_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    with torch.no_grad():  # ← Saves a bit of memory (no gradient tracking)
        outputs = rewriter_model.generate(
            inputs.input_ids.to("cpu"),
            attention_mask=inputs.attention_mask.to("cpu"),
            max_new_tokens=128,      # ← IMPORTANT: shorter = much faster + lower peak memory
            num_beams=1,             # Greedy decoding = fastest on CPU
            do_sample=False,
            early_stopping=True,
            repetition_penalty=1.2   # Optional: reduces repetition
        )

    rewritten = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean prefix if model repeats it
    rewritten = rewritten.replace(prefix, "").strip()

    return {"toxicity": True, "rewrite": rewritten, "original": text}

# ================== Notes for Render deployment ==================
# Run locally for testing:   uvicorn server:app --reload --port 8000
# For Render: create a new Web Service → Python → set start command:
#   uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1
# (only 1 worker → saves memory on free tier)