# server.py  (rename script.py → server.py if you want — it's clearer)
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import joblib
import torch
import os
import uvicorn  # ← ADD THIS (needed for manual port binding on Render)

# ================== CONFIG ==================
MODEL_DIR = os.path.abspath("models/rewriter_model_fp16")   # your compressed fp16 folder

TOXICITY_MODEL_PATH = "models/toxicity_classifier/classifier.pkl"

# ================== Load models at startup ==================
print("Loading toxicity classifier...")
toxicity_model = joblib.load(TOXICITY_MODEL_PATH)
print("Toxicity classifier loaded!")

print("Loading T5 rewriter (fp16 mode)...")
rewriter_model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)

rewriter_tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
rewriter_model.eval()

print("Rewriter model loaded successfully!")

# ================== FastAPI setup ==================
app = FastAPI(title="Text Processing API")

class RequestBody(BaseModel):
    text: str

@app.post("/process/")
async def process_text(body: RequestBody):
    text = body.text.strip()

    # Toxicity check
    pred_prob = toxicity_model.predict_proba([text])[0][1]
    is_toxic = pred_prob >= 0.4

    if not is_toxic:
        return {"toxicity": False, "original": text}

    # Rewrite if toxic
    prefix = "rewrite: "
    input_text = prefix + text

    inputs = rewriter_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    with torch.no_grad():
        outputs = rewriter_model.generate(
            inputs.input_ids.to("cpu"),
            attention_mask=inputs.attention_mask.to("cpu"),
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
            repetition_penalty=1.2
        )

    rewritten = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)
    rewritten = rewritten.replace(prefix, "").strip()

    return {"toxicity": True, "rewrite": rewritten, "original": text}

# ================== Render-friendly run block ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))           # ← Render sets PORT
    uvicorn.run(
        "server:app",                                      # ← important: "filename:app"
        host="0.0.0.0",                                    # ← must be 0.0.0.0
        port=port,
        workers=1,                                         # ← free tier: only 1 worker!
        log_level="info"
    )