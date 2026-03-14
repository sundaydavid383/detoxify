# rewrite_trainer_tsv_fixed.py
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# ================== 1. Config ==================
model_name = "google/flan-t5-small"
dataset_path = "train.tsv"  # ParadeTox TSV
save_dir = "models/rewriter_model_cpu_light"
num_train_epochs = 1        # reduce for CPU / free-tier
batch_size = 2              # keep small for CPU
max_input_length = 64       # shorter = faster
max_target_length = 64
sample_fraction = 0.2       # take 20% of dataset for testing

# ================== 2. Load Dataset ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "train.tsv")
df = pd.read_csv(dataset_path, sep='\t')
df = df.fillna("")

# Sample smaller dataset for fast CPU training
df = df.sample(frac=sample_fraction, random_state=42)

# Rename columns to match expected names
df = df.rename(columns={
    "en_toxic_comment": "input_text",
    "en_neutral_comment": "target_text"
})

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# ================== 3. Load Tokenizer & Model ==================
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ================== 4. Tokenize Dataset ==================
def tokenize(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=max_input_length)
    targets = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=max_target_length)
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = targets["input_ids"]
    return batch

dataset = dataset.map(tokenize, batched=True, batch_size=8)
dataset = dataset.remove_columns(["input_text", "target_text", "__index_level_0__"])

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ================== 5. Data Collator ==================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ================== 6. Training Arguments ==================
training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    fp16=False,
    report_to="none",
    remove_unused_columns=True
)

# ================== 7. Trainer ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ================== 8. Train ==================
print("🏋️‍♂️ Starting lightweight CPU training...")
trainer.train()

# ================== 9. Save Model ==================
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\n✅ Model saved at {save_dir}")

# ================== 10. Quick Test ==================
print("\n🔍 Quick test:")
test_texts = [
    "You are stupid and ugly",
    "I hate this",
    "Thank you for your help",
    "This idea is terrible"
]

for text in test_texts:
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64)
    rewritten = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"→ Original: {text}")
    print(f"  Rewritten: {rewritten}\n")