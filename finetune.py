import unsloth  # MUST be first

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
import os
import json
import torch

# ==================================================
# PATHS — update these to your AIRAWAT paths
# ==================================================

TRAIN_FILE         = "AG/train.jsonl"
VAL_FILE           = "AG/val.jsonl"
TEST_FILE          = "AG/test.jsonl"

FORMATTED_SAVE_DIR = "AG/formatted_data"
CHECKPOINT_DIR     = "AG/checkpoints"
ADAPTER_SAVE_DIR   = "AG/qlora_adapters_qwen2.5_7b"
MERGED_MODEL_DIR   = None   # set a path to merge after training, else None

# ==================================================
# SETTINGS — updated for A100
# ==================================================

MAX_SEQ_LENGTH = 8192           # A100 handles this comfortably
LOAD_IN_4BIT   = False          # A100 has enough VRAM — no need for 4bit
DTYPE          = torch.bfloat16 # A100 natively supports bf16

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = (
    "नीचे दी गई सार्वभौमिक अर्थपरक निरूपण (USR) संरचना का उपयोग करके "
    "एक स्वाभाविक लगने वाला हिंदी वाक्य बनाइए। "
    "महत्वपूर्ण: नीचे दिए गए USR खंडों का उपयोग करके एक पूर्ण वाक्य बनाइए। "
    "प्रत्येक USR खंड की सही सामग्री का उपयोग निर्दिष्ट क्रम में करना अनिवार्य है। "
    "दी गई संरचना में मौजूद जानकारी के अलावा कोई अतिरिक्त जानकारी न जोड़ें।"
)

# ==================================================
# STEP 1 — LOAD JSONL FILES
# Uses your chapter-level split from split_data.py
# ==================================================

print("\n" + "="*55)
print("STEP 1 — Loading datasets")
print("="*55)

def load_jsonl_as_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)

train_raw = load_jsonl_as_dataset(TRAIN_FILE)
val_raw   = load_jsonl_as_dataset(VAL_FILE)
test_raw  = load_jsonl_as_dataset(TEST_FILE)

print(f"Train : {len(train_raw)} windows")
print(f"Val   : {len(val_raw)} windows")
print(f"Test  : {len(test_raw)} windows")

# ==================================================
# STEP 2 — LOAD BASE MODEL
# ==================================================

print("\n" + "="*55)
print("STEP 2 — Loading base model")
print("="*55)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

print(f"✅ Model loaded | dtype={DTYPE} | 4bit={LOAD_IN_4BIT}")

# ==================================================
# STEP 3 — ATTACH LORA ADAPTERS
# ==================================================

print("\n" + "="*55)
print("STEP 3 — Attaching LoRA adapters")
print("="*55)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha                 = 16,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

# ==================================================
# STEP 4 — FORMAT DATASETS
# ==================================================

print("\n" + "="*55)
print("STEP 4 — Formatting datasets")
print("="*55)

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for input_text, output_text in zip(examples["input"], examples["output"]):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.endswith(EOS_TOKEN):
            text += EOS_TOKEN
        texts.append(text)
    return {"text": texts}

train_dataset = train_raw.map(formatting_prompts_func, batched=True)
val_dataset   = val_raw.map(formatting_prompts_func,   batched=True)
test_dataset  = test_raw.map(formatting_prompts_func,  batched=True)

print(f"✅ Formatted | Train={len(train_dataset)} Val={len(val_dataset)} Test={len(test_dataset)}")
print("\nSample (first 500 chars):")
print(train_dataset[0]["text"][:500])

# Save formatted datasets to disk (useful for resuming)
os.makedirs(FORMATTED_SAVE_DIR, exist_ok=True)
train_dataset.save_to_disk(os.path.join(FORMATTED_SAVE_DIR, "train_hf"))
val_dataset.save_to_disk(  os.path.join(FORMATTED_SAVE_DIR, "val_hf"))
test_dataset.save_to_disk( os.path.join(FORMATTED_SAVE_DIR, "test_hf"))
print(f"✅ Saved formatted datasets to {FORMATTED_SAVE_DIR}")

# ==================================================
# STEP 5 — TRAINER
# ==================================================

print("\n" + "="*55)
print("STEP 5 — Setting up trainer")
print("="*55)

os.makedirs(CHECKPOINT_DIR,   exist_ok=True)
os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],

    args = SFTConfig(
        output_dir             = CHECKPOINT_DIR,

        dataset_text_field     = "text",
        max_seq_length         = MAX_SEQ_LENGTH,
        packing                = True,      # packs short examples efficiently

        # A100 — larger batch than T4
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,    # effective batch = 8

        num_train_epochs       = 5,
        warmup_steps           = 30,
        learning_rate          = 2e-4,
        weight_decay           = 0.01,

        optim                  = "adamw_8bit",
        lr_scheduler_type      = "linear",

        fp16                   = False,     # A100 uses bf16 not fp16
        bf16                   = True,

        logging_steps          = 10,

        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        save_total_limit       = 3,         # keep only last 3 checkpoints

        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,

        dataloader_pin_memory  = False,
        seed                   = 3407,
    ),
)

# ==================================================
# STEP 6 — TRAIN
# ==================================================

print("\n" + "="*55)
print("STEP 6 — Training")
print("="*55)

checkpoint_exists = (
    os.path.exists(CHECKPOINT_DIR) and
    any(f.startswith("checkpoint-") for f in os.listdir(CHECKPOINT_DIR))
)

if checkpoint_exists:
    print("📂 Resuming from existing checkpoint...")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("🆕 Fresh training run")
    trainer_stats = trainer.train()

print("✅ Training complete")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ==================================================
# STEP 7 — SAVE ADAPTERS
# ==================================================

print("\n" + "="*55)
print("STEP 7 — Saving adapters")
print("="*55)

model.save_pretrained(ADAPTER_SAVE_DIR)
tokenizer.save_pretrained(ADAPTER_SAVE_DIR)
print(f"✅ Adapters saved to {ADAPTER_SAVE_DIR}")

# ==================================================
# STEP 8 — OPTIONAL MERGE
# ==================================================

if MERGED_MODEL_DIR:
    print(f"\n🔀 Merging adapters → {MERGED_MODEL_DIR}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print("✅ Merged model saved")
else:
    print("\nℹ️  Merge skipped (MERGED_MODEL_DIR=None)")

# ==================================================
# STEP 9 — FINAL EVAL ON TEST SET
# ==================================================

print("\n" + "="*55)
print("STEP 9 — Final evaluation on test set")
print("="*55)

trainer.eval_dataset = test_dataset
test_results = trainer.evaluate()
print("Test set results:")
print(test_results)

# ==================================================
# INFERENCE REMINDER
# ==================================================

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 How to load your trained model later:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = {MAX_SEQ_LENGTH},
    load_in_4bit   = False,
)
model = PeftModel.from_pretrained(model, "{ADAPTER_SAVE_DIR}")
FastLanguageModel.for_inference(model)

inputs  = tokenizer("your USR input", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
