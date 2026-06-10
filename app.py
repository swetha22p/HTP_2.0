import unsloth  # MUST be first

from unsloth import FastLanguageModel
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# ==================================================
# PATHS
# ==================================================

DATA_FILE          = "AG/llm_training_data.jsonl"
FORMATTED_SAVE_DIR = "AG/formatted_data_v2"
CHECKPOINT_DIR     = "AG/checkpoints_v2"

# Adapter-only save (small — ~50 MB instead of ~4 GB)
ADAPTER_SAVE_DIR   = "AG/qlora_adapters_qwen2.5_7b"

# Optional: merged model path (only needed for inference/deployment)
# Leave as None to skip merging entirely and save time/space.
MERGED_MODEL_DIR   = None   # set to a path string to enable merge+save

# ==================================================
# T4 SETTINGS
# ==================================================

MAX_SEQ_LENGTH = 2048   # safe for T4 16 GB; bump to 1536/2048 only after OOM-free run

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = (
    "नीचे दिए गए यूनिवर्सल स्ट्रक्चर रिप्रेजेंटेशन (USR) स्ट्रक्चर से एक नेचुरल लगने वाला हिंदी वाक्य बनाएं।"
    "ज़रूरी: नीचे दिए गए USR सेगमेंट का इस्तेमाल करके एक पूरा वाक्य बनाएं।"
    "आपको हर USR सेगमेंट का सही कंटेंट बताए गए क्रम में इस्तेमाल करना होगा।"
    "ऐसी कोई भी एक्स्ट्रा जानकारी न जोड़ें जो दिए गए स्ट्रक्चर में मौजूद न हो।"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("📂 Loading dataset...")

raw_dataset   = load_dataset("json", data_files=DATA_FILE, split="train")
train_testval = raw_dataset.train_test_split(test_size=0.10, seed=3407)
test_val      = train_testval["test"].train_test_split(test_size=0.50, seed=3407)

split_dataset = DatasetDict({
    "train" : train_testval["train"],
    "val"   : test_val["train"],
    "test"  : test_val["test"],
})

print(split_dataset)

# ==================================================
# LOAD BASE MODEL IN 4-BIT  ← this is the "Q" in QLoRA
# The base model weights are quantized to 4-bit and
# are COMPLETELY FROZEN throughout training.
# Only the small LoRA adapter weights (added next)
# will receive gradient updates.
# ==================================================

print("📥 Loading base model in 4-bit (frozen)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length   = MAX_SEQ_LENGTH,
    dtype            = None,        # auto → FP16 on T4 (no BF16 on Turing arch)
    load_in_4bit     = True,        # quantize base model → ~4-5 GB VRAM
)

# ==================================================
# ATTACH LORA ADAPTERS  ← this is the "LoRA" in QLoRA
#
# Only these adapter parameters are trainable.
# The 4-bit base model is untouched.
#
# Trainable params ≈ 2 × r × d_model × num_layers
# For Qwen2.5-7B with r=16: ~40 M params vs 7B total
# That's < 0.6% of the model — very fast, very low VRAM.
# ==================================================

print("🔧 Attaching LoRA adapters (only these will train)...")

model = FastLanguageModel.get_peft_model(
    model,
    r                        = 16,
    target_modules           = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha               = 16,
    lora_dropout             = 0,
    bias                     = "none",
    use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM on T4
    random_state             = 3407,
)

# Confirm only adapters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
      f"({100*trainable/total:.2f}%)")

# ==================================================
# FORMATTING FUNCTION
# ==================================================

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

# ==================================================
# FORMAT & SAVE DATASETS
# ==================================================

print("🔄 Formatting datasets...")

train_dataset = split_dataset["train"].map(formatting_prompts_func, batched=True)
val_dataset   = split_dataset["val"].map(formatting_prompts_func,   batched=True)
test_dataset  = split_dataset["test"].map(formatting_prompts_func,  batched=True)

print(f"Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")
print("\nSample formatted text (first 500 chars):")
print(train_dataset[0]["text"][:500])

os.makedirs(FORMATTED_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR,     exist_ok=True)
os.makedirs(ADAPTER_SAVE_DIR,   exist_ok=True)

train_dataset.save_to_disk(os.path.join(FORMATTED_SAVE_DIR, "train_hf"))
val_dataset.save_to_disk(  os.path.join(FORMATTED_SAVE_DIR, "val_hf"))
test_dataset.save_to_disk( os.path.join(FORMATTED_SAVE_DIR, "test_hf"))
print("✅ Datasets saved")

# ==================================================
# TRAINER
# ==================================================

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
        packing                = True,

        # T4: batch=1 + accum=8 keeps effective batch=8
        # while halving per-step activation memory vs batch=2
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,

        num_train_epochs       = 5,
        warmup_steps           = 30,
        learning_rate          = 2e-4,
        weight_decay           = 0.01,

        optim                  = "adamw_8bit",
        lr_scheduler_type      = "linear",

        # T4 = Turing arch, no BF16 — auto-selects FP16
        fp16                   = not torch.cuda.is_bf16_supported(),
        bf16                   = torch.cuda.is_bf16_supported(),

        logging_steps          = 10,

        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        save_total_limit       = 3,

        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,

        # Prevents CPU-RAM pressure on Colab (~12 GB system RAM)
        dataloader_pin_memory  = False,

        seed                   = 3407,
    ),
)

# ==================================================
# TRAIN (resume if checkpoint exists)
# ==================================================

print("🚀 Starting QLoRA training (adapters only)...")

checkpoint_exists = (
    os.path.exists(CHECKPOINT_DIR) and
    any(f.startswith("checkpoint-") for f in os.listdir(CHECKPOINT_DIR))
)

if checkpoint_exists:
    print("📂 Resuming from checkpoint...")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("🆕 Fresh training run")
    trainer_stats = trainer.train()

print("✅ Training complete")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ==================================================
# SAVE — ADAPTERS ONLY (~50 MB, not the full model)
#
# This saves only the trained LoRA weights.
# To run inference later, load the original base model
# and these adapters together (see inference note below).
# ==================================================

print(f"💾 Saving LoRA adapters to {ADAPTER_SAVE_DIR} ...")
model.save_pretrained(ADAPTER_SAVE_DIR)      # saves adapter_config.json + adapter weights
tokenizer.save_pretrained(ADAPTER_SAVE_DIR)  # saves tokenizer alongside adapters
print("🎉 Adapters saved")

# --------------------------------------------------
# OPTIONAL: Merge adapters into base model and save
# full merged weights (needed for deployment/GGUF).
# This requires extra VRAM — skip on T4 if OOM.
# --------------------------------------------------

if MERGED_MODEL_DIR:
    print(f"🔀 Merging adapters into base model → {MERGED_MODEL_DIR} ...")
    merged_model = model.merge_and_unload()   # fuses adapters into base weights
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print("✅ Merged model saved")
else:
    print("ℹ️  Merge skipped (MERGED_MODEL_DIR=None). "
          "To load for inference use:\n"
          "  model = PeftModel.from_pretrained(base_model, ADAPTER_SAVE_DIR)")

# ==================================================
# FINAL EVAL LOSS
# ==================================================

print("🧪 Final evaluation on test set...")
results = trainer.evaluate(test_dataset)
print(results)

# ==================================================
# INFERENCE USAGE REMINDER
# ==================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 How to load your trained model later:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from unsloth import FastLanguageModel
from peft import PeftModel

# 1. Load original 4-bit base model (frozen)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
)

# 2. Load your trained adapters on top
model = PeftModel.from_pretrained(model, ADAPTER_SAVE_DIR)
FastLanguageModel.for_inference(model)

# 3. Run inference
inputs = tokenizer("your USR input", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
