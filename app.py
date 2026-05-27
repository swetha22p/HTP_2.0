import os
import torch
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ==================================================
# 1. DEFINE YOUR PERSISTENT CHECKPOINT FOLDER
# ==================================================
checkpoint_dir   = "/content/drive/MyDrive/AG/checkpoints_n1"
final_drive_path = "/content/drive/MyDrive/AG/finetuned_model_qwen2.5_7b_FINAL"

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],

    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps                = 30,
        num_train_epochs            = 5,
        learning_rate               = 2e-4,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = checkpoint_dir,
        dataset_text_field          = "text",
        packing                     = True,
        max_seq_length              = 1024,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
    ),
)

# ==================================================
# 2. SMART TRAINING / RESUMING LOGIC
# ==================================================
print("🚀 Preparing to train...")

checkpoints_exist = False
if os.path.exists(checkpoint_dir):
    checkpoint_folders = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
    if len(checkpoint_folders) > 0:
        checkpoints_exist = True

if checkpoints_exist:
    print(f"📂 Found existing checkpoints in {checkpoint_dir}.")
    print("🔄 RESUMING training from the latest checkpoint...")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("🆕 No existing checkpoints found. STARTING fresh training...")
    trainer_stats = trainer.train()

print("✅ Training complete! Best model loaded into memory.")
print(f"📊 Peak VRAM used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ==================================================
# 3. SAVE FINAL BEST MODEL
# ==================================================
print(f"💾 Saving final model to: {final_drive_path}...")
model.save_pretrained(final_drive_path)
tokenizer.save_pretrained(final_drive_path)
print("🎉 Model successfully saved!")
