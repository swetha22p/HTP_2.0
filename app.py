import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# ==================================================
# PATHS
# ==================================================
DATA_FILE = "AG/llm_training_data.jsonl"

FORMATTED_SAVE_DIR = "AG/formatted_data_clean1"

CHECKPOINT_DIR = "AG/checkpoints_n1"

FINAL_MODEL_DIR = "AG/finetuned_model_qwen2.5_7b_FINAL"

# ==================================================
# SYSTEM PROMPT
# ==================================================
SYSTEM_PROMPT = """
 "आप एक हिंदी वाक्य जनरेटर हैं जो USR (Universal Semantic Representation) सिमेंटिक डेटा से स्वाभाविक और व्याकरणिक रूप से सही हिंदी वाक्य बनाता है।\n\n"

    "इनपुट फ़ॉर्मेट की समझ\n\n"

    "USR की प्रत्येक पंक्ति वाक्य में एक अवधारणा (concept) का प्रतिनिधित्व करती है।\n\n"

    "कॉलम 1: अवधारणा / शाब्दिक इकाई — इसमें वास्तविक सिमेंटिक इकाई होती है (जैसे, वायु_1, अत्यधिक_2)।\n"

    "कॉलम 2: इंडेक्स — अवधारणा का अद्वितीय पहचानकर्ता, जिसका उपयोग निर्भरता संबंधों (dependency relations) को जोड़ने के लिए किया जाता है (जैसे, 1, 2, 3)।\n"

    "कॉलम 3: सिमेंटिक श्रेणी — अवधारणा का सिमेंटिक प्रकार (जैसे, anim, place, obj)।\n"

    "कॉलम 4: मॉर्फो-सिमेंटिक जानकारी — व्याकरणिक जानकारी जैसे वचन (pl), लिंग, TAM, और अन्य रूपात्मक विशेषताएँ।\n"

    "कॉलम 5: निर्भरता संबंध — यह बताता है कि वर्तमान अवधारणा किसी दूसरी अवधारणा से कैसे संबंधित है, इसके लिए target_index:relation फ़ॉर्मेट का उपयोग किया जाता है।\n\n"

    "उदाहरण:\n"

    "\"7:k1\" → वर्तमान अवधारणा, इंडेक्स 7 पर स्थित अवधारणा का कर्ता (subject) है।\n"

    "\"9:k2\" → वर्तमान अवधारणा, इंडेक्स 9 पर स्थित अवधारणा का कर्म (object) है।\n"

    "\"14:r6\" → वर्तमान अवधारणा, अवधारणा 14 को संशोधित करती है या उस पर स्वामित्व दर्शाती है।\n"

    "\"15:k7\" → वर्तमान अवधारणा, अवधारणा 15 के लिए स्थान/समय व्यक्त करती है।\n"

    "\"quant\" → मात्रा संशोधक (quantity modifier)।\n"

    "\"rt\", \"rcdelim\", \"vkvn\", \"verbalizer\" आदि की व्याख्या USR निर्भरता परंपराओं के अनुसार की जानी चाहिए।\n\n"

    "कॉलम 6: सह-संदर्भ / विमर्श जानकारी — यह खंडों (segments) के बीच के संदर्भों या विमर्श संबंधों को इंगित करता है (जैसे, coref, pariNAma)। सर्वनामों को स्पष्ट करने और वाक्य में निरंतरता बनाए रखने के लिए इसका उपयोग करें।\n"

    "कॉलम 7: वक्ता का दृष्टिकोण — इसमें परिप्रेक्ष्य संबंधी जानकारी होती है, जैसे ज़ोर (emphasis) या दूरी (distal आदि)।\n"

    "कॉलम 8: दायरा (Scope) — जहाँ लागू हो, वहाँ यह सिमेंटिक दायरे को परिभाषित करता है।\n"

    "कॉलम 9: वाक्य का प्रकार — यह निर्धारित करता है कि आउटपुट किस प्रकार का होना चाहिए: कथनात्मक, प्रश्नवाचक, नकारात्मक आदि।\n\n"

    "निर्माण निर्देश\n\n"

    "1. केवल USR में मौजूद जानकारी का ही उपयोग करें।\n"

    "2. कोई नए तथ्य, उदाहरण, या स्पष्टीकरण न जोड़ें और न ही कोई छूटी हुई जानकारी मनगढ़ंत रूप से शामिल करें।\n"

    "3. USR में दी गई कोई भी जानकारी न छोड़ें।\n"

    "4. USR की सभी पंक्तियों को ठीक उसी क्रम में संसाधित करें जिस क्रम में वे दी गई हैं।\n"

    "5. k1, k2, k7, r6, rt, rcdelim, quant, vkvn, verbalizer आदि जैसे निर्भरता संबंधों का उपयोग करके वाक्य का अर्थ निकालें।\n"

    "6. स्वाभाविक और प्रवाहपूर्ण हिंदी लिखें, न कि शब्द-दर-शब्द अनुवाद।\n"

    "7. विमर्श (discourse) और सह-संदर्भ (coreference) को सही ढंग से हल करें, और संज्ञाओं की बार-बार पुनरावृत्ति करने के बजाय उचित सर्वनामों का उपयोग करें।\n"

    "8. सभी अवधारणाओं को एक सुसंगत वाक्य में पिरोएँ और उनके तार्किक प्रवाह को बनाए रखें।\n"

    "9. वाक्य-खंडों के बीच के संबंधों को बनाए रखें, न कि उन्हें अलग-अलग टुकड़ों के रूप में लिखें।\n"

    "10. संख्या, अन्वय (agreement), काल-पक्ष-वृत्ति (TAM), और व्याकरणिक रूप जैसी रूप-अर्थ संबंधी जानकारी का ध्यान रखें।\n"

    "11. जब भी निर्भरता संबंधों द्वारा संकेत दिया गया हो, तो सापेक्ष-सहसंबंधी संरचनाओं और अंतर्निहित उपवाक्यों को बनाए रखें।\n"

    "12. वाक्य के प्रकार का सख्ती से पालन करें (उदाहरण: प्रश्नवाचक → प्रश्न के रूप में)।\n"

    "13. आउटपुट के रूप में केवल अंतिम हिंदी वाक्य दें, और उसके अलावा कुछ भी नहीं।"
"""

# ==================================================
# LOAD & SPLIT DATA
# ==================================================
print("📂 Loading dataset...")

raw_dataset = load_dataset(
    "json",
    data_files=DATA_FILE,
    split="train"
)

train_testval = raw_dataset.train_test_split(
    test_size=0.1,
    seed=3407
)

test_val = train_testval["test"].train_test_split(
    test_size=0.5,
    seed=3407
)

split_dataset = DatasetDict({
    "train": train_testval["train"],
    "val": test_val["train"],
    "test": test_val["test"]
})

print(split_dataset)

# ==================================================
# FORMATTING FUNCTION
# ==================================================
def formatting_prompts_func(examples):
    texts = []

    ids = examples.get(
        "id",
        ["unknown"] * len(examples["input"])
    )

    for segment_id, input_text, output_text in zip(
        ids,
        examples["input"],
        examples["output"]
    ):

        cleaned_output_lines = []

        for line in output_text.splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) >= 6:
                cleaned_line = (
                    f"{parts[0]}\t"
                    f"{parts[1]}\t"
                    f"{parts[4]}\t"
                    f"{parts[5]}"
                )
                cleaned_output_lines.append(cleaned_line)

            else:
                cleaned_output_lines.append(line)

        processed_output = "\n".join(cleaned_output_lines)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": processed_output
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        texts.append(text)

    return {"text": texts}

# ==================================================
# FORMAT DATASETS
# ==================================================
print("🔄 Formatting datasets...")

train_dataset = split_dataset["train"].map(
    formatting_prompts_func,
    batched=True
)

val_dataset = split_dataset["val"].map(
    formatting_prompts_func,
    batched=True
)

test_dataset = split_dataset["test"].map(
    formatting_prompts_func,
    batched=True
)

print(
    f"✅ Train: {len(train_dataset)} | "
    f"Val: {len(val_dataset)} | "
    f"Test: {len(test_dataset)}"
)

# ==================================================
# SAVE FORMATTED DATA
# ==================================================
os.makedirs(FORMATTED_SAVE_DIR, exist_ok=True)

train_dataset.save_to_disk(
    os.path.join(FORMATTED_SAVE_DIR, "train_hf")
)

val_dataset.save_to_disk(
    os.path.join(FORMATTED_SAVE_DIR, "val_hf")
)

test_dataset.save_to_disk(
    os.path.join(FORMATTED_SAVE_DIR, "test_hf")
)

train_dataset.to_json(
    os.path.join(FORMATTED_SAVE_DIR, "train.jsonl")
)

val_dataset.to_json(
    os.path.join(FORMATTED_SAVE_DIR, "val.jsonl")
)

test_dataset.to_json(
    os.path.join(FORMATTED_SAVE_DIR, "test.jsonl")
)

print("✅ Formatted datasets saved.")

# ==================================================
# TRAINER
# ==================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ],

    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        warmup_steps=30,
        num_train_epochs=5,

        learning_rate=2e-4,

        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        logging_steps=10,

        optim="adamw_8bit",

        weight_decay=0.01,
        lr_scheduler_type="linear",

        seed=3407,

        output_dir=CHECKPOINT_DIR,

        dataset_text_field="text",

        packing=True,
        max_seq_length=1024,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=3,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)

# ==================================================
# RESUME IF CHECKPOINT EXISTS
# ==================================================
print("🚀 Preparing training...")

checkpoints_exist = False

if os.path.exists(CHECKPOINT_DIR):

    checkpoint_folders = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("checkpoint-")
    ]

    if len(checkpoint_folders) > 0:
        checkpoints_exist = True

if checkpoints_exist:

    print("📂 Checkpoint found.")
    print("🔄 Resuming training...")

    trainer_stats = trainer.train(
        resume_from_checkpoint=True
    )

else:

    print("🆕 Starting fresh training...")

    trainer_stats = trainer.train()

# ==================================================
# TRAINING COMPLETE
# ==================================================
print("✅ Training completed.")

print(
    f"📊 Peak VRAM Used: "
    f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB"
)

# ==================================================
# SAVE BEST MODEL
# ==================================================
print(f"💾 Saving model to {FINAL_MODEL_DIR}")

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("🎉 Final model saved successfully!")

# ==================================================
# OPTIONAL EVALUATION
# ==================================================
print("🧪 Running final evaluation...")

results = trainer.evaluate(test_dataset)

print(results)
