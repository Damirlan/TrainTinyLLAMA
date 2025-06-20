import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# === Константы ===
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_merged"
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./saiga-mistral-lora-output"

# === LoRA настройки ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# === Загрузка модели и токенизатора ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # важно для корректной генерации

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16  # если не поддерживается — замените на float16
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False  # отключаем cache для совместимости с LoRA

# === Загрузка и токенизация датасета ===
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize(example):
    full_text = f"{example['prompt']}{example['response']}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)

dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

# === Аргументы тренировки ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1.5,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01,
    bf16=True,  # если не поддерживается, замени на fp16=True
    fp16=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    report_to="none",
    save_safetensors=True,
    load_best_model_at_end=False
)

# === SFTTrainer ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# === Автоматическое возобновление с последнего чекпоинта ===
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [ckpt for ckpt in os.listdir(OUTPUT_DIR) if ckpt.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
        print(f"⚠️ Обнаружен чекпоинт: {last_checkpoint}. Продолжаем обучение.")

trainer.train(resume_from_checkpoint=last_checkpoint)

# === Сохранение финальной модели ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

