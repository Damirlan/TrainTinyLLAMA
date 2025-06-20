import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# === Константы ===
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_merged"
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./saiga-mistral-lora-output"

# === 8-bit квантование (BitsAndBytes) ===
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

# === Загрузка и токенизация датасета ===
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize(example):
    full_text = f"{example['prompt']}{example['response']}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)

dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

# === Аргументы тренировки ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

# === Обучение ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

