import os
import gc
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

# 🔁 Очистка памяти
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🔧 Параметры
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_merged"
CSV_PATH = "reviews_with_replies.csv"
OUTPUT_DIR = "./saiga-mistral-reply-lora"
MAX_LENGTH = 512

# 📊 Загрузка данных
df = pd.read_csv(CSV_PATH)

# 🧾 Форматируем как диалог
def format_row(row):
    prompt = (
        f"<s>Пользователь:\n"
        f"Рейтинг: {row['rating']}\n"
        f"Товар: {row['product_name']}\n"
        f"Отзыв: {row['review_text']}\n"
        f"Ассистент:"
    )
    return {"text": f"{prompt} {row['reply']}</s>"}

formatted = [format_row(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(formatted)

# 🔠 Токенизация
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# 🧠 LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 🧠 BitsAndBytes 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 📥 Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

# 📚 Коллатор
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ⚙️ Аргументы обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    report_to="none"
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 🧠 Обучение
trainer.train()

# 💾 Сохранение
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

