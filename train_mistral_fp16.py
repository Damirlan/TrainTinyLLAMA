import os
import gc
import pandas as pd
import torch

from trl import SFTTrainer

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

# 🔁 Очистка памяти
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🔧 Параметры
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_merged"
CSV_PATH = "reviews_with_replies.csv"
OUTPUT_DIR = "./saiga-mistral-fp16"
MAX_LENGTH = 512

# 📊 Загрузка данных
df = pd.read_csv(CSV_PATH)

# 🧾 Форматирование как диалог
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

# 🔠 Токенизация с добавлением labels
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    result = tokenizer(
        example["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )
    result["labels"] = result["input_ids"].copy()
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
        "labels": result["labels"]
    }

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 🧠 LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 📥 Загрузка модели (float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

print("📦 Проверяем вход:")
sample = tokenized_dataset[0]

input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to("cuda")
attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long).to("cuda")
labels = torch.tensor([sample["labels"]], dtype=torch.long).to("cuda")

model.eval()
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    print("✅ ПОЛУЧЕННЫЙ LOSS:", out.loss)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ⚙️ Аргументы обучения (с label_names!)
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
training_args.model_max_length = MAX_LENGTH  # 👈 добавь это вручную


# 🚀 Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    dataset_text_field=None,  # уже токенизировано
    packing=False
)

# 🧠 Обучение
trainer.train()

# 💾 Сохранение
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

