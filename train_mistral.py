import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

# 1. Настройки
model_name = "IlyaGusev/saiga_mistral_7b_merged"
output_dir = "./saiga-mistral-reply-lora"
csv_path = "reviews_with_replies.csv"

# 2. Загрузка данных
df = pd.read_csv(csv_path)

# 3. Форматируем данные
def format_row(row):
    prompt = (
        f"<s>Пользователь:\n"
        f"Рейтинг: {row['rating']}\n"
        f"Товар: {row['product_name']}\n"
        f"Отзыв: {row['review_text']}\n"
        f"Ассистент:"
    )
    completion = row['reply']
    return {"text": f"{prompt} {completion}</s>"}

formatted = [format_row(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(formatted)

# 4. Токенизация
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# 5. Настройка LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 6. Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model = get_peft_model(model, peft_config)

# 7. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 8. Аргументы обучения
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none"
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 10. Обучение
trainer.train()

# 11. Сохранение модели
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

