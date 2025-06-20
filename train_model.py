import os
import pandas as pd
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Расширение управления CUDA-памятью (опционально, но полезно)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Загрузка данных
print("Загрузка CSV...")
df = pd.read_csv("reviews_with_replies.csv", encoding="utf-8")
print(f"Данные загружены: {len(df)} записей")

# 2. Форматирование примеров
def format_example(example):
    prompt = (
        f"Рейтинг: {example['rating']}\n"
        f"Товар: {example['product_name']}\n"
        f"Отзыв: {example['review_text']}\n"
        "Ответ:"
    )
    completion = example['reply']
    return {"text": prompt + " " + completion}

print("Форматируем примеры...")
texts = df.apply(format_example, axis=1).tolist()
texts_for_dataset = [x["text"] for x in texts]
print(f"Пример форматированного текста:\n{texts_for_dataset[0][:300]}...")

dataset = Dataset.from_dict({"text": texts_for_dataset})

# 3. Определение устройства и dtype
use_gpu = torch.cuda.is_available()
device_map = "auto" if use_gpu else None
torch_dtype = torch.float16 if use_gpu else torch.float32
print(f"Используемое устройство: {'GPU' if use_gpu else 'CPU'}")

# 4. Загрузка токенизатора и модели
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Загружаем токенизатор и модель...")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
    offload_folder="offload" if use_gpu else None,
    offload_state_dict=use_gpu,
)

# Включаем gradient checkpointing (экономия памяти)
model.gradient_checkpointing_enable()

# 5. Токенизация
print("Токенизируем данные...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Пример токенизированных данных:\n{tokenized_datasets[0]}")

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Параметры тренировки
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    bf16=True,  # ✅ A5000 поддерживает bfloat16
    report_to="none",
    load_best_model_at_end=False,
    logging_dir="./logs",
)
# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print("🚀 Начинаем обучение...")
trainer.train()
print("✅ Обучение завершено.")

# 9. Сохраняем модель и токенизатор
print("💾 Сохраняем модель...")
trainer.save_model("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
print("✅ Модель и токенизатор сохранены.")

