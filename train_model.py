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

# Чтобы уменьшить фрагментацию памяти CUDA (опционально)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Загружаем данные
print("Загрузка CSV...")
df = pd.read_csv("reviews_with_replies.csv", encoding='utf-8')
print(f"Данные загружены: {len(df)} записей")

# 2. Форматируем примеры
def format_example(example):
    prompt = (f"Рейтинг: {example['rating']}\n"
              f"Товар: {example['product_name']}\n"
              f"Отзыв: {example['review_text']}\n"
              "Ответ:")
    completion = example['reply']
    return {"text": prompt + " " + completion}

print("Форматируем примеры...")
texts = df.apply(format_example, axis=1).tolist()
texts_for_dataset = [x["text"] for x in texts]
print(f"Пример форматированного текста:\n{texts_for_dataset[0][:300]}...")

dataset = Dataset.from_dict({"text": texts_for_dataset})

# 3. Загружаем токенизатор и модель
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Загружаем токенизатор и модель...")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",       # offload части модели на диск
    offload_state_dict=True,        # выгрузка состояния оптимизатора
    torch_dtype=torch.float16       # 16-битная точность (fp16)
)

# Включаем gradient checkpointing для экономии памяти
model.gradient_checkpointing_enable()

# 4. Токенизируем
print("Токенизируем данные...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"Пример токенизированных данных:\n{tokenized_datasets[0]}")

# 5. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Настройки тренировки
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Уменьшаем batch size из-за ограничений памяти
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=False,                    # GTX 1660 не стабильно работает с fp16 через AMP, поэтому False
    report_to="none",
    load_best_model_at_end=False,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print("🚀 Начинаем обучение на GPU...")
trainer.train()
print("Обучение завершено.")

# 8. Сохраняем модель
print("Сохраняем модель...")
trainer.save_model("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
print("Модель сохранена.")
