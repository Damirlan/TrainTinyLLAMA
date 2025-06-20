import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# === Путь к сохранённой модели ===
MODEL_PATH = "./saiga-mistral-lora-output"
BASE_MODEL = "IlyaGusev/saiga_mistral_7b_merged"

# === Загрузка токенизатора и модели с LoRA ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

# === Отзывы для теста ===
reviews = [
    "Куртка шикарная! Удобная, лёгкая и теплая.",
    "Пальто пришло не того цвета, но сидит отлично.",
    "Хорошая ветровка за свою цену. Муж доволен.",
]

# === Шаблон prompt'а (тот же, что в обучении) ===
def build_prompt(review_text):
    return (
        "<|system|>\nВы — вежливый представитель интернет-магазина. "
        "Напишите развернутый ответ на отзыв клиента.\n"
        f"<|user|>\nОтзыв: {review_text}\n<|assistant|>"
    )

# === Генерация
for i, review in enumerate(reviews, 1):
    prompt = build_prompt(review)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n🔹 Отзыв {i}:\n📝 {review}\n📣 Ответ модели:\n{response}\n" + "-" * 60)

