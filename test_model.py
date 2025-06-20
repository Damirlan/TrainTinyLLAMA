import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Путь к дообученной модели
MODEL_PATH = "./tinyllama-finetuned"

# Загружаем токенизатор и модель
print("🔄 Загружаем модель и токенизатор...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)


model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32  # ✅ Без fp16
).to("cpu")  # ✅ без CUDA

model.eval()

# Примеры для тестирования
examples = [
    {
        "rating": 5,
        "product_name": "пуховик",
        "review_text": "Очень тёплый и лёгкий. Доставка быстрая. Рекомендую!"
    },
    {
        "rating": 2,
        "product_name": "часы",
        "review_text": "Через неделю перестали работать. Очень разочарован."
    },
    {
        "rating": 4,
        "product_name": "наушники",
        "review_text": "Звук хороший, но немного давят на уши при длительном использовании."
    }
]

# Генерация ответа
def generate_reply(example):
    prompt = (
        f"Рейтинг: {example['rating']}\n"
        f"Товар: {example['product_name']}\n"
        f"Отзыв: {example['review_text']}\n"
        "Ответ:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()  # убираем prompt из начала

# Основной цикл
print("\n📊 Результаты генерации:\n" + "-"*50)
for i, ex in enumerate(examples, 1):
    reply = generate_reply(ex)
    print(f"[Пример {i}]")
    print(f"🛒 Товар: {ex['product_name']}")
    print(f"⭐️ Рейтинг: {ex['rating']}")
    print(f"📝 Отзыв: {ex['review_text']}")
    print(f"🤖 Ответ модели: {reply}\n" + "-"*50)

