import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Используем оригинальную модель с Hugging Face
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("🔄 Загружаем оригинальную модель и токенизатор...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Тестовые примеры
examples = [
    {
        "rating": 5,
        "product_name": "зимняя куртка",
        "review_text": "Очень тёплая и удобная. Ношу с удовольствием!"
    },
    {
        "rating": 2,
        "product_name": "беспроводные наушники",
        "review_text": "Через неделю один наушник перестал работать."
    }
]

# Генерация ответа
def generate_reply(example):
    prompt = (
        f"Ты — вежливый русскоязычный помощник магазина.\n"
        f"Рейтинг: {example['rating']}\n"
        f"Товар: {example['product_name']}\n"
        f"Отзыв: {example['review_text']}\n"
        f"Ответ:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    response = generated[len(prompt):].strip().split("Рейтинг:")[0].strip()
    return response

# Основной цикл
print("\n📊 Ответы оригинальной модели:\n" + "-"*50)
for i, ex in enumerate(examples, 1):
    reply = generate_reply(ex)
    print(f"[Пример {i}]")
    print(f"🛒 Товар: {ex['product_name']}")
    print(f"⭐️ Рейтинг: {ex['rating']}")
    print(f"📝 Отзыв: {ex['review_text']}")
    print(f"🤖 Ответ модели: {reply}\n" + "-"*50)

