import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_NAME = "IlyaGusev/saiga_mistral_7b_merged"

print("🔄 Загружаем модель и токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

gen_conf = GenerationConfig.from_pretrained(MODEL_NAME)

# Примеры отзывов на русском
examples = [
    {
        "rating": 5,
        "product_name": "тёплый свитер",
        "review_text": "Очень мягкий и приятный, идеально на зиму!"
    },
    {
        "rating": 2,
        "product_name": "пылесос",
        "review_text": "Слабое всасывание, неприятный шум."
    },
]

def generate_reply(example):
    prompt = (
        f"Ты — вежливый помощник интернет-магазина.\n"
        f"Рейтинг: {example['rating']}\n"
        f"Товар: {example['product_name']}\n"
        f"Отзыв: {example['review_text']}\n"
        f"Ответ:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=gen_conf,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = text[len(prompt):].split("Рейтинг:")[0].strip()
    return reply

print("\n📊 Ответы модели Saiga-Mistral:\n" + "-"*50)
for i, ex in enumerate(examples, 1):
    reply = generate_reply(ex)
    print(f"[Пример {i}]")
    print(f"Товар: {ex['product_name']}, Рейтинг: {ex['rating']}")
    print(f"Отзыв: {ex['review_text']}")
    print(f"Ответ: {reply}\n" + "-"*50)

