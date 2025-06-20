import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("🔄 Loading model and tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# English instruction-style examples
examples = [
    {
        "instruction": "Write a short and polite response to this customer review:\n\nRating: 5\nProduct: winter jacket\nReview: Very warm and comfortable. I love wearing it!"
    },
    {
        "instruction": "Write a helpful reply to this customer review:\n\nRating: 2\nProduct: wireless headphones\nReview: One earbud stopped working after a week."
    }
]

def generate_reply(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = generated[len(prompt):].strip()
    return reply

# Run evaluation
print("\n📊 English Generation Results:\n" + "-" * 50)
for i, ex in enumerate(examples, 1):
    reply = generate_reply(ex)
    print(f"[Example {i}]")
    print(f"📝 Instruction:\n{ex['instruction']}\n")
    print(f"🤖 Model Response:\n{reply}\n" + "-" * 50)

