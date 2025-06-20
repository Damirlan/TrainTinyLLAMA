from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "IlyaGusev/saiga_mistral_7b_merged"
lora_path = "./lora-checkpoint"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(model, "./output_dir")  # путь к temp-папке модели после обучения

model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)

print("✅ LoRA веса и токенизатор сохранены в", lora_path)

