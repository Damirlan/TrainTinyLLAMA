import pandas as pd
import json

INPUT_CSV = "reviews_with_replies.csv"      # замените на имя вашего файла
OUTPUT_JSONL = "train_data.jsonl"

df = pd.read_csv(INPUT_CSV)

with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
    for _, row in df.iterrows():
        prompt = (
            "<|system|>\nВы — вежливый представитель интернет-магазина. "
            "Напишите развернутый ответ на отзыв клиента.\n"
            f"<|user|>\nОтзыв: {row['review_text'].strip()}\n<|assistant|>"
        )
        response = row['reply'].strip()
        json.dump({"prompt": prompt, "response": response}, f_out, ensure_ascii=False)
        f_out.write('\n')

print(f"Сохранено в {OUTPUT_JSONL}")

