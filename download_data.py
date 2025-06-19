import csv
from db_handler import PostgresHandler  # Импортируй свой класс

def export_reviews_with_replies_to_csv(csv_filepath: str):
    query = """
    SELECT rating, product_name, review_text, reply
    FROM reviews
    WHERE reply IS NOT NULL AND reply <> ''
    """

    with PostgresHandler() as db:
        results = db.fetch_results(query)

    if results:
        with open(csv_filepath, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rating', 'product_name', 'review_text', 'reply'])
            writer.writerows(results)
        print(f"Данные успешно экспортированы в {csv_filepath}")
    else:
        print("Нет данных для экспорта.")

if __name__ == "__main__":
    export_reviews_with_replies_to_csv("reviews_with_replies.csv")