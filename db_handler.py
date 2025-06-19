import psycopg2
from psycopg2 import Error
from contextlib import contextmanager
from config import DB_CONFIG


class PostgresHandler:
    def __init__(self, config: dict = DB_CONFIG):
        self.config = config
        self.connection = None
        self.cursor = None

    def __enter__(self):
        """Установить соединение и вернуть объект класса."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрыть соединение и курсор."""
        self.close()

    def connect(self):
        """Установить соединение с базой данных."""
        try:
            self.connection = psycopg2.connect(
                dbname=self.config.get('dbname'),
                user=self.config.get('user'),
                password=self.config.get('password'),
                host=self.config.get('host'),
                port=self.config.get('port')
            )
            if self.connection:
                self.cursor = self.connection.cursor()
                print("Подключение к PostgreSQL установлено.")
        except Exception as e:
            print(f"Ошибка подключения: {e}")

    def close(self):
        """Закрыть курсор и соединение."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Соединение с PostgreSQL закрыто.")

    def execute_query(self, query, params=None):
        """Выполнить SQL-запрос."""
        try:
            self.cursor.execute(query, params or ())
            self.connection.commit()
            print("Запрос выполнен успешно.")
        except Exception as e:
            print(f"Ошибка выполнения запроса: {e}")

    def fetch_results(self, query, params=None):
        """Выполнить запрос и вернуть результаты."""
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return None


if __name__ == '__main__':
    # Пример запроса: выборка всех отзывов
    test_query = "SELECT * FROM reviews LIMIT 5;"

    with PostgresHandler() as db:
        results = db.fetch_results(test_query)
        if results:
            print("Результаты выборки:")
            for row in results:
                print(row)
        else:
            print("Нет результатов или произошла ошибка.")

