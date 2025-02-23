# python3 analyze_query_with_chroma.py


import json
from typing import Dict, Any

from .OpenAIChatBot import OpenAIChatBot
from .PromptManager import PromptManager
from .ChromaDBManager import ChromaDBManager  # <-- Подключайте вашу реализацию
from .contradiction_detector import process_documents_in_parallel

def analyze_query_with_chroma(query_text: str):
    """
    Функция для анализа входного текста (query_text) с помощью OpenAIChatBot,
    PromptManager и ChromaDBManager, обрабатывая документы параллельно.
    """
    # 1. Создаём экземпляры
    bot = OpenAIChatBot()
    prompt_manager = PromptManager()
    db_manager = ChromaDBManager(
        env_path="/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                 "code_2/Umnico_Widget_Test/copilot_service/.env",
        persist_directory="/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                         "code_2/Umnico_Widget_Test/ChromaDB/chroma_db",
        collection_name="my_collection",  # имя вашей коллекции
        model_name="text-embedding-3-small"
    )

    # 2. Примерная JSON-схема (можете подставить свою)
    schema: Dict[str, Any] = {
        "name": "test_analysis_schema",
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Короткий анализ документа"
                }
            },
            "additionalProperties": False
        }
    }

    # 3. Вызываем параллельную обработку
    results = process_documents_in_parallel(
        bot=bot,
        prompt_manager=prompt_manager,
        db_manager=db_manager,
        query_text=query_text,
        schema=schema,
        n_results=10
    )

    # 4. Печатаем и возвращаем результаты
    print("Результаты параллельной обработки (с реальным ChromaDBManager):\n")
    for i, res in enumerate(results, start=1):
        print(f"=== Документ #{i} ===")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print()

    return results

# Пример использования
if __name__ == "__main__":
    user_query = "Стоимость временных коронок у нас составляет 5000 доларов сша за штуку "
    analyze_query_with_chroma(user_query)