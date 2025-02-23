import json
from typing import Dict, Any



from OpenAIChatBot import OpenAIChatBot
from PromptManager import PromptManager
from ChromaDBManager import ChromaDBManager  # <-- Подключаем ваш реальный класс
from contradiction_detector import process_documents_in_parallel

def main():
    # 1. Создаём экземпляры
    bot = OpenAIChatBot()
    prompt_manager = PromptManager()
    db_manager = ChromaDBManager(  # при необходимости укажите свои параметры
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

    # 3. Текст запроса
    query_text = "какова стоимость временной коронки ? "

    # 4. Вызываем параллельную обработку
    results = process_documents_in_parallel(
        bot=bot,
        prompt_manager=prompt_manager,
        db_manager=db_manager,
        query_text=query_text,
        schema=schema,
        n_results=10
    )

    # 5. Печатаем результаты
    print("Результаты параллельной обработки (с реальным ChromaDBManager):\n")
    for i, res in enumerate(results, start=1):
        print(f"=== Документ #{i} ===")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print()

if __name__ == "__main__":
    main()