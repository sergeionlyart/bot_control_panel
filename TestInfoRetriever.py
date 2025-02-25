# TestInfoRetriever.py
# python3 -m copilot_service.tools.TestInfoRetriever

"""
Ниже краткая цепочка того, как run_classification_and_build_markdown формирует итоговый ответ:
  1. Создание необходимых объектов
     - OpenAIChatBot (файл OpenAIChatBot.py) — для взаимодействия с LLM.
     - PromptManager (файл PromptManager.py) — формирует тексты/промты.
     - ChromaDBManager (файл ChromaDBManager.py) — управляет поиском и хранением документов в базе Chroma.
  2. Определение JSON-схемы
     Описывает структуру, в которой будет возвращён результат классификации (например, поля document_id, category, explanation).
  3. Вызов process_documents_in_parallel
     - Находится в chroma_parallel_analyzer_search.py.
     - Под капотом:
       1) Запрашивает документы из ChromaDB через ChromaDBManager.
       2) Для каждого документа формирует промт (с помощью PromptManager и текста из PROMPT.py).
       3) Параллельно вызывает OpenAI-модель (через OpenAIChatBot) для определения категории.
  4. Фильтрация результатов (категории 1 и 2)
     Из списка классификаций отбираются только те, где category == 1 или category == 2.
  5. Повторное обращение к ChromaDB
     Снова через ChromaDBManager.get_documents_json, чтобы получить тексты и ответы документов,
     нужных для формирования итогового Markdown.
  6. Сборка итогового Markdown
     Функция формирует строки с заголовками («Сообшение», «Response») и данными для каждого
     подходящего документа. Возвращает результат одной строкой в формате Markdown.
"""

import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Импорт необходимых классов
# -----------------------------
from copilot_service.OpenAIChatBot import OpenAIChatBot
from copilot_service.PromptManager import PromptManager
from copilot_service.ChromaDBManager import ChromaDBManager
from copilot_service.chroma_parallel_analyzer_search import process_documents_in_parallel

# Синхронный клиент для MongoDB
from pymongo import MongoClient

# -----------------------------
# Класс для сохранения/обновления в MongoDB (синхронно, через PyMongo)
# -----------------------------
class DialogHistoryManager:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """
        Подключается к MongoDB и получает коллекцию для хранения/обновления документов.
        СИНХРОННЫЙ вариант с использованием PyMongo.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logging.info(
            f"[DialogHistoryManager] Connected to Mongo at {mongo_uri}, "
            f"DB='{db_name}', Collection='{collection_name}'"
        )

    def add_or_update_vector_document(
        self,
        request_id: str,
        cat_1: List[str],
        cat_2: List[str]
    ) -> None:
        """
        Сохраняет (или обновляет) документ в MongoDB (синхронно).
        Если документа с данным request_id не существует — создаёт,
        иначе добавляет новые значения в списки cat_1 и cat_2.
        """
        existing_doc = self.collection.find_one({"request_id": request_id})
        if existing_doc:
            # Обновляем (добавляем новые значения)
            self.collection.update_one(
                {"request_id": request_id},
                {
                    # $each добавляет сразу несколько элементов в массив
                    "$push": {
                        "cat_1": {"$each": cat_1},
                        "cat_2": {"$each": cat_2}
                    }
                }
            )
            logging.info(f"Документ с request_id={request_id} обновлён (добавлены новые cat_1/cat_2).")
        else:
            # Создаём новый документ
            new_doc = {
                "request_id": request_id,
                "cat_1": cat_1,
                "cat_2": cat_2
            }
            self.collection.insert_one(new_doc)
            logging.info(f"Создан новый документ с request_id={request_id}.")

# -----------------------------
# СИНХРОННАЯ функция vector_search
# -----------------------------
def vector_search(query_text: str, request_id: str = None) -> str:
    """
    Поисковая функция (синхронная), которая:
      1) Создаёт bot, prompt_manager, db_manager.
      2) Принимает на вход текст запроса (query_text) и JSON-схему (schema).
      3) Вызывает process_documents_in_parallel(...).
      4) Собирает ID документов, у которых category = 1 или 2.
      5) Извлекает эти документы из ChromaDBManager.
      6) Формирует и возвращает строку в формате Markdown
         (сначала документы категории 1, потом 2).

    Дополнительно:
      - При наличии request_id сохраняет списки cat1_ids и cat2_ids в MongoDB
        с помощью метода add_or_update_vector_document (синхронно).
    """
    logger.info(f"[vector_search] called with query_text={query_text!r}, request_id={request_id!r}")

    # 1. Создаём экземпляры
    bot = OpenAIChatBot()
    prompt_manager = PromptManager()
    db_manager = ChromaDBManager(
        env_path="/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                 "code_2/Umnico_Widget_Test/copilot_service/.env",
        persist_directory="/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                         "code_2/Umnico_Widget_Test/ChromaDB/chroma_db",
        collection_name="my_collection",
        model_name="text-embedding-3-small"
    )

    # 2. Определяем схему для JSON-ответов
    schema: Dict[str, Any] = {
        "name": "doc_classification_schema",
        "schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Идентификатор документа (doc ID)."
                },
                "category": {
                    "type": "number",
                    "description": "Категория (1, 2 или 3)."
                },
                "explanation": {
                    "type": "string",
                    "description": "Объяснение выбора категории."
                }
            },
            "required": ["document_id", "category"],
            "additionalProperties": False
        }
    }

    # 3. Выполняем параллельную обработку
    results = process_documents_in_parallel(
        bot=bot,
        prompt_manager=prompt_manager,
        db_manager=db_manager,
        query_text=query_text,
        schema=schema,
        n_results=10,           # Сколько документов достать (можно менять)
        model="gpt-4o-mini",   # Пример модели
        temperature=0.0
    )

    # 4. Отфильтровываем нужные категории (1 и 2), вытаскиваем их IDs
    cat1_ids = []
    cat2_ids = []

    for item in results:
        response_json = item.get("response_json", {})
        doc_id = response_json.get("document_id")
        category = response_json.get("category")
        if category == 1:
            cat1_ids.append(doc_id)
        elif category == 2:
            cat2_ids.append(doc_id)

    # Сохраняем данные в MongoDB (синхронно), если передан request_id
    if request_id and (cat1_ids or cat2_ids):
        dialog_history_manager = DialogHistoryManager(
            mongo_uri="mongodb://localhost:27017",  # Замените на ваш MongoDB URI
            db_name="Chat_bot",                    # Имя базы данных
            collection_name="vector_search"        # Имя коллекции
        )
        dialog_history_manager.add_or_update_vector_document(
            request_id=request_id,
            cat_1=cat1_ids,
            cat_2=cat2_ids
        )

    if not cat1_ids and not cat2_ids:
        return "Нет документов с категорией 1 или 2."

    # 5. Получаем документы из db_manager и строим map {doc_id: {...}}
    doc_json_str = db_manager.get_documents_json(query_text, n_results=10)
    doc_dict = json.loads(doc_json_str)
    documents = doc_dict.get("documents", [])

    doc_map = {}
    for doc in documents:
        d_id = doc.get("id")
        doc_map[d_id] = {
            "text": doc.get("text", ""),
            "response": doc.get("response", "")
        }

    # 6. Формируем финальную Markdown-строку
    lines = []
    lines.append("# Список релевантных документов\n")

    # --- Категория 1 ---
    if cat1_ids:
        for d_id in cat1_ids:
            doc_info = doc_map.get(d_id)
            if not doc_info:
                continue  # если документ не найден
            lines.append("**Сообшение:**\n")
            lines.append(f"> {doc_info['text']}\n")
            lines.append("\n**Response:**\n")
            lines.append(f"> {doc_info['response']}\n")
            lines.append("---\n")

    # --- Категория 2 ---
    if cat2_ids:
        for d_id in cat2_ids:
            doc_info = doc_map.get(d_id)
            if not doc_info:
                continue
            lines.append("**Сообшение:**\n")
            lines.append(f"> {doc_info['text']}\n")
            lines.append("\n**Response:**\n")
            lines.append(f"> {doc_info['response']}\n")
            lines.append("---\n")

    final_markdown = "\n".join(lines)
    return final_markdown


# Пример вызова функции vector_search (синхронно)
if __name__ == "__main__":
    query_text = "какая самая крупная планета в солнечной системе ? "
    request_id = "unique_request_123"
    final_md = vector_search(query_text, request_id)
    print(final_md)