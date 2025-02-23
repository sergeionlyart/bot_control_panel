
#!/usr/bin/env python3
# Скрипт: python3 get_chroma_document_ID.py
# Выводит документ с ID="doc001" в виде JSON (включая весь embedding),
# при этом конвертирует embedding (ndarray) в список.

import os
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

ENV_PATH = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/copilot_service/.env"
PERSIST_DIRECTORY = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/ChromaDB/chroma_db"
COLLECTION_NAME = "my_collection"
MODEL_NAME = "text-embedding-3-small"

DOC_ID = "doc002"

def get_chroma_collection():
    """
    Загружает переменные окружения, инициализирует клиент ChromaDB и возвращает коллекцию.
    """
    load_dotenv(dotenv_path=ENV_PATH)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Не удалось найти OPENAI_API_KEY в .env")

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    openai_embed = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=MODEL_NAME
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_embed
    )
    return collection

def get_full_document(doc_id: str):
    """
    Получает документ по его doc_id из ChromaDB и возвращает всю информацию:
      - ID
      - Текст документа
      - Метаданные
      - Эмбеддинг (numpy.ndarray -> list)
    Возвращает словарь с данными документа или None, если документ не найден.
    """
    collection = get_chroma_collection()

    result = collection.get(
        ids=[doc_id],
        include=["documents", "metadatas", "embeddings"]
    )

    # Проверяем, найден ли документ
    if not result["ids"] or not result["ids"][0]:
        return None

    # Извлекаем данные (Chroma возвращает их как списки для batch-запросов)
    retrieved_id = result["ids"][0]
    retrieved_text = result["documents"][0]
    retrieved_metadata = result["metadatas"][0]
    retrieved_embedding = result["embeddings"][0]

    # Если это numpy.ndarray — конвертируем в список Python
    if isinstance(retrieved_embedding, np.ndarray):
        retrieved_embedding = retrieved_embedding.tolist()

    doc_info = {
        "id": retrieved_id,
        "text": retrieved_text,
        "metadata": retrieved_metadata,
       # "embedding": retrieved_embedding
    }
    return doc_info


from typing import List, Dict, Any
import numpy as np

def get_full_documents(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Получает документы по их doc_ids из ChromaDB и возвращает список словарей:
      - ID
      - Текст документа
      - Метаданные
      - Эмбеддинг (конвертируется из np.ndarray в list, если требуется)
      
    Если какие-то документы не найдены, Chroma вернёт пустой слот, поэтому при
    необходимости нужно самостоятельно проверять, есть ли документ в ответе.
    """
    # Получаем коллекцию Chroma
    collection = get_chroma_collection()

    # Если список пуст — сразу вернём пустой список
    if not doc_ids:
        return []

    # Делаем один запрос в Chroma, передаём весь список ID
    result = collection.get(
        ids=doc_ids,
        include=["documents", "metadatas", "embeddings"]
    )

    # Если ничего не найдено
    if not result["ids"]:
        return []

    documents_info = []
    # Разбираем результаты по индексам
    for i in range(len(result["ids"])):
        # В зависимости от того, как Chroma обрабатывает "не найденные" ID,
        # возможно, стоит проверять наличие result["ids"][i] и т.д.
        retrieved_id = result["ids"][i]
        retrieved_text = result["documents"][i]
        retrieved_metadata = result["metadatas"][i]
        retrieved_embedding = result["embeddings"][i]

        # Конвертируем эмбеддинг из np.ndarray в list, если нужно
        if isinstance(retrieved_embedding, np.ndarray):
            retrieved_embedding = retrieved_embedding.tolist()

        documents_info.append({
            "id": retrieved_id,
            "text": retrieved_text,
            "metadata": retrieved_metadata,
            #"embedding": retrieved_embedding
        })

    return documents_info

if __name__ == "__main__":
    # Пример вызова функции для одного документа (как в исходном коде)
    doc_data = get_full_document(DOC_ID)
    if doc_data is None:
        print(f"Документ с ID='{DOC_ID}' не найден.")
    else:
        print("Одиночный документ:")
        print(json.dumps(doc_data, ensure_ascii=False, indent=2))
    
    # Пример вызова функции для нескольких документов
    doc_ids = ["doc001", "doc002"]  # Замените своими реальными ID
    docs_data = get_full_documents(doc_ids)
    
    if not docs_data:
        print(f"Ни один из документов {doc_ids} не найден.")
    else:
        print("\nСписок документов:")
        print(json.dumps(docs_data, ensure_ascii=False, indent=2))