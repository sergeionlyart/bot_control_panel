# update_delete_document_ChromaDB.py

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional

# Пути и название коллекции подставьте свои:
ENV_PATH = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/copilot_service/.env"
PERSIST_DIRECTORY = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/ChromaDB/chroma_db"
COLLECTION_NAME = "my_collection"
MODEL_NAME = "text-embedding-3-small"  # или модель, с которой создавалась коллекция

def get_chroma_collection():
    """
    Возвращает объект коллекции ChromaDB, используя указанные пути и env.
    """
    # 1. Загрузить переменные окружения (ключ OpenAI и т.д.)
    load_dotenv(dotenv_path=ENV_PATH)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Не удалось найти OPENAI_API_KEY в .env")

    # 2. Инициализировать клиента ChromaDB
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # 3. Настроить функцию эмбеддинга (если нужно)
    openai_embed = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=MODEL_NAME
    )

    # 4. Получить коллекцию
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_embed
    )
    return collection


def add_or_update_document(
    new_text: str, 
    new_metadata: dict = None, 
    doc_id: Optional[str] = None
):
    """
    Добавляет новый или обновляет существующий документ в ChromaDB.
    
    Параметры:
    -----------
    - new_text: str 
        Текст документа (при обновлении заменяет старый текст).
    - new_metadata: dict, optional 
        Метаданные (при обновлении замещают старые метаданные).
    - doc_id: Optional[str], optional
        Идентификатор документа:
        - Если None (по умолчанию), генерируется новый уникальный ID в формате 'doc001', 'doc002', ...
        - Если строка, но документа с таким ID нет, будет добавлен новый документ с этим ID.
        - Если строка и документ с таким ID есть, он будет обновлён.
    """
    collection = get_chroma_collection()
    
    # Список всех ID в коллекции
    all_docs = collection.get()
    existing_ids = all_docs["ids"]  # Пример: ['doc001', 'doc002', ...]

    def get_next_doc_id():
        """
        Находит максимальный числовой суффикс среди документов формата 'docXXX'
        и возвращает следующий ID 'docXXX' + 1.
        """
        max_num = 0
        for existing_id in existing_ids:
            # Допустим, формат всегда "docXYZ"
            if existing_id.startswith("doc"):
                numeric_part = existing_id[3:]  # берем часть после 'doc'
                try:
                    num = int(numeric_part)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    # Если встретился некорректный формат, пропускаем
                    pass
        return f"doc{(max_num + 1):03d}"

    # Если doc_id не передан (или None), нужно сгенерировать новый
    if doc_id is None:
        new_id = get_next_doc_id()
        collection.add(
            ids=[new_id],
            documents=[new_text],
            metadatas=[new_metadata if new_metadata else {}]
        )
        print(f"Новый документ с ID={new_id} успешно добавлен.")
        return

    # doc_id указан явно
    if doc_id in existing_ids:
        # Если документ с таким ID есть, обновляем
        collection.update(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata if new_metadata else {}]
        )
        print(f"Документ с ID={doc_id} успешно обновлён.")
    else:
        # Если документа с таким ID не существует, просто добавляем с этим ID
        collection.add(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata if new_metadata else {}]
        )
        print(f"Новый документ с заданным ID={doc_id} успешно добавлен.")


def delete_document(doc_id: str):
    """
    Удаляет документ из ChromaDB по его doc_id.
    """
    collection = get_chroma_collection()
    
    # Удаляем документ по ID
    collection.delete(ids=[doc_id])
    print(f"Документ с ID={doc_id} успешно удалён.")


