"""
Общий класс для работы с ChromaDB.
Объединяет логику get_chroma_document_ID.py, query_documents_ChromaDB.py и update_delete_document_ChromaDB.py,
а также содержит метод add_document для добавления новых документов с проверкой дубликатов.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import numpy as np


class ChromaDBManager:
    def __init__(
        self,
        env_path: str = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                       "code_2/Umnico_Widget_Test/copilot_service/.env",
        persist_directory: str = "/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/"
                                 "code_2/Umnico_Widget_Test/ChromaDB/chroma_db",
        collection_name: str = "my_collection",
        model_name: str = "text-embedding-3-small"
    ):
        """
        При инициализации:
          - Загружаем переменные окружения из .env.
          - Создаём PersistentClient для ChromaDB.
          - Получаем коллекцию (collection_name) с нужной моделью (model_name).
        """
        self.env_path = env_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name

        # Загрузка окружения (ключи и т.д.)
        load_dotenv(dotenv_path=self.env_path)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Не удалось найти OPENAI_API_KEY в .env")

        # Инициализация ChromaDB-клиента
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Настройка функции эмбеддинга
        self.openai_embed = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=self.model_name
        )

        # Получаем (или создаём) коллекцию
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.openai_embed
        )

    def add_document(self, doc_id: str, doc_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавляет новый документ в коллекцию ChromaDB, если документа с таким ID ещё нет.
          - doc_id: уникальный идентификатор документа (например, "doc010")
          - doc_text: содержимое документа (текст)
          - metadata: словарь с метаданными (author, date, tags и т.д.)
          
          СОЗДАВАЕМЫЙ ДОКУМЕНТ:

{
   "id": <строка>,
   "document": <строка с текстом>,
   "metadata": {
       "имя_колонки_2": <значение>,
       "имя_колонки_3": <значение>,
       ...
   },
   "embedding": <вектор_эмбеддинга_из_OpenAI>  # генерируется автоматически
}     
          
          
        """
        if metadata is None:
            metadata = {}

        # Проверяем, нет ли уже документа с таким doc_id
        existing_docs = self.collection.get(ids=[doc_id])
        if existing_docs["ids"] and existing_docs["ids"][0] == doc_id:
            print(f"Документ с ID={doc_id} уже существует! Добавление нового документа невозможно.")
            return

        # Если документа нет, добавляем его
        self.collection.add(
            documents=[doc_text],
            ids=[doc_id],
            metadatas=[metadata]
        )
        print(f"Документ с ID={doc_id} успешно добавлен.")

    def get_full_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает документ по его doc_id из ChromaDB и возвращает всю информацию:
          - ID
          - Текст документа
          - Метаданные
          - (При необходимости можно вернуть embedding)
        Возвращает словарь с данными документа или None, если документ не найден.
        """
        result = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]  # embeddings при желании можно убрать
        )

        # Проверяем, найден ли документ
        if not result["ids"] or not result["ids"][0]:
            return None

        # Извлекаем данные (Chroma возвращает списки для batch-запросов)
        retrieved_id = result["ids"][0]
        retrieved_text = result["documents"][0]
        retrieved_metadata = result["metadatas"][0]
        retrieved_embedding = result["embeddings"][0]

        # При необходимости конвертируем embedding из numpy.ndarray в list
        if isinstance(retrieved_embedding, np.ndarray):
            retrieved_embedding = retrieved_embedding.tolist()

        return {
            "id": retrieved_id,
            "text": retrieved_text,
            "metadata": retrieved_metadata,
            # "embedding": retrieved_embedding  # добавьте, если нужно
        }

    def query_top_n_documents(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Ищет ближайшие n_results документов по текстовому запросу (query_text) в ChromaDB.
        Возвращает список словарей вида:
          {
            "id": <doc_id>,
            "document": <текст документа>,
            "metadata": <dict>,
          }
        """
        result = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]  # при необходимости можно вернуть embeddings
        )

        if not result["ids"] or not result["ids"][0]:
            # Нет найденных документов
            return []

        returned_docs = []
        # Извлекаем результаты (Chroma возвращает списки списков)
        for i in range(len(result["ids"][0])):
            doc_id = result["ids"][0][i]
            doc_text = result["documents"][0][i]
            doc_metadata = result["metadatas"][0][i]

            returned_docs.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": doc_metadata
            })

        return returned_docs

    def get_documents_json(self, query_text: str, n_results: int = 3) -> str:
        """
        Выполняет поиск документов (query_top_n_documents) и преобразует результат в JSON-строку вида:
          {
            "documents": [
              {
                "id": "<doc_id>",
                "text": "<Исходный текст>",
                "response": "<system_response>"
              },
              ...
            ]
          }
        """
        docs = self.query_top_n_documents(query_text, n_results=n_results)

        final_docs = []
        for d in docs:
            doc_id = d["id"]
            doc_text = d["document"]
            # Извлекаем 'system_response' (или пустую строку, если нет такого ключа)
            system_response = d["metadata"].get("system_response", "")

            final_docs.append({
                "id": doc_id,
                "text": doc_text,
                "response": system_response
            })

        final_data = {
            "documents": final_docs
        }

        return json.dumps(final_data, ensure_ascii=False, indent=2)

    def update_document(self, doc_id: str, new_text: str, new_metadata: Optional[Dict] = None) -> None:
        """
        Обновляет существующий документ в ChromaDB по его doc_id.
          - doc_id: Идентификатор документа (например, "doc001")
          - new_text: Новый (обновлённый) текст документа
          - new_metadata: Словарь с обновлёнными метаданными. Если не нужно, передавать пустой dict
        """
        if new_metadata is None:
            new_metadata = {}

        self.collection.update(
            ids=[doc_id],
            documents=[new_text],
            metadatas=[new_metadata]
        )
        print(f"Документ с ID={doc_id} успешно обновлён.")

    def delete_document(self, doc_id: str) -> None:
        """
        Удаляет документ из ChromaDB по его doc_id.
        """
        self.collection.delete(ids=[doc_id])
        print(f"Документ с ID={doc_id} успешно удалён.")