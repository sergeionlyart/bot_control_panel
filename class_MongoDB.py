# python3 class_MongoDB.py

from pymongo import MongoClient
import logging

class DataStorageManager:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """
        Подключается к MongoDB и получает коллекцию для хранения/обновления документов.
        СИНХРОННЫЙ вариант с использованием PyMongo.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logging.info(
            f"[DataStorageManager] Connected to Mongo at {mongo_uri}, "
            f"DB='{db_name}', Collection='{collection_name}'"
        )
    
    def get_cat_lists(self, unique_id: str):
        """
        Ищет документы, у которых request_id == unique_id,
        затем собирает значения из cat_1 и cat_2 (ожидаются списки),
        убирает дубликаты и возвращает два списка:
        (список_для_cat1, список_для_cat2).
        """
        cat_1_accumulator = set()
        cat_2_accumulator = set()
        
        # Ищем все документы с нужным request_id
        cursor = self.collection.find({"request_id": unique_id})
        
        for doc in cursor:
            # Получаем значения cat_1 и cat_2 как списки
            cat_1_values = doc.get("cat_1", [])
            cat_2_values = doc.get("cat_2", [])
            
            # Добавляем значения в множества для устранения дублей
            cat_1_accumulator.update(cat_1_values)
            cat_2_accumulator.update(cat_2_values)
        
        # Преобразуем множества обратно в списки
        cat_1_list = list(cat_1_accumulator)
        cat_2_list = list(cat_2_accumulator)
        
        return cat_1_list, cat_2_list


# Пример использования:
if __name__ == "__main__":
    data_storage_manager = DataStorageManager(
        mongo_uri="mongodb://localhost:27017",  
        db_name="Chat_bot",                     
        collection_name="vector_search"
    )

    unique_id = "d6ef1740-d5ed-46ec-95bd-a61894d3f6eb"
    cat1_list, cat2_list = data_storage_manager.get_cat_lists(unique_id)
    print("cat_1:", cat1_list)
    print("cat_2:", cat2_list)