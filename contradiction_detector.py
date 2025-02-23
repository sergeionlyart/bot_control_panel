# contradiction_detector.py


import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from .PROMPT import contradiction_detector

def process_documents_in_parallel(
    bot,            # Экземпляр OpenAIChatBot
    prompt_manager, # Экземпляр PromptManager
    db_manager,     # Экземпляр ChromaDBManager (где есть метод get_documents_json)
    query_text: str,
    schema: Dict[str, Any],
    n_results: int = 3,
    model: str = "gpt-4o-mini",   # по умолчанию
    temperature: float = 0.0           # по умолчанию
) -> List[Dict[str, Any]]:
    """
    1) Выполняет поиск документов в ChromaDBManager -> JSON с полем 'documents'.
    2) Преобразует каждый элемент массива 'documents' в отдельный dict (id, text, response).
    3) Для каждого документа строит prompt с помощью prompt_manager.build_prompt_with_json(...) и 
       вызывает bot.generate_chat_response_with_jsonschema(...).
    4) Запускает все вызовы параллельно через ThreadPoolExecutor для снижения задержки.
    5) Возвращает список результатов (по одному на каждый документ).
    """

    # Шаг 1. Получаем JSON со списком документов
    doc_json_str = db_manager.get_documents_json(query_text, n_results=n_results)
    doc_dict = json.loads(doc_json_str)
    documents = doc_dict.get("documents", [])
    print("Полученные документ:", documents)
    if not documents:
        return []

    results = []
    with ThreadPoolExecutor() as executor:
        future_to_doc = {}
        for doc in documents:
            user_messages = prompt_manager.build_contradiction_detector_json(
                prompt=contradiction_detector,
                json_data=doc,
                user_query=query_text
            )
            # Передаём нужные параметры:
            future = executor.submit(
                bot.generate_chat_response_with_jsonschema,
                user_messages,
                schema,
                model="gpt-4o",
                temperature=0,
                seed=42
            )
            future_to_doc[future] = doc

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                data = future.result()
            except Exception as e:
                data = {"error": str(e), "doc_id": doc.get("id")}
            results.append(data)

    return results