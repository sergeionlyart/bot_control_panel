# Project Structure
```
/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel
├── ChromaDBManager.py
├── OpenAIChatBot.py
├── PROMPT.py
├── PromptManager.py
├── README.md
├── __init__.py
├── analyze_query_with_chroma.py
├── class_MongoDB.py
├── contradiction_detector.py
├── export_to_md.sh
├── get_chroma_document_ID.py
├── main.py
├── main_2.py
├── project_bot_control_panel.md
├── tab1
│   ├── __init__.py
│   └── tab1_code.py
├── tab2
│   ├── __init__.py
│   └── tab2_code.py
├── tab3
│   ├── __init__.py
│   └── tab3_code.py
├── tab4
│   ├── __init__.py
│   └── tab4_code.py
├── tab5
│   ├── __init__.py
│   └── tab5_code.py
├── test.py
└── update_delete_document_ChromaDB.py

6 directories, 26 files
```

# Source Code

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab1/tab1_code.py
```python
# python3 main_2.py

# python3 -m bot_control_panel.main_2
import gradio as gr
import time
import uuid
from typing import List

# Импорты, учитывая что класс и функции лежат на уровень выше:
from copilot_service.conversation_manager_5 import get_multiagent_answer
from ..class_MongoDB import DataStorageManager
from ..get_chroma_document_ID import get_full_document
from ..update_delete_document_ChromaDB import add_or_update_document, delete_document

# CSS для изменения цветов кнопок по variant="primary"/"secondary"
css = """
.gradio-container button.gr-button.gr-button-secondary {
    background-color: red !important;
    color: white !important;
}

.gradio-container button.gr-button.gr-button-primary {
    background-color: green !important;
    color: white !important;
}
"""

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for file in message["files"]:
        history.append({"role": "user", "content": {"path": file}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    """
    Генератор, формирующий ответ для пользователя.
    """
    user_text = ""
    for msg in reversed(history):
        if msg["role"] == "user" and isinstance(msg["content"], str):
            user_text = msg["content"]
            break

    unique_id = str(uuid.uuid4())
    answer = get_multiagent_answer(
        user_text=user_text,
        history_messages=history,
        request_id=unique_id
    )
    
    history.append({"role": "assistant", "content": ""})
    for character in answer:
        history[-1]["content"] += character
        # Небольшая задержка для имитации «печатания»
        time.sleep(0.01)
        yield history, unique_id

def remove_last_interaction(history):
    """
    Удаляем последнее сообщение (и ассистента, и пользователя, если есть).
    """
    if history and history[-1]["role"] == "assistant":
        history.pop()
    if history and history[-1]["role"] == "user":
        history.pop()
    return history

def decide_button_variant(lst: List[str]) -> str:
    """
    Если список непустой — возвращаем 'primary' (зелёная),
    если пустой — 'secondary' (красная).
    """
    return "primary" if lst else "secondary"

def update_cat_fields(request_id: str):
    """
    Вызывается после ответа бота. 
    Получаем из БД списки cat_1_list, cat_2_list
    и формируем строки для текстовых полей + задаём цвет кнопок.
    
    Также возвращаем сами списки (cat_1_list, cat_2_list),
    чтобы их сохранить в состояниях (State) и использовать при кликах на кнопки.
    """
    manager = DataStorageManager(
        mongo_uri="mongodb://localhost:27017",
        db_name="Chat_bot",
        collection_name="vector_search"
    )

    # Получаем списки документов для обеих категорий
    cat_1_list, cat_2_list = manager.get_cat_lists(request_id)
    print(f"[DEBUG] Получены cat_1_list: {cat_1_list}")
    print(f"[DEBUG] Получены cat_2_list: {cat_2_list}")

    # Формируем строки для текстовых полей
    if cat_1_list:
        cat_1_str = "cat_1_list:\n" + "\n".join(cat_1_list)
    else:
        cat_1_str = "cat_1_list: документов не найдено"

    if cat_2_list:
        cat_2_str = "cat_2_list:\n" + "\n".join(cat_2_list)
    else:
        cat_2_str = "cat_2_list: документов не найдено"

    # Определяем, какими будут кнопки: красными или зелёными
    cat_1_variant = decide_button_variant(cat_1_list)
    cat_2_variant = decide_button_variant(cat_2_list)

    # Возвращаем: тексты для Textbox-ов, новые варианты для кнопок,
    # а также списки документов (для последующего использования при кликах).
    return (
        cat_1_str,                       # cat_1_text (новая строка)
        cat_2_str,                       # cat_2_text (новая строка)
        gr.update(variant=cat_1_variant),  # cat_1_button → цвет
        gr.update(variant=cat_2_variant),  # cat_2_button → цвет
        cat_1_list,                      # cat_1_list_state
        cat_2_list                       # cat_2_list_state
    )

def cat_1_click_action(cat_1_list):
    """
    При клике на кнопку cat_1:
    - Берём список cat_1_list (список doc_id)
    - Вызываем get_full_document(...) для каждого doc_id
    - Склеиваем результаты в одну строку и возвращаем
    """
    if not cat_1_list:
        return "Список cat_1 пуст! Документов нет."
    result_lines = []
    for doc_id in cat_1_list:
        doc_text = get_full_document(doc_id)
        result_lines.append(f"=== Документ {doc_id} ===\n{doc_text}\n")
    return "\n".join(result_lines)

def cat_2_click_action(cat_2_list):
    """
    При клике на кнопку cat_2:
    - Берём список cat_2_list (список doc_id)
    - Вызываем get_full_document(...) для каждого doc_id
    - Склеиваем результаты в одну строку и возвращаем
    """
    if not cat_2_list:
        return "Список cat_2 пуст! Документов нет."
    result_lines = []
    for doc_id in cat_2_list:
        doc_text = get_full_document(doc_id)
        result_lines.append(f"=== Документ {doc_id} ===\n{doc_text}\n")
    return "\n".join(result_lines)

def create_tab1():
    """
    Возвращаем объект gr.Blocks со всем интерфейсом, идентичным исходному коду.
    """
    with gr.Blocks(css=css) as tab1:
        
        gr.Markdown("## Моделирование работы бота с контролем получаемых документов из векторного поиска")
        
        chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")
        request_id_state = gr.State()  # Для хранения request_id

        # Состояния для списков документов по категориям
        cat_1_list_state = gr.State()
        cat_2_list_state = gr.State()

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Введите сообщение или загрузите файл...",
            show_label=False,
            sources=["microphone", "upload"]
        )
        
        remove_button = gr.Button(value="Удалить последнее сообщение")

        gr.Markdown("### Категории документов, где cat_1 документы содержашие прямую информацию на сообшение и cat_2 — содержат косвенную информацию ")

        # Первая строка (cat_1_button + cat_1_text)
        with gr.Row():
            cat_1_button = gr.Button(
                value="cat_1", 
                scale=1,        
                min_width=0,
                variant="secondary"  # Изначально красная
            )
            cat_1_text = gr.Textbox(
                label="Документы cat_1", 
                lines=3, 
                value="Здесь появятся документы cat_1", 
                interactive=False,
                scale=1,
                min_width=0
            )

        # Вторая строка (cat_2_button + cat_2_text)
        with gr.Row():
            cat_2_button = gr.Button(
                value="cat_2", 
                scale=1,
                min_width=0,
                variant="secondary"  # Изначально красная
            )
            cat_2_text = gr.Textbox(
                label="Документы cat_2",
                lines=3,
                value="Здесь появятся документы cat_2", 
                interactive=False,
                scale=1,
                min_width=0
            )

        # Поле, куда выводим результат нажатия кнопок cat_1 / cat_2
        cat_output = gr.Textbox(
            label="Что бы увидеть содержания найденных документов нажмите соотвесвтуюшие кнопки cat_1 / cat_2",
            value="Проверка содержания документов",
            interactive=False
        )

        # Кнопка «Удалить последнее сообщение»
        remove_button.click(
            fn=remove_last_interaction,
            inputs=chatbot,
            outputs=chatbot
        )

        # Сабмит сообщения (ввод пользователя + файлы)
        chat_msg = chat_input.submit(
            fn=add_message, 
            inputs=[chatbot, chat_input], 
            outputs=[chatbot, chat_input]
        )

        # Генерация ответа бота (функция bot)
        bot_msg = chat_msg.then(
            fn=bot,
            inputs=chatbot,
            outputs=[chatbot, request_id_state],
        )
        
        # Снова делаем поле ввода интерактивным после генерации ответа
        bot_msg.then(
            fn=lambda: gr.MultimodalTextbox(interactive=True),
            inputs=None,
            outputs=[chat_input]
        )

        # Обновляем текст в полях категорий, цвет кнопок и состояния со списками
        bot_msg.then(
            fn=update_cat_fields,
            inputs=request_id_state,
            outputs=[cat_1_text, cat_2_text, cat_1_button, cat_2_button, cat_1_list_state, cat_2_list_state]
        )

        # Лайки / дизлайки
        chatbot.like(print_like_dislike, None, None, like_user_message=True)

        # При нажатии на кнопку cat_1
        cat_1_button.click(
            fn=cat_1_click_action,
            inputs=cat_1_list_state,  
            outputs=cat_output
        )

        # При нажатии на кнопку cat_2
        cat_2_button.click(
            fn=cat_2_click_action,
            inputs=cat_2_list_state,
            outputs=cat_output
        )

    # Возвращаем полностью настроенный интерфейс в виде Blocks
    return tab1```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab1/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/PromptManager.py
```python
import json


class PromptManager:
    """
    Класс для подготовки массива messages (list[dict]), 
    который затем будет передаваться в OpenAIChatBot.
    """
    
    def build_contradiction_detector_json(self, prompt: str, json_data: dict, user_query: str) -> list:
        """
        Формирует сообщение, где содержатся инструкции классификатора, 
        вопрос (сообщение) пользователя и блок тройных кавычек с анализируемым JSON-документом.

        :param prompt: Текст системного промта (инструкции для классификатора).
        :param json_data: Документ (dict, list и т.п.), который нужно сериализовать в JSON.
        :param user_query: Сообщение пользователя (строка), относительно которого
                           необходимо проводить классификацию документа.
        :return: список сообщений в формате [{'role': 'user', 'content': '...'}].
        """
        # Превращаем объект в красивый (многострочный) JSON
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

        # Формируем итоговый контент
        # prompt здесь содержит основные инструкции, например:
        #   "Вы — ассистент, выполняющий роль классификатора документов по их релевантности ... (и т.д.)"
        # См. детальную структуру ниже
        content = (
            f"{prompt}\n\n"
            f"**Новый документ:** «{user_query}»\n\n"
            f"**Релевантные документы из базы** (каждый документ в формате: идентификатор, сообшение пользователья - ответ на сообшение пользователя ):\"\"\"{json_str}\"\"\""
        )

        return [
            {
                "role": "user",
                "content": content
            }
        ]
    
    def build_prompt_with_json(self, prompt: str, json_data: dict, user_query: str) -> list:
        """
        Формирует сообщение, где содержатся инструкции классификатора, 
        вопрос (сообщение) пользователя и блок тройных кавычек с анализируемым JSON-документом.

        :param prompt: Текст системного промта (инструкции для классификатора).
        :param json_data: Документ (dict, list и т.п.), который нужно сериализовать в JSON.
        :param user_query: Сообщение пользователя (строка), относительно которого
                           необходимо проводить классификацию документа.
        :return: список сообщений в формате [{'role': 'user', 'content': '...'}].
        """
        # Превращаем объект в красивый (многострочный) JSON
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

        # Формируем итоговый контент
        # prompt здесь содержит основные инструкции, например:
        #   "Вы — ассистент, выполняющий роль классификатора документов по их релевантности ... (и т.д.)"
        # См. детальную структуру ниже
        content = (
            f"{prompt}\n\n"
            f"Вопрос (сообщение) пользователя: «{user_query}»\n\n"
            f"Анализируемый документ: \"\"\"{json_str}\"\"\""
        )

        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def build_simple_messages(self, user_message: str) -> list:
        """
        Сценарий: только пользовательское сообщение,
        без дополнительных инструкций.
        """
        return [
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_developer_messages(self, user_message: str, developer_instruction: str) -> list:
        """
        Сценарий: есть developer-инструкция + пользовательское сообщение.
        """
        return [
            {
                "role": "developer",
                "content": developer_instruction
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_json_schema_messages(self, user_message: str) -> list:
        """
        Сценарий: сообщить модели, что нужно вывести результат в JSON.
        Можно расширить под ваши схемы.
        """
        return [
            {
                "role": "developer",
                "content": "You will output JSON according to the provided schema."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_conversation_with_history(self, user_message: str, conversation_history: list) -> list:
        """
        Сценарий: добавляем уже имеющуюся историю диалога (conversation_history),
        а затем текущее сообщение user.
        
        conversation_history — это list[dict], где каждый словарь имеет 
        роль 'user' или 'assistant' и 'content'.
        """
        # Копируем историю, затем добавляем новое сообщение
        messages = []
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        return messages```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/update_delete_document_ChromaDB.py
```python
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


```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/test.py
```python
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
    main()```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/ChromaDBManager.py
```python
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
        print(f"Документ с ID={doc_id} успешно удалён.")```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/class_MongoDB.py
```python
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
    print("cat_2:", cat2_list)```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab5/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab5/tab5_code.py
```python
import gradio as gr

def create_tab5():
    with gr.Blocks() as tab5:
        gr.Markdown("## Инструкция")
        gr.Markdown(
            """
            Здесь будут находиться инструкции или документация по работе с приложением.
            На данный момент раздел является тестовой заглушкой, вы можете разместить 
            здесь любую информацию, которая поможет пользователям разобраться, как 
            пользоваться вашим сервисом.
            """
        )

        # Можно добавить тестовый контент, например, условный список рекомендаций
        gr.Markdown(
            """
            ### Пример раздела рекомендаций:
            1. Перед загрузкой файлов убедитесь, что они имеют корректный формат.
            2. После загрузки данных перейдите во вкладку "Тестиорование ответов" для проверки работы модели.
            3. Для добавления новых документов используйте соответствующую вкладку.
            4. При необходимости редактирования или удаления - перейдите в раздел "Редактирование и удаление документов".
            """
        )
    return tab5```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab2/tab2_code.py
```python
import gradio as gr
from typing import Optional
import json

# Импортируем нужные функции
from ..update_delete_document_ChromaDB import add_or_update_document
from ..analyze_query_with_chroma import analyze_query_with_chroma
from ..get_chroma_document_ID import get_full_document

def create_tab2():
    with gr.Blocks() as tab2:
        gr.Markdown("## Добавляем документ в базу данных")
        
        # Кастомный CSS для настройки ширины текстовых полей
        gr.HTML(
            """
            <style>
                .custom-textbox textarea {
                    width: 500px; /* Установите желаемую ширину */
                }
            </style>
            """
        )
        
        # Поле для ввода пользовательского вопроса/сообщения
        user_question = gr.Textbox(
            lines=4, 
            label="Вопрос-сообщение пользователя", 
            elem_classes="custom-textbox"
        )

        # Поле для ввода ответа на вопрос/сообщение пользователя
        user_answer = gr.Textbox(
            lines=4, 
            label="Ответ на вопрос-сообщение пользователя", 
            elem_classes="custom-textbox"
        )

        # Поле для отображения результата сохранения
        save_result = gr.Textbox(label="Результат операции", interactive=False)
        
        # Новое поле для вывода результатов анализа
        analysis_result = gr.Textbox(
            label="Результат анализа на противоречия",
            lines=10,
            interactive=False
        )

        # --- Функции-обработчики кнопок ---
        
        # 1) Сохранение документа в базе данных
        def on_save(user_q, user_ans):
            add_or_update_document(
                new_text=user_q,  
                new_metadata={"system_response": user_ans},
                doc_id=None
            )
            return "Документ успешно сохранён в базе данных!"

        # 2) Очистка полей
        def clear_fields():
            return "", "", "", ""

        # 3) Анализ на противоречия
        def on_analyze(user_q, user_ans):
            # Собираем строку для анализа
            combined_text = f"{user_q}\n-----\n{user_ans}"
            
            # Вызываем функцию анализа
            results = analyze_query_with_chroma(combined_text)
            if not results:
                return "Результатов нет или функция вернула пустой список."

            # Сюда будем собирать описание каждого документа
            analysis_details = []
            
            for i, doc in enumerate(results, start=1):
                # Предполагаем, что внутри doc есть ключ 'response_json',
                # в котором содержится анализ, и usage и т.д.
                response_json = doc.get("response_json", {})
                
                # response_json["analysis"] может быть JSON-строкой или уже объектом
                analysis_data = response_json.get("analysis", {})
                
                # Попробуем спарсить, если это строка
                if isinstance(analysis_data, str):
                    try:
                        analysis_data = json.loads(analysis_data)
                    except json.JSONDecodeError:
                        # Если почему-то парсинг не удался, оставляем как есть
                        pass
                
                # Из analysis_data достаём поля
                doc_id = analysis_data.get("doc_id")
                contradiction_found = analysis_data.get("contradiction_found", False)
                contradiction_description = analysis_data.get("contradiction_description", "нет описания")

                # Проверяем наличие противоречия
                if contradiction_found:
                    # Получаем содержимое документа по doc_id
                    doc_info = get_full_document(doc_id)
                    # Формируем описательную строку для вывода
                    info_text = (
                        f"Документ #{i} (ID: {doc_id}) содержит ПРOТИВОРЕЧИЕ.\n"
                        f"Описание противоречия: {contradiction_description}\n"
                        f"Содержимое документа:\n{doc_info.get('text', 'Текст не найден.')}\n"
                        "------------------------------------"
                    )
                else:
                    # Если нет противоречий, просто выводим описание
                    info_text = (
                        f"Документ #{i} (ID: {doc_id}) - Нет противоречий.\n"
                        f"Описание: {contradiction_description}\n"
                        "------------------------------------"
                    )
                
                analysis_details.append(info_text)
            
            # Возвращаем текст для поля analysis_result
            return "\n".join(analysis_details)

        # --- Размещение кнопок в одной строке ---
        with gr.Row():
            save_button = gr.Button("Сохранить документ в базе данных")
            analyze_button = gr.Button("Анализ на противоречия")
            clear_button = gr.Button("Очистить поля")

        # --- Привязка кнопок ---
        # Кнопка "Сохранить документ в базе данных"
        save_button.click(
            fn=on_save,
            inputs=[user_question, user_answer],
            outputs=save_result
        )

        # Кнопка "Очистить поля"
        clear_button.click(
            fn=clear_fields,
            inputs=None,
            outputs=[user_question, user_answer, save_result, analysis_result]
        )

        # Кнопка "Анализ на противоречия"
        analyze_button.click(
            fn=on_analyze,
            inputs=[user_question, user_answer],
            outputs=analysis_result
        )

    return tab2```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab2/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab3/tab3_code.py
```python
import gradio as gr

# Импорт необходимых функций
from ..get_chroma_document_ID import get_full_document
from ..update_delete_document_ChromaDB import add_or_update_document, delete_document

def create_tab3():
    """
    Создаёт вкладку с интерфейсом для работы с документами:
      1) Ввод doc_id (до 50 символов)
      2) Кнопка "Редактировать документ" -> вызывает get_full_document(doc_id)
      3) Кнопка "Удалить документ" -> вызывает delete_document(doc_id)
      4) Текстовое поле для вывода данных документа
      5) Поле "Новая редакция сообщения пользователя"
      6) Поле "Новая редакция ответа"
      7) Кнопка "Сохранить документ" (удаляет старый и пересохраняет)
      8) Кнопка "Очистить поля", обнуляющая все поля ввода/вывода
    """
    with gr.Blocks() as tab3:
        gr.Markdown("## Работа с документами редактирование и удаление")

        # 1) Поле ввода doc_id (до 50 символов)
        doc_id_input = gr.Textbox(
            label="Идентификатор документа (doc_id)",
            max_length=50,
            placeholder="Введите идентификатор документа (doc_id)"
        )

        # Кнопки "Редактировать", "Удалить", "Очистить"
        with gr.Row():
            edit_button = gr.Button("Редактировать документ")
            delete_button = gr.Button("Удалить документ")
            clear_button = gr.Button("Очистить поля")  # Новая кнопка очистки

        # Поле для отображения информации о документе
        doc_info_output = gr.Textbox(
            label="Информация о документе",
            lines=5,
            interactive=False
        )

        # Поле "Новая редакция сообщения пользователя"
        new_user_text_input = gr.Textbox(
            label="Новая редакция сообщения пользователя",
            lines=3,
            placeholder="Введите новую редакцию текста, который был внесён пользователем"
        )

        # Поле "Новая редакция ответа"
        new_answer_text_input = gr.Textbox(
            label="Новая редакция ответа",
            lines=3,
            placeholder="Введите новую редакцию ответа, который генерирует ваш сервис/бот"
        )

        # Кнопка "Сохранить документ"
        save_button = gr.Button("Сохранить документ")

        # ============= ЛОГИКА ФУНКЦИЙ =============

        def edit_document(doc_id):
            """
            Нажатие "Редактировать документ".
            Загружает данные документа и возвращает их в виде строки.
            """
            try:
                doc_info = get_full_document(doc_id)
                if not doc_info:
                    return "Документ не найден или не удалось загрузить данные."
                
                doc_str = (
                    f"ID: {doc_info.get('id', '')}\n"
                    f"Text: {doc_info.get('text', '')}\n"
                    f"Metadata: {doc_info.get('metadata', '')}"
                )
                return doc_str
            except Exception as e:
                return f"Ошибка при получении документа: {str(e)}"

        def remove_document(doc_id):
            """
            Нажатие "Удалить документ".
            Удаляет документ и возвращает сообщение об успехе/неуспехе.
            """
            try:
                delete_document(doc_id)
                return "Документ успешно удалён."
            except Exception as e:
                return f"Ошибка при удалении документа: {str(e)}"

        def save_document(doc_id, new_user_text, new_answer_text):
            """
            Нажатие "Сохранить документ".
            Сначала удаляет старый документ (если он есть),
            затем создаёт новый с обновлёнными данными.
            """
            try:
                # Удаляем старый (если существует)
                try:
                    delete_document(doc_id)
                except Exception:
                    pass  # Игнорируем ошибку, если документа нет

                # Создаём документ
                new_metadata = {"system_response": new_answer_text}

                add_or_update_document(
                    new_text=new_user_text,
                    new_metadata=new_metadata,
                    doc_id=doc_id
                )
                return "Документ успешно пересоздан (удалён и сохранён заново)."
            except Exception as e:
                return f"Ошибка при сохранении документа: {str(e)}"

        def clear_fields():
            """
            Функция очистки всех полей.
            Возвращает пустые значения для:
              - doc_id_input
              - doc_info_output
              - new_user_text_input
              - new_answer_text_input
            """
            return "", "", "", ""

        # ============= ПРИВЯЗКА КНОПОК =============

        # "Редактировать документ" -> показывает информацию о документе
        edit_button.click(
            fn=edit_document,
            inputs=doc_id_input,
            outputs=doc_info_output
        )

        # "Удалить документ" -> удаляет документ
        delete_button.click(
            fn=remove_document,
            inputs=doc_id_input,
            outputs=doc_info_output
        )

        # "Сохранить документ"
        save_button.click(
            fn=save_document,
            inputs=[doc_id_input, new_user_text_input, new_answer_text_input],
            outputs=doc_info_output
        )

        # "Очистить поля" -> обнуляет все поля
        clear_button.click(
            fn=clear_fields,
            inputs=None,
            outputs=[doc_id_input, doc_info_output, new_user_text_input, new_answer_text_input]
        )

    return tab3```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab3/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab4/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab4/tab4_code.py
```python
import gradio as gr
import pandas as pd

def convert_document_to_csv(file) -> list:
    """
    Тестовая функция для проверки работоспособности.
    В реальном случае здесь должна быть логика:
      - чтение TXT или DOC/DOCX
      - извлечение из них контента
      - преобразование в CSV-формат (например, pandas.DataFrame и т.д.)

    Сейчас возвращаем список списков (аналог строк CSV).
    Первая строка — заголовки, последующие — данные.
    """
    # Пример: 3 колонки и несколько строк.
    fake_data = [
        ["Колонка1", "Колонка2", "Колонка3"],
        ["Значение11", "Значение12", "Значение13"],
        ["Значение21", "Значение22", "Значение23"],
    ]
    return fake_data

def create_tab4():
    with gr.Blocks() as tab4:
        gr.Markdown("## Загрузка данных")
        gr.Markdown(
            """
            Здесь можно загрузить файлы или данные, которые будут использоваться вашим приложением.
            Данный раздел пока что является заглушкой для тестирования.
            """
        )
        
        file_input = gr.File(label="Выберите файл(ы) для загрузки", file_count="multiple")
        upload_button = gr.Button("Загрузить")

        output_text = gr.Textbox(label="Результат загрузки")
        # Здесь будем отображать CSV-данные
        output_df = gr.DataFrame(
            label="Проверка конвертации в CSV",
            interactive=True
        )

        def upload_files(files):
            """
            1. Проверяем, выбраны ли файлы.
            2. Берём первый файл (для примера).
            3. Конвертируем его в CSV (тестовая функция).
            4. Возвращаем текст о количестве файлов и саму таблицу.
            """
            if not files:
                return "Файлы не выбраны.", None
            
            # Берём первый файл (для упрощения).
            file = files[0]
            
            # Вызываем тестовую функцию конвертации.
            csv_data = convert_document_to_csv(file)
            
            # Опционально можно сразу превратить csv_data в pandas.DataFrame,
            # но gr.DataFrame умеет работать и со списком списков.
            # Если хотите pandas, делаем так:
            # df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            # return (f"Загружен файл: {file.name}", df)

            return (f"Загружено файлов: {len(files)}\nОбрабатываем: {file.name}", csv_data)

        # Настраиваем, чтобы при клике вызывалась наша функция,
        # а результат выводился в Textbox и DataFrame.
        upload_button.click(
            upload_files,
            inputs=[file_input],
            outputs=[output_text, output_df]
        )
        
    return tab4


# Если вы запускаете весь скрипт напрямую (например, python main.py),
# в конце можно запустить Gradio:
if __name__ == "__main__":
    demo = create_tab4()
    demo.launch()```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/get_chroma_document_ID.py
```python

#!/usr/bin/env python3
# Скрипт: get_chroma_document_ID.py
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
        print(json.dumps(docs_data, ensure_ascii=False, indent=2))```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/OpenAIChatBot.py
```python


import os
import base64
from typing import Optional, Dict, Any, Union
from io import BufferedReader, BytesIO

# Если используете пакет python-dotenv для чтения .env:
from dotenv import load_dotenv
load_dotenv()

# Официальная Python-библиотека OpenAI, соответствующая актуальной документации:
from openai import OpenAI


class OpenAIChatBot:
    """
    Пример класса, который демонстрирует методы:
      1) Отправка текстового сообщения и получение ответа.
      2) Отправка текстового сообщения в JSON формате.
      3) Передача изображения (в base64) для анализа.
      4) Транскрибация аудио и генерация ответа на полученный текст.
      5) Получение векторных эмбеддингов для текста.
      6) Преобразование текста в речь (TTS).

    Параметры temperature, top_p, presence_penalty и т.д.
    можно задавать при инициализации или прямо в методах.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        default_text_model: str = "gpt-4o",
        default_temperature: float = 0,
        default_max_tokens: int = 5000,
        default_top_p: float = 1.0,
        default_presence_penalty: float = 0.0,
        default_frequency_penalty: float = 0.0,
        default_n: int = 1,
    ):
        """
        :param openai_api_key: Ключ для доступа к OpenAI API. Если не указан, берём из окружения.
        :param default_text_model: Модель, используемая по умолчанию при генерации текстовых ответов.
        :param default_temperature: Параметр 'temperature' (стохастичность вывода).
        :param default_max_tokens: Лимит на кол-во генерируемых токенов ответа.
        :param default_top_p: Параметр 'top_p' (nucleus sampling).
        :param default_presence_penalty: Штраф за «присутствие» темы (presence penalty).
        :param default_frequency_penalty: Штраф за частоту упоминаний (frequency penalty).
        :param default_n: Сколько вариантов ответа генерировать за один запрос.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не задан ни напрямую, ни в переменных окружения.")

        # Инициализируем клиент
        self.client = OpenAI(api_key=self.api_key)

        # Параметры по умолчанию
        self.default_text_model = default_text_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_top_p = default_top_p
        self.default_presence_penalty = default_presence_penalty
        self.default_frequency_penalty = default_frequency_penalty
        self.default_n = default_n

    # -------------------------------------------------------------------------------------------
    # 1) Метод для генерации текстового ответа (chat.completions)
    # -------------------------------------------------------------------------------------------
    def generate_text_response(
        self,
        user_message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Отправляет текст в модель, возвращает ответ + usage.

        :param user_message: Текст запроса.
        :param model: Название модели (по умолчанию self.default_text_model).
        :param temperature, max_tokens, top_p, presence_penalty, frequency_penalty, n:
                      Переопределения параметров генерации.
        :return: dict с ключами "response_text" и "usage" (при наличии).
        """
        chosen_model = model or self.default_text_model
        chosen_temperature = temperature if (temperature is not None) else self.default_temperature
        chosen_max_tokens = max_tokens if (max_tokens is not None) else self.default_max_tokens
        chosen_top_p = top_p if (top_p is not None) else self.default_top_p
        chosen_presence_penalty = (presence_penalty if (presence_penalty is not None)
                                   else self.default_presence_penalty)
        chosen_frequency_penalty = (frequency_penalty if (frequency_penalty is not None)
                                    else self.default_frequency_penalty)
        chosen_n = n if (n is not None) else self.default_n

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "user", "content": user_message}
            ],
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            top_p=chosen_top_p,
            presence_penalty=chosen_presence_penalty,
            frequency_penalty=chosen_frequency_penalty,
            n=chosen_n,
            store=True  # рекомендует документация, если хотим хранить историю
        )

        answer_text = response.choices[0].message.content if response.choices else ""

        usage_data = {}
        if hasattr(response, "usage") and response.usage is not None:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 2) Генерация JSON-ответа, используя response_format = json_schema
    # -------------------------------------------------------------------------------------------
    def generate_json_response(
        self,
        user_message: str,
        json_schema: Dict[str, Any],
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Генерация структурированного JSON согласно schema (см. раздел "Generate JSON data" из документации).

        :param user_message: входной текст
        :param json_schema: описание схемы JSON
        :param model: Модель для JSON-вывода (например "gpt-4o-2024-08-06")
        :param temperature: как правило 0, чтобы модель не «фантазировала»
        :return: {"response_json": dict, "usage": {...}} — JSON + usage-токены
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You will output JSON according to the provided schema."},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            },
            temperature=temperature,
            store=True
        )

        # Пытаемся парсить:
        import json
        response_str = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed_json = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_json = {"error": "JSON parse error", "raw_response": response_str}

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_json": parsed_json,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 3) Анализ изображения (байты/base64) через chat-completions
    # -------------------------------------------------------------------------------------------
    def analyze_image(
        self,
        image_bytes_base64: str,
        user_question: str = "What is in this image?",
        model: str = "gpt-4o-mini",
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """
        Принимает base64-изображение (можно сделать data:image/*;base64,...) + вопрос. Возвращает описание.

        :param image_bytes_base64: Base64-кодированное изображение (возможно с префиксом 'data:image/...').
        :param user_question: вопрос
        :param model: например "gpt-4o" или "gpt-4o-mini"
        :param detail: "low", "high" или "auto" — детализация
        :return: {"response_text": str, "usage": {...}}
        """
        if not image_bytes_base64.startswith("data:image"):
            # Если нет префикса, добавим
            image_bytes_base64 = f"data:image/jpeg;base64,{image_bytes_base64}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_bytes_base64,
                                "detail": detail
                            }
                        }
                    ],
                },
            ],
            store=True,
        )

        answer_text = response.choices[0].message.content if response.choices else ""
        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 4) Обработка голосового сообщения: транскрибация + ответ
    # -------------------------------------------------------------------------------------------
    def handle_voice_message(
        self,
        audio_file: Union[str, BufferedReader],
        question: str = "Можешь ответить на это сообщение?",
        transcription_model: str = "whisper-1",
        text_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        1) Транскрибирует аудио (whisper).
        2) Использует транскрибированный текст + question => chat.completions.

        :return: {
            "transcribed_text": str,
            "response_text": str,
            "usage_transcription": None,  # у Whisper нет поля usage
            "usage_chat": {...}           # usage от chat
        }
        """
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                transcription_resp = self.client.audio.transcriptions.create(
                    file=f,
                    model=transcription_model,
                )
        else:
            transcription_resp = self.client.audio.transcriptions.create(
                file=audio_file,
                model=transcription_model,
            )

        transcribed_text = transcription_resp.text

        # Формируем финальный prompt
        combined_prompt = f"Пользователь сказал (голосом): '{transcribed_text}'.\nВопрос: {question}"

        # Генерируем ответ
        text_model = text_model or self.default_text_model
        chat_answer = self.generate_text_response(
            user_message=combined_prompt,
            model=text_model
        )

        return {
            "transcribed_text": transcribed_text,
            "response_text": chat_answer["response_text"],
            "usage_transcription": None,        # нет usage для audio.transcriptions
            "usage_chat": chat_answer["usage"]  # usage от chat
        }

    # -------------------------------------------------------------------------------------------
    # 5) Получение эмбеддинга (embedding) для текста
    # -------------------------------------------------------------------------------------------
    def get_text_embedding(
        self,
        text: str,
        embedding_model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Генерация векторных эмбеддингов (embeddings) для указанного текста.

        :param text: Входной текст
        :param embedding_model: ID модели, напр. 'text-embedding-3-small'
        :param dimensions: Если нужно укоротить вектор, укажите кол-во измерений.
        :return: {"embedding": List[float], "usage": {...}}
        """
        kwargs = {
            "model": embedding_model,
            "input": text,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = self.client.embeddings.create(**kwargs)
        embedding_vector = response.data[0].embedding

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "embedding": embedding_vector,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 6) Преобразование текста в речь (Text-To-Speech) - TTS
    # -------------------------------------------------------------------------------------------
    def text_to_speech(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: Optional[str] = None,  # например "mp3" (по умолчанию), "wav", "aac" и т.п.
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Превращает текст в аудио. Возвращает байты, либо сохраняет в файл, если указан save_path.

        :param text: Исходная текстовая строка
        :param model: Модель TTS (напр. 'tts-1' или 'tts-1-hd')
        :param voice: Голос ('alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer')
        :param response_format: Формат аудио (см. доку: 'mp3' по умолчанию, 'opus', 'aac', 'flac', 'wav', 'pcm' и т.д.)
        :param save_path: Путь для записи файла; если None, вернём байты
        :return: {
            "audio_content": bytes | None,
            "voice_used": str,
            "model_used": str,
            "note": "..."
        }
        """
        # Формируем запрос к audio.speech.create
        create_kwargs = {
            "model": model,
            "voice": voice,
            "input": text
        }
        if response_format:  # Если хотим явно указать формат
            create_kwargs["response_format"] = response_format

        # Получаем объект-ответ, который поддерживает stream_to_file(...)
        response = self.client.audio.speech.create(**create_kwargs)

        audio_content = None
        if save_path:
            # Сохраняем напрямую
            response.stream_to_file(save_path)
        else:
            # Запишем в BytesIO, чтобы получить байты "на руки"
            buffer = BytesIO()
            response.stream_to_file(buffer)
            audio_content = buffer.getvalue()

        return {
            "audio_content": audio_content,
            "voice_used": voice,
            "model_used": model,
            "note": (
                "По документации TTS не возвращает usage. Формат по умолчанию mp3, "
                "либо задан через response_format."
            )
        }
        
        
     # -------------------------------------------------------------------------------------------
    # НОВЫЙ МЕТОД: Генерация ответа из заранее подготовленного messages
    # -------------------------------------------------------------------------------------------
    def generate_chat_response_from_messages(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,  # <-- Добавляем seed
    ) -> Dict[str, Any]:
        """
        Получает на вход список messages (список dict), 
        затем вызывает chat.completions.

        Пример использования:
            pm = PromptManager()
            msgs = pm.build_simple_messages("Hello!")
            bot.generate_chat_response_from_messages(msgs)

        Возвращает dict: {
          "response_text": str,
          "usage": {...}  # при наличии
        }
        """
        chosen_model = model or self.default_text_model
        chosen_temperature = temperature if (temperature is not None) else self.default_temperature
        chosen_max_tokens = max_tokens if (max_tokens is not None) else self.default_max_tokens
        chosen_top_p = top_p if (top_p is not None) else self.default_top_p
        chosen_presence_penalty = (presence_penalty if (presence_penalty is not None)
                                   else self.default_presence_penalty)
        chosen_frequency_penalty = (frequency_penalty if (frequency_penalty is not None)
                                    else self.default_frequency_penalty)
        chosen_n = n if (n is not None) else self.default_n

        # Выполняем запрос
        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            top_p=chosen_top_p,
            presence_penalty=chosen_presence_penalty,
            frequency_penalty=chosen_frequency_penalty,
            n=chosen_n,
            store=True
        )

        # Извлекаем текст ответа
        answer_text = ""
        if response.choices and response.choices[0].message:
            answer_text = response.choices[0].message.content

        # Извлекаем usage (если есть)
        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # НОВЫЙ МЕТОД: аналог для JSON-схем, если хотите жёстко задать response_format
    # -------------------------------------------------------------------------------------------
    def generate_chat_response_with_jsonschema(
        self,
        messages: list,
        json_schema: Dict[str, Any],
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        seed: Optional[int] = None,  # <-- Добавляем seed
    ) -> Dict[str, Any]:
        """
        Если хотите использовать заранее сформированное messages, 
        но с жёстким требованием, что модель должна выдавать JSON по schema.
        
        Пример:
            pm = PromptManager()
            msgs = pm.build_json_schema_messages("My text ...")
            schema = {...}
            result = bot.generate_chat_response_with_jsonschema(msgs, schema)
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            },
            temperature=temperature,
            store=True
        )

        import json

        response_str = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed_json = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_json = {"error": "JSON parse error", "raw_response": response_str}

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_json": parsed_json,
            "usage": usage_data
        }   ```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/PROMPT.py
```python

contradiction_detector = """
Ты – продвинутый помощник, эксперт в области анализа текстов. Твоя задача – оценивать новый документ (новую информацию) на предмет наличия противоречий с уже имеющимися документами из базы знаний. 

Ты должен:
1. Получать на вход текст нового документа и набор (выборку) наиболее похожих или релевантных документов (из векторной базы данных).
2. Сравнивать новые утверждения с утверждениями из ранее внесённых документов.
3. Выносить заключение:
   a) Есть ли противоречия (прямые или косвенные).
   b) Если противоречия есть, в чём именно они состоят.
   c) Если противоречий нет, указать, что всё согласуется.

Отвечай максимально чётко и структурированно. Избегай избыточных формулировок, не сочиняй фактов. Если некоторые утверждения не могут быть проверены из-за отсутствия релевантного материала в базе, отдельно указывай это. 

Ниже я предоставляю тебе:

1. **Новый документ**, который нужно проверить на противоречия.
2. **Набор релевантных документов** (или фрагментов), извлечённых из векторной базы.

Твоя задача – сравнить текст нового документа с уже имеющимися документами и определить, существует ли между ними противоречие.

--------------

**Требования к ответу**:
1. Для каждого документа укажите :
   - '"doc_id"': идентификатор документа
   - `"contradiction_found"`: true или false
   - `"contradiction_description"`: краткое описание обнаруженного противоречия или текст «нет противоречий» (либо пометка «недостаточно данных»)
2. Вывести ответ строго в виде JSON без дополнительного оформления.
3. Не добавлять никакой другой текст, кроме требуемого JSON-объекта.
"""```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/analyze_query_with_chroma.py
```python
# python3 analyze_query_with_chroma.py


import json
from typing import Dict, Any

from OpenAIChatBot import OpenAIChatBot
from PromptManager import PromptManager
from ChromaDBManager import ChromaDBManager  # <-- Подключайте вашу реализацию
from contradiction_detector import process_documents_in_parallel

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
    analyze_query_with_chroma(user_query)```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/contradiction_detector.py
```python



import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from PROMPT import contradiction_detector

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
                model=model,
                temperature=temperature,
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

    return results```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/main.py
```python
# python3 main.py 
# или
# python3 -m bot_control_panel.main

import gradio as gr

# Импортируем функции, создающие вкладки
from .tab1.tab1_code import create_tab1
from .tab2.tab2_code import create_tab2
from .tab3.tab3_code import create_tab3

# Новые импорты
from .tab4.tab4_code import create_tab4
from .tab5.tab5_code import create_tab5

def main():
    # Создаём существующие вкладки
    tab1 = create_tab1()     # Первая вкладка: ваш мультимодальный чат-бот
    tab2 = create_tab2()     # Вторая вкладка: тестовый интерфейс
    tab3 = create_tab3()     # Третья вкладка: тестовый интерфейс
    
    # Создаём дополнительные вкладки
    tab4 = create_tab4()     # Четвёртая вкладка: Загрузка данных
    tab5 = create_tab5()     # Пятая вкладка: Инструкция
    
    # Объединяем их в одну TabbedInterface
    demo = gr.TabbedInterface(
        [tab1, tab2, tab3, tab4, tab5],
        [
            "Тестиорование ответов", 
            "Добавление документов", 
            "Редактиование и удаление документов",
            "Загрузка данных",
            "Инструкция",
        ]
    )

    # Запускаем приложение
    demo.launch()

if __name__ == "__main__":
    main()```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/main_2.py
```python
# python3 main_2.py

# python3 -m bot_control_panel.main_2
import gradio as gr
import time
import uuid
from typing import List

from copilot_service.conversation_manager_5 import get_multiagent_answer
from .class_MongoDB import DataStorageManager
from .get_chroma_document_ID import get_full_document
from .update_delete_document_ChromaDB import add_or_update_document, delete_document

# <-- Если метод get_full_document(...) находится в другом файле,
#     раскомментируйте и замените "your_module" на нужное:
# from your_module import get_full_document

# ======== ВРЕМЕННАЯ ЗАГЛУШКА =========


# CSS для изменения цветов кнопок по variant="primary"/"secondary"
css = """
button.primary {
    background-color: green !important; /* Зеленая кнопка */
    color: white !important;
}

button.secondary {
    background-color: red !important;   /* Красная кнопка */
    color: white !important;
}
"""

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for file in message["files"]:
        history.append({"role": "user", "content": {"path": file}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    """
    Генератор, формирующий ответ для пользователя.
    """
    user_text = ""
    for msg in reversed(history):
        if msg["role"] == "user" and isinstance(msg["content"], str):
            user_text = msg["content"]
            break

    unique_id = str(uuid.uuid4())
    answer = get_multiagent_answer(
        user_text=user_text,
        history_messages=history,
        request_id=unique_id
    )
    
    history.append({"role": "assistant", "content": ""})
    for character in answer:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history, unique_id

def remove_last_interaction(history):
    """
    Удаляем последнее сообщение (и ассистента, и пользователя, если есть).
    """
    if history and history[-1]["role"] == "assistant":
        history.pop()
    if history and history[-1]["role"] == "user":
        history.pop()
    return history

def decide_button_variant(lst: List[str]) -> str:
    """
    Если список непустой — возвращаем 'primary' (зелёная),
    если пустой — 'secondary' (красная).
    """
    return "primary" if lst else "secondary"

def update_cat_fields(request_id: str):
    """
    Вызывается после ответа бота. 
    Получаем из БД списки cat_1_list, cat_2_list
    и формируем строки для текстовых полей + задаём цвет кнопок.
    
    Также возвращаем сами списки (cat_1_list, cat_2_list),
    чтобы их сохранить в состоянии (State) и использовать при кликах на кнопки.
    """
    manager = DataStorageManager(
        mongo_uri="mongodb://localhost:27017",
        db_name="Chat_bot",
        collection_name="vector_search"
    )

    # Получаем списки документов для обеих категорий
    cat_1_list, cat_2_list = manager.get_cat_lists(request_id)
    print(f"[DEBUG] Получены cat_1_list: {cat_1_list}")
    print(f"[DEBUG] Получены cat_2_list: {cat_2_list}")

    # Формируем строки для текстовых полей
    if cat_1_list:
        cat_1_str = "cat_1_list:\n" + "\n".join(cat_1_list)
    else:
        cat_1_str = "cat_1_list: документов не найдено"

    if cat_2_list:
        cat_2_str = "cat_2_list:\n" + "\n".join(cat_2_list)
    else:
        cat_2_str = "cat_2_list: документов не найдено"

    # Определяем, какими будут кнопки: красными или зелёными
    cat_1_variant = decide_button_variant(cat_1_list)
    cat_2_variant = decide_button_variant(cat_2_list)

    # Возвращаем: тексты для Textbox-ов, новые варианты для кнопок,
    # а также списки документов (для последующего использования при кликах).
    return (
        cat_1_str,                      # cat_1_text
        cat_2_str,                      # cat_2_text
        gr.update(variant=cat_1_variant),  # cat_1_button (обновление цвета)
        gr.update(variant=cat_2_variant),  # cat_2_button (обновление цвета)
        cat_1_list,                     # Сохраняем список doc_id для cat_1
        cat_2_list                      # Сохраняем список doc_id для cat_2
    )

def cat_1_click_action(cat_1_list):
    """
    При клике на кнопку cat_1:
    - Берём список cat_1_list (список doc_id)
    - Вызываем get_full_document(...) для каждого doc_id
    - Склеиваем результаты в одну строку и возвращаем
    """
    if not cat_1_list:
        return "Список cat_1 пуст! Документов нет."

    result_lines = []
    for doc_id in cat_1_list:
        doc_text = get_full_document(doc_id)
        result_lines.append(f"=== Документ {doc_id} ===\n{doc_text}\n")

    return "\n".join(result_lines)

def cat_2_click_action(cat_2_list):
    """
    При клике на кнопку cat_2:
    - Берём список cat_2_list (список doc_id)
    - Вызываем get_full_document(...) для каждого doc_id
    - Склеиваем результаты в одну строку и возвращаем
    """
    if not cat_2_list:
        return "Список cat_2 пуст! Документов нет."

    result_lines = []
    for doc_id in cat_2_list:
        doc_text = get_full_document(doc_id)
        result_lines.append(f"=== Документ {doc_id} ===\n{doc_text}\n")

    return "\n".join(result_lines)


##################  INTERFACE  ##################
with gr.Blocks(css=css) as demo:
    gr.Markdown("## Мультимодальный чат-бот с поддержкой Markdown, изображений, аудио и видео")
    
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")
    request_id_state = gr.State()  # Состояние для хранения request_id
    
    # Дополнительные state для хранения списков doc_id по каждой категории
    cat_1_list_state = gr.State()
    cat_2_list_state = gr.State()

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Введите сообщение или загрузите файл...",
        show_label=False,
        sources=["microphone", "upload"]
    )
    
    remove_button = gr.Button(value="Удалить последнее сообщение")

    gr.Markdown("### Управление категориями (по 50% ширины на кнопку и поле)")

    # Первая строка (cat_1_button + cat_1_text)
    with gr.Row():
        cat_1_button = gr.Button(
            value="cat_1", 
            scale=1,        
            min_width=0,
            variant="secondary"  # Изначально красная
        )
        cat_1_text = gr.Textbox(
            label="Документы cat_1", 
            lines=3, 
            value="Здесь появятся документы cat_1", 
            interactive=False,
            scale=1,
            min_width=0
        )

    # Вторая строка (cat_2_button + cat_2_text)
    with gr.Row():
        cat_2_button = gr.Button(
            value="cat_2", 
            scale=1,
            min_width=0,
            variant="secondary"  # Изначально красная
        )
        cat_2_text = gr.Textbox(
            label="Документы cat_2",
            lines=3,
            value="Здесь появятся документы cat_2", 
            interactive=False,
            scale=1,
            min_width=0
        )

    # Поле, куда выводим результат нажатия кнопок cat_1 / cat_2
    cat_output = gr.Textbox(
        label="Результат нажатия кнопки cat_1 / cat_2",
        value="Здесь будет выводиться тестовый текст",
        interactive=False
    )

    # Кнопка удаления последнего сообщения
    remove_button.click(
        fn=remove_last_interaction,
        inputs=chatbot,
        outputs=chatbot
    )

    # Ввод сообщения → добавление сообщения в историю
    chat_msg = chat_input.submit(
        fn=add_message, 
        inputs=[chatbot, chat_input], 
        outputs=[chatbot, chat_input]
    )
    
    # Генератор, формирующий ответ
    bot_msg = chat_msg.then(
        fn=bot,
        inputs=chatbot,
        outputs=[chatbot, request_id_state],
    )
    
    # Делаем поле ввода снова интерактивным после окончания генерации
    bot_msg.then(
        fn=lambda: gr.MultimodalTextbox(interactive=True),
        inputs=None,
        outputs=[chat_input]
    )

    # После ответа бота обновляем текстовые поля + цвет кнопок + state со списками doc_id
    bot_msg.then(
        fn=update_cat_fields,
        inputs=request_id_state,
        outputs=[cat_1_text, cat_2_text, cat_1_button, cat_2_button, cat_1_list_state, cat_2_list_state]
    )

    # Лайки / дизлайки сообщений
    chatbot.like(print_like_dislike, None, None, like_user_message=True)

    # При нажатии на кнопку cat_1:
    #  - В качестве inputs -> cat_1_list_state (список doc_id)
    #  - Вызываем cat_1_click_action -> возвращает текст, который отобразим в cat_output
    cat_1_button.click(
        fn=cat_1_click_action,
        inputs=cat_1_list_state,  
        outputs=cat_output
    )

    # При нажатии на кнопку cat_2:
    cat_2_button.click(
        fn=cat_2_click_action,
        inputs=cat_2_list_state,
        outputs=cat_output
    )

demo.launch()```

