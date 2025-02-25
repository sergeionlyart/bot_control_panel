# Project Structure
```
/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel
├── ChromaDBManager.py
├── OpenAIChatBot.py
├── PROMPT.py
├── PromptManager.py
├── README.md
├── TestInfoRetriever.py
├── __init__.py
├── analyze_query_with_chroma.py
├── chroma_parallel_analyzer_search.py
├── class_MongoDB.py
├── contradiction_detector.py
├── conversation_manager_5.py
├── export_to_md.sh
├── generate_qa_pairs.py
├── get_chroma_document_ID.py
├── main.py
├── main_2.py
├── monitor_usage.py
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
├── tab6
│   ├── __init__.py
│   └── tab6_code.py
├── test.py
└── update_delete_document_ChromaDB.py

7 directories, 33 files
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
from ..conversation_manager_5 import get_multiagent_answer
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

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab6/__init__.py
```python
```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/tab6/tab6_code.py
```python
# bot_control_panel/tab6/tab6_code.py

# bot_control_panel/tab6/tab6_code.py

# bot_control_panel/tab6/tab6_code.py

import gradio as gr
import os
from dotenv import load_dotenv
import pandas as pd

# Импортируем нужные функции из вашего скрипта monitor_usage.py
from bot_control_panel.monitor_usage import fetch_usage_data, fetch_costs_data

def create_tab6():
    """
    Возвращает компонент Gradio, который будет отображаться во вкладке 'Стоимость'.
    Здесь оставлен только блок, который отображает Usage и Costs (с умножением показателей на 100)
    при нажатии на кнопку "Обновить данные".
    """

    with gr.Blocks() as tab6:
        # --- Заголовок страницы (по желанию можете убрать или переименовать) ---
        gr.Markdown("## Стоимость")

        # --- Раздел для получения данных из monitor_usage.py ---
        gr.Markdown("### Статистика Usage & Costs ")

        usage_info = gr.Textbox(
            label="Usage ",
            placeholder="Здесь будет отображаться суммарная статистика по Usage...",
            lines=6
        )
        costs_info = gr.Textbox(
            label="Costs ",
            placeholder="Здесь будет отображаться суммарная статистика по Costs...",
            lines=6
        )

        def refresh_data():
            """
            Вызываем функции fetch_usage_data / fetch_costs_data,
            суммируем нужные показатели
            и возвращаем два текстовых блока со сводкой.
            """
            load_dotenv()
            api_key = os.getenv("OPENAI_ADMIN_KEY")
            if not api_key:
                usage_summary = "Ошибка: нет ключа API (OPENAI_ADMIN_KEY) в .env!"
                costs_summary = "Ошибка: нет ключа API (OPENAI_ADMIN_KEY) в .env!"
                return usage_summary, costs_summary

            # 1) Usage
            usage_df = fetch_usage_data(api_key, days_ago=30, limit=7)
            if usage_df.empty:
                usage_summary = "Данных по Usage нет или не удалось получить."
            else:
                total_input_tokens = usage_df["input_tokens"].sum()
                total_output_tokens = usage_df["output_tokens"].sum()
                total_requests = usage_df["num_model_requests"].sum()

                # Умножаем на 100
                total_input_tokens_x100 = total_input_tokens * 10
                total_output_tokens_x100 = total_output_tokens * 10
                total_requests_x100 = total_requests * 10

                usage_summary = (
                    f"Суммарные значения Usage (за 30 дней):\n"
                    f" - Входные токены : {total_input_tokens_x100:,}\n"
                    f" - Выходные токены : {total_output_tokens_x100:,}\n"
                    f" - Вызовы модели : {total_requests_x100:,}\n"
                )

            # 2) Costs
            costs_df = fetch_costs_data(api_key, days_ago=30, limit=30)
            if costs_df.empty:
                costs_summary = "Данных по Costs нет или не удалось получить."
            else:
                total_cost = costs_df["amount_value"].sum()
                total_cost_x100 = total_cost * 10

                # Обычно "usd", но оставляем на случай других валют
                currency_set = set(costs_df["currency"].unique())
                currency_str = ", ".join(currency_set)

                costs_summary = (
                    f"Суммарные затраты (за 30 дней),:\n"
                    f" - Итого: {total_cost_x100:.2f} {currency_str}\n"
                )

            return usage_summary, costs_summary

        refresh_button = gr.Button("Обновить данные")
        refresh_button.click(
            fn=refresh_data,
            inputs=None,
            outputs=[usage_info, costs_info]
        )

    return tab6```

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
        return messages
    
    
    
    
    
    
    
    
    ```

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

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/monitor_usage.py
```python
# python3 monitor_usage.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import requests
import json
import pandas as pd
from dotenv import load_dotenv

def get_data_from_api(url: str, params: dict, api_key: str) -> list:
    """
    Универсальная функция для получения (и пагинации) данных от Usage или Costs API.
    Возвращает список словарей (bucket'ов).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_data = []
    page_cursor = None

    while True:
        if page_cursor:
            params["page"] = page_cursor  # добавляем курсор пагинации при необходимости

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data_json = response.json()
            # Складываем данные из текущей "страницы" в общий список
            all_data.extend(data_json.get("data", []))

            # Проверяем, есть ли "next_page"
            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break
        else:
            print(f"Ошибка при запросе {url}: {response.status_code} {response.text}")
            break
    
    return all_data

def fetch_usage_data(api_key: str, days_ago: int = 30, limit: int = 7) -> pd.DataFrame:
    """
    Получаем статистику по токенам из Usage API за последние days_ago дней.
    Параметр limit задаёт, на сколько "корзин" (bucket'ов) делить период.
    Возвращаем DataFrame со сводкой.
    """
    url = "https://api.openai.com/v1/organization/usage/completions"

    # вычислим стартовое время
    start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

    params = {
        "start_time": start_time,  # обязательный параметр
        "bucket_width": "1d",      # группируем по суткам
        "limit": limit,            # сколько временных сегментов (дней) выгружаем
        # при желании можно раскомментировать и указать дополнительные фильтры:
        # "group_by": ["model", "project_id"],
        # "models": ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"],
        # "project_ids": ["proj_..."],
        # ...
    }

    data = get_data_from_api(url, params, api_key)

    if not data:
        print("Не удалось получить данные Usage API или данных нет.")
        return pd.DataFrame()

    # Парсим полученные "bucket"-данные в список записей
    records = []
    for bucket in data:
        st = bucket.get("start_time")
        et = bucket.get("end_time")
        results = bucket.get("results", [])
        for r in results:
            records.append(
                {
                    "start_time": st,
                    "end_time": et,
                    "input_tokens": r.get("input_tokens", 0),
                    "output_tokens": r.get("output_tokens", 0),
                    "input_cached_tokens": r.get("input_cached_tokens", 0),
                    "input_audio_tokens": r.get("input_audio_tokens", 0),
                    "output_audio_tokens": r.get("output_audio_tokens", 0),
                    "num_model_requests": r.get("num_model_requests", 0),
                    "project_id": r.get("project_id"),
                    "user_id": r.get("user_id"),
                    "api_key_id": r.get("api_key_id"),
                    "model": r.get("model"),
                    "batch": r.get("batch"),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("Usage API вернул пустой список результатов.")
        return pd.DataFrame()

    # Преобразуем Unix-время в человекочитаемый формат
    df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s")
    df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s")

    # Удобно вывести в начало DataFrame читаемые даты
    cols = [
        "start_datetime",
        "end_datetime",
        "start_time",
        "end_time",
        "input_tokens",
        "output_tokens",
        "input_cached_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "num_model_requests",
        "project_id",
        "user_id",
        "api_key_id",
        "model",
        "batch",
    ]
    return df[cols]

def fetch_costs_data(api_key: str, days_ago: int = 30, limit: int = 30) -> pd.DataFrame:
    """
    Получаем статистику по затратам (в USD) из Costs API за последние days_ago дней.
    Параметр limit задаёт, на сколько 'bucket' мы делим период (обычно до 30).
    Возвращаем DataFrame со сводкой.
    """
    url = "https://api.openai.com/v1/organization/costs"

    start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

    params = {
        "start_time": start_time,
        "bucket_width": "1d",  # Costs API на момент статьи поддерживает в основном '1d'
        "limit": limit,
        # При желании можно указать group_by, например:
        # "group_by": ["line_item"],
    }

    data = get_data_from_api(url, params, api_key)

    if not data:
        print("Не удалось получить данные Costs API или данных нет.")
        return pd.DataFrame()

    records = []
    for bucket in data:
        st = bucket.get("start_time")
        et = bucket.get("end_time")
        results = bucket.get("results", [])
        for r in results:
            amount_info = r.get("amount", {})
            records.append(
                {
                    "start_time": st,
                    "end_time": et,
                    "amount_value": amount_info.get("value", 0),
                    "currency": amount_info.get("currency", "usd"),
                    "line_item": r.get("line_item"),
                    "project_id": r.get("project_id"),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("Costs API вернул пустой список результатов.")
        return pd.DataFrame()

    df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s")
    df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s")

    # Переставим колонки для удобства
    cols = [
        "start_datetime",
        "end_datetime",
        "start_time",
        "end_time",
        "amount_value",
        "currency",
        "line_item",
        "project_id",
    ]
    return df[cols]

def main():
    # Загружаем .env и берём API ключ
    load_dotenv()
    api_key = os.getenv("OPENAI_ADMIN_KEY")

    if not api_key:
        print("OPENAI_ADMIN_KEY не найден в .env!")
        return
    
    # 1) Получаем и выводим Usage Data
    print("="*60)
    print("Запрашиваем Usage API...")
    usage_df = fetch_usage_data(api_key, days_ago=30, limit=7)
    if not usage_df.empty:
        print("\nПример полученных данных (Usage):")
        print(usage_df.head(10).to_string(index=False))  # показываем несколько строк
        print(f"\nВсего строк в usage_df: {len(usage_df)}")

        # Суммарные значения по токенам
        total_input_tokens = usage_df["input_tokens"].sum()
        total_output_tokens = usage_df["output_tokens"].sum()
        total_requests = usage_df["num_model_requests"].sum()

        print("\nСуммарная статистика по Usage за период:")
        print(f" - Входные токены: {total_input_tokens:,}")
        print(f" - Выходные токены: {total_output_tokens:,}")
        print(f" - Всего вызовов модели: {total_requests:,}")
    else:
        print("Данных по Usage нет или не удалось получить.")

    # 2) (Опционально) Получаем и выводим Costs Data
    print("="*60)
    print("Запрашиваем Costs API...")
    costs_df = fetch_costs_data(api_key, days_ago=30, limit=30)
    if not costs_df.empty:
        print("\nПример полученных данных (Costs):")
        print(costs_df.head(10).to_string(index=False))
        print(f"\nВсего строк в costs_df: {len(costs_df)}")

        # Суммарные затраты
        total_cost = costs_df["amount_value"].sum()
        currency_set = set(costs_df["currency"].unique())
        currency_str = ", ".join(currency_set)

        print("\nСуммарные затраты за период:")
        print(f" - Затрачено: {total_cost:.2f} {currency_str}")
    else:
        print("Данных по затратам нет или не удалось получить.")

    print("="*60)
    print("Скрипт завершён.")

if __name__ == "__main__":
    main()```

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
from ..get_chroma_document_ID import get_full_documents

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
        
        user_question = gr.Textbox(
            lines=4, 
            label="Вопрос-сообщение пользователя", 
            elem_classes="custom-textbox"
        )

        user_answer = gr.Textbox(
            lines=4, 
            label="Ответ на вопрос-сообщение пользователя", 
            elem_classes="custom-textbox"
        )

        save_result = gr.Textbox(label="Результат операции", interactive=False)
        
        analysis_result = gr.Textbox(
            label="Результат анализа на противоречия",
            lines=10,
            interactive=False
        )

        def on_save(user_q, user_ans):
            add_or_update_document(
                new_text=user_q,  
                new_metadata={"system_response": user_ans},
                doc_id=None
            )
            return "Документ успешно сохранён в базе данных!"

        def clear_fields():
            return "", "", "", ""

        def on_analyze(user_q, user_ans):
            combined_text = f"{user_q}\n-----\n{user_ans}"
            
            results = analyze_query_with_chroma(combined_text)
            if not results:
                return "Результатов нет или функция вернула пустой список."

            contradictory_doc_ids = []
            contradiction_info_list = []

            for doc in results:
                response_json = doc.get("response_json", {})
                
                doc_id = response_json.get("doc_id")
                contradiction_found = response_json.get("contradiction_found", False)
                contradiction_description = response_json.get("contradiction_description", "")

                # Если нужные поля лежат в 'analysis'
                if doc_id is None and "analysis" in response_json:
                    try:
                        analysis_data = response_json["analysis"]
                        if isinstance(analysis_data, str):
                            analysis_data = json.loads(analysis_data)
                        if isinstance(analysis_data, dict):
                            doc_id = analysis_data.get("doc_id")
                            contradiction_found = analysis_data.get("contradiction_found", False)
                            contradiction_description = analysis_data.get("contradiction_description", "")
                    except json.JSONDecodeError:
                        pass

                if doc_id and contradiction_found:
                    contradictory_doc_ids.append(doc_id)
                    contradiction_info_list.append({
                        "doc_id": doc_id,
                        "contradiction_description": contradiction_description
                    })

            if not contradictory_doc_ids:
                return "Нет документов с противоречиями"

            docs_data = get_full_documents(contradictory_doc_ids)

            doc_dict = {}
            for d in docs_data:
                doc_dict[d["id"]] = d

            analysis_details = []
            for info in contradiction_info_list:
                doc_id = info["doc_id"]
                contradiction_description = info["contradiction_description"]

                doc_obj = doc_dict.get(doc_id, {})
                doc_text = doc_obj.get("text", "Текст не найден.")
                doc_metadata = doc_obj.get("metadata", {})

                # Получаем только system_response
                system_response = doc_metadata.get("system_response", "(system_response отсутствует)")

                text_block = (
                    f"doc_id: {doc_id}\n"
                    f"contradiction_found: True\n"
                    f"contradiction_description: {contradiction_description}\n"
                    "Содержимое документа:\n"
                    f"{doc_text}\n"
                    "Ответ :\n"
                    f"{system_response}\n"
                    "------------------------------------"
                )
                analysis_details.append(text_block)

            return "\n".join(analysis_details)

        with gr.Row():
            save_button = gr.Button("Сохранить документ в базе данных")
            analyze_button = gr.Button("Анализ на противоречия")
            clear_button = gr.Button("Очистить поля")

        save_button.click(
            fn=on_save,
            inputs=[user_question, user_answer],
            outputs=save_result
        )

        clear_button.click(
            fn=clear_fields,
            inputs=None,
            outputs=[user_question, user_answer, save_result, analysis_result]
        )

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
import os
import json
from typing import Optional, Dict, Any, List

import gradio as gr

# Для работы с .docx/.doc
import docx
import mammoth

from bot_control_panel.PromptManager import PromptManager
from bot_control_panel.OpenAIChatBot import OpenAIChatBot
from bot_control_panel.PROMPT import text_prompt
from bot_control_panel.generate_qa_pairs import generate_qa_pairs

# Новые модули, упомянутые в задании (скорректируйте пути при необходимости)
from ..analyze_query_with_chroma import analyze_query_with_chroma
from ..get_chroma_document_ID import get_full_documents

# Функция для добавления/обновления документов в ChromaDB
from ..update_delete_document_ChromaDB import add_or_update_document


def read_file_content(file_path: str) -> str:
    print("[DEBUG] Вызов read_file_content, file_path =", file_path)
    if not file_path:
        print("[DEBUG] Пустой путь, возвращаем пустую строку")
        return ""

    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        print(f"[DEBUG] Расширение файла: {ext}")

        if ext == ".docx":
            print("[DEBUG] Читаем .docx через python-docx")
            doc_obj = docx.Document(file_path)
            full_text = []
            for para in doc_obj.paragraphs:
                full_text.append(para.text)
            content = "\n".join(full_text)
            return content

        elif ext == ".doc":
            print("[DEBUG] Читаем .doc через mammoth")
            with open(file_path, "rb") as f:
                result = mammoth.convert_to_html(f)
            html_content = result.value
            import re
            text_only = re.sub(r"<[^>]*>", "", html_content)
            return text_only.strip()

        else:
            print("[DEBUG] Пытаемся читать файл как txt")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    except Exception as e:
        print("[ERROR] Ошибка при чтении файла:", e)
        return "[Не удалось прочитать как текст]"


def create_tab4():
    """
    Страница с загрузкой одного файла (.doc, .docx, .txt).
    - Нажатие "Отправить в ChatBot" -> Генерация JSON (ожидаем список).
    - Если модель вернёт dict, оборачиваем его в список.
    - Выводим результат в текстбоксах + чекбоксах.
    """

    with gr.Blocks() as tab4:
        gr.Markdown("## Обработка файла через ChatBot (JSON-ответ)")

        file_input_chat = gr.File(
            label="Файл для ChatBot",
            file_count="single"
        )
        process_button = gr.Button("Отправить в ChatBot")

        chat_status = gr.Textbox(label="Статус/Usage ChatBot", interactive=False)
        response_json_state = gr.State([])

        textboxes = []
        checkboxes = []
        widgets_for_json = []
        MAX_OBJECTS = 100

        # Создаём 100 рядов, в каждом — TextBox (интерактивный) и Checkbox
        for i in range(MAX_OBJECTS):
            with gr.Row(visible=False) as row:
                # Ставим interactive=True, чтобы пользователь мог редактировать текст
                tbox = gr.Textbox(label=f"JSON объект #{i+1}", interactive=True, lines=5)
                cbox = gr.Checkbox(label="Выбрать", value=False)
            textboxes.append(tbox)
            checkboxes.append(cbox)
            widgets_for_json.append(row)

        # ---------------------------------------------------------------------
        # Функция для анализа Q/A и противоречий
        # ---------------------------------------------------------------------
        def analyze_qa_object(obj: Dict[str, Any]) -> str:
            """
            Принимает на вход объект вида:
            {
                "id": "doc001",
                "question": "...",
                "answer": "..."
            }
            Возвращает строку с результатом анализа (противоречия и т.д.).
            """

            question = obj.get("question", "")
            answer = obj.get("answer", "")

            # Склеиваем question/answer через разделитель
            combined_text = f"{question} //// {answer}"

            # Вызываем analyze_query_with_chroma
            results = analyze_query_with_chroma(combined_text)
            if not results:
                return "Результатов нет или функция вернула пустой список."

            # Ищем противоречивые документы
            contradictory_doc_ids = []
            contradiction_info_list = []

            for doc in results:
                response_json_ = doc.get("response_json", {})
                doc_id = response_json_.get("doc_id")
                contradiction_found = response_json_.get("contradiction_found", False)
                contradiction_description = response_json_.get("contradiction_description", "")

                # Если нужные поля лежат глубже в 'analysis'
                if doc_id is None and "analysis" in response_json_:
                    try:
                        analysis_data = response_json_["analysis"]
                        if isinstance(analysis_data, str):
                            analysis_data = json.loads(analysis_data)
                        if isinstance(analysis_data, dict):
                            doc_id = analysis_data.get("doc_id")
                            contradiction_found = analysis_data.get("contradiction_found", False)
                            contradiction_description = analysis_data.get("contradiction_description", "")
                    except json.JSONDecodeError:
                        pass

                if doc_id and contradiction_found:
                    contradictory_doc_ids.append(doc_id)
                    contradiction_info_list.append({
                        "doc_id": doc_id,
                        "contradiction_description": contradiction_description
                    })

            if not contradictory_doc_ids:
                return "Нет документов с противоречиями"

            # Получаем полные данные документов
            docs_data = get_full_documents(contradictory_doc_ids)
            doc_dict = {}
            for d in docs_data:
                doc_dict[d["id"]] = d

            # Формируем итоговый блок по каждому противоречивому документу
            analysis_details = []
            for info in contradiction_info_list:
                doc_id_ = info["doc_id"]
                contradiction_description_ = info["contradiction_description"]

                doc_obj = doc_dict.get(doc_id_, {})
                doc_text = doc_obj.get("text", "Текст не найден.")
                doc_metadata = doc_obj.get("metadata", {})

                system_response = doc_metadata.get("system_response", "(system_response отсутствует)")

                text_block = (
                    f"doc_id: {doc_id_}\n"
                    f"contradiction_found: True\n"
                    f"contradiction_description: {contradiction_description_}\n"
                    "Содержимое документа:\n"
                    f"{doc_text}\n"
                    "Ответ:\n"
                    f"{system_response}\n"
                    "------------------------------------"
                )
                analysis_details.append(text_block)

            return "\n".join(analysis_details)

        # ---------------------------------------------------------------------
        # Функция call_chatbot (из старой рабочей версии)
        # ---------------------------------------------------------------------
        def call_chatbot(files):
            print("[DEBUG] call_chatbot вызван, files =", files)
            if not files:
                status_msg = "Файл не выбран."
                print("[DEBUG] files is None или пусто.")
                return status_msg, [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []

            # Проверяем тип files
            if isinstance(files, str):
                file_path = files
            elif isinstance(files, list):
                if not files:
                    print("[DEBUG] files - пустой список.")
                    return "Файл не выбран.", [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []
                first_elem = files[0]
                if isinstance(first_elem, str):
                    file_path = first_elem
                elif isinstance(first_elem, dict) and "name" in first_elem:
                    file_path = first_elem["name"]
                else:
                    print("[ERROR] Неизвестный формат элемента в files:", first_elem)
                    return "[Не удалось прочитать]", [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []
            else:
                print("[ERROR] files имеет тип:", type(files), "ожидали str или list.")
                return "[Не удалось прочитать]", [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []

            print("[DEBUG] Итоговый file_path =", file_path)
            user_message = read_file_content(file_path)
            print("[DEBUG] Первые 200 символов считанного текста:", user_message[:200])

            # Логика PromptManager (не обязательно для генерации Q/A, но оставим для совместимости)
            pm = PromptManager()
            messages = pm.build_developer_messages(
                user_message=user_message,
                developer_instruction=text_prompt
            )
            print("[DEBUG] Сформированные messages (не используется при generate_qa_pairs, но сохраняется):", messages)

            print("[DEBUG] *** Перед вызовом generate_qa_pairs ***")
            try:
                response_json = generate_qa_pairs(user_message)
                print("ДОКУМЕНТЫ:", response_json)
            except Exception as e:
                err_msg = f"[ERROR] При работе generate_qa_pairs возникла ошибка: {e}"
                print(err_msg)
                return err_msg, [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []

            print("[DEBUG] *** После вызова generate_qa_pairs ***")
            print("[DEBUG] Получен response_json:", response_json)

            usage_data = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

            # Проверяем, если функция вернула dict, превращаем в список
            if isinstance(response_json, dict):
                print("[DEBUG] Функция вернула dict, оборачиваем в список.")
                response_json = [response_json]

            if not isinstance(response_json, list):
                status_msg = f"Ожидали JSON-массив (list), получили: {type(response_json)}"
                print("[DEBUG]", status_msg)
                return status_msg, [], [gr.update(visible=False) for _ in range(MAX_OBJECTS)], []

            status_msg = (
                f"Функция generate_qa_pairs вернула {len(response_json)} объект(ов).\n"
                f"Prompt tokens: {usage_data.get('prompt_tokens')}, "
                f"Completion tokens: {usage_data.get('completion_tokens')}, "
                f"Total tokens: {usage_data.get('total_tokens')}"
            )
            print("[DEBUG] status_msg:", status_msg)

            # Показ/скрытие строк
            updates_for_rows = []
            for i in range(MAX_OBJECTS):
                updates_for_rows.append(
                    gr.update(visible=True) if i < len(response_json) else gr.update(visible=False)
                )

            # Возвращаем: 4 значения
            # 1) Статус,
            # 2) Собственно JSON (response_json_data),
            # 3) Видимость рядов,
            # 4) Доп. копия (необязательно, но так сделан код).
            return status_msg, response_json, updates_for_rows, response_json

        # ---------------------------------------------------------------------
        # call_chatbot_enhanced - добавляет в каждый Textbox результат анализа Q/A
        # ---------------------------------------------------------------------
        def call_chatbot_enhanced(files):
            print("[DEBUG] call_chatbot_enhanced вызван.")
            status_msg, response_json_data, rows_visibility, raw_json_data = call_chatbot(files)

            textboxes_updates = []
            for i in range(MAX_OBJECTS):
                if i < len(response_json_data):
                    # Преобразуем сам JSON-объект в строку
                    obj_str = json.dumps(response_json_data[i], indent=2, ensure_ascii=False)

                    # Анализируем question/answer
                    analysis_str = analyze_qa_object(response_json_data[i])

                    # Запишем всё в один TextBox
                    combined_output = (
                        f"{obj_str}\n\n"
                        f"=== Результат анализа ===\n"
                        f"{analysis_str}"
                    )
                    # Если не хотим каждый раз перезаписывать, можем убрать value=...
                    textboxes_updates.append(gr.update(value=combined_output, visible=True))
                else:
                    textboxes_updates.append(gr.update(value="", visible=False))

            # Возвращаем: 
            # 1) status_msg, 
            # 2) response_json_data (сохраняем в State), 
            # 3) Установки для row (видимость), 
            # 4) Установки для textboxes (value и visible)
            return [status_msg, response_json_data] + rows_visibility + textboxes_updates

        # Кликаем — вызываем call_chatbot_enhanced, выводим в текстовые блоки
        process_button.click(
            fn=call_chatbot_enhanced,
            inputs=[file_input_chat],
            outputs=[
                chat_status,
                response_json_state,
                *widgets_for_json,
                *textboxes
            ]
        )

        gr.Markdown("---")

        confirm_button = gr.Button("Подтвердить выбор")
        confirm_result = gr.Textbox(label="Результат подтверждения", interactive=False)

        # ---------------------------------------------------------------------
        # Новая логика в confirm_selection - получить и сохранить 
        # как выбранные объекты, так и пользовательские правки из TextBox
        # ---------------------------------------------------------------------
        def confirm_selection(response_json, *checkboxes_and_texts):
            print("[DEBUG] confirm_selection вызван.")
            if not response_json:
                print("[DEBUG] Нет JSON для выбора.")
                return "Нет данных для подтверждения."

            # Первые 100 аргументов - значения чекбоксов
            checkboxes_values = checkboxes_and_texts[:MAX_OBJECTS]
            # Следующие 100 - строки из текстовых полей
            textboxes_values  = checkboxes_and_texts[MAX_OBJECTS:]

            selected_objs = []
            for i, (obj, checked) in enumerate(zip(response_json, checkboxes_values)):
                if checked:
                    # Считываем текущее содержимое TextBox
                    edited_text = textboxes_values[i]
                    # Можно сохранить в поле "edited_text", если хотим
                    obj["edited_text"] = edited_text
                    selected_objs.append(obj)

            print("=== Выбранные объекты (с учётом правок) ===")
            print(json.dumps(selected_objs, ensure_ascii=False, indent=2))
            print("=== /конец ===")

            # Если ничего не выбрано
            if not selected_objs:
                return "Вы не выбрали ни одного объекта!"

            # Сохраняем в ChromaDB как новые документы
            for doc_data in selected_objs:
                # Вместо (question + answer) используем то, что пользователь отредактировал
                combined_text = doc_data["edited_text"]

                new_metadata = {
                    "original_id": doc_data.get("id", "no_id"),
                    "source": "user_upload",  # любая ваша метка
                }

                add_or_update_document(
                    new_text=combined_text,
                    new_metadata=new_metadata,
                    doc_id=None
                )

            return f"Сохранено документов: {len(selected_objs)}"

        # Входными данными на «Подтвердить выбор» теперь будут:
        # 1) response_json_state
        # 2) все чекбоксы
        # 3) все текстовые поля
        inputs_to_confirm = [response_json_state] + checkboxes + textboxes

        confirm_button.click(
            fn=confirm_selection,
            inputs=inputs_to_confirm,
            outputs=confirm_result
        )

    return tab4


if __name__ == "__main__":
    demo = create_tab4()
    demo.launch()```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/get_chroma_document_ID.py
```python

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
        print(json.dumps(docs_data, ensure_ascii=False, indent=2))```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/OpenAIChatBot.py
```python
# OpenAIChatBot.py

import os
import base64
from typing import Optional, Dict, Any, Union
from io import BufferedReader, BytesIO

from dotenv import load_dotenv
load_dotenv()

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
        chosen_presence_penalty = (
            presence_penalty if (presence_penalty is not None)
            else self.default_presence_penalty
        )
        chosen_frequency_penalty = (
            frequency_penalty if (frequency_penalty is not None)
            else self.default_frequency_penalty
        )
        chosen_n = n if (n is not None) else self.default_n

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": user_message}],
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

        text_model = text_model or self.default_text_model
        chat_answer = self.generate_text_response(
            user_message=combined_prompt,
            model=text_model
        )

        return {
            "transcribed_text": transcribed_text,
            "response_text": chat_answer["response_text"],
            "usage_transcription": None,
            "usage_chat": chat_answer["usage"]
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
        response_format: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Превращает текст в аудио. Возвращает байты, либо сохраняет в файл, если указан save_path.

        :param text: Исходная текстовая строка
        :param model: Модель TTS (напр. 'tts-1' или 'tts-1-hd')
        :param voice: Голос ('alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer')
        :param response_format: Формат аудио ('mp3' по умолчанию, 'opus', 'aac', 'flac', 'wav' и т.д.)
        :param save_path: Путь для записи файла; если None, вернём байты
        :return: {
            "audio_content": bytes | None,
            "voice_used": str,
            "model_used": str,
            "note": "..."
        }
        """
        create_kwargs = {
            "model": model,
            "voice": voice,
            "input": text
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        response = self.client.audio.speech.create(**create_kwargs)

        audio_content = None
        if save_path:
            response.stream_to_file(save_path)
        else:
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
        chosen_presence_penalty = (
            presence_penalty if (presence_penalty is not None)
            else self.default_presence_penalty
        )
        chosen_frequency_penalty = (
            frequency_penalty if (frequency_penalty is not None)
            else self.default_frequency_penalty
        )
        chosen_n = n if (n is not None) else self.default_n

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

        answer_text = ""
        if response.choices and response.choices[0].message:
            answer_text = response.choices[0].message.content

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
            response_format={"type": "json_object"},
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
        }

    def generate_chat_response_with_jsonschema_2_FINAL(
        self,
        messages: list,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Пример: хотим получить массив объектов {id, question, answer}.
        """
        json_schema = {
            "name": "QuestionAnswerPairs",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"}
                    },
                    "required": ["id", "question", "answer"],
                    "additionalProperties": False
                }
            }
        }

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
        }```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/chroma_parallel_analyzer_search.py
```python
# python3 chroma_parallel_analyzer_search.py


import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from copilot_service.PROMPT import chroma_parallel_analyzer_search

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
            user_messages = prompt_manager.build_prompt_with_json(
                prompt=chroma_parallel_analyzer_search,
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
"""



text_prompt = """
**Формулировка задания (расширенная)**

1. Цель

На основе полного исходного текста, описывающего работу медцентра (правила, процедуры, расценки, алгоритмы для 
персонала), нужно сформировать пары «вопрос–ответ». Каждая пара должна отражать конкретную часть 
(логический блок) исходного текста, чтобы при дальнейшем использовании в системе (векторный поиск, LLM) можно было
эффективно отвечать на вопросы пациентов. Вопросы формируйте в разговорной манере (как если бы спрашивал пациент), ответы 
формируйте как профисиональный врачь/администратор клиники.

2. Объём и структура исходного текста

- Текст может содержать общие сведения (импланты, стоимость, сроки и т.д.), а также детальные описания 
(шаги лечения, протоколы для сотрудников).
- **Важно**: если объём или детализация текста велика, **разбейте** его на все логические блоки; для **каждого** 
блока должна быть **своя** пара «вопрос–ответ».

3. Требования к формату пар «вопрос–ответ»

- **Единая нумерация**: Каждая пара имеет уникальный идентификатор вида `doc001`, `doc002`, `doc003` …, и так далее,
без пропусков.
- **Полнота информации**: Нельзя придумывать информацию, не содержащуюся в тексте, и нельзя упускать важные детали, 
имеющиеся в нём.
- **Связь с исходным документом**: Каждая пара отражает конкретный смысловой фрагмент текста для точного поиска.
- **Точность формулировок**: Ответ должен соответствовать исходному тексту (без искажения смысла), но допускается 
лёгкая адаптация для удобства чтения.
- **Сжатость вопроса**: Формулируется так, как если бы спрашивал пациент, и охватывает ровно ту информацию, которая
описана в данном фрагменте.
- **Стиль ответа**: Ответ составляется от лица клиники; лаконично, информативно, с упором на важные детали 
(стоимость, сроки, условия и т.д.).
- **Количество пар**:количесвто созданных пар должно перекрывать все тему , всю инфномацию изложенную в тексте ,
то есть созданные пары должны передать смысл всего документа ,не мение 20  но не более 50 пар . Если блоков/важных 
тем **больше**, формируйте соразмерно больше пар, чтобы каждая тема была представлена отдельно.

4. Шаги по выполнению

1. **Анализ текста**: Разделить документ на логические блоки (темы, процедуры, услуги, частые вопросы и т.д.).  
2. **Формирование вопросов**: Для **каждого** такого фрагмента составить вопрос, отражающий суть именно этого блока.  
3. **Запись ответа**: На вопрос дать ответ, взятый из содержания того же блока.  
4. **Проверка полноты**: Убедиться, что **все ключевые моменты** (стоимость, процедуры, сроки и пр.) попали как
минимум в одну пару «вопрос–ответ».  
5. **Редактирование**: Привести пары к единому формату, удалить дублирующиеся повторы.

5. Результат

- Итогом получается массив (список) из **N** пар  количесвто пар должно обеспечить передачу всех смыслов , 
всей инофрмации в переланном тексте для анализа ).  
- Каждая пара «накрывает» часть исходной информации, чтобы в совокупности охватить весь документ.
- Формат ответа **строго** в виде **JSON-массива**, не мение 20  но не более 50 пар, где каждый элемент — объект вида:
  ```json
  {
    "id": "doc033",
    "question": "[Вопрос]",
    "answer": "[Ответ]"
  }
  
  и т.д.
	•	Никаких дополнительных полей не добавляйте — только id, question, answer.

Пример итогового JSON:

[
  {
    "id": "doc001",
    "question": "Какие импланты используются?",
    "answer": "..."
  },
  {
    "id": "doc002",
    "question": "Какова стоимость онлайн консультации?",
    "answer": "..."
  },
  ...
]
ФОРМАТ ВЫВОДА: строго JSON, без поясняющего текста вне массива.

ТЕКСТ ДЛЯ АНАЛИЗА:

"""

text_prompt_2 = """ 
**Формулировка задания**
	1.	Что делать:
	•	Возьмите полный текст о работе медцентра (правила, процедуры, цены и т.д.).
	•	Разделите его на логические блоки (каждый блок — отдельная тема или смысловой фрагмент).
	•	Для каждого блока сформируйте пару «вопрос–ответ».
	2.	Как формировать пары:
	•	Вопрос: пишите от лица пациента (разговорный стиль, напрямую по теме блока).
	•	Ответ: давайте как специалист клиники, с чёткими деталями (стоимость, сроки, условия).
	•	Не выдумывайте информацию, которой нет в исходном тексте.
	•	Создавайте максимум возможных пар (но не менее 20 и не более 50), чтобы охватить все блоки.
	3.	Требования к оформлению:
	•	Используйте единую нумерацию: doc001, doc002, doc003 и т.д. без пропусков.
	•	Форматируйте результат строго как JSON-массив из объектов вида:

{
  "id": "doc001",
  "question": "[Вопрос]",
  "answer": "[Ответ]"
}


	•	Никаких лишних полей и пояснений за пределами массива не добавляйте.

Создайте максимум информативных «вопрос–ответ» пар, покрывающих весь текст.
"""




medical_services_pricing = """   

# Прайс-лист стоматологических услуг (Варшава)

| Услуга                                     | Описание                                                                                 | Цена (PLN)      |
|--------------------------------------------|------------------------------------------------------------------------------------------|-----------------|
| **Консультация стоматолога**              | Первичный осмотр, сбор анамнеза, определение текущего состояния зубов и дёсен           | 150             |
| **Рентген (панорамный снимок)**           | Полный обзор зубных рядов для диагностики                                               | 80              |
| **Гигиеническая чистка**                   | Удаление зубного налёта и камня, полировка                                              | 200             |
| **Пломбирование (композитный материал)**   | Устранение кариеса и восстановление структуры зуба                                      | 250–400         |
| **Эстетическая пломба (светоотверждаемая)**| Более эстетичный вариант пломбы, с подгонкой цвета под оттенок зуба                      | 350–450         |
| **Лечение корневых каналов (1 канал)**     | Эндодонтическое лечение, включает обработку и пломбирование корневого канала            | 600–800         |
| **Лечение корневых каналов (3 канала)**    | Стоимость за сложную процедуру при множественных каналах (обычно на коренных зубах)     | 900–1,200       |
| **Удаление зуба (простое)**               | Простая экстракция зуба без осложнений                                                  | 250–350         |
| **Удаление зуба (сложное)**               | Экстракция зуба мудрости, ретинированного зуба или с осложнениями                       | 500–800         |
| **Протезирование (металлокерамическая коронка)** | Классический вариант коронки с металлическим каркасом                                   | 800–1,000       |
| **Керамическая коронка (цельнокерамическая)** | Полностью керамическая коронка с улучшенной эстетикой                                    | 1,200–1,500     |
| **Имплантация**                            | Установка импланта (исключая стоимость коронки), зависит от используемой системы имплантации | 2,500–4,000     |
| **Профессиональное отбеливание зубов**     | Отбеливание в клинике специальными составами и лампами                                  | 800–1,000       |
| **Установка брекетов (металлические)**     | Полный курс установки и начальной коррекции прикуса                                      | 3,500–5,000     |
| **Установка брекетов (керамические)**      | Эстетическая версия брекет-системы, менее заметная                                       | 5,000–7,000     |
| **Стоматологический массаж дёсен**         | Массаж и укрепление тканей для профилактики пародонтита                                 | 80–150          |

---

## Дополнительная информация

- Точная стоимость услуг может варьироваться в зависимости от состояния зубов и выбранных материалов.
- Итоговая цена определяется после консультации и детального плана лечения.
- Некоторые услуги могут частично или полностью покрываться страховой компанией; уточняйте детали у администратора клиники.
- Данный прайс-лист является примерным и не отражает актуальные расценки конкретного медицинского учреждения.



"""```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/analyze_query_with_chroma.py
```python
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
    analyze_query_with_chroma(user_query)```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/contradiction_detector.py
```python
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

    return results```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/conversation_manager_5.py
```python

# conversation_manager_5.py

"""последняя рабочая версия с применением Context Object"""

import os
import json
import inspect
import logging
import sys
from dataclasses import dataclass, field
from inspect import Parameter
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv


# Пример вектора поиска
from .TestInfoRetriever import vector_search
from .PROMPT import medical_services_pricing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.debug("Логирование DEBUG включено. Теперь вы будете видеть подробные сообщения в консоли.")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env или окружении.")

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-4o"

# -----------------------------------------------------------------------------
# 1. Определяем класс Context
# -----------------------------------------------------------------------------
@dataclass
class RequestContext:
    request_id: str
    lead_id: int
    user_text: str
    history_messages: List[dict]
    channel: Optional[str] = None

    # Пример полей для хранения cat1_ids, cat2_ids
    cat1_ids: List[str] = field(default_factory=list)
    cat2_ids: List[str] = field(default_factory=list)
    # Добавляйте любые поля, которые захотите отслеживать через контекст


# -----------------------------------------------------------------------------
# Вспомогательная функция: из списка пар {user_message, assistant_message}
# делаем [{'role':..., 'content':...}, ...]
# -----------------------------------------------------------------------------
def build_history_messages_from_db(records: List[dict]) -> List[dict]:
    history = []
    for r in records:
        user_msg = r.get("user_message", "").strip()
        assistant_msg = r.get("assistant_message", "").strip()

        if user_msg:
            history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            history.append({"role": "assistant", "content": assistant_msg})

    return history


# -----------------------------------------------------------------------------
# 2. Инструменты и класс Agent
# -----------------------------------------------------------------------------
def clinic_data_lookup(criteria: str, request_id: str = None):
    """Инструмент для поиска информации о клинике услугах и правилах работы."""
    answer = vector_search(criteria, request_id=request_id)
    return f"Информация полученная из базы знаний : {answer}"

def sql_search(criteria: str):
    """Инструмент для получения прайс-листа."""
    return f"информация полученная из прескуранта : {medical_services_pricing}"

def submit_ticket(description: str):
    """Создать тикет (просто фейк)."""
    return f"(Fake) Тикет создан: {description}"

def calendar_api(action: str, dateTime: str = "", managerId: str = ""):
    """Инструмент календаря: 'check_avail' или 'book_slot'."""
    if action == "check_avail":
        return f"(Fake) calendar_api: Проверяем слот {dateTime} для manager={managerId}"
    elif action == "book_slot":
        return f"(Fake) calendar_api: Слот {dateTime} забронирован (manager={managerId})"
    return "(Fake) calendar_api: неизвестное действие"

def transfer_to_medical_agent() -> str:
    return "HandoffToMedicalAgent"

def transfer_to_legal_agent() -> str:
    return "HandoffToLegalAgent"

def transfer_to_super_agent() -> str:
    return "HandoffToSuperAgent"

def transfer_to_finance_agent() -> str:
    return "HandoffToFinanceAgent"


class Agent:
    def __init__(self, name: str, instructions: str, tools: list, model: str = MODEL_NAME):
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.model = model


# -----------------------------------------------------------------------------
# 3. Схема для инструментов (function-calling)
# -----------------------------------------------------------------------------
def function_to_schema(func) -> dict:
    """
    Генерирует схему инструмента (функции) в формате "type":"function"
    по документации OpenAI.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    properties = {}
    required_fields = []

    for param_name, param in signature.parameters.items():
        annotation = param.annotation if param.annotation is not Parameter.empty else str
        json_type = type_map.get(annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is Parameter.empty:
            required_fields.append(param_name)

    description = (func.__doc__ or "").strip()

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False
            }
        }
    }


# -----------------------------------------------------------------------------
# 4. Объявляем агентов
# -----------------------------------------------------------------------------
super_agent = Agent(
    name="Super Agent",
    instructions=(
        """Вы — Супер Агент (резервный). Обрабатывайте любые запросы, выходящие за рамки других агентов.
как опытный администратор медцентра для получения информации всегда используйте инструмент 'clinic_data_lookup'"""
    ),
    tools=[clinic_data_lookup],
    model=MODEL_NAME
)

finance_agent = Agent(
    name="Finance Agent",
    instructions=(
        """Вы Агент, отвечающий за стоимость услуг в медцентре.
1) Вызовите 'sql_search', чтобы получить прайс или стоимость.
"""
    ),
    tools=[sql_search],
    model=MODEL_NAME
)

medical_agent = Agent(
    name="Medical Agent",
    instructions=(
        """Вы Medical Agent администратор в медцентре.
Ваша задача — максимально точно отвечать на вопросы посетителей.
Всегда используйте 'clinic_data_lookup' для поиска мед. информации.
"""
    ),
    tools=[clinic_data_lookup],
    model=MODEL_NAME
)

legal_agent = Agent(
    name="Legal Agent",
    instructions=(
        """Вы Legal Agent. Отвечаете на юридические вопросы (контракты, GDPR и т.п.).
Если вопрос не юридический — вызывайте transfer_to_super_agent.
"""
    ),
    tools=[transfer_to_super_agent],
    model=MODEL_NAME
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        """Вы — Triage Agent.
Правила:
1) Вопросы про работу медцентра, порядок оказания услуг, стоимость => transfer_to_medical_agent (1 раз).
2) Если вопрос юридический => transfer_to_legal_agent (1 раз).
3) Если вопрос про стоимость — можно вызвать Finance Agent (1 раз).
4) Иначе => transfer_to_super_agent (1 раз).
"""
    ),
    tools=[
        transfer_to_medical_agent,
        transfer_to_legal_agent,
        transfer_to_super_agent,
        transfer_to_finance_agent
    ],
    model=MODEL_NAME
)


# -----------------------------------------------------------------------------
# 5. Функция map_handoff_to_agent(...)
# -----------------------------------------------------------------------------
def map_handoff_to_agent(handoff_string: str) -> Agent:
    if handoff_string == "HandoffToMedicalAgent":
        return medical_agent
    elif handoff_string == "HandoffToLegalAgent":
        return legal_agent
    elif handoff_string == "HandoffToFinanceAgent":
        return finance_agent
    elif handoff_string == "HandoffToSuperAgent":
        return super_agent
    return super_agent


# -----------------------------------------------------------------------------
# 6. execute_tool_call с подмешиванием request_id из ctx
# -----------------------------------------------------------------------------
def execute_tool_call(function_name: str, args: dict, tools_map: dict, agent_name: str, ctx: RequestContext):
    """
    Добавляем ctx, чтобы подмешивать request_id, lead_id и т.д. в аргументы инструмента,
    если инструмент их поддерживает.
    """
    logger.debug(f"[execute_tool_call] {agent_name} вызывает '{function_name}' c аргументами {args}")

    if function_name not in tools_map:
        logger.warning(f"{agent_name} tried to call unknown function {function_name}")
        return f"[Warning] Unknown function: {function_name}"

    # Подмешиваем request_id, если у функции в сигнатуре есть такой параметр
    sig = inspect.signature(tools_map[function_name])
    if "request_id" in sig.parameters:
        args.setdefault("request_id", ctx.request_id)

    result = tools_map[function_name](**args)
    logger.debug(f"[execute_tool_call] Результат '{function_name}': {result}")
    return result


# -----------------------------------------------------------------------------
# 7. run_subagent принимает ctx вместо user_message, history_messages
# -----------------------------------------------------------------------------
def run_subagent(agent: Agent, ctx: RequestContext) -> str:
    """
    Запуск одного саб-агента. Если агент вызывает handoff, переходим к другому.
    """
    logger.debug(f"[run_subagent] START, agent='{agent.name}' user_message='{ctx.user_text}'")

    # Формируем массив сообщений из ctx.history_messages + dev instructions + текущее user_text
    messages = []
    if ctx.history_messages:
        messages.extend(ctx.history_messages)

    messages.append({"role": "developer", "content": agent.instructions})
    messages.append({"role": "user", "content": ctx.user_text})

    all_messages = messages[:]

    tools_map = {t.__name__: t for t in agent.tools}
    tool_schemas = [function_to_schema(t) for t in agent.tools]

    # Вместо set() используем счетчик, чтобы дать возможность инструменту вызываться несколько раз
    from collections import defaultdict
    used_tools_count = defaultdict(int)

    max_rounds = 2
    consecutive_empty = 0

    for step in range(max_rounds):
        logger.debug(f"[run_subagent] step={step}, agent='{agent.name}'")

        if tool_schemas:
            response = client.chat.completions.create(
                model=agent.model,
                messages=all_messages,
                tools=tool_schemas,
                temperature=0
            )
        else:
            response = client.chat.completions.create(
                model=agent.model,
                messages=all_messages,
                temperature=0
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        # Добавляем ответ ассистента (без инструментов) в историю
        all_messages.append({"role": "assistant", "content": content})
        logger.debug(f"[run_subagent] agent='{agent.name}' content='{content}', tool_calls={tool_calls}")

        if not content and not tool_calls:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        if consecutive_empty >= 1:
            logger.debug("[run_subagent] Дважды пустой ответ — завершаем.")
            break

        # Обрабатываем вызовы инструментов
        if not tool_calls:
            # Нет вызовов инструментов, значит это финальный ответ
            break

        for tc in tool_calls:
            fn_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except:
                args = {}

            # --- Разрешаем вызывать один и тот же инструмент до 2 раз ---
            if used_tools_count[fn_name] >= 2:
                logger.debug(f"[run_subagent] {agent.name}: инструмент '{fn_name}' уже вызывался >= 2 раз, пропускаем.")
                continue

            used_tools_count[fn_name] += 1

            result = execute_tool_call(fn_name, args, tools_map, agent.name, ctx)
            logger.debug(f"[run_subagent] Результат вызова инструмента '{fn_name}': {result}")

            # Если мы получили HandoffTo..., переходим к другому агенту
            if result.startswith("HandoffTo"):
                new_agent = map_handoff_to_agent(result)
                logger.debug(f"[run_subagent] handoff -> {new_agent.name}")
                # Перезапускаем subagent (рекурсивно) с тем же контекстом
                return run_subagent(new_agent, ctx)
            else:
                # Добавляем вывод инструмента как очередное сообщение assistant
                all_messages.append({
                    "role": "assistant",
                    "content": f"[Tool output] {result}"
                })

    # Собираем финальный ответ
    final_answer = ""
    for i in reversed(all_messages):
        if i["role"] == "assistant" and not i["content"].startswith("[Tool output]"):
            final_answer = i["content"]
            break

    logger.debug(f"[run_subagent] END, final_answer='{final_answer}'")
    return final_answer


# -----------------------------------------------------------------------------
# 8. run_triage_and_collect тоже принимает ctx
# -----------------------------------------------------------------------------
def run_triage_and_collect(triage_agent: Agent, ctx: RequestContext) -> List[Dict[str, str]]:
    """
    Запускаем Triage Agent и собираем partial ответы от суб-агентов.
    Теперь разрешаем повторный вызов одного и того же инструмента
    (например, transfer_to_super_agent) до 2 раз.
    """
    logger.debug("[run_triage_and_collect] START")

    triage_msgs = []
    if ctx.history_messages:
        triage_msgs.extend(ctx.history_messages)

    # Добавляем инструкции триаж-агента и текущее сообщение пользователя
    triage_msgs.append({"role": "developer", "content": triage_agent.instructions})
    triage_msgs.append({"role": "user", "content": ctx.user_text})

    partials = []

    # Создаём мапу "имя_функции -> сама функция" для быстрого вызова
    tools_map = {t.__name__: t for t in triage_agent.tools}
    tool_schemas = [function_to_schema(t) for t in triage_agent.tools]

    # Вместо set() используем счётчик, чтобы разрешить до 2 вызовов одного и того же инструмента
    from collections import defaultdict
    used_tools_count = defaultdict(int)

    max_rounds = 2
    consecutive_empty = 0

    for step in range(max_rounds):
        #logger.debug(f"[run_triage_and_collect] step={step}")

        # Делаем запрос к модели с учётом tools
        if tool_schemas:
            response = client.chat.completions.create(
                model=triage_agent.model,
                messages=triage_msgs,
                tools=tool_schemas,
                temperature=0.0
            )
        else:
            # Если у агента нет инструментов — обычный чат без function-calling
            response = client.chat.completions.create(
                model=triage_agent.model,
                messages=triage_msgs,
                temperature=0.0
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        triage_msgs.append({"role": "assistant", "content": content})
        #logger.debug(f"[run_triage_and_collect] content='{content}', tool_calls={tool_calls}")

        if not content and not tool_calls:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        if consecutive_empty >= 1:
            #logger.debug("[run_triage_and_collect] Дважды пусто — завершаем.")
            break

        # Если нет вызовов инструментов, значит это финальный ответ триаж-агента
        if not tool_calls:
            break

        # Обрабатываем инструментальные вызовы
        for tc in tool_calls:
            fn_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except:
                args = {}

            # --- проверяем счётчик для инструмента ---
            if used_tools_count[fn_name] >= 2:
                logger.debug(f"[run_triage_and_collect] '{fn_name}' уже вызывался >= 2 раз, пропускаем.")
                continue
            used_tools_count[fn_name] += 1

            # Если функция (инструмент) не зарегистрирована
            if fn_name not in tools_map:
                logger.warning(f"Unknown function: {fn_name}")
                triage_msgs.append({"role": "assistant", "content": f"Unknown function: {fn_name}"})
                continue

            # Вызываем инструмент
            result = tools_map[fn_name](**args)
            logger.debug(f"[run_triage_and_collect] result={result}")

            # Если инструмент возвращает HandoffTo..., передаём запрос соответствующему агенту
            if result.startswith("HandoffTo"):
                subagent = map_handoff_to_agent(result)
                partial_answer = run_subagent(subagent, ctx)
                partials.append({"agent": subagent.name, "answer": partial_answer})
                triage_msgs.append({
                    "role": "assistant",
                    "content": f"[Triage -> {subagent.name}] partial ok"
                })
            else:
                # Иначе это «обычный» инструмент: выводим результат
                triage_msgs.append({
                    "role": "assistant",
                    "content": f"[Tool output] {result}"
                })

    #logger.debug(f"[run_triage_and_collect] END, partials={partials}")
    return partials


# -----------------------------------------------------------------------------
# 9. final_aggregation — пока оставим без ctx, но можно и туда передавать
# -----------------------------------------------------------------------------
def final_aggregation(
    partials: List[Dict[str, str]],
    user_message: str,
    history_messages: Optional[List[dict]] = None
) -> str:
    """
    Формируем итоговый запрос к GPT, включая:
    - Основную инструкцию (developer)
    - partials (промежуточные ответы)
    - Историю диалога (history_messages)
    - Сообщение пользователя
    """
    logger.debug("[final_aggregation] START")

    partials_str = json.dumps(partials, ensure_ascii=False, indent=2)

    if not history_messages:
        history_messages = []
    history_json = json.dumps(history_messages, ensure_ascii=False, indent=2)

    developer_content = f"""
Вы Врач-администратор колцентра в медцентре. Используйте предоставленные "** ДАННЫЕ НА ОСВНОАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ 
ОТВЕТ  :"
и вопрос пользователя для формулирования дружелюбного ответа. Формируйте ответы от своего имени, не ссылаясь на других 
администраторов, работников медцентра, или других агентов. Ваши ответы должны быть в формате разговора с пациентом и не 
начинайтесь с приветствия, кроме случаев первого сообщения от пользователя.

При формировании ответа используйте только информацию из раздела "ДАННЫЕ НА ОСНОВАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ ОТВЕТ" и 
никогда не придумывайте ответы. Если информации недостаточно, сообщите пользователю, что вы не можете ответить на его вопрос.

# Шаги

1. Прочитайте вопрос пользователя и соответствующие Partial-ответы.
2. Определите информацию, имеющую отношение к вопросу из предоставленных данных.
3. Составьте ответ, используя только доступную информацию.
4. Информацию, которой нет в предоставленных данных, не добавляйте в ответ.
5. Если информации недостаточно, вежливо сообщите об этом пользователю.

# Формат ответа

Ответ должен быть в виде дружелюбного и вежливого текста, содержащего только информацию из предоставленных данных.

**Инструкция  - Алгоритм ответа на вопросы типа : “** **“Как я могу записаться на прием?”:**

1. **Приветствие и уточнение запроса**

🔹 “Подскажите, к какому специалисту или на какую процедуру вы хотите записаться?”

(Важно понять, нужен ли врач-стоматолог, терапевт, узкий специалист или диагностика.)

Если эта информация получена ранее переходим к следующему шагу . 

2. **Сбор основной информации**

🔹 “Как вас зовут?” (ФИО)

🔹 “Есть ли у вас предпочтения по дате и времени?”

🔹 “Вы уже были у нас раньше или записываетесь впервые?”

Если эта информация получена ранее переходим к следующему шагу . 

3. **Проверка доступного времени и предложения вариантов**

🔹 Проверить в системе расписание врача.

🔹 “На ближайшие дни есть свободное время: [Среда 8-00 и Четверг 15-00]. Какой вам удобнее?”

Если эта информация получена ранее переходим к следующему шагу . 

4. **Подтверждение записи**

🔹 “Подтверждаю вашу запись на [дата, время].”

🔹 “Вам отправить напоминание по SMS или мессенджеру?”

Если эта информация получена ранее переходим к следующему шагу . 

5. **Дополнительные инструкции**

🔹 “Приходите за 10–15 минут до приема, возьмите с собой документы (если нужно).”

🔹 “Если у вас изменятся планы, дайте нам знать заранее.”

Если эта информация получена ранее переходим к следующему шагу . 

6. **Заключение**

🔹 “Спасибо, что выбрали нашу клинику! Если появятся вопросы, звоните или пишите.”

** ДАННЫЕ НА ОСВНОАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ ОТВЕТ  :
{partials_str}

ИСТОРИЯ ДИАЛОГА (ДЛЯ КОНТЕКСТА):
{history_json}

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ, НА КОТОРОЕ НУЖНО ОТВЕТИТЬ:
{user_message}
"""

    print("ИТОГОВАЯ ИНСТРУКЦИЯ:", developer_content)

    messages = [
        {
            "role": "developer",
            "content": developer_content
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0
    )
    final_text = response.choices[0].message.content or ""
    return final_text


# -----------------------------------------------------------------------------
# 10. get_multiagent_answer, который принимает (и создаёт) RequestContext
# -----------------------------------------------------------------------------
def get_multiagent_answer(
    user_text: str,
    lead_id: int = None,
    channel: str = None,
    history_messages: Optional[List[dict]] = None,
    request_id: str = None
) -> str:
    """
    Главная точка входа (синхронная).
    Теперь история передаётся извне через history_messages.
    Мы создаём RequestContext и передаём его в triage & subagents.
    """
    #logger.info(f"[get_multiagent_answer] called, user_text='{user_text}', request_id={request_id}")

    if not history_messages:
        history_messages = []

    # Сформируем контекст
    ctx = RequestContext(
        request_id=request_id or "noRequestId",
        lead_id=lead_id or 0,
        user_text=user_text,
        history_messages=history_messages,
        channel=channel
    )

    # Логируем историю
    #logger.info(
    #    "[get_multiagent_answer] ИСТОРИЯ ПЕРЕПИСКИ:\n"
    #    + json.dumps(ctx.history_messages, ensure_ascii=False, indent=2)
    #)

    # 1) Запуск триажа (передаём ctx)
    partials = run_triage_and_collect(triage_agent, ctx)

    # 2) Финальная агрегация
    final_text = final_aggregation(
        partials,
        ctx.user_text,
        history_messages=ctx.history_messages
    )

    logger.info(f"[get_multiagent_answer] completed, final_text='{final_text}'")

    # Пример: ctx.cat1_ids, ctx.cat2_ids уже могут быть собраны (если вы дописали логику в vector_search)
    # Вы можете их вернуть наружу, если нужно.

    return final_text```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/main.py
```python
# python3 main.py 
# или
# python3 -m bot_control_panel.main

# bot_control_panel/main.py

import gradio as gr

# Импортируем функции, создающие вкладки
from .tab1.tab1_code import create_tab1
from .tab2.tab2_code import create_tab2
from .tab3.tab3_code import create_tab3
from .tab4.tab4_code import create_tab4
from .tab5.tab5_code import create_tab5

# Новый импорт для вкладки 'Стоимость'
from .tab6.tab6_code import create_tab6

def main():
    # Создаём существующие вкладки
    tab1 = create_tab1()
    tab2 = create_tab2()
    tab3 = create_tab3()
    tab4 = create_tab4()
    tab5 = create_tab5()

    # Создаём новую вкладку «Стоимость»
    tab6 = create_tab6()

    # Объединяем все вкладки в одну TabbedInterface
    demo = gr.TabbedInterface(
        [tab1, tab2, tab3, tab4, tab5, tab6],
        [
            "Тестиорование ответов", 
            "Добавление документов", 
            "Редактиование и удаление документов",
            "Загрузка данных",
            "Инструкция",
            "Стоимость"
        ]
    )

    # Запускаем приложение
    demo.launch()

if __name__ == "__main__":
    main()```

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/generate_qa_pairs.py
```python
# python3 generate_qa_pairs.py
# generate_qa_pairs.py
# python3 generate_qa_pairs.py
import os
from dotenv import load_dotenv

import openai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .PROMPT import text_prompt_2

class QAPair(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор, например doc001, doc002 и т.д.")
    question: str = Field(..., description="Вопрос, сформулированный с точки зрения пациента.")
    answer: str = Field(..., description="Короткий, точный ответ от лица клиники.")

class QAPairs(BaseModel):
    qa_pairs: List[QAPair] = Field(
        ...,
        description="Список пар QAPair, отражающих разные смысловые блоки исходного текста.",
    )

def chunk_text_by_paragraphs(text: str, max_length: int = 30000) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        p_len = len(paragraph)
        if p_len > max_length:
            # Упрощённо добавляем как отдельный блок
            chunks.append(paragraph)
            continue
        if current_length + p_len + 2 > max_length:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = p_len
        else:
            current_chunk.append(paragraph)
            current_length += p_len + 2

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

def _process_block(client: OpenAI, block_text: str, block_index: int) -> List[dict]:
    """
    Отправляет один блок текста в модель и возвращает список пар Q&A.
    Если возникает отказ или другая ошибка, выбрасывает исключение.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"{text_prompt_2}\n{block_text}"
            }
        ],
        response_format=QAPairs,
        temperature=0.5
    )

    if response.choices[0].message.refusal:
        raise ValueError(
            f"Модель отказалась сформировать ответ (блок #{block_index}):\n"
            + response.choices[0].message.refusal
        )
    # Парсим ответ в модель QAPairs
    qa_pairs_model: QAPairs = response.choices[0].message.parsed
    # Преобразуем во "flat" формат list[dict]
    return [item.dict() for item in qa_pairs_model.qa_pairs]

def generate_qa_pairs(medcenter_text: str, max_workers: int = 4) -> List[dict]:
    """
    Делим текст на блоки и отправляем их в OpenAI параллельно (потоками).
    Возвращаем общий список словарей [{id, question, answer}, ...].
    """

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Не удалось найти ключ OpenAI в переменной окружения OPENAI_API_KEY.")

    client = OpenAI()
    text_blocks = chunk_text_by_paragraphs(medcenter_text, max_length=30000)

    all_qa_list = []

    # ### Параллелизация запросов к API ###
    # Функция executor.map() тоже подойдёт, но as_completed() даёт
    # возможность обрабатывать результаты по мере готовности.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, block in enumerate(text_blocks, start=1):
            futures.append(executor.submit(_process_block, client, block, idx))

        # Собираем результаты
        for future in as_completed(futures):
            block_qa_list = future.result()  # если были ошибки, тут будет исключение
            all_qa_list.extend(block_qa_list)

    return all_qa_list

# Пример использования
if __name__ == "__main__":
    text_for_analysis = """
    Здесь разместите реальный (или тестовый) текст о работе медцентра.
    1. Мы используем только сертифицированные импланты ...
    2. Консультация стоит 1000 руб. ...
    
    3. Общие условия приёма: пациенту необходимо предварительно заполнить анкету...
    
    4. В случае отмены записи менее чем за 24 часа ...
    ... и т.д.
    """

    try:
        # Запустим с 4 параллельными потоками
        result = generate_qa_pairs(text_for_analysis, max_workers=4)
        print(result)
    except Exception as e:
        print("Ошибка при генерации Q&A пар:", str(e))```

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

## /Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel/TestInfoRetriever.py
```python
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
    print(final_md)```

