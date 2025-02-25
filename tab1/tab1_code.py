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
    return tab1