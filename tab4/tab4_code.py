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
# (Убедитесь, что путь корректный: 'bot_control_panel.' - или другая структура проекта)


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

        for i in range(MAX_OBJECTS):
            with gr.Row(visible=False) as row:
                tbox = gr.Textbox(label=f"JSON объект #{i+1}", interactive=False, lines=5)
                cbox = gr.Checkbox(label="Выбрать", value=False)
            textboxes.append(tbox)
            checkboxes.append(cbox)
            widgets_for_json.append(row)

        # ---------------------------------------------------------------------
        # Новая функция для анализа Q/A и противоречий
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

                    # Новый шаг: анализируем question/answer
                    analysis_str = analyze_qa_object(response_json_data[i])

                    # Записываем всё в один Textbox
                    combined_output = (
                        f"{obj_str}\n\n"
                        f"=== Результат анализа ===\n"
                        f"{analysis_str}"
                    )
                    textboxes_updates.append(gr.update(value=combined_output, visible=True))
                else:
                    textboxes_updates.append(gr.update(value="", visible=False))

            return [status_msg, response_json_data] + rows_visibility + textboxes_updates

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
        # Новая логика в confirm_selection - сохранение выбранных документов в ChromaDB
        # ---------------------------------------------------------------------
        def confirm_selection(response_json, *checkbox_values):
            print("[DEBUG] confirm_selection вызван.")
            if not response_json:
                print("[DEBUG] Нет JSON для выбора.")
                return "Нет данных для подтверждения."

            selected_objs = []
            for obj, is_checked in zip(response_json, checkbox_values):
                if is_checked:
                    selected_objs.append(obj)

            print("=== Выбранные объекты ===")
            print(json.dumps(selected_objs, ensure_ascii=False, indent=2))
            print("=== /конец ===")

            # Если ничего не выбрано
            if not selected_objs:
                return "Вы не выбрали ни одного объекта!"

            # Сохраняем как новые документы (не используя исходный "id")
            # чтобы ChromaDB сгенерировала свой doc_id
            for doc_data in selected_objs:
                question = doc_data.get("question", "")
                answer = doc_data.get("answer", "")
                # Сформируем основной текст: в данном случае объединим вопрос+ответ
                combined_text = f"Question: {question}\nAnswer: {answer}"

                # Метаданные (можно передать что угодно, напр. исходный id)
                # doc_data.get("id") - это старый ID, мы его не используем как doc_id,
                # но можем сохранить в метадате для справки
                new_metadata = {
                    "original_id": doc_data.get("id", "no_id"),
                    "source": "user_upload",  # или любая другая метка
                }

                # Вызываем add_or_update_document с doc_id=None
                # => внутри функция сама вызовет get_next_doc_id()
                add_or_update_document(
                    new_text=combined_text,
                    new_metadata=new_metadata,
                    doc_id=None  # явно указываем, что не используем старый ID
                )

            return f"Сохранено документов: {len(selected_objs)}"

        inputs_to_confirm = [response_json_state] + checkboxes
        confirm_button.click(
            fn=confirm_selection,
            inputs=inputs_to_confirm,
            outputs=confirm_result
        )

    return tab4


if __name__ == "__main__":
    demo = create_tab4()
    demo.launch()