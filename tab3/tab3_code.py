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

    return tab3