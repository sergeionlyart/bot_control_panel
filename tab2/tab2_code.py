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

    return tab2