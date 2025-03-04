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
    return tab5