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
    main()