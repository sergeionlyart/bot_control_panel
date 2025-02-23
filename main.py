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
    main()