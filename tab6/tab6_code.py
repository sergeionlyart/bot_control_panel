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

    return tab6