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
    main()