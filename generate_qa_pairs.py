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
        print("Ошибка при генерации Q&A пар:", str(e))