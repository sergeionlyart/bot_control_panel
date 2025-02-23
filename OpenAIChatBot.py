# OpenAIChatBot.py

import os
import base64
from typing import Optional, Dict, Any, Union
from io import BufferedReader, BytesIO

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


class OpenAIChatBot:
    """
    Пример класса, который демонстрирует методы:
      1) Отправка текстового сообщения и получение ответа.
      2) Отправка текстового сообщения в JSON формате.
      3) Передача изображения (в base64) для анализа.
      4) Транскрибация аудио и генерация ответа на полученный текст.
      5) Получение векторных эмбеддингов для текста.
      6) Преобразование текста в речь (TTS).

    Параметры temperature, top_p, presence_penalty и т.д.
    можно задавать при инициализации или прямо в методах.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        default_text_model: str = "gpt-4o",
        default_temperature: float = 0,
        default_max_tokens: int = 5000,
        default_top_p: float = 1.0,
        default_presence_penalty: float = 0.0,
        default_frequency_penalty: float = 0.0,
        default_n: int = 1,
    ):
        """
        :param openai_api_key: Ключ для доступа к OpenAI API. Если не указан, берём из окружения.
        :param default_text_model: Модель, используемая по умолчанию при генерации текстовых ответов.
        :param default_temperature: Параметр 'temperature' (стохастичность вывода).
        :param default_max_tokens: Лимит на кол-во генерируемых токенов ответа.
        :param default_top_p: Параметр 'top_p' (nucleus sampling).
        :param default_presence_penalty: Штраф за «присутствие» темы (presence penalty).
        :param default_frequency_penalty: Штраф за частоту упоминаний (frequency penalty).
        :param default_n: Сколько вариантов ответа генерировать за один запрос.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не задан ни напрямую, ни в переменных окружения.")

        # Инициализируем клиент
        self.client = OpenAI(api_key=self.api_key)

        # Параметры по умолчанию
        self.default_text_model = default_text_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_top_p = default_top_p
        self.default_presence_penalty = default_presence_penalty
        self.default_frequency_penalty = default_frequency_penalty
        self.default_n = default_n

    # -------------------------------------------------------------------------------------------
    # 1) Метод для генерации текстового ответа (chat.completions)
    # -------------------------------------------------------------------------------------------
    def generate_text_response(
        self,
        user_message: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Отправляет текст в модель, возвращает ответ + usage.

        :param user_message: Текст запроса.
        :param model: Название модели (по умолчанию self.default_text_model).
        :param temperature, max_tokens, top_p, presence_penalty, frequency_penalty, n:
                      Переопределения параметров генерации.
        :return: dict с ключами "response_text" и "usage" (при наличии).
        """
        chosen_model = model or self.default_text_model
        chosen_temperature = temperature if (temperature is not None) else self.default_temperature
        chosen_max_tokens = max_tokens if (max_tokens is not None) else self.default_max_tokens
        chosen_top_p = top_p if (top_p is not None) else self.default_top_p
        chosen_presence_penalty = (
            presence_penalty if (presence_penalty is not None)
            else self.default_presence_penalty
        )
        chosen_frequency_penalty = (
            frequency_penalty if (frequency_penalty is not None)
            else self.default_frequency_penalty
        )
        chosen_n = n if (n is not None) else self.default_n

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": user_message}],
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            top_p=chosen_top_p,
            presence_penalty=chosen_presence_penalty,
            frequency_penalty=chosen_frequency_penalty,
            n=chosen_n,
            store=True  # рекомендует документация, если хотим хранить историю
        )

        answer_text = response.choices[0].message.content if response.choices else ""

        usage_data = {}
        if hasattr(response, "usage") and response.usage is not None:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 2) Генерация JSON-ответа, используя response_format = json_schema
    # -------------------------------------------------------------------------------------------
    def generate_json_response(
        self,
        user_message: str,
        json_schema: Dict[str, Any],
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Генерация структурированного JSON согласно schema (см. раздел "Generate JSON data" из документации).

        :param user_message: входной текст
        :param json_schema: описание схемы JSON
        :param model: Модель для JSON-вывода (например "gpt-4o-2024-08-06")
        :param temperature: как правило 0, чтобы модель не «фантазировала»
        :return: {"response_json": dict, "usage": {...}} — JSON + usage-токены
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You will output JSON according to the provided schema."},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            },
            temperature=temperature,
            store=True
        )

        import json
        response_str = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed_json = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_json = {"error": "JSON parse error", "raw_response": response_str}

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_json": parsed_json,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 3) Анализ изображения (байты/base64) через chat-completions
    # -------------------------------------------------------------------------------------------
    def analyze_image(
        self,
        image_bytes_base64: str,
        user_question: str = "What is in this image?",
        model: str = "gpt-4o-mini",
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """
        Принимает base64-изображение (можно сделать data:image/*;base64,...) + вопрос. Возвращает описание.

        :param image_bytes_base64: Base64-кодированное изображение (возможно с префиксом 'data:image/...').
        :param user_question: вопрос
        :param model: например "gpt-4o" или "gpt-4o-mini"
        :param detail: "low", "high" или "auto" — детализация
        :return: {"response_text": str, "usage": {...}}
        """
        if not image_bytes_base64.startswith("data:image"):
            image_bytes_base64 = f"data:image/jpeg;base64,{image_bytes_base64}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_bytes_base64,
                                "detail": detail
                            }
                        }
                    ],
                },
            ],
            store=True,
        )

        answer_text = response.choices[0].message.content if response.choices else ""
        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 4) Обработка голосового сообщения: транскрибация + ответ
    # -------------------------------------------------------------------------------------------
    def handle_voice_message(
        self,
        audio_file: Union[str, BufferedReader],
        question: str = "Можешь ответить на это сообщение?",
        transcription_model: str = "whisper-1",
        text_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        1) Транскрибирует аудио (whisper).
        2) Использует транскрибированный текст + question => chat.completions.

        :return: {
            "transcribed_text": str,
            "response_text": str,
            "usage_transcription": None,  # у Whisper нет поля usage
            "usage_chat": {...}           # usage от chat
        }
        """
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                transcription_resp = self.client.audio.transcriptions.create(
                    file=f,
                    model=transcription_model,
                )
        else:
            transcription_resp = self.client.audio.transcriptions.create(
                file=audio_file,
                model=transcription_model,
            )

        transcribed_text = transcription_resp.text

        # Формируем финальный prompt
        combined_prompt = f"Пользователь сказал (голосом): '{transcribed_text}'.\nВопрос: {question}"

        text_model = text_model or self.default_text_model
        chat_answer = self.generate_text_response(
            user_message=combined_prompt,
            model=text_model
        )

        return {
            "transcribed_text": transcribed_text,
            "response_text": chat_answer["response_text"],
            "usage_transcription": None,
            "usage_chat": chat_answer["usage"]
        }

    # -------------------------------------------------------------------------------------------
    # 5) Получение эмбеддинга (embedding) для текста
    # -------------------------------------------------------------------------------------------
    def get_text_embedding(
        self,
        text: str,
        embedding_model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Генерация векторных эмбеддингов (embeddings) для указанного текста.

        :param text: Входной текст
        :param embedding_model: ID модели, напр. 'text-embedding-3-small'
        :param dimensions: Если нужно укоротить вектор, укажите кол-во измерений.
        :return: {"embedding": List[float], "usage": {...}}
        """
        kwargs = {
            "model": embedding_model,
            "input": text,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = self.client.embeddings.create(**kwargs)
        embedding_vector = response.data[0].embedding

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "embedding": embedding_vector,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # 6) Преобразование текста в речь (Text-To-Speech) - TTS
    # -------------------------------------------------------------------------------------------
    def text_to_speech(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Превращает текст в аудио. Возвращает байты, либо сохраняет в файл, если указан save_path.

        :param text: Исходная текстовая строка
        :param model: Модель TTS (напр. 'tts-1' или 'tts-1-hd')
        :param voice: Голос ('alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer')
        :param response_format: Формат аудио ('mp3' по умолчанию, 'opus', 'aac', 'flac', 'wav' и т.д.)
        :param save_path: Путь для записи файла; если None, вернём байты
        :return: {
            "audio_content": bytes | None,
            "voice_used": str,
            "model_used": str,
            "note": "..."
        }
        """
        create_kwargs = {
            "model": model,
            "voice": voice,
            "input": text
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        response = self.client.audio.speech.create(**create_kwargs)

        audio_content = None
        if save_path:
            response.stream_to_file(save_path)
        else:
            buffer = BytesIO()
            response.stream_to_file(buffer)
            audio_content = buffer.getvalue()

        return {
            "audio_content": audio_content,
            "voice_used": voice,
            "model_used": model,
            "note": (
                "По документации TTS не возвращает usage. Формат по умолчанию mp3, "
                "либо задан через response_format."
            )
        }

    # -------------------------------------------------------------------------------------------
    # НОВЫЙ МЕТОД: Генерация ответа из заранее подготовленного messages
    # -------------------------------------------------------------------------------------------
    def generate_chat_response_from_messages(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,  # <-- Добавляем seed
    ) -> Dict[str, Any]:
        """
        Получает на вход список messages (список dict), 
        затем вызывает chat.completions.

        Пример использования:
            pm = PromptManager()
            msgs = pm.build_simple_messages("Hello!")
            bot.generate_chat_response_from_messages(msgs)

        Возвращает dict: {
          "response_text": str,
          "usage": {...}  # при наличии
        }
        """
        chosen_model = model or self.default_text_model
        chosen_temperature = temperature if (temperature is not None) else self.default_temperature
        chosen_max_tokens = max_tokens if (max_tokens is not None) else self.default_max_tokens
        chosen_top_p = top_p if (top_p is not None) else self.default_top_p
        chosen_presence_penalty = (
            presence_penalty if (presence_penalty is not None)
            else self.default_presence_penalty
        )
        chosen_frequency_penalty = (
            frequency_penalty if (frequency_penalty is not None)
            else self.default_frequency_penalty
        )
        chosen_n = n if (n is not None) else self.default_n

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            top_p=chosen_top_p,
            presence_penalty=chosen_presence_penalty,
            frequency_penalty=chosen_frequency_penalty,
            n=chosen_n,
            store=True
        )

        answer_text = ""
        if response.choices and response.choices[0].message:
            answer_text = response.choices[0].message.content

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_text": answer_text,
            "usage": usage_data
        }

    # -------------------------------------------------------------------------------------------
    # НОВЫЙ МЕТОД: аналог для JSON-схем, если хотите жёстко задать response_format
    # -------------------------------------------------------------------------------------------
    def generate_chat_response_with_jsonschema(
        self,
        messages: list,
        json_schema: Dict[str, Any],
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        seed: Optional[int] = None,  # <-- Добавляем seed
    ) -> Dict[str, Any]:
        """
        Если хотите использовать заранее сформированное messages, 
        но с жёстким требованием, что модель должна выдавать JSON по schema.
        
        Пример:
            pm = PromptManager()
            msgs = pm.build_json_schema_messages("My text ...")
            schema = {...}
            result = bot.generate_chat_response_with_jsonschema(msgs, schema)
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            store=True
        )

        import json
        response_str = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed_json = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_json = {"error": "JSON parse error", "raw_response": response_str}

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_json": parsed_json,
            "usage": usage_data
        }

    def generate_chat_response_with_jsonschema_2_FINAL(
        self,
        messages: list,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Пример: хотим получить массив объектов {id, question, answer}.
        """
        json_schema = {
            "name": "QuestionAnswerPairs",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"}
                    },
                    "required": ["id", "question", "answer"],
                    "additionalProperties": False
                }
            }
        }

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            },
            temperature=temperature,
            store=True
        )

        import json
        response_str = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed_json = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_json = {"error": "JSON parse error", "raw_response": response_str}

        usage_data = {}
        if hasattr(response, "usage") and response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return {
            "response_json": parsed_json,
            "usage": usage_data
        }