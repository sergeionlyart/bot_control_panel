import json


class PromptManager:
    """
    Класс для подготовки массива messages (list[dict]), 
    который затем будет передаваться в OpenAIChatBot.
    """
    
    def build_contradiction_detector_json(self, prompt: str, json_data: dict, user_query: str) -> list:
        """
        Формирует сообщение, где содержатся инструкции классификатора, 
        вопрос (сообщение) пользователя и блок тройных кавычек с анализируемым JSON-документом.

        :param prompt: Текст системного промта (инструкции для классификатора).
        :param json_data: Документ (dict, list и т.п.), который нужно сериализовать в JSON.
        :param user_query: Сообщение пользователя (строка), относительно которого
                           необходимо проводить классификацию документа.
        :return: список сообщений в формате [{'role': 'user', 'content': '...'}].
        """
        # Превращаем объект в красивый (многострочный) JSON
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

        # Формируем итоговый контент
        # prompt здесь содержит основные инструкции, например:
        #   "Вы — ассистент, выполняющий роль классификатора документов по их релевантности ... (и т.д.)"
        # См. детальную структуру ниже
        content = (
            f"{prompt}\n\n"
            f"**Новый документ:** «{user_query}»\n\n"
            f"**Релевантные документы из базы** (каждый документ в формате: идентификатор, сообшение пользователья - ответ на сообшение пользователя ):\"\"\"{json_str}\"\"\""
        )

        return [
            {
                "role": "user",
                "content": content
            }
        ]
    
    def build_prompt_with_json(self, prompt: str, json_data: dict, user_query: str) -> list:
        """
        Формирует сообщение, где содержатся инструкции классификатора, 
        вопрос (сообщение) пользователя и блок тройных кавычек с анализируемым JSON-документом.

        :param prompt: Текст системного промта (инструкции для классификатора).
        :param json_data: Документ (dict, list и т.п.), который нужно сериализовать в JSON.
        :param user_query: Сообщение пользователя (строка), относительно которого
                           необходимо проводить классификацию документа.
        :return: список сообщений в формате [{'role': 'user', 'content': '...'}].
        """
        # Превращаем объект в красивый (многострочный) JSON
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

        # Формируем итоговый контент
        # prompt здесь содержит основные инструкции, например:
        #   "Вы — ассистент, выполняющий роль классификатора документов по их релевантности ... (и т.д.)"
        # См. детальную структуру ниже
        content = (
            f"{prompt}\n\n"
            f"Вопрос (сообщение) пользователя: «{user_query}»\n\n"
            f"Анализируемый документ: \"\"\"{json_str}\"\"\""
        )

        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def build_simple_messages(self, user_message: str) -> list:
        """
        Сценарий: только пользовательское сообщение,
        без дополнительных инструкций.
        """
        return [
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_developer_messages(self, user_message: str, developer_instruction: str) -> list:
        """
        Сценарий: есть developer-инструкция + пользовательское сообщение.
        """
        return [
            {
                "role": "developer",
                "content": developer_instruction
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_json_schema_messages(self, user_message: str) -> list:
        """
        Сценарий: сообщить модели, что нужно вывести результат в JSON.
        Можно расширить под ваши схемы.
        """
        return [
            {
                "role": "developer",
                "content": "You will output JSON according to the provided schema."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

    def build_conversation_with_history(self, user_message: str, conversation_history: list) -> list:
        """
        Сценарий: добавляем уже имеющуюся историю диалога (conversation_history),
        а затем текущее сообщение user.
        
        conversation_history — это list[dict], где каждый словарь имеет 
        роль 'user' или 'assistant' и 'content'.
        """
        # Копируем историю, затем добавляем новое сообщение
        messages = []
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        return messages
    
    
    
    
    
    
    
    
    