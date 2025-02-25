
# conversation_manager_5.py

"""последняя рабочая версия с применением Context Object"""

import os
import json
import inspect
import logging
import sys
from dataclasses import dataclass, field
from inspect import Parameter
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv


# Пример вектора поиска
from .TestInfoRetriever import vector_search
from .PROMPT import medical_services_pricing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.debug("Логирование DEBUG включено. Теперь вы будете видеть подробные сообщения в консоли.")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env или окружении.")

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-4o"

# -----------------------------------------------------------------------------
# 1. Определяем класс Context
# -----------------------------------------------------------------------------
@dataclass
class RequestContext:
    request_id: str
    lead_id: int
    user_text: str
    history_messages: List[dict]
    channel: Optional[str] = None

    # Пример полей для хранения cat1_ids, cat2_ids
    cat1_ids: List[str] = field(default_factory=list)
    cat2_ids: List[str] = field(default_factory=list)
    # Добавляйте любые поля, которые захотите отслеживать через контекст


# -----------------------------------------------------------------------------
# Вспомогательная функция: из списка пар {user_message, assistant_message}
# делаем [{'role':..., 'content':...}, ...]
# -----------------------------------------------------------------------------
def build_history_messages_from_db(records: List[dict]) -> List[dict]:
    history = []
    for r in records:
        user_msg = r.get("user_message", "").strip()
        assistant_msg = r.get("assistant_message", "").strip()

        if user_msg:
            history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            history.append({"role": "assistant", "content": assistant_msg})

    return history


# -----------------------------------------------------------------------------
# 2. Инструменты и класс Agent
# -----------------------------------------------------------------------------
def clinic_data_lookup(criteria: str, request_id: str = None):
    """Инструмент для поиска информации о клинике услугах и правилах работы."""
    answer = vector_search(criteria, request_id=request_id)
    return f"Информация полученная из базы знаний : {answer}"

def sql_search(criteria: str):
    """Инструмент для получения прайс-листа."""
    return f"информация полученная из прескуранта : {medical_services_pricing}"

def submit_ticket(description: str):
    """Создать тикет (просто фейк)."""
    return f"(Fake) Тикет создан: {description}"

def calendar_api(action: str, dateTime: str = "", managerId: str = ""):
    """Инструмент календаря: 'check_avail' или 'book_slot'."""
    if action == "check_avail":
        return f"(Fake) calendar_api: Проверяем слот {dateTime} для manager={managerId}"
    elif action == "book_slot":
        return f"(Fake) calendar_api: Слот {dateTime} забронирован (manager={managerId})"
    return "(Fake) calendar_api: неизвестное действие"

def transfer_to_medical_agent() -> str:
    return "HandoffToMedicalAgent"

def transfer_to_legal_agent() -> str:
    return "HandoffToLegalAgent"

def transfer_to_super_agent() -> str:
    return "HandoffToSuperAgent"

def transfer_to_finance_agent() -> str:
    return "HandoffToFinanceAgent"


class Agent:
    def __init__(self, name: str, instructions: str, tools: list, model: str = MODEL_NAME):
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.model = model


# -----------------------------------------------------------------------------
# 3. Схема для инструментов (function-calling)
# -----------------------------------------------------------------------------
def function_to_schema(func) -> dict:
    """
    Генерирует схему инструмента (функции) в формате "type":"function"
    по документации OpenAI.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    properties = {}
    required_fields = []

    for param_name, param in signature.parameters.items():
        annotation = param.annotation if param.annotation is not Parameter.empty else str
        json_type = type_map.get(annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is Parameter.empty:
            required_fields.append(param_name)

    description = (func.__doc__ or "").strip()

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False
            }
        }
    }


# -----------------------------------------------------------------------------
# 4. Объявляем агентов
# -----------------------------------------------------------------------------
super_agent = Agent(
    name="Super Agent",
    instructions=(
        """Вы — Супер Агент (резервный). Обрабатывайте любые запросы, выходящие за рамки других агентов.
как опытный администратор медцентра для получения информации всегда используйте инструмент 'clinic_data_lookup'"""
    ),
    tools=[clinic_data_lookup],
    model=MODEL_NAME
)

finance_agent = Agent(
    name="Finance Agent",
    instructions=(
        """Вы Агент, отвечающий за стоимость услуг в медцентре.
1) Вызовите 'sql_search', чтобы получить прайс или стоимость.
"""
    ),
    tools=[sql_search],
    model=MODEL_NAME
)

medical_agent = Agent(
    name="Medical Agent",
    instructions=(
        """Вы Medical Agent администратор в медцентре.
Ваша задача — максимально точно отвечать на вопросы посетителей.
Всегда используйте 'clinic_data_lookup' для поиска мед. информации.
"""
    ),
    tools=[clinic_data_lookup],
    model=MODEL_NAME
)

legal_agent = Agent(
    name="Legal Agent",
    instructions=(
        """Вы Legal Agent. Отвечаете на юридические вопросы (контракты, GDPR и т.п.).
Если вопрос не юридический — вызывайте transfer_to_super_agent.
"""
    ),
    tools=[transfer_to_super_agent],
    model=MODEL_NAME
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        """Вы — Triage Agent.
Правила:
1) Вопросы про работу медцентра, порядок оказания услуг, стоимость => transfer_to_medical_agent (1 раз).
2) Если вопрос юридический => transfer_to_legal_agent (1 раз).
3) Если вопрос про стоимость — можно вызвать Finance Agent (1 раз).
4) Иначе => transfer_to_super_agent (1 раз).
"""
    ),
    tools=[
        transfer_to_medical_agent,
        transfer_to_legal_agent,
        transfer_to_super_agent,
        transfer_to_finance_agent
    ],
    model=MODEL_NAME
)


# -----------------------------------------------------------------------------
# 5. Функция map_handoff_to_agent(...)
# -----------------------------------------------------------------------------
def map_handoff_to_agent(handoff_string: str) -> Agent:
    if handoff_string == "HandoffToMedicalAgent":
        return medical_agent
    elif handoff_string == "HandoffToLegalAgent":
        return legal_agent
    elif handoff_string == "HandoffToFinanceAgent":
        return finance_agent
    elif handoff_string == "HandoffToSuperAgent":
        return super_agent
    return super_agent


# -----------------------------------------------------------------------------
# 6. execute_tool_call с подмешиванием request_id из ctx
# -----------------------------------------------------------------------------
def execute_tool_call(function_name: str, args: dict, tools_map: dict, agent_name: str, ctx: RequestContext):
    """
    Добавляем ctx, чтобы подмешивать request_id, lead_id и т.д. в аргументы инструмента,
    если инструмент их поддерживает.
    """
    logger.debug(f"[execute_tool_call] {agent_name} вызывает '{function_name}' c аргументами {args}")

    if function_name not in tools_map:
        logger.warning(f"{agent_name} tried to call unknown function {function_name}")
        return f"[Warning] Unknown function: {function_name}"

    # Подмешиваем request_id, если у функции в сигнатуре есть такой параметр
    sig = inspect.signature(tools_map[function_name])
    if "request_id" in sig.parameters:
        args.setdefault("request_id", ctx.request_id)

    result = tools_map[function_name](**args)
    logger.debug(f"[execute_tool_call] Результат '{function_name}': {result}")
    return result


# -----------------------------------------------------------------------------
# 7. run_subagent принимает ctx вместо user_message, history_messages
# -----------------------------------------------------------------------------
def run_subagent(agent: Agent, ctx: RequestContext) -> str:
    """
    Запуск одного саб-агента. Если агент вызывает handoff, переходим к другому.
    """
    logger.debug(f"[run_subagent] START, agent='{agent.name}' user_message='{ctx.user_text}'")

    # Формируем массив сообщений из ctx.history_messages + dev instructions + текущее user_text
    messages = []
    if ctx.history_messages:
        messages.extend(ctx.history_messages)

    messages.append({"role": "developer", "content": agent.instructions})
    messages.append({"role": "user", "content": ctx.user_text})

    all_messages = messages[:]

    tools_map = {t.__name__: t for t in agent.tools}
    tool_schemas = [function_to_schema(t) for t in agent.tools]

    # Вместо set() используем счетчик, чтобы дать возможность инструменту вызываться несколько раз
    from collections import defaultdict
    used_tools_count = defaultdict(int)

    max_rounds = 2
    consecutive_empty = 0

    for step in range(max_rounds):
        logger.debug(f"[run_subagent] step={step}, agent='{agent.name}'")

        if tool_schemas:
            response = client.chat.completions.create(
                model=agent.model,
                messages=all_messages,
                tools=tool_schemas,
                temperature=0
            )
        else:
            response = client.chat.completions.create(
                model=agent.model,
                messages=all_messages,
                temperature=0
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        # Добавляем ответ ассистента (без инструментов) в историю
        all_messages.append({"role": "assistant", "content": content})
        logger.debug(f"[run_subagent] agent='{agent.name}' content='{content}', tool_calls={tool_calls}")

        if not content and not tool_calls:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        if consecutive_empty >= 1:
            logger.debug("[run_subagent] Дважды пустой ответ — завершаем.")
            break

        # Обрабатываем вызовы инструментов
        if not tool_calls:
            # Нет вызовов инструментов, значит это финальный ответ
            break

        for tc in tool_calls:
            fn_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except:
                args = {}

            # --- Разрешаем вызывать один и тот же инструмент до 2 раз ---
            if used_tools_count[fn_name] >= 2:
                logger.debug(f"[run_subagent] {agent.name}: инструмент '{fn_name}' уже вызывался >= 2 раз, пропускаем.")
                continue

            used_tools_count[fn_name] += 1

            result = execute_tool_call(fn_name, args, tools_map, agent.name, ctx)
            logger.debug(f"[run_subagent] Результат вызова инструмента '{fn_name}': {result}")

            # Если мы получили HandoffTo..., переходим к другому агенту
            if result.startswith("HandoffTo"):
                new_agent = map_handoff_to_agent(result)
                logger.debug(f"[run_subagent] handoff -> {new_agent.name}")
                # Перезапускаем subagent (рекурсивно) с тем же контекстом
                return run_subagent(new_agent, ctx)
            else:
                # Добавляем вывод инструмента как очередное сообщение assistant
                all_messages.append({
                    "role": "assistant",
                    "content": f"[Tool output] {result}"
                })

    # Собираем финальный ответ
    final_answer = ""
    for i in reversed(all_messages):
        if i["role"] == "assistant" and not i["content"].startswith("[Tool output]"):
            final_answer = i["content"]
            break

    logger.debug(f"[run_subagent] END, final_answer='{final_answer}'")
    return final_answer


# -----------------------------------------------------------------------------
# 8. run_triage_and_collect тоже принимает ctx
# -----------------------------------------------------------------------------
def run_triage_and_collect(triage_agent: Agent, ctx: RequestContext) -> List[Dict[str, str]]:
    """
    Запускаем Triage Agent и собираем partial ответы от суб-агентов.
    Теперь разрешаем повторный вызов одного и того же инструмента
    (например, transfer_to_super_agent) до 2 раз.
    """
    logger.debug("[run_triage_and_collect] START")

    triage_msgs = []
    if ctx.history_messages:
        triage_msgs.extend(ctx.history_messages)

    # Добавляем инструкции триаж-агента и текущее сообщение пользователя
    triage_msgs.append({"role": "developer", "content": triage_agent.instructions})
    triage_msgs.append({"role": "user", "content": ctx.user_text})

    partials = []

    # Создаём мапу "имя_функции -> сама функция" для быстрого вызова
    tools_map = {t.__name__: t for t in triage_agent.tools}
    tool_schemas = [function_to_schema(t) for t in triage_agent.tools]

    # Вместо set() используем счётчик, чтобы разрешить до 2 вызовов одного и того же инструмента
    from collections import defaultdict
    used_tools_count = defaultdict(int)

    max_rounds = 2
    consecutive_empty = 0

    for step in range(max_rounds):
        #logger.debug(f"[run_triage_and_collect] step={step}")

        # Делаем запрос к модели с учётом tools
        if tool_schemas:
            response = client.chat.completions.create(
                model=triage_agent.model,
                messages=triage_msgs,
                tools=tool_schemas,
                temperature=0.0
            )
        else:
            # Если у агента нет инструментов — обычный чат без function-calling
            response = client.chat.completions.create(
                model=triage_agent.model,
                messages=triage_msgs,
                temperature=0.0
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        triage_msgs.append({"role": "assistant", "content": content})
        #logger.debug(f"[run_triage_and_collect] content='{content}', tool_calls={tool_calls}")

        if not content and not tool_calls:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        if consecutive_empty >= 1:
            #logger.debug("[run_triage_and_collect] Дважды пусто — завершаем.")
            break

        # Если нет вызовов инструментов, значит это финальный ответ триаж-агента
        if not tool_calls:
            break

        # Обрабатываем инструментальные вызовы
        for tc in tool_calls:
            fn_name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except:
                args = {}

            # --- проверяем счётчик для инструмента ---
            if used_tools_count[fn_name] >= 2:
                logger.debug(f"[run_triage_and_collect] '{fn_name}' уже вызывался >= 2 раз, пропускаем.")
                continue
            used_tools_count[fn_name] += 1

            # Если функция (инструмент) не зарегистрирована
            if fn_name not in tools_map:
                logger.warning(f"Unknown function: {fn_name}")
                triage_msgs.append({"role": "assistant", "content": f"Unknown function: {fn_name}"})
                continue

            # Вызываем инструмент
            result = tools_map[fn_name](**args)
            logger.debug(f"[run_triage_and_collect] result={result}")

            # Если инструмент возвращает HandoffTo..., передаём запрос соответствующему агенту
            if result.startswith("HandoffTo"):
                subagent = map_handoff_to_agent(result)
                partial_answer = run_subagent(subagent, ctx)
                partials.append({"agent": subagent.name, "answer": partial_answer})
                triage_msgs.append({
                    "role": "assistant",
                    "content": f"[Triage -> {subagent.name}] partial ok"
                })
            else:
                # Иначе это «обычный» инструмент: выводим результат
                triage_msgs.append({
                    "role": "assistant",
                    "content": f"[Tool output] {result}"
                })

    #logger.debug(f"[run_triage_and_collect] END, partials={partials}")
    return partials


# -----------------------------------------------------------------------------
# 9. final_aggregation — пока оставим без ctx, но можно и туда передавать
# -----------------------------------------------------------------------------
def final_aggregation(
    partials: List[Dict[str, str]],
    user_message: str,
    history_messages: Optional[List[dict]] = None
) -> str:
    """
    Формируем итоговый запрос к GPT, включая:
    - Основную инструкцию (developer)
    - partials (промежуточные ответы)
    - Историю диалога (history_messages)
    - Сообщение пользователя
    """
    logger.debug("[final_aggregation] START")

    partials_str = json.dumps(partials, ensure_ascii=False, indent=2)

    if not history_messages:
        history_messages = []
    history_json = json.dumps(history_messages, ensure_ascii=False, indent=2)

    developer_content = f"""
Вы Врач-администратор колцентра в медцентре. Используйте предоставленные "** ДАННЫЕ НА ОСВНОАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ 
ОТВЕТ  :"
и вопрос пользователя для формулирования дружелюбного ответа. Формируйте ответы от своего имени, не ссылаясь на других 
администраторов, работников медцентра, или других агентов. Ваши ответы должны быть в формате разговора с пациентом и не 
начинайтесь с приветствия, кроме случаев первого сообщения от пользователя.

При формировании ответа используйте только информацию из раздела "ДАННЫЕ НА ОСНОВАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ ОТВЕТ" и 
никогда не придумывайте ответы. Если информации недостаточно, сообщите пользователю, что вы не можете ответить на его вопрос.

# Шаги

1. Прочитайте вопрос пользователя и соответствующие Partial-ответы.
2. Определите информацию, имеющую отношение к вопросу из предоставленных данных.
3. Составьте ответ, используя только доступную информацию.
4. Информацию, которой нет в предоставленных данных, не добавляйте в ответ.
5. Если информации недостаточно, вежливо сообщите об этом пользователю.

# Формат ответа

Ответ должен быть в виде дружелюбного и вежливого текста, содержащего только информацию из предоставленных данных.

**Инструкция  - Алгоритм ответа на вопросы типа : “** **“Как я могу записаться на прием?”:**

1. **Приветствие и уточнение запроса**

🔹 “Подскажите, к какому специалисту или на какую процедуру вы хотите записаться?”

(Важно понять, нужен ли врач-стоматолог, терапевт, узкий специалист или диагностика.)

Если эта информация получена ранее переходим к следующему шагу . 

2. **Сбор основной информации**

🔹 “Как вас зовут?” (ФИО)

🔹 “Есть ли у вас предпочтения по дате и времени?”

🔹 “Вы уже были у нас раньше или записываетесь впервые?”

Если эта информация получена ранее переходим к следующему шагу . 

3. **Проверка доступного времени и предложения вариантов**

🔹 Проверить в системе расписание врача.

🔹 “На ближайшие дни есть свободное время: [Среда 8-00 и Четверг 15-00]. Какой вам удобнее?”

Если эта информация получена ранее переходим к следующему шагу . 

4. **Подтверждение записи**

🔹 “Подтверждаю вашу запись на [дата, время].”

🔹 “Вам отправить напоминание по SMS или мессенджеру?”

Если эта информация получена ранее переходим к следующему шагу . 

5. **Дополнительные инструкции**

🔹 “Приходите за 10–15 минут до приема, возьмите с собой документы (если нужно).”

🔹 “Если у вас изменятся планы, дайте нам знать заранее.”

Если эта информация получена ранее переходим к следующему шагу . 

6. **Заключение**

🔹 “Спасибо, что выбрали нашу клинику! Если появятся вопросы, звоните или пишите.”

** ДАННЫЕ НА ОСВНОАНИИ КОТОРЫХ ВЫ ДОЛЖНЫ СФОРМИРОВАТЬ ОТВЕТ  :
{partials_str}

ИСТОРИЯ ДИАЛОГА (ДЛЯ КОНТЕКСТА):
{history_json}

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ, НА КОТОРОЕ НУЖНО ОТВЕТИТЬ:
{user_message}
"""

    print("ИТОГОВАЯ ИНСТРУКЦИЯ:", developer_content)

    messages = [
        {
            "role": "developer",
            "content": developer_content
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0
    )
    final_text = response.choices[0].message.content or ""
    return final_text


# -----------------------------------------------------------------------------
# 10. get_multiagent_answer, который принимает (и создаёт) RequestContext
# -----------------------------------------------------------------------------
def get_multiagent_answer(
    user_text: str,
    lead_id: int = None,
    channel: str = None,
    history_messages: Optional[List[dict]] = None,
    request_id: str = None
) -> str:
    """
    Главная точка входа (синхронная).
    Теперь история передаётся извне через history_messages.
    Мы создаём RequestContext и передаём его в triage & subagents.
    """
    #logger.info(f"[get_multiagent_answer] called, user_text='{user_text}', request_id={request_id}")

    if not history_messages:
        history_messages = []

    # Сформируем контекст
    ctx = RequestContext(
        request_id=request_id or "noRequestId",
        lead_id=lead_id or 0,
        user_text=user_text,
        history_messages=history_messages,
        channel=channel
    )

    # Логируем историю
    #logger.info(
    #    "[get_multiagent_answer] ИСТОРИЯ ПЕРЕПИСКИ:\n"
    #    + json.dumps(ctx.history_messages, ensure_ascii=False, indent=2)
    #)

    # 1) Запуск триажа (передаём ctx)
    partials = run_triage_and_collect(triage_agent, ctx)

    # 2) Финальная агрегация
    final_text = final_aggregation(
        partials,
        ctx.user_text,
        history_messages=ctx.history_messages
    )

    logger.info(f"[get_multiagent_answer] completed, final_text='{final_text}'")

    # Пример: ctx.cat1_ids, ctx.cat2_ids уже могут быть собраны (если вы дописали логику в vector_search)
    # Вы можете их вернуть наружу, если нужно.

    return final_text