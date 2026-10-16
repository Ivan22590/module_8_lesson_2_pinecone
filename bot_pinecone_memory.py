"""
bot_pinecone_memory.py — Telegram-бот-помощник с памятью в векторной базе Pinecone.

Отличие от bot_with_memory.py: вместо SQLite-тезисов, подгружаемых «все подряд»,
используется семантический поиск в векторной базе Pinecone (модуль pine.py).
Перед каждым ответом бот:

    1. Превращает сообщение пользователя в вектор (get_embedding);
    2. Находит топ-10 самых близких по смыслу воспоминаний этого пользователя
       (query с фильтром по user_id в metadata);
    3. Вставляет их в системный промпт как «долгосрочную память».

После каждого ответа модель выделяет ключевые тезисы диалога, которые
превращаются в векторы и сохраняются в Pinecone вместе с метаданными
(user_id, текст тезиса). База переживает перезапуски — бот помнит
интересы пользователя хоть через неделю, хоть через год, и вспоминает
именно те факты, которые релевантны текущему вопросу.

Переменные окружения (см. .env.example):
    TG_BOT_API_KEY    — токен Telegram-бота от @BotFather (обязательный);
    PROXYAPI_API_KEY  — ключ ProxyAPI (обязательный);
    PROXYAPI_API_BASE — базовый URL ProxyAPI (по умолчанию https://api.proxyapi.ru/v1);
    LLM_MODEL         — имя модели (по умолчанию openai/gpt-5.6-luna);
    LLM_TIMEOUT       — таймаут запроса к LLM в секундах (по умолчанию 60);
    LLM_MAX_RETRIES   — число автоматических повторов запроса (по умолчанию 2);
    PINECONE_API_KEY  — ключ Pinecone (обязательный);
    PINECONE_INDEX    — имя индекса памяти (по умолчанию bot-memory);
    RECALL_TOP_K      — сколько воспоминаний подгружать в промпт (по умолчанию 10);
    HISTORY_LIMIT     — размер краткой истории диалога в памяти (по умолчанию 20).

Запуск: python bot_pinecone_memory.py
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import os
import re
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError

from pine import PineconeClient, PineconeClientError, get_embedding as _sync_get_embedding

# ================================================================
# Конфигурация
# ================================================================

load_dotenv()

DEFAULT_PROXYAPI_BASE = "https://api.proxyapi.ru/v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_INDEX_NAME = "bot-memory"
DEFAULT_RECALL_TOP_K = 10
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small выдаёт 1536-мерные векторы

TG_MESSAGE_LIMIT = 4096  # жёсткий лимит длины сообщения Telegram, символов

TYPING_REFRESH_SECONDS = 5.0  # Telegram показывает статус «печатает…» примерно 5 секунд

SYSTEM_PROMPT_BASE = """\
Ты — полезный AI-ассистент в Telegram-боте.
Отвечай на языке пользователя (русский — если вопрос на русском).
Будь кратким и информативным, не повторяй вопрос и не перечисляй то, \
что уже написал пользователь.
Если запрос неясен — задай уточняющий вопрос, не пытайся угадать.
Используй простую разметку: **жирный**, *курсив*, `код`, блоки кода через тройные обратные кавычки.

Помимо сообщения для пользователя ты всегда выделяешь ключевые тезисы диалога.\
"""

MEMORY_SECTION_TEMPLATE = """\
Долгосрочная память о пользователе — воспоминания из прошлых диалогов, \
отобранные по смыслу для текущего вопроса:
{recalled}

Учитывай эти воспоминания в ответах: если тема пересекается с прошлыми \
интересами, можно опираться на них и уточнять детали.\
"""

logger = logging.getLogger("bot")


class SettingsError(RuntimeError):
    """Ошибка конфигурации: отсутствуют или некорректны переменные окружения."""


def _env_float(name: str, default: float) -> float:
    """Читает вещественное число из переменной окружения.

    Args:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная не задана.

    Returns:
        Значение переменной либо default, если переменная пуста.

    Raises:
        SettingsError: Если значение не удаётся преобразовать во float.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"Переменная {name}={raw!r} не является числом.") from exc


def _env_int(name: str, default: int) -> int:
    """Читает целое число из переменной окружения.

    Args:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная не задана.

    Returns:
        Значение переменной либо default, если переменная пуста.

    Raises:
        SettingsError: Если значение не удаётся преобразовать в int.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"Переменная {name}={raw!r} не является целым числом.") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки бота, читаются из переменных окружения (файл .env).

    Attributes:
        tg_bot_token: Токен Telegram-бота, полученный у @BotFather.
        proxyapi_key: API-ключ ProxyAPI для доступа к LLM.
        proxyapi_base: Базовый URL OpenAI-совместимого API.
        model: Идентификатор модели LLM.
        timeout: Таймаут одного запроса к LLM, в секундах.
        max_retries: Число автоматических повторов запроса при сбое.
        pinecone_api_key: API-ключ векторной базы Pinecone.
        index_name: Имя индекса Pinecone для долговременной памяти.
        recall_top_k: Сколько релевантных воспоминаний подгружать в промпт.
        history_limit: Размер краткой истории диалога в оперативной памяти.
    """

    tg_bot_token: str
    proxyapi_key: str
    proxyapi_base: str
    model: str
    timeout: float
    max_retries: int
    pinecone_api_key: str
    index_name: str
    recall_top_k: int
    history_limit: int

    @classmethod
    def from_env(cls) -> Settings:
        """Считывает настройки из переменных окружения.

        Returns:
            Заполненный экземпляр настроек.

        Raises:
            SettingsError: Если обязательная переменная отсутствует
                или переменная числового типа задана в неверном формате.
        """
        missing = [
            name
            for name in ("TG_BOT_API_KEY", "PROXYAPI_API_KEY", "PINECONE_API_KEY")
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise SettingsError(
                "Отсутствуют обязательные переменные окружения: "
                f"{', '.join(missing)}. Создайте .env из .env.example и заполните значения."
            )
        history_limit = _env_int("HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)
        if history_limit < 2:
            raise SettingsError("HISTORY_LIMIT должен быть не меньше 2 сообщений.")
        recall_top_k = _env_int("RECALL_TOP_K", DEFAULT_RECALL_TOP_K)
        if not 1 <= recall_top_k <= 50:
            raise SettingsError("RECALL_TOP_K должен быть в диапазоне 1..50.")
        return cls(
            tg_bot_token=os.environ["TG_BOT_API_KEY"].strip(),
            proxyapi_key=os.environ["PROXYAPI_API_KEY"].strip(),
            proxyapi_base=os.getenv("PROXYAPI_API_BASE", DEFAULT_PROXYAPI_BASE).strip(),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip(),
            timeout=_env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            max_retries=_env_int("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            pinecone_api_key=os.environ["PINECONE_API_KEY"].strip(),
            index_name=os.getenv("PINECONE_INDEX", DEFAULT_INDEX_NAME).strip()
            or DEFAULT_INDEX_NAME,
            recall_top_k=recall_top_k,
            history_limit=history_limit,
        )


def create_llm_client(settings: Settings) -> AsyncOpenAI:
    """Создаёт клиент ProxyAPI (OpenAI-совместимый) с параметрами из настроек.

    Args:
        settings: Настройки бота.

    Returns:
        Готовый асинхронный клиент LLM.
    """
    return AsyncOpenAI(
        api_key=settings.proxyapi_key,
        base_url=settings.proxyapi_base,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )


def create_pinecone_client(settings: Settings) -> PineconeClient:
    """Создаёт клиент векторной базы Pinecone с параметрами из настроек.

    Индекс при отсутствии создаётся автоматически (serverless).

    Args:
        settings: Настройки бота.

    Returns:
        Готовый клиент Pinecone.

    Raises:
        PineconeClientError: Если не удалось подключиться или создать индекс.
    """
    return PineconeClient(
        index_name=settings.index_name,
        dimension=EMBEDDING_DIMENSION,
        api_key=settings.pinecone_api_key,
    )


# ================================================================
# Структурированный ответ модели
# ================================================================


class BotReply(BaseModel):
    """Структурированный ответ модели: JSON с тезисами и сообщением.

    Модель используется как response_format — API генерирует из неё
    JSON-schema и валидирует ответ, поэтому «сырой» JSON приходит уже
    проверенным.

    Attributes:
        theses: Ключевые тезисы текущего диалога для сохранения в Pinecone.
        message: Сообщение для отправки пользователю в чат (тезисы он не видит).
    """

    theses: list[str] = Field(
        description=(
            "Ключевые моменты диалога для долговременной памяти, каждый в виде "
            "короткой самодостаточной фразы от третьего лица: "
            "'Пользователь хочет поехать в [место]', "
            "'Пользователь работает программистом', "
            "'День рождения Саши — 15 марта' и т.д. "
            "Если новых тезисов нет — пустой список."
        )
    )
    message: str = Field(
        description=(
            "Только текст ответа пользователю, на его языке и без упоминания тезисов; "
            "простая разметка Telegram: **жирный**, *курсив*, `код`, ```блоки кода```."
        )
    )


# ================================================================
# Форматирование и нарезка ответа
# ================================================================

# Regex прекомпилированы заранее: повторная компиляция в re.sub на каждом вызове
# израсходовала бы лишнее время на каждый ответ бота.
_FENCED_CODE_RE = re.compile(r"```(.+?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

# Токены, по которым допустим разрез HTML-текста: собственные теги разметки и перевод строки.
# Пользовательский текст сюда попасть не может — перед конвертацией он экранируется,
# поэтому «сырые» угловые скобки превращаются в безобидные &lt; и &gt;.
_HTML_TOKEN_RE = re.compile(r"</?(?:b|i|code|pre)>|\n")

# Плейсхолдеры кода внутри JSON-вопроса (для ручного разбора fallback)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _md_to_html(text: str) -> str:
    """Экранирует HTML-символы и преобразует простую Markdown-разметку в HTML Telegram.

    Поддерживаются: ```блоки кода```, `код`, **жирный**, *курсив*.
    Экранирование выполняется первым, чтобы пользовательский текст
    не мог сломать итоговую разметку; содержимое код-фрагментов
    от дополнительной разметки защищено.

    Args:
        text: Ответ модели в Markdown.

    Returns:
        HTML-строка, безопасная для parse_mode=HTML в Telegram.
    """
    text = html.escape(text)

    # Код-фрагменты убираем за плейсхолдеры: иначе regex'ы жирного/курсива
    # могут замотать уже вставленные <code>/<pre> и нарушить вложенность тегов
    # (например, текст вида `a*` и `b*`).
    stashed: list[str] = []

    def stash(fragment: str) -> str:
        """Прячет готовый HTML-фрагмент кода до финальной сборки."""
        stashed.append(fragment)
        return f"\x00{len(stashed) - 1}\x00"

    text = _FENCED_CODE_RE.sub(lambda m: stash(f"<pre>{m.group(1)}</pre>"), text)
    text = _INLINE_CODE_RE.sub(lambda m: stash(f"<code>{m.group(1)}</code>"), text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    return _PLACEHOLDER_RE.sub(lambda m: stashed[int(m.group(1))], text)


def _split_html_text(source: str, limit: int = TG_MESSAGE_LIMIT) -> list[str]:
    """Разбивает HTML-текст на части длиной не более limit символов.

    Разрез выполняется по границам строк и тегов, поэтому сами теги и
    HTML-сущности никогда не разрываются пополам. Незакрытые на границе теги
    форматирования (<b>, <i>, <code>, <pre>) закрываются в конце части и снова
    открываются в начале следующей — визуальное форматирование сохраняется.

    Args:
        source: HTML-текст, результат :func:`_md_to_html`.
        limit: Максимальная длина одной части.

    Returns:
        Список частей, каждая не длиннее limit; разметка в каждой части
        сбалансирована, поэтому части можно отправлять независимо.
    """

    def closers() -> str:
        """Возвращает закрывающие теги для открытых в данный момент."""
        return "".join(f"</{name}>" for name in reversed(open_tags))

    def openers() -> str:
        """Возвращает открывающие теги для повторного открытия в новой части."""
        return "".join(f"<{name}>" for name in open_tags)

    def flush() -> None:
        """Сохраняет текущую часть (закрывая открытые теги) и начинает новую."""
        nonlocal current, current_len
        chunks.append("".join(current) + closers())
        current = [openers()]
        current_len = len(openers())

    def add_run(run: str) -> None:
        """Укладывает текст без переводов строк в текущую часть.

        Если текст не помещается, он разрезается: остаток уходит в следующую
        часть, где открытые теги переоткрываются. Свободное место считается
        с запасом на закрывающие теги, чтобы итоговая часть не превысила лимит.
        """
        nonlocal current_len
        while run:
            free = limit - current_len - len(closers())
            if len(run) <= free:
                current.append(run)
                current_len += len(run)
                return
            take = max(free, 0)
            current.append(run[:take])
            run = run[take:]
            flush()  # после него current_len становится равным длине openers() < limit

    chunks: list[str] = []
    current: list[str] = []  # фрагменты текущей части
    current_len = 0
    open_tags: list[str] = []  # имена открытых тегов в порядке их открытия
    pos = 0

    for match in _HTML_TOKEN_RE.finditer(source):
        text = source[pos : match.start()]
        token = match.group()
        pos = match.end()

        add_run(text)

        if token == "\n":
            if current_len + 1 + len(closers()) > limit:
                flush()
            current.append("\n")
            current_len += 1
            continue

        # Тег: учитываем его длину и запас под закрывающие теги
        if current_len + len(token) + len(closers()) > limit:
            flush()
        current.append(token)
        current_len += len(token)
        if token.startswith("</"):
            name = token[2:-1]
            if open_tags and open_tags[-1] == name:
                open_tags.pop()
        else:
            open_tags.append(token[1:-1])

    add_run(source[pos:])  # хвост после последнего токена

    remainder = "".join(current)
    if remainder.strip():
        chunks.append(remainder + closers())
    return chunks


def _strip_tags(text: str) -> str:
    """Удаляет теги разметки и восстанавливает экранированные символы.

    Используется как запасной вариант, если Telegram отклонил HTML-версию.

    Args:
        text: HTML-строка с тегами <b>, <i>, <code>, <pre>.

    Returns:
        Обычный текст без разметки.
    """
    stripped = re.sub(r"</?(?:b|i|code|pre)>", "", text)
    return html.unescape(stripped)


async def _keep_typing(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Поддерживает индикатор «печатает…», пока бот ждёт ответ от LLM.

    Telegram показывает статус около 5 секунд, поэтому отправка chat action
    повторяется с интервалом TYPING_REFRESH_SECONDS до установки stop-флага.

    Args:
        bot: Экземпляр бота.
        chat_id: Идентификатор чата, в котором ждём ответ.
        stop: Событие-флаг: как только установлено, фоновая задача завершается.
    """
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except TelegramForbiddenError:
            return  # пользователь заблокировал бота — индикатор не нужен
        try:
            await asyncio.wait_for(stop.wait(), timeout=TYPING_REFRESH_SECONDS)
        except asyncio.TimeoutError:
            continue


async def _send_safely(message: Message, chunk: str) -> None:
    """Отправляет одну часть ответа в чат, устойчиво к сбоям Telegram.

    Порядок действий: повтор отправки при ограничении частоты (RetryAfter),
    обычная отправка HTML; при отклонении разметки — отправка очищенного
    текста; при блокировке бота пользователем — тихое завершение.

    Args:
        message: Исходное сообщение пользователя (ответ уходит в тот же чат).
        chunk: HTML-фрагмент ответа модели.
    """
    try:
        await message.answer(chunk, parse_mode=ParseMode.HTML)
    except TelegramRetryAfter as exc:
        logger.warning("Telegram ограничил частоту запросов, повтор через %s с", exc.retry_after)
        await asyncio.sleep(exc.retry_after)
        try:
            await message.answer(chunk, parse_mode=ParseMode.HTML)
        except Exception:
            logger.error("Не удалось отправить ответ в чат %s", message.chat.id, exc_info=True)
    except TelegramForbiddenError:
        user_id = message.from_user.id if message.from_user else "?"
        logger.info("Пользователь %s заблокировал бота — ответ не отправлен", user_id)
    except TelegramBadRequest as exc:
        logger.warning("Telegram отклонил HTML (%s), отправляю обычный текст", exc)
        try:
            await message.answer(_strip_tags(chunk), parse_mode=None)
        except Exception:
            logger.error("Не удалось отправить ответ в чат %s", message.chat.id, exc_info=True)
    except Exception:
        logger.error("Не удалось отправить ответ в чат %s", message.chat.id, exc_info=True)


# ================================================================
# Работа с LLM: структурированный вывод + fallback
# ================================================================


def build_system_prompt(recalled: Sequence[str]) -> str:
    """Собирает системный промпт: базовое поведение + релевантные воспоминания.

    Args:
        recalled: Воспоминания пользователя, найденные по смыслу в Pinecone.

    Returns:
        Полный текст системного промпта для запроса к модели.
    """
    if not recalled:
        return SYSTEM_PROMPT_BASE
    recalled_lines = "\n".join(f"- {fact}" for fact in recalled)
    return (
        SYSTEM_PROMPT_BASE
        + "\n\n"
        + MEMORY_SECTION_TEMPLATE.format(recalled=recalled_lines)
    )


def _parse_reply_json(content: str) -> BotReply:
    """Разбирает JSON-ответ модели вручную (fallback-путь).

    Убирает ```json-обёртку и возможные поясные слова вокруг объекта,
    затем валидирует данные Pydantic-моделью.

    Args:
        content: Текст ответа модели, содержащий JSON с ключами theses/message.

    Returns:
        Проверенный объект BotReply.

    Raises:
        ValidationError: Если JSON отсутствует или не соответствует схеме.
    """
    text = content.strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return BotReply.model_validate_json(text)


async def ask_llm(
    client: AsyncOpenAI,
    settings: Settings,
    system_prompt: str,
    history: list[dict[str, str]],
    user_input: str,
) -> BotReply:
    """Получает структурированный ответ модели на сообщение пользователя.

    Основной путь — встроенный структурированный вывод
    (client.chat.completions.parse + response_format=BotReply): API сам
    генерирует JSON-schema из модели и валидирует ответ. Если провайдер
    не поддерживает response_format — повторный запрос с явным требованием
    вернуть JSON и ручной разбор.

    Args:
        client: Асинхронный клиент ProxyAPI.
        settings: Настройки бота.
        system_prompt: Системный промпт с воспоминаниями пользователя.
        history: Краткая история диалога (сообщения user/assistant).
        user_input: Новое сообщение пользователя.

    Returns:
        Объект BotReply с тезисами диалога и сообщением для чата.

    Raises:
        ValidationError: Если модель вернула некорректный JSON (fallback).
        Exception: Ошибки транспорта или API (обрабатываются вызывающим кодом).
    """
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_input}]

    try:
        completion = await client.chat.completions.parse(
            model=settings.model,
            messages=messages,
            response_format=BotReply,
            extra_body={"reasoning_effort": "none"},
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
        content = completion.choices[0].message.content or ""
        logger.warning("Провайдер вернул ответ без .parsed, пробую ручной разбор JSON")
        return _parse_reply_json(content)
    except BadRequestError as exc:
        logger.warning("Провайдер не поддерживает response_format (%s), использую ручной JSON", exc)

    fallback_messages = [
        {
            "role": "system",
            "content": system_prompt
            + "\n\nОтветь СТРОГО одним JSON-объектом без пояснений, "
            'с двумя ключами: "theses" — массив строк с тезисами и '
            '"message" — строка с сообщением пользователю.',
        },
        *history,
        {"role": "user", "content": user_input},
    ]
    completion = await client.chat.completions.create(
        model=settings.model,
        messages=fallback_messages,
        extra_body={"reasoning_effort": "none"},
    )
    return _parse_reply_json(completion.choices[0].message.content or "")


# ================================================================
# Работа с Pinecone: память пользователя
# ================================================================

# История per-chat: ограниченная очередь сообщений user/assistant.
# Долгосрочная память живёт отдельно — в Pinecone (см. pine.py).
_HISTORY: dict[int, deque[dict[str, str]]] = {}


def _get_history(chat_id: int, limit: int) -> deque[dict[str, str]]:
    """Возвращает (создавая при необходимости) краткую историю диалога чата.

    Args:
        chat_id: Идентификатор чата.
        limit: Максимум сообщений в истории (старые отбрасываются).

    Returns:
        Очередь сообщений в формате aiogram {"role": ..., "content": ...}.
    """
    return _HISTORY.setdefault(chat_id, deque(maxlen=limit))


def _thesis_id(user_id: int, thesis: str) -> str:
    """Генерирует стабильный идентификатор вектора для тезиса пользователя.

    Стабильный id (тот же текст → тот же id) позволяет upsert перезаписывать
    дубликаты вместо накопления одинаковых воспоминаний.

    Args:
        user_id: Идентификатор пользователя Telegram.
        thesis: Текст тезиса.

    Returns:
        Идентификатор вида ``user_<id>_<md5>``.
    """
    digest = hashlib.md5(thesis.strip().lower().encode("utf-8")).hexdigest()
    return f"user_{user_id}_{digest}"


class PineconeMemory:
    """Асинхронная обёртка над синхронным PineconeClient.

    Все вызовы выполняются в отдельном потоке через asyncio.to_thread,
    чтобы не блокировать event loop aiogram.

    Attributes:
        client: Синхронный клиент из модуля pine.py.
        top_k: Сколько релевантных воспоминаний возвращать при запросе.
    """

    def __init__(self, client: PineconeClient, top_k: int = DEFAULT_RECALL_TOP_K) -> None:
        """Сохраняет клиент и настраивает размер выборки воспоминаний.

        Args:
            client: Синхронный клиент Pinecone.
            top_k: Сколько воспоминаний подгружать перед каждым ответом.
        """
        self.client = client
        self.top_k = top_k

    async def recall(self, user_id: int, user_input: str) -> list[str]:
        """Находит топ-K воспоминаний пользователя, релевантных сообщению.

        Вектор сообщения вычисляется функцией get_embedding (модуль pine.py),
        поиск выполняется с фильтром по user_id, чтобы пользователи
        не видели чужих воспоминаний.

        Args:
            user_id: Идентификатор пользователя Telegram.
            user_input: Текущее сообщение пользователя (для поиска по смыслу).

        Returns:
            Список текстов воспоминаний, отсортированных по релевантности
            (самое релевантное — первым); пустой список, если памяти нет.
        """
        try:
            vector = await asyncio.to_thread(_sync_get_embedding, user_input)
        except Exception as exc:
            logger.error("Не удалось получить эмбеддинг для recall: %s", exc, exc_info=True)
            return []

        def _query() -> list[dict[str, str]]:
            """Ищет релевантные векторы пользователя в Pinecone."""
            return self.client.query(
                vector=vector,
                top_k=self.top_k,
                filter={"user_id": {"$eq": str(user_id)}},
                include_metadata=True,
            )

        try:
            matches = await asyncio.to_thread(_query)
        except PineconeClientError as exc:
            logger.error("Ошибка поиска воспоминаний в Pinecone: %s", exc, exc_info=True)
            return []

        recalled: list[str] = []
        for match in matches:
            text = str((match.get("metadata") or {}).get("text", "")).strip()
            if text:
                recalled.append(text)
        return recalled

    async def remember(self, user_id: int, theses: Sequence[str]) -> int:
        """Сохраняет тезисы диалога в Pinecone с фильтром по user_id.

        Каждый тезис превращается в вектор, id строится из хеша текста —
        повторные тезисы перезаписывают существующие векторы, не раздувая память.

        Args:
            user_id: Идентификатор пользователя Telegram.
            theses: Тексты тезисов, сгенерированные моделью.

        Returns:
            Число реально записанных (upserted) векторов.
        """
        cleaned = [t.strip() for t in theses if t and t.strip()]
        if not cleaned:
            return 0

        vectors: list[tuple[str, list[float], dict[str, str]]] = []
        for thesis in cleaned:
            try:
                vector = await asyncio.to_thread(_sync_get_embedding, thesis)
            except Exception as exc:
                logger.warning("Тезис пропущен (ошибка эмбеддинга): %s", exc)
                continue
            vectors.append(
                (_thesis_id(user_id, thesis), vector, {"user_id": str(user_id), "text": thesis})
            )

        if not vectors:
            return 0

        def _upsert() -> int:
            """Записывает приготовленные векторы в Pinecone одним пакетом."""
            return self.client.upsert(vectors=vectors)

        try:
            count = await asyncio.to_thread(_upsert)
        except PineconeClientError as exc:
            logger.error("Не удалось сохранить тезисы в Pinecone: %s", exc, exc_info=True)
            return 0
        logger.info("user=%s: сохранено воспоминаний %s", user_id, count)
        return count

    async def forget(self, user_id: int) -> None:
        """Удаляет все воспоминания пользователя из Pinecone.

        Args:
            user_id: Идентификатор пользователя Telegram.
        """
        user_filter = {"user_id": {"$eq": str(user_id)}}

        def _delete() -> None:
            """Удаляет все векторы пользователя по metadata-фильтру."""
            self.client.delete_by_filter(user_filter)

        try:
            await asyncio.to_thread(_delete)
        except PineconeClientError as exc:
            logger.error("Ошибка удаления воспоминаний из Pinecone: %s", exc, exc_info=True)
            return
        logger.info("Память пользователя %s удалена из Pinecone", user_id)

    async def count_user_memories(self, user_id: int) -> int:
        """Считает воспоминания конкретного пользователя.

        Serverless-индексы Pinecone не поддерживают describe_index_stats
        с metadata-фильтром, поэтому используется перечисление id:
        все тезисы пользователя имеют префикс ``user_<id>_``.

        Args:
            user_id: Идентификатор пользователя Telegram.

        Returns:
            Число векторов пользователя; 0 при ошибке запроса.
        """
        prefix = f"user_{user_id}_"

        def _count() -> int:
            """Возвращает число векторов пользователя по префиксу id."""
            return len(self.client.list_ids(prefix=prefix))

        try:
            return await asyncio.to_thread(_count)
        except PineconeClientError as exc:
            logger.error("Ошибка подсчёта воспоминаний Pinecone: %s", exc, exc_info=True)
            return 0


# ================================================================
# Хендлеры
# ================================================================


async def handle_start(message: Message) -> None:
    """Отвечает приветствием на команду /start.

    Args:
        message: Сообщение пользователя с командой /start.
    """
    await message.answer(
        "👋 <b>Привет! Я умный ассистент с векторной памятью.</b>\n\n"
        "После каждого ответа я запоминаю ключевые факты о вас в базе Pinecone, "
        "а перед каждым ответом поднимаю только <b>релевантные</b> воспоминания "
        "— те, что по смыслу связаны с вашим вопросом.\n\n"
        "Команды:\n"
        "/clear — стереть мою память о вас;\n"
        "/stats — сколько воспоминаний я храню о вас."
    )


async def handle_clear(message: Message, memory: PineconeMemory) -> None:
    """Полностью сбрасывает память пользователя: историю и воспоминания в Pinecone.

    Args:
        message: Сообщение пользователя с командой /clear.
        memory: Обёртка над Pinecone из Dispatcher.
    """
    chat_id = message.chat.id
    _HISTORY.pop(chat_id, None)
    await memory.forget(chat_id)
    await message.answer(
        "🧹 <b>Память очищена.</b>\n\n"
        "Я забыл краткую историю диалога и все воспоминания из векторной базы. "
        "Начните диалог заново — и я снова начну вас запоминать."
    )


async def handle_stats(message: Message, memory: PineconeMemory) -> None:
    """Сообщает, сколько воспоминаний хранится в базе о пользователе.

    Args:
        message: Сообщение пользователя с командой /stats.
        memory: Обёртка над Pinecone из Dispatcher.
    """
    chat_id = message.chat.id
    count = await memory.count_user_memories(chat_id)
    await message.answer(
        f"📊 <b>Память о вас:</b> {count} воспоминаний в векторной базе.\n\n"
        "Перед каждым ответом я поднимаю до "
        f"{memory.top_k} самых релевантных из них."
    )


async def handle_message(
    message: Message,
    bot: Bot,
    llm_client: AsyncOpenAI,
    settings: Settings,
    memory: PineconeMemory,
) -> None:
    """Обрабатывает текстовое сообщение: вспоминает, отвечает и запоминает.

    Цикл: семантический поиск воспоминаний → сборка системного промпта →
    запрос структурированного ответа → отправка message в чат →
    сохранение новых тезисов в Pinecone.

    Args:
        message: Текстовое сообщение пользователя (фильтр F.text).
        bot: Экземпляр бота (aiogram передаёт автоматически).
        llm_client: Клиент LLM, переданный в Dispatcher при запуске.
        settings: Настройки бота, переданные в Dispatcher при запуске.
        memory: Обёртка над Pinecone, переданная в Dispatcher при запуске.
    """
    user_input = (message.text or "").strip()
    if not user_input:
        await message.answer("Сообщение пустое 🙂 Напишите что-нибудь текстом.")
        return

    chat_id = message.chat.id
    history = _get_history(chat_id, settings.history_limit)

    # Релевантные воспоминания для текущего запроса (семантический поиск)
    try:
        recalled = await memory.recall(chat_id, user_input)
    except Exception as exc:
        logger.error("Ошибка recall для chat_id=%s: %s", chat_id, exc, exc_info=True)
        recalled = []
    if recalled:
        logger.info("chat_id=%s: в промпт поднято %s воспоминаний", chat_id, len(recalled))

    system_prompt = build_system_prompt(recalled)

    # Пока ждём модель, поддерживаем индикатор «печатает…» фоновой задачей
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(bot, chat_id, stop_typing))
    reply: BotReply | None = None
    try:
        reply = await ask_llm(llm_client, settings, system_prompt, list(history), user_input)
    except ValidationError as exc:
        logger.error("Модель вернула некорректный JSON для chat_id=%s: %s", chat_id, exc)
        await message.answer("⚠️ Модель вернула ответ в неожиданном формате. Попробуйте ещё раз.")
    except Exception as exc:
        logger.error("Ошибка LLM для chat_id=%s: %s", chat_id, exc, exc_info=True)
        await message.answer("⚠️ Не удалось получить ответ от модели. Попробуйте позже.")
    finally:
        stop_typing.set()
        await typing_task

    if reply is None:
        return

    # Обновляем краткую историю и сохраняем новые тезисы в векторную память
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply.message})
    saved = await memory.remember(chat_id, reply.theses)
    logger.info(
        "chat_id=%s: тезисов в ответе %s, сохранено %s, в истории %s сообщ.",
        chat_id, len(reply.theses), saved, len(history),
    )

    for chunk in _split_html_text(_md_to_html(reply.message)):
        await _send_safely(message, chunk)


# ================================================================
# Точка входа
# ================================================================


def _setup_logging() -> None:
    """Настраивает логирование: INFO в консоль, без шума от HTTP-клиентов."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "openai", "pinecone"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


async def run_bot(settings: Settings) -> None:
    """Инициализирует память в Pinecone, создаёт бота и запускает long polling.

    При старте проверяется соединение с индексом (он создаётся при
    отсутствии) и запрашивается общая статистика.

    Args:
        settings: Валидные настройки бота.
    """
    pinecone_client = create_pinecone_client(settings)
    stats = pinecone_client.describe_index_stats()
    logger.info(
        "Векторная база подключена: индекс '%s', векторов всего %s",
        settings.index_name, stats.get("total_vector_count"),
    )

    memory = PineconeMemory(pinecone_client, top_k=settings.recall_top_k)

    llm_client = create_llm_client(settings)
    bot = Bot(
        token=settings.tg_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(
        llm_client=llm_client,
        settings=settings,
        memory=memory,
    )
    dp.message.register(handle_start, CommandStart())
    dp.message.register(handle_clear, Command("clear"))
    dp.message.register(handle_stats, Command("stats"))
    dp.message.register(handle_message, F.text)

    logger.info("Бот запущен (модель %s, память: индекс '%s'). Ожидаю сообщения...",
                settings.model, settings.index_name)
    try:
        await dp.start_polling(bot)
    finally:
        await llm_client.close()
        await bot.session.close()


async def main() -> None:
    """Точка входа: настройка логирования, проверка конфигурации, запуск бота.

    Raises:
        SystemExit: Если конфигурация некорректна (сообщение уже записано в лог).
    """
    _setup_logging()
    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    await run_bot(settings)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")