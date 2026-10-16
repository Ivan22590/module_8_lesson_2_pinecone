"""
bot_without_memory.py — Telegram-бот с AI-ассистентом на базе LLM (без памяти диалога).

Единственный модуль проекта: конфигурация, клиент LLM, форматирование ответа,
хендлеры aiogram и точка входа. Каждый запрос обрабатывается независимо —
история диалога не сохраняется и контекст между сообщениями не передаётся.

Переменные окружения (см. .env.example):
    TG_BOT_API_KEY   — токен Telegram-бота от @BotFather (обязательный);
    PROXYAPI_API_KEY — ключ ProxyAPI (обязательный);
    PROXYAPI_API_BASE — базовый URL ProxyAPI (по умолчанию https://api.proxyapi.ru/v1);
    LLM_MODEL        — имя модели (по умолчанию openai/gpt-5.6-luna);
    LLM_TIMEOUT      — таймаут запроса к LLM в секундах (по умолчанию 60);
    LLM_MAX_RETRIES  — число автоматических повторов запроса (по умолчанию 2).

Запуск: python bot_without_memory.py
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ================================================================
# Конфигурация
# ================================================================

load_dotenv()

DEFAULT_PROXYAPI_BASE = "https://api.proxyapi.ru/v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2

TG_MESSAGE_LIMIT = 4096  # жёсткий лимит длины сообщения Telegram, символов

TYPING_REFRESH_SECONDS = 5.0  # Telegram показывает статус «печатает…» примерно 5 секунд

SYSTEM_PROMPT = """\
Ты — полезный AI-ассистент в Telegram-боте.
Отвечай на языке пользователя (русский — если вопрос на русском).
Будь кратким и информативным, не повторяй вопрос и не перечисляй то, \
что уже написал пользователь.
Если запрос неясен — задай уточняющий вопрос, не пытайся угадать.
Используй простую разметку: **жирный**, *курсив*, `код`, блоки кода через тройные обратные кавычки.\
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
    """

    tg_bot_token: str
    proxyapi_key: str
    proxyapi_base: str
    model: str
    timeout: float
    max_retries: int

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
            for name in ("TG_BOT_API_KEY", "PROXYAPI_API_KEY")
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise SettingsError(
                "Отсутствуют обязательные переменные окружения: "
                f"{', '.join(missing)}. Создайте .env из .env.example и заполните значения."
            )
        return cls(
            tg_bot_token=os.environ["TG_BOT_API_KEY"].strip(),
            proxyapi_key=os.environ["PROXYAPI_API_KEY"].strip(),
            proxyapi_base=os.getenv("PROXYAPI_API_BASE", DEFAULT_PROXYAPI_BASE).strip(),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip(),
            timeout=_env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            max_retries=_env_int("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        )


def create_llm_client(settings: Settings) -> AsyncOpenAI:
    """Создаёт клиент ProxyAPI (OpenAI-совместимый) с параметрами из настроек.

    Таймаут и число повторов задаются здесь, чтобы бот не завис надолго
    при недоступности LLM-провайдера.

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


# ================================================================
# Работа с LLM
# ================================================================


async def ask_llm(client: AsyncOpenAI, settings: Settings, user_input: str) -> str:
    """Возвращает ответ LLM на одно сообщение пользователя (без контекста диалога).

    Args:
        client: Асинхронный клиент ProxyAPI.
        settings: Настройки бота (идентификатор модели).
        user_input: Текст сообщения пользователя.

    Returns:
        Ответ модели без внешних пробелов; если модель вернула пустой текст —
        строка-заглушка с пояснением.

    Raises:
        Exception: Ошибки транспорта или API пробрасываются вызывающему коду,
            который логирует сбой и уведомляет пользователя.
    """
    response = await client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        # Отключаем рассуждение (reasoning) — для чат-бота нужен быстрый прямой ответ
        extra_body={"reasoning_effort": "none"},
    )
    content = response.choices[0].message.content
    return (content or "").strip() or "Модель вернула пустой ответ."


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
# Хендлеры
# ================================================================


async def handle_start(message: Message) -> None:
    """Отвечает приветствием на команду /start.

    Args:
        message: Сообщение пользователя с командой /start.
    """
    await message.answer(
        "👋 <b>Привет! Я AI-ассистент на базе LLM.</b>\n\n"
        "Просто напишите мне что угодно — вопрос, задачу, текст для редактирования. "
        "Отвечу на вашем языке."
    )


async def handle_message(
    message: Message,
    bot: Bot,
    llm_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """Обрабатывает текстовое сообщение: спрашивает модель и отвечает в чат.

    Args:
        message: Текстовое сообщение пользователя (фильтр F.text).
        bot: Экземпляр бота (aiogram передаёт автоматически).
        llm_client: Клиент LLM, переданный в Dispatcher при запуске.
        settings: Настройки бота, переданные в Dispatcher при запуске.
    """
    user_input = (message.text or "").strip()
    if not user_input:
        await message.answer("Сообщение пустое 🙂 Напишите что-нибудь текстом.")
        return

    # Пока ждём модель, поддерживаем индикатор «печатает…» фоновой задачей
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(bot, message.chat.id, stop_typing))
    answer: str | None = None
    try:
        answer = await ask_llm(llm_client, settings, user_input)
    except Exception as exc:
        logger.error("Ошибка LLM для chat_id=%s: %s", message.chat.id, exc, exc_info=True)
        await message.answer("⚠️ Не удалось получить ответ от модели. Попробуйте позже.")
    finally:
        stop_typing.set()
        await typing_task

    if answer is None:
        return

    for chunk in _split_html_text(_md_to_html(answer)):
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
    for noisy_logger in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


async def run_bot(settings: Settings) -> None:
    """Создаёт бота, диспетчер и запускает long polling до остановки.

    LLM-клиент и настройки передаются в диспетчер и попадают в хендлеры
    через механизм инъекции зависимостей aiogram.

    Args:
        settings: Валидные настройки бота.
    """
    llm_client = create_llm_client(settings)
    bot = Bot(
        token=settings.tg_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(llm_client=llm_client, settings=settings)
    dp.message.register(handle_start, CommandStart())
    dp.message.register(handle_message, F.text)

    logger.info("Бот запущен (модель %s). Ожидаю сообщения...", settings.model)
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
