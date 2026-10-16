"""
database.py — долговременная память бота на SQLite.

Хранит «тезисы» диалогов каждого пользователя персистентно: таблица создаётся
при первом сообщении пользователя, тезисы добавляются после каждого ответа
бота и подгружаются в системный промпт при следующих запросах. База данных
переживает перезапуски приложения — бот помнит интересы пользователя хоть
через неделю, хоть через год.

Схема:
    users            — каталог всех известных пользователей;
    user_<id>        — таблица конкретного пользователя со своими тезисами.

Все публичные функции асинхронные: sqlite3 работает в отдельном потоке
через asyncio.to_thread, чтобы не блокировать event loop aiogram.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from collections.abc import Sequence

DEFAULT_DB_PATH = "memory.db"  # база по умолчанию — рядом со скриптом

# Имена таблиц пользователей строятся только для неотрицательных целых id;
# этот regex используется как вторая линия защиты перед подстановкой в SQL
_USER_TABLE_RE = re.compile(r"user_\d+")

logger = logging.getLogger("bot.database")

_db_path: str = DEFAULT_DB_PATH


class DatabaseError(RuntimeError):
    """Ошибка работы с базой данных памяти."""


def _user_table_name(user_id: int) -> str:
    """Возвращает безопасное имя таблицы пользователя.

    SQLite не позволяет передать имя таблицы параметром запроса, поэтому
    оно строится только из проверенного целого числа — SQL-инъекция
    через идентификатор пользователя невозможна.

    Args:
        user_id: Идентификатор пользователя Telegram.

    Returns:
        Имя таблицы вида ``user_<id>``.

    Raises:
        DatabaseError: Если user_id не является неотрицательным целым числом.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, (int, str)):
        raise DatabaseError(f"Некорректный идентификатор пользователя: {user_id!r}")
    if isinstance(user_id, str) and not user_id.isdigit():
        raise DatabaseError(f"Идентификатор пользователя должен быть целым числом: {user_id!r}")
    user_id = int(user_id)
    if user_id < 0:
        raise DatabaseError(f"Идентификатор пользователя не может быть отрицательным: {user_id}")
    return f"user_{user_id}"


def _connect() -> sqlite3.Connection:
    """Открывает соединение с базой данных памяти.

    Returns:
        Соединение sqlite3 с фабрикой строк ``sqlite3.Row``.
    """
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn


async def init_db(db_path: str | None = None) -> None:
    """Инициализирует базу: путь, режим WAL, каталог пользователей.

    Args:
        db_path: Путь к файлу базы. Если None — используется текущий путь
            (переменная MEMORY_DB_PATH или ``memory.db``).
    """
    global _db_path
    if db_path:
        _db_path = db_path

    def _init() -> None:
        with _connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id    INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    await asyncio.to_thread(_init)
    logger.info("База памяти инициализирована: %s", os.path.abspath(_db_path))


async def ensure_user_table(user_id: int) -> None:
    """Создаёт таблицу пользователя при первом его обращении.

    Args:
        user_id: Идентификатор пользователя Telegram.
    """
    table = _user_table_name(user_id)

    def _ensure() -> None:
        with _connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    thesis     TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (int(user_id),))

    await asyncio.to_thread(_ensure)


async def save_theses(user_id: int, theses: Sequence[str]) -> int:
    """Сохраняет новые тезисы диалога в таблицу пользователя.

    Дубликаты отбрасываются на уровне ``UNIQUE`` и ``INSERT OR IGNORE`` —
    повторяющиеся тезисы не накапливаются от ответа к ответу.

    Args:
        user_id: Идентификатор пользователя Telegram.
        theses: Тексты тезисов, сгенерированные моделью.

    Returns:
        Число реально сохранённых (новых) тезисов.

    Raises:
        DatabaseError: Если таблица не удалось создать или записать данные.
    """
    cleaned = [thesis.strip() for thesis in theses if thesis and thesis.strip()]
    if not cleaned:
        return 0
    table = _user_table_name(user_id)

    def _save() -> int:
        try:
            with _connect() as conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        thesis     TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (int(user_id),))
                cursor = conn.executemany(
                    f"INSERT OR IGNORE INTO {table} (thesis) VALUES (?)",
                    [(thesis,) for thesis in cleaned],
                )
                return max(cursor.rowcount, 0)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Не удалось сохранить тезисы пользователя {user_id}: {exc}") from exc

    return await asyncio.to_thread(_save)


async def load_theses(user_id: int) -> list[str]:
    """Загружает все сохранённые тезисы пользователя для системного промпта.

    Args:
        user_id: Идентификатор пользователя Telegram.

    Returns:
        Список тезисов в порядке их сохранения; пустой список, если
        у пользователя ещё нет таблицы или тезисов.
    """
    table = _user_table_name(user_id)

    def _load() -> list[str]:
        with _connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return []
            rows = conn.execute(f"SELECT thesis FROM {table} ORDER BY id").fetchall()
            return [row["thesis"] for row in rows]

    return await asyncio.to_thread(_load)


async def clear_memory(user_id: int) -> None:
    """Полностью стирает долговременную память пользователя.

    Удаляет его таблицу с тезисами и запись из каталога пользователей.

    Args:
        user_id: Идентификатор пользователя Telegram.
    """
    table = _user_table_name(user_id)

    def _clear() -> None:
        with _connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("DELETE FROM users WHERE user_id = ?", (int(user_id),))

    await asyncio.to_thread(_clear)
    logger.info("Память пользователя %s очищена", user_id)


async def count_users() -> int:
    """Возвращает число пользователей в каталоге (для лога при старте)."""

    def _count() -> int:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
            return int(row["n"])

    return await asyncio.to_thread(_count)


async def count_theses() -> int:
    """Возвращает суммарное число сохранённых тезисов (для лога при старте)."""

    def _count() -> int:
        with _connect() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'user\\_%' ESCAPE '\\'"
            ).fetchall()
            total = 0
            for row in tables:
                name = row["name"]
                if not _USER_TABLE_RE.fullmatch(name):
                    logger.warning("Пропускаю подозрительную таблицу: %s", name)
                    continue
                total += int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            return total

    return await asyncio.to_thread(_count)
