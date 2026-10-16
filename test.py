"""
test.py — демонстрация наполнения векторного хранилища Pinecone.

Создаёт эмбеддинги для списка фактов об автомобилях Nissan и Volkswagen «Жук»,
и загружает их в индекс «lesson» размерности 1536.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from pine import PineconeClient

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Функция получения эмбеддинга
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None


def get_embedding(text: str) -> list[float]:
    """Возвращает векторное представление (эмбеддинг) текста.

    Использует OpenAI-совместимый API ProxyAPI и модель
    ``openai/text-embedding-3-small`` (размерность 1536).

    Args:
        text: Исходный текст, для которого нужен эмбеддинг.

    Returns:
        Список чисел-компонент вектора длиной 1536.

    Raises:
        RuntimeError: Если PROXYAPI_API_KEY не задан либо запрос к API
            завершился ошибкой.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("PROXYAPI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PROXYAPI_API_KEY не задан: установите переменную окружения"
            )
        _openai_client = OpenAI(
            base_url=os.getenv("PROXYAPI_API_BASE", "https://api.proxyapi.ru/v1"),
            api_key=api_key,
        )

    try:
        response = _openai_client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=text,
        )
    except Exception as exc:
        raise RuntimeError(f"Ошибка получения эмбеддинга: {exc}") from exc

    return response.data[0].embedding


# ---------------------------------------------------------------------------
# 2. Список демонстрационных фраз
# ---------------------------------------------------------------------------

PHRASES: list[str] = [
    # --- 10 фактов про Nissan ---
    "Компания Nissan была основана в 1933 году, хотя её предшественник Jitsuyo Jidosha существовал с 1919 года.",
    "Первая модель Datsun 12 Phaeton сошла с конвейера в 1933 году — в год основания компании Nissan.",
    "Nissan является одним из основателей альянса Renault-Nissan-Mitsubishi, действующего с 1999 года.",
    "Легендарный Nissan Skyline GT-R, получивший прозвище «Godzilla», впервые появился в 1989 году.",
    "Nissan Leaf стал одним из самых продаваемых полностью электрических автомобилей в истории.",
    "Nissan является пионером в разработке систем безопасности ProPILOT для автономного вождения.",
    "Nissan GT-R R35, представленный в 2007 году, считается одним из самых быстрых серийных автомобилей в своём классе.",
    "Nissan Patrol — легендарный внедорожник, выпускаемый с 1951 года и считающийся одним из старейших серийных внедорожников.",
    "Nissan Micra, первый компактный автомобиль компании, выпускается с 1982 года в нескольких поколениях.",
    "Nissan участвует в гонках Le Mans, а Nissan GT-R LM NISMO стартовал в 2015 году с гибридной силовой установкой.",
    # --- 10 фактов про Volkswagen «Жук» (Beetle/Käfer) ---
    "Прототип Volkswagen Beetle был разработан Фердинандом Порше по заказу правительства Германии в 1934 году.",
    "Первый серийный Volkswagen Beetle сошёл с конвейера в 1938 году, а массовое производство началось после войны.",
    "Volkswagen официально не называл автомобиль «Жук» — в Германии он назывался Käfer, а в США — просто Beetle.",
    "К 1972 году Volkswagen Beetle опередил Ford Model T, став самым продаваемым автомобилем в истории.",
    "Всего было выпущено более 21,5 миллиона автомобилей Volkswagen Beetle — мировой рекорд для одной модели.",
    "Volkswagen Beetle последнего поколения (Typ 1) выпускался до 2003 года на заводе в Мексике под названием «Beetle».",
    "В 1953 году у Volkswagen Beetle появилось знаменитое «овальное» заднее окно, увеличившее обзор на 23 процента.",
    "Volkswagen Beetle в исполнении актёра-робота стал культовым персонажем — «Herbie the Love Bug».",
    "Volkswagen Beetle с его оппозитным 4-цилиндровым двигателем воздушного охлаждения прославился своей надёжностью.",
    "Современный Volkswagen Beetle выпускался с 2011 по 2019 год в кузовах седан, купе и кабриолет.",
]


# ---------------------------------------------------------------------------
# 3. Основной цикл: загрузка фраз в Pinecone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Подключаемся к индексу «lesson» (автоматически создастся при первом запуске)
    client = PineconeClient(index_name="lesson", dimension=1536)

    print(f"Начинаем наполнение индекса «lesson» ({len(PHRASES)} фраз)...\n")

    for i, phrase in enumerate(PHRASES, start=1):
        # Получаем вектор
        vec = get_embedding(phrase)
        print(f"[{i}/{len(PHRASES)}] Фраза: «{phrase[:40]}...» → вектор dim={len(vec)}")

        # Отправляем в Pinecone: id = порядковый номер, metadata = сама фраза
        client.upsert(
            vectors=[(str(i), vec, {"text": phrase})],
        )

    # Выводим итоговую статистику индекса
    print("\n--- Статистика индекса «lesson» ---")
    stats: dict[str, Any] = client.describe_index_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✅ Индекс «lesson» успешно наполнен!")

