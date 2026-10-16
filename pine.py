"""
pine.py — клиент для векторной базы данных Pinecone.

Предоставляет класс PineconeClient для подключения к индексу, записи,
чтения, поиска и удаления векторов. Автоматически создаёт индекс при
отсутствии (serverless-спецификация).

Пример использования:
    from pine import PineconeClient

    client = PineconeClient(index_name="my-index", dimension=1536)
    client.upsert([("id1", [0.1, 0.2, ...], {"text": "привет"})])
    results = client.query([0.1, 0.2, ...], top_k=5)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, PineconeError, ServerlessSpec

logger = logging.getLogger("bot.pinecone")

DEFAULT_UPSERT_BATCH_SIZE = 100

EMBEDDING_MODEL = "openai/text-embedding-3-small"  # 1536-мерные эмбеддинги
EMBEDDING_DIMENSION = 1536

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
        PineconeClientError: Если PROXYAPI_API_KEY не задан либо запрос
            к API эмбеддингов завершился ошибкой.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("PROXYAPI_API_KEY")
        if not api_key:
            raise PineconeClientError(
                "PROXYAPI_API_KEY не задан: установите переменную окружения"
            )
        _openai_client = OpenAI(
            base_url=os.getenv("PROXYAPI_API_BASE", "https://api.proxyapi.ru/v1"),
            api_key=api_key,
        )

    try:
        response = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception as exc:
        raise PineconeClientError(f"Не удалось получить эмбеддинг текста: {exc}") from exc

    return response.data[0].embedding


class PineconeClientError(RuntimeError):
    """Ошибка работы с векторной базой Pinecone."""


class PineconeClient:
    """Клиент для работы с векторным индексом Pinecone.

    Инкапсулирует подключение, создание индекса (при необходимости),
    операции upsert / fetch / query / delete / describe_index_stats.

    Attributes:
        index_name: Имя индекса Pinecone.
        dimension: Размерность векторов в индексе.
        namespace: Пространство имён по умолчанию для всех операций.
    """

    def __init__(
        self,
        index_name: str,
        dimension: int,
        *,
        api_key: str | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
        namespace: str = "",
    ) -> None:
        """Инициализирует клиент и подключается к индексу.

        Если индекс с указанным именем отсутствует, он автоматически
        создаётся с serverless-спецификацией.

        Args:
            index_name: Имя индекса Pinecone.
            dimension: Размерность векторов (должна совпадать с размерностью
                эмбеддингов, например 1536 для text-embedding-3-small).
            api_key: API-ключ Pinecone. Если None, берётся из переменной
                окружения ``PINECONE_API_KEY``.
            cloud: Облачный провайдер для создания индекса (``aws``,
                ``gcp``, ``azure``). Используется только при создании.
            region: Регион облака для создания индекса.
            metric: Метрика расстояния: ``cosine``, ``euclidean`` или
                ``dotproduct``.
            namespace: Пространство имён по умолчанию для операций.

        Raises:
            PineconeClientError: Если api_key не передан и отсутствует в
                окружении, либо не удалось подключиться/создать индекс.
        """
        resolved_key = api_key or os.getenv("PINECONE_API_KEY")
        if not resolved_key:
            raise PineconeClientError(
                "PINECONE_API_KEY не задан: передайте api_key или установите переменную окружения"
            )

        self.index_name = index_name
        self.dimension = dimension
        self.namespace = namespace

        try:
            self._pc = Pinecone(api_key=resolved_key)
        except PineconeError as exc:
            raise PineconeClientError(f"Не удалось инициализировать клиент Pinecone: {exc}") from exc

        self._index = self._ensure_index(
            index_name=index_name,
            dimension=dimension,
            cloud=cloud,
            region=region,
            metric=metric,
        )
        logger.info("Подключено к индексу Pinecone: %s", index_name)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _ensure_index(
        self,
        index_name: str,
        dimension: int,
        cloud: str,
        region: str,
        metric: str,
    ) -> Any:
        """Возвращает объект индекса, создав его при отсутствии.

        Args:
            index_name: Имя индекса.
            dimension: Размерность векторов.
            cloud: Облачный провайдер.
            region: Регион облака.
            metric: Метрика расстояния.

        Returns:
            Объект ``pinecone.Index``, готовый к операциям.

        Raises:
            PineconeClientError: При ошибке создания или получения индекса.
        """
        try:
            if not self._pc.has_index(index_name):
                logger.info("Индекс '%s' не найден — создаём serverless...", index_name)
                self._pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                logger.info("Индекс '%s' создан (dimension=%d, metric=%s)", index_name, dimension, metric)

            return self._pc.Index(index_name)
        except PineconeError as exc:
            raise PineconeClientError(
                f"Не удалось получить/создать индекс '{index_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def upsert(
        self,
        vectors: Sequence[tuple[str, Sequence[float]] | tuple[str, Sequence[float], Mapping[str, Any]]],
        *,
        namespace: str | None = None,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> int:
        """Записывает (или обновляет) векторы в индекс пакетами.

        Args:
            vectors: Последовательность кортежей ``(id, values)`` или
                ``(id, values, metadata)``, где:

                - ``id`` — уникальный идентификатор вектора (str);
                - ``values`` — компоненты вектора (Sequence[float]);
                - ``metadata`` — необязательный словарь метаданных.

            namespace: Пространство имён. Если None — используется
                ``self.namespace``.
            batch_size: Размер пакета для отправки (по умолчанию 100).

        Returns:
            Общее количество записанных векторов.

        Raises:
            PineconeClientError: При ошибке записи.
        """
        ns = namespace if namespace is not None else self.namespace
        total_count = 0

        for start in range(0, len(vectors), batch_size):
            batch = list(vectors[start : start + batch_size])
            try:
                response = self._index.upsert(vectors=batch, namespace=ns)
                total_count += response.upserted_count
            except PineconeError as exc:
                raise PineconeClientError(
                    f"Ошибка upsert (пакет {start // batch_size + 1}): {exc}"
                ) from exc

        logger.info("Upsert: %d векторов в namespace='%s'", total_count, ns)
        return total_count

    def fetch(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Читает векторы по их идентификаторам.

        Args:
            ids: Список идентификаторов векторов для получения.
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.

        Returns:
            Словарь ``{id: {"values": [...], "metadata": {...}}}`` только для
            найденных векторов.

        Raises:
            PineconeClientError: При ошибке чтения.
        """
        ns = namespace if namespace is not None else self.namespace
        try:
            response = self._index.fetch(ids=list(ids), namespace=ns)
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка fetch (ids={list(ids)}): {exc}") from exc

        result: dict[str, dict[str, Any]] = {}
        for vec_id, vec in (response.vectors or {}).items():
            result[vec_id] = {
                "values": vec.values if vec.values else [],
                "metadata": dict(vec.metadata) if vec.metadata else {},
            }
        return result

    def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """Выполняет поиск ближайших соседей по вектору запроса.

        Args:
            vector: Запрос-вектор (должен совпадать по размерности с индексом).
            top_k: Количество возвращаемых результатов (по умолчанию 10).
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.
            filter: Фильтр по метаданным в формате Pinecone, например
                ``{"genre": "drama", "year": {"$gte": 2020}}``.
            include_metadata: Включать ли метаданные в результаты
                (по умолчанию True).
            include_values: Включать ли компоненты векторов в результаты
                (по умолчанию False, чтобы уменьшить объём ответа).

        Returns:
            Список словарей ``{"id": ..., "score": ..., "metadata": ...,
            "values": ...}``, отсортированных по убыванию score.

        Raises:
            PineconeClientError: При ошибке поиска.
        """
        ns = namespace if namespace is not None else self.namespace
        try:
            response = self._index.query(
                vector=list(vector),
                top_k=top_k,
                namespace=ns,
                filter=dict(filter) if filter else None,
                include_metadata=include_metadata,
                include_values=include_values,
            )
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка query (top_k={top_k}): {exc}") from exc

        results: list[dict[str, Any]] = []
        for match in (response.matches or []):
            entry: dict[str, Any] = {
                "id": match.id,
                "score": match.score,
            }
            if include_metadata and match.metadata:
                entry["metadata"] = dict(match.metadata)
            if include_values and match.values:
                entry["values"] = match.values
            results.append(entry)
        return results

    def delete_by_ids(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> None:
        """Удаляет векторы по списку идентификаторов.

        Args:
            ids: Идентификаторы векторов для удаления.
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.

        Raises:
            PineconeClientError: При ошибке удаления.
        """
        ns = namespace if namespace is not None else self.namespace
        try:
            self._index.delete(ids=list(ids), namespace=ns)
            logger.info("Удалено %d векторов (ids) из namespace='%s'", len(ids), ns)
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка delete_by_ids: {exc}") from exc

    def delete_by_filter(
        self,
        filter: Mapping[str, Any],
        *,
        namespace: str | None = None,
    ) -> None:
        """Удаляет векторы, удовлетворяющие фильтру по метаданным.

        Args:
            filter: Фильтр в формате Pinecone, например
                ``{"category": "outdated"}`` или
                ``{"year": {"$lt": 2020}}``.
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.

        Raises:
            PineconeClientError: При ошибке удаления.
        """
        ns = namespace if namespace is not None else self.namespace
        try:
            self._index.delete(filter=dict(filter), namespace=ns)
            logger.info("Удалено по фильтру %s из namespace='%s'", filter, ns)
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка delete_by_filter ({filter}): {exc}") from exc

    def delete_all(self, *, namespace: str | None = None) -> None:
        """Удаляет **все** векторы из указанного пространства имён.

        Args:
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.

        Raises:
            PineconeClientError: При ошибке удаления.
        """
        ns = namespace if namespace is not None else self.namespace
        try:
            self._index.delete(delete_all=True, namespace=ns)
            logger.info("Все векторы удалены из namespace='%s'", ns)
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка delete_all (namespace='{ns}'): {exc}") from exc

    def list_ids(self, prefix: str | None = None, *, namespace: str | None = None) -> list[str]:
        """Перечисляет идентификаторы векторов в индексе (с префиксом).

        Serverless-индексы не поддерживают describe_index_stats с фильтром
        по metadata, поэтому для подсчёта векторов пользователя используется
        перечисление id по префиксу.

        Args:
            prefix: Необязательный префикс идентификаторов, например
                ``"user_42_"``.
            namespace: Пространство имён. Если None — используется
                ``self.namespace``.

        Returns:
            Список идентификаторов векторов, удовлетворяющих префиксу.

Raises:
            PineconeClientError: При ошибке запроса.
        """
        ns = namespace if namespace is not None else self.namespace
        ids: list[str] = []
        try:
            pages = self._index.list(prefix=prefix, namespace=ns)
            for page in pages:
                vectors = getattr(page, "vectors", None) or []
                ids.extend(v.id for v in vectors if getattr(v, "id", None))
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка list_ids (prefix={prefix!r}): {exc}") from exc
        return ids

    def describe_index_stats(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Возвращает статистику индекса: число векторов, пространства имён и т.д.

        Args:
            filter: Необязательный фильтр по метаданным — статистика только
                для векторов, удовлетворяющих фильтру.

        Returns:
            Словарь вида::

                {
                    "total_vector_count": int,
                    "dimension": int | None,
                    "metric": str | None,
                    "namespaces": {"<ns>": {"vector_count": int}, ...},
                }

        Raises:
            PineconeClientError: При ошибке запроса статистики.
        """
        try:
            stats = self._index.describe_index_stats(
                filter=dict(filter) if filter else None,
            )
        except PineconeError as exc:
            raise PineconeClientError(f"Ошибка describe_index_stats: {exc}") from exc

        namespaces: dict[str, dict[str, int]] = {}
        for ns_name, summary in (stats.namespaces or {}).items():
            namespaces[ns_name] = {"vector_count": summary.vector_count}

        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "metric": stats.metric,
            "namespaces": namespaces,
        }
