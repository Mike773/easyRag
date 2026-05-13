"""Эмбеддинги поверх LangChain.

Поддерживаются два провайдера, выбираемых через ``settings.embed_provider``:
* ``openai``   — ``langchain_openai.OpenAIEmbeddings`` (любой OpenAI-совместимый endpoint).
* ``gigachat`` — ``langchain_gigachat.GigaChatEmbeddings``.

Размерность фиксируется через ``settings.embed_dim``. В реальном режиме мы доверяем
ответу провайдера; в mock-режиме генерируем детерминированный нормированный вектор
заявленной размерности — для офлайн-тестов.
"""
import asyncio
import hashlib
import math
from typing import Any

from easyrag.config import Provider, get_settings


class EmbeddingClient:
    def __init__(self) -> None:
        s = get_settings()
        self._mock = s.embed_mock
        self._provider: Provider = s.embed_provider
        self._model = s.embed_model
        self._dim = s.embed_dim
        self._embedder: Any = None if self._mock else _build_embedder(s)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed_many([text])
        return out[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._mock:
            return [_deterministic_vector(t, self._dim) for t in texts]

        if self._embedder is None:
            raise RuntimeError("EmbeddingClient is in mock mode; real client not initialised")

        # LangChain Embeddings: предпочитаем aembed_documents, иначе fallback в thread.
        aembed = getattr(self._embedder, "aembed_documents", None)
        if aembed is not None:
            return await aembed(texts)
        return await asyncio.to_thread(self._embedder.embed_documents, texts)

    async def aclose(self) -> None:
        # У LangChain-эмбеддингов нет общего close(); httpx-клиенты внутри
        # ChatOpenAI/GigaChat закрываются на финализаторе.
        return None

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _build_embedder(settings: Any) -> Any:
    if settings.embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {"model": settings.embed_model}
        if settings.embed_api_key:
            kwargs["api_key"] = settings.embed_api_key
        if settings.embed_base_url:
            kwargs["base_url"] = settings.embed_base_url
        # Если задана нестандартная размерность — попросим её у API
        # (поддерживается моделями text-embedding-3-*).
        if settings.embed_dim:
            kwargs["dimensions"] = settings.embed_dim
        return OpenAIEmbeddings(**kwargs)

    if settings.embed_provider == "gigachat":
        from langchain_gigachat import GigaChatEmbeddings

        kwargs = {
            "model": settings.embed_model,
            "credentials": settings.gigachat_credentials,
            "scope": settings.gigachat_scope,
            "verify_ssl_certs": settings.gigachat_verify_ssl,
        }
        return GigaChatEmbeddings(**kwargs)

    raise ValueError(f"Unknown embed_provider: {settings.embed_provider!r}")


def _deterministic_vector(text: str, dim: int) -> list[float]:
    """Псевдослучайный, но воспроизводимый вектор — для офлайн-тестов.

    Из sha256(text) разворачиваем поток float'ов и нормализуем до unit-сферы.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw: list[float] = []
    while len(raw) < dim:
        seed = hashlib.sha256(seed).digest()
        for i in range(0, len(seed), 4):
            if len(raw) >= dim:
                break
            chunk = int.from_bytes(seed[i : i + 4], "big", signed=False)
            raw.append((chunk / 0xFFFFFFFF) * 2.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


_default_embeddings: EmbeddingClient | None = None


def get_embeddings() -> EmbeddingClient:
    global _default_embeddings
    if _default_embeddings is None:
        _default_embeddings = EmbeddingClient()
    return _default_embeddings


def set_embeddings_for_tests(client: EmbeddingClient) -> None:
    global _default_embeddings
    _default_embeddings = client


def reset_embeddings() -> None:
    global _default_embeddings
    _default_embeddings = None


__all__ = [
    "EmbeddingClient",
    "get_embeddings",
    "set_embeddings_for_tests",
    "reset_embeddings",
]
