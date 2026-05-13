"""HTTP-клиент эмбеддингов (OpenAI-совместимый формат /v1/embeddings).

Размерность фиксируется через config.embed_dim (по умолчанию 2560 — Qwen3-Embedding-4B).
"""
import hashlib
import math
from typing import Any

import httpx

from easyrag.config import get_settings


class EmbeddingClient:
    def __init__(self) -> None:
        s = get_settings()
        self._mock = s.embed_mock
        self._url = s.embed_url
        self._model = s.embed_model
        self._dim = s.embed_dim
        headers = {"Content-Type": "application/json"}
        if s.embed_api_key:
            headers["Authorization"] = f"Bearer {s.embed_api_key}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=headers)

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

        payload: dict[str, Any] = {"model": self._model, "input": texts}
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-совместимый ответ: {"data": [{"embedding": [...]}, ...]}
        return [item["embedding"] for item in data["data"]]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


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
