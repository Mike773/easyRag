"""Smoke-тесты шага 0: импорты живы, моки возвращают ожидаемые формы."""
import pytest

from easyrag.config import get_settings
from easyrag.llm.client import LLMClient
from easyrag.llm.embeddings import EmbeddingClient


def test_settings_load():
    s = get_settings()
    assert s.embed_dim > 0
    assert s.llm_provider in {"openai", "gigachat"}
    assert s.embed_provider in {"openai", "gigachat"}
    assert "asyncpg" in s.db_dsn


def test_models_import():
    from easyrag.db import models  # noqa: F401

    # Все 9 таблиц должны быть зарегистрированы в metadata.
    table_names = set(models.Base.metadata.tables.keys())
    expected = {
        "source_doc",
        "source_chunk",
        "wiki_page",
        "wiki_section",
        "wiki_link",
        "section_provenance",
        "entity_candidate",
        "abbreviation",
        "query_gap",
    }
    assert expected.issubset(table_names), f"missing: {expected - table_names}"


@pytest.mark.asyncio
async def test_llm_mock_returns_schema_shape():
    llm = LLMClient()
    out = await llm.call_json(
        system="ты ассистент",
        user="привет",
        tool_name="echo",
        tool_description="вернуть структуру",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "items": {"type": "array"},
                "ok": {"type": "boolean"},
            },
            "required": ["answer", "items", "ok"],
        },
    )
    assert set(out.keys()) == {"answer", "items", "ok"}
    assert isinstance(out["answer"], str)
    assert isinstance(out["items"], list)
    assert isinstance(out["ok"], bool)


@pytest.mark.asyncio
async def test_embed_mock_dim_and_determinism():
    e = EmbeddingClient()
    v1 = await e.embed_one("привет мир")
    v2 = await e.embed_one("привет мир")
    v3 = await e.embed_one("другой текст")
    assert len(v1) == get_settings().embed_dim
    assert v1 == v2  # детерминированность
    assert v1 != v3  # разные тексты — разные векторы
    # норма ≈ 1
    s = sum(x * x for x in v1) ** 0.5
    assert abs(s - 1.0) < 1e-6
