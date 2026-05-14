"""Smoke-тесты шага 0: импорты живы, моки возвращают ожидаемые формы."""
import pytest

from easyrag.config import get_settings
from easyrag.llm.client import LLMClient
from easyrag.llm.embeddings import EmbeddingClient
from easyrag.wiki.backlinker import BackfillResult, backfill_links


def test_settings_load():
    s = get_settings()
    assert s.embed_dim > 0
    assert s.llm_provider in {"openai", "gigachat"}
    assert s.embed_provider in {"openai", "gigachat"}
    assert "asyncpg" in s.db_dsn


def test_backlinker_public_api():
    # Регресс-якорь: backfill_links и BackfillResult импортируются и совместимы
    # с конфиг-флагом — отключённый backlink_enabled даёт пустой результат
    # без обращения к БД/LLM.
    import asyncio

    assert callable(backfill_links)
    assert BackfillResult().relinked == ()
    assert BackfillResult().skipped == ()
    assert BackfillResult().relinked_count == 0

    # При выключенном backlink_enabled функция возвращает пустой результат
    # сразу — даже без session: до запроса к БД не доходим.
    import easyrag.wiki.backlinker as bl
    from easyrag.config import get_settings

    settings = get_settings()
    original = settings.backlink_enabled
    settings.backlink_enabled = False
    try:
        out = asyncio.run(
            bl.backfill_links(session=None, exclude_slugs=("a", "b"))  # type: ignore[arg-type]
        )
        assert out == BackfillResult()
    finally:
        settings.backlink_enabled = original


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
                "items": {"type": "array", "items": {"type": "string"}},
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

    def _cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # Тот же текст — близкие векторы (моки точно равны, OpenAI ≈ равны).
    assert _cos(v1, v2) > 0.999
    # Разные тексты — заметно расходятся.
    assert _cos(v1, v3) < 0.95
    # норма ≈ 1
    s = sum(x * x for x in v1) ** 0.5
    assert abs(s - 1.0) < 1e-2
