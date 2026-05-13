"""Шаг 2: чанкер и extractor (без БД, моки LLM/эмбеддингов)."""
from __future__ import annotations

import json

import pytest

from easyrag.ingest import chunk_text, extract_entities
from easyrag.ingest.prompts import (
    ENTITY_EXTRACTION_SCHEMA,
    build_extraction_user_prompt,
)


# --- chunker ---

def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_keeps_paragraph_boundaries():
    body = "Параграф один.\n\nПараграф два.\n\nПараграф три."
    chunks = chunk_text(body, target_size=1000, max_size=2000, overlap=50)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.ord == 0
    assert c.char_start == 0
    assert c.char_end == len(body)
    assert c.text == body


def test_chunk_offsets_round_trip():
    body = "Первый абзац.\n\nВторой абзац подлиннее.\n\nТретий."
    chunks = chunk_text(body, target_size=20, max_size=40, overlap=5)
    for c in chunks:
        assert body[c.char_start:c.char_end] == c.text
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_chunk_splits_long_paragraph_with_overlap():
    body = "A" * 5000
    chunks = chunk_text(body, target_size=1000, max_size=1500, overlap=100)
    # Должно получиться несколько чанков, последний кончается на len(body).
    assert len(chunks) > 1
    assert chunks[-1].char_end == len(body)
    # Соседние чанки перекрываются.
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.char_start < prev.char_end


def test_chunk_groups_short_paragraphs():
    body = "\n\n".join(f"абзац {i} с небольшим текстом." for i in range(20))
    chunks = chunk_text(body, target_size=200, max_size=400, overlap=20)
    # Несколько параграфов сольются в один чанк, но в сумме покроют весь текст.
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(body)
    # Никакой chunk.text не пустой
    assert all(c.text.strip() for c in chunks)


def test_chunk_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("hi", target_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("hi", target_size=100, max_size=50)


# --- prompts ---

def test_extraction_schema_is_strict_json_schema():
    # Должен быть object с required "entities".
    assert ENTITY_EXTRACTION_SCHEMA["type"] == "object"
    assert ENTITY_EXTRACTION_SCHEMA["required"] == ["entities"]
    item = ENTITY_EXTRACTION_SCHEMA["properties"]["entities"]["items"]
    assert set(item["required"]) == {"name", "descriptor", "statements"}
    assert item["properties"]["statements"]["type"] == "array"
    # JSON-сериализуема — гарантирует, что схема улетит в LangChain без сюрпризов.
    json.dumps(ENTITY_EXTRACTION_SCHEMA, ensure_ascii=False)


def test_extraction_user_prompt_wraps_body():
    prompt = build_extraction_user_prompt("Текст документа.", source_hint="readme.txt")
    assert "readme.txt" in prompt
    assert "<fragment>" in prompt and "</fragment>" in prompt
    assert "Текст документа." in prompt


def test_extraction_user_prompt_without_hint():
    prompt = build_extraction_user_prompt("Просто текст.")
    assert "Источник" not in prompt
    assert "Просто текст." in prompt


# --- extractor (mock LLM) ---

class _StubLLM:
    """Не дергает сеть — отдаёт заранее заданный JSON по схеме."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def call_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


@pytest.mark.asyncio
async def test_extract_entities_passes_prompts_to_llm():
    stub = _StubLLM({"entities": []})
    out = await extract_entities(
        "Договор № 7.", source_hint="contracts/7.txt", llm=stub  # type: ignore[arg-type]
    )
    assert out == []
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["tool_name"] == "save_entities"
    assert "contracts/7.txt" in call["user"]
    assert "Договор № 7." in call["user"]
    assert call["input_schema"]["required"] == ["entities"]


@pytest.mark.asyncio
async def test_extract_entities_skips_empty_text():
    stub = _StubLLM({"entities": []})
    assert await extract_entities("", llm=stub) == []  # type: ignore[arg-type]
    assert await extract_entities("   \n", llm=stub) == []  # type: ignore[arg-type]
    # пустой текст не должен дергать LLM
    assert stub.calls == []


@pytest.mark.asyncio
async def test_extract_entities_coerces_messy_payload(monkeypatch):
    # Подменяем LLMClient.call_json, чтобы проверить нормализацию ответа.
    payload = {
        "entities": [
            {
                "name": "  ООО Ромашка  ",
                "descriptor": " Поставщик. ",
                "statements": [
                    "Заключила договор № 7.",
                    "",  # пустое — должно отсеяться
                    "Юридический адрес: Москва.",
                ],
            },
            {
                # дубль по имени (с другим регистром/пробелами) — должен схлопнуться
                "name": "ооо ромашка",
                "descriptor": "Дубль",
                "statements": "одной строкой",
            },
            {
                # пустое имя — выбрасываем целиком
                "name": "",
                "descriptor": "anything",
                "statements": [],
            },
            "not-a-dict",  # мусор — игнор
        ]
    }

    class _Fake:
        async def call_json(self, **_kwargs):
            return payload

    out = await extract_entities("любой текст", llm=_Fake())  # type: ignore[arg-type]
    assert len(out) == 1
    e = out[0]
    assert e.name == "ООО Ромашка"
    assert e.descriptor == "Поставщик."
    assert e.statements == ("Заключила договор № 7.", "Юридический адрес: Москва.")
