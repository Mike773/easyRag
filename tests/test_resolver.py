"""Шаг 4: тесты резолвера — схемы merge, prompt-сборка, coercion.

Полная интеграция (vector-search → upsert_page → embed → provenance) требует
Postgres + pgvector и проверяется руками. Здесь — детерминированная
"вокруг pgvector" часть.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyrag.query.prompts import (
    RESOLVE_JUDGE_SCHEMA,
    RESOLVE_JUDGE_SYSTEM,
    RESOLVE_JUDGE_TOOL_NAME,
    WIKI_MERGE_SCHEMA,
    WIKI_MERGE_SYSTEM,
    WIKI_MERGE_TOOL_NAME,
    build_merge_user_prompt,
    build_resolve_judge_user_prompt,
)
from easyrag.query.resolver import (
    _coerce_aliases,
    _coerce_body,
    _judge_target,
    _pick_title,
)

# --- schemas ---

def test_merge_schema_is_strict_json_schema():
    assert WIKI_MERGE_SCHEMA["type"] == "object"
    assert set(WIKI_MERGE_SCHEMA["required"]) == {"body_md", "aliases"}
    assert WIKI_MERGE_SCHEMA["properties"]["aliases"]["items"]["type"] == "string"
    json.dumps(WIKI_MERGE_SCHEMA, ensure_ascii=False)


def test_merge_tool_name_stable():
    assert WIKI_MERGE_TOOL_NAME == "save_wiki_page"


def test_merge_system_forbids_invention():
    # Регресс-якорь: модель НЕ должна выдумывать факты при merge.
    assert "Не выдумывай" in WIKI_MERGE_SYSTEM
    # И должна использовать [[ссылки]] на другие сущности.
    assert "[[" in WIKI_MERGE_SYSTEM and "]]" in WIKI_MERGE_SYSTEM


# --- merge user prompt ---

def test_build_merge_prompt_creates_new_page_when_body_empty():
    prompt = build_merge_user_prompt(
        title="ООО Ромашка",
        current_body="",
        current_aliases=[],
        new_descriptors=["Поставщик в договоре №7"],
        new_statements=["Заключила договор № 7.", "Юридический адрес: Москва."],
        source_uris=["contracts/7.txt"],
    )
    assert "ООО Ромашка" in prompt
    assert "Текущего тела страницы нет" in prompt
    assert "Заключила договор № 7." in prompt
    assert "Поставщик в договоре №7" in prompt
    assert "contracts/7.txt" in prompt


def test_build_merge_prompt_includes_current_body_and_aliases():
    current = "## Реквизиты\n\nИНН: 7700000000.\n"
    prompt = build_merge_user_prompt(
        title="ООО Ромашка",
        current_body=current,
        current_aliases=["Ромашка"],
        new_descriptors=[],
        new_statements=["Юридический адрес: Москва."],
    )
    assert "<current_body>" in prompt and "</current_body>" in prompt
    assert "ИНН: 7700000000." in prompt
    assert "Текущие алиасы: Ромашка" in prompt


def test_build_merge_prompt_includes_existing_entities_catalog():
    prompt = build_merge_user_prompt(
        title="Машенька",
        current_body="",
        current_aliases=[],
        new_descriptors=[],
        new_statements=["Машенька встретила медведя."],
        existing_entities=[
            ("медведь", ["косолапый"]),
            ("лягушка-квакушка", []),
        ],
    )
    assert "Существующие сущности" in prompt
    assert "- медведь (псевдонимы: косолапый)" in prompt
    assert "- лягушка-квакушка" in prompt


def test_build_merge_prompt_skips_catalog_when_empty():
    # Анти-регрессия: пустой каталог не должен порождать секцию.
    prompt = build_merge_user_prompt(
        title="Машенька",
        current_body="",
        current_aliases=[],
        new_descriptors=[],
        new_statements=["Машенька встретила медведя."],
        existing_entities=[],
    )
    assert "Существующие сущности" not in prompt


def test_build_merge_prompt_handles_empty_statements():
    # Резолвер может вызвать merge для уточнения алиасов без новых утверждений.
    prompt = build_merge_user_prompt(
        title="ООО Ромашка",
        current_body="## Overview\n\nфакт.\n",
        current_aliases=["Ромашка"],
        new_descriptors=[],
        new_statements=[],
    )
    assert "Новых утверждений нет" in prompt


# --- coerce body ---

def test_coerce_body_returns_llm_value_when_non_empty():
    out = _coerce_body(
        {"body_md": "  ## Заголовок\n\nтело\n  "},
        statements_fallback=["x"],
        title="T",
    )
    assert out == "## Заголовок\n\nтело"


def test_coerce_body_falls_back_to_bulleted_statements_when_empty():
    out = _coerce_body(
        {"body_md": ""},
        statements_fallback=["факт один", "факт два"],
        title="T",
    )
    assert "## Overview" in out
    assert "- факт один" in out
    assert "- факт два" in out


def test_coerce_body_returns_empty_when_no_fallback():
    # Если LLM пуст и statements'ов нет — нечего записать; страница останется
    # технически пустой. Это корректно: апстрим (resolver) не зовёт merge без
    # statements'ов и без существующего тела, но защита всё равно нужна.
    assert _coerce_body({}, statements_fallback=[], title="T") == ""


def test_coerce_body_handles_garbage_input():
    assert _coerce_body(None, statements_fallback=[], title="T") == ""  # type: ignore[arg-type]
    assert _coerce_body({"body_md": 5}, statements_fallback=[], title="T") == ""


# --- coerce aliases ---

def test_coerce_aliases_merges_sources_and_dedups():
    raw = {"aliases": ["Ромашка", "ромашка", "Romashka LLC"]}
    out = _coerce_aliases(
        raw,
        current=["Ромашка", "RML"],
        new_names=["ООО Ромашка", "Ромашка"],
        title="ООО Ромашка",
    )
    # title удалён из алиасов
    assert "ООО Ромашка" not in out
    # case-insensitive dedup сохраняет ПЕРВЫЙ встретившийся вариант написания
    lowered = [a.casefold() for a in out]
    assert lowered.count("ромашка") == 1
    assert "RML" in out or "rml" in lowered
    assert "Romashka LLC" in out


def test_coerce_aliases_skips_garbage():
    raw = {"aliases": ["valid", 5, "", "   "]}
    out = _coerce_aliases(raw, current=[], new_names=[], title="T")
    assert out == ["valid"]


def test_coerce_aliases_handles_garbage_input():
    assert _coerce_aliases(None, [], [], "T") == []  # type: ignore[arg-type]
    assert _coerce_aliases({"aliases": "not-a-list"}, [], [], "T") == []
    assert _coerce_aliases({}, ["existing"], ["new"], "T") == ["existing", "new"]


# --- pick title ---

def test_pick_title_takes_longest():
    class _Stub:
        def __init__(self, name: str) -> None:
            self.name = name

    # outcome не используется в _pick_title, так что подменяем заглушкой
    items = [
        (_Stub("Ромашка"), None),
        (_Stub("ООО «Ромашка»"), None),
        (_Stub("ООО Ромашка"), None),
    ]
    assert _pick_title(items) == "ООО «Ромашка»"  # type: ignore[arg-type]


def test_pick_title_fallback_when_empty():
    class _Stub:
        def __init__(self, name: str) -> None:
            self.name = name

    items = [(_Stub(""), None), (_Stub("   "), None)]
    assert _pick_title(items) == "Без названия"  # type: ignore[arg-type]


# --- resolve-judge schema + prompt ---


def test_resolve_judge_schema_is_strict_json_schema():
    assert RESOLVE_JUDGE_SCHEMA["type"] == "object"
    assert set(RESOLVE_JUDGE_SCHEMA["required"]) == {"decision", "reasoning"}
    assert RESOLVE_JUDGE_SCHEMA["properties"]["decision"]["enum"] == [
        "existing", "new", "ambiguous",
    ]
    json.dumps(RESOLVE_JUDGE_SCHEMA, ensure_ascii=False)


def test_resolve_judge_tool_name_stable():
    assert RESOLVE_JUDGE_TOOL_NAME == "decide_entity_target"


def test_resolve_judge_system_forbids_invented_slug():
    # Регресс-якорь: судья не должен галлюцинировать slug.
    assert "Не выдумывай" in RESOLVE_JUDGE_SYSTEM


def test_build_resolve_judge_prompt_includes_candidate_and_options():
    prompt = build_resolve_judge_user_prompt(
        candidate_name="старик",
        candidate_descriptor="главный герой сказки",
        candidate_statements=["тянет репку", "зовёт бабку"],
        options=[
            ("дед", "Дед", ["старик"], "## Описание\n\nГлавный герой..."),
            ("баба", "Баба", [], "## Описание\n\nСупруга деда..."),
        ],
    )
    assert "старик" in prompt
    assert "главный герой сказки" in prompt
    assert "тянет репку" in prompt
    assert "slug=дед" in prompt
    assert "title=Дед" in prompt
    assert "aliases: старик" in prompt
    assert "slug=баба" in prompt
    assert RESOLVE_JUDGE_TOOL_NAME in prompt


def test_build_resolve_judge_prompt_handles_empty_options():
    prompt = build_resolve_judge_user_prompt(
        candidate_name="X",
        candidate_descriptor="",
        candidate_statements=[],
        options=[],
    )
    assert "Существующих страниц-претендентов нет" in prompt


# --- _judge_target ---


class _StubLLM:
    """Минимальный стаб LLMClient для тестов: возвращает заранее заданный dict."""

    def __init__(self, response: dict | Exception) -> None:
        self._response = response
        self.call_count = 0

    async def call_json(self, **_kwargs: object) -> dict:
        self.call_count += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _stub_candidate(name: str = "старик") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        descriptor="главный герой",
        statements=["факт один"],
    )


def _stub_options() -> list[tuple[str, str, list[str], str]]:
    return [
        ("дед", "Дед", ["старик"], "Главный герой сказки."),
        ("баба", "Баба", [], "Супруга деда."),
    ]


@pytest.mark.asyncio
async def test_judge_target_existing_with_valid_slug():
    llm = _StubLLM({"decision": "existing", "slug": "дед", "reasoning": "синонимы"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "existing"
    assert slug == "дед"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_judge_target_existing_with_invalid_slug_falls_back_to_ambiguous():
    # Защита от галлюцинации: модель назвала slug, которого не было в опциях.
    llm = _StubLLM({"decision": "existing", "slug": "неведомая", "reasoning": "x"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "ambiguous"
    assert slug is None


@pytest.mark.asyncio
async def test_judge_target_existing_without_slug_falls_back_to_ambiguous():
    llm = _StubLLM({"decision": "existing", "reasoning": "x"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "ambiguous"
    assert slug is None


@pytest.mark.asyncio
async def test_judge_target_new():
    llm = _StubLLM({"decision": "new", "reasoning": "не та же сущность"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "new"
    assert slug is None


@pytest.mark.asyncio
async def test_judge_target_ambiguous_passthrough():
    llm = _StubLLM({"decision": "ambiguous", "reasoning": "недостаточно данных"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "ambiguous"
    assert slug is None


@pytest.mark.asyncio
async def test_judge_target_llm_error_falls_back_to_ambiguous():
    llm = _StubLLM(RuntimeError("network down"))
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "ambiguous"
    assert slug is None


@pytest.mark.asyncio
async def test_judge_target_garbage_response_falls_back_to_ambiguous():
    llm = _StubLLM({"decision": "yeet", "reasoning": "x"})
    decision, slug = await _judge_target(
        llm=llm, candidate=_stub_candidate(), options=_stub_options(),
    )
    assert decision == "ambiguous"
    assert slug is None
