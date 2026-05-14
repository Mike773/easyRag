"""Шаг 4: тесты для query-пайплайна — схемы, prompt-сборка, парсинг ответа.

Без БД: проверяется поведение «вокруг» pgvector-зависимой части. Реальный
retrieval требует поднятого Postgres + pgvector и тестируется руками.
"""
from __future__ import annotations

import json
from uuid import uuid4

from easyrag.query.pipeline import (
    Citation,
    QueryResult,
    _coerce_answer,
    _coerce_citations,
    _match_citations,
)
from easyrag.query.prompts import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM,
    ANSWER_TOOL_NAME,
    build_answer_user_prompt,
)
from easyrag.query.retrieval import RetrievedSection


# --- schemas ---

def test_answer_schema_is_strict_json_schema():
    assert ANSWER_SCHEMA["type"] == "object"
    assert set(ANSWER_SCHEMA["required"]) == {"answer", "citations"}
    cit_item = ANSWER_SCHEMA["properties"]["citations"]["items"]
    assert cit_item["type"] == "object"
    assert set(cit_item["required"]) == {"slug", "anchor"}
    # serializable end-to-end
    json.dumps(ANSWER_SCHEMA, ensure_ascii=False)


def test_answer_system_mentions_no_data_path():
    # Защита от регресса: модель должна знать, что отвечать «нет данных» — ок.
    assert "Нет данных" in ANSWER_SYSTEM or "нет данных" in ANSWER_SYSTEM
    # И что цитаты — пары slug+anchor.
    assert "slug" in ANSWER_SYSTEM and "anchor" in ANSWER_SYSTEM


# --- user prompt ---

def _mk_section(slug: str, anchor: str, body: str = "контент секции") -> RetrievedSection:
    return RetrievedSection(
        section_id=uuid4(),
        page_id=uuid4(),
        slug=slug,
        anchor=anchor,
        page_title=f"Page {slug}",
        section_title=f"Section {anchor}",
        body_md=body,
        similarity=0.9,
        source="vector",
    )


def test_build_answer_prompt_includes_sections_and_question():
    sections = [_mk_section("dogovor", "storony", "Тело про стороны.")]
    prompt = build_answer_user_prompt(
        question="Кто стороны договора?",
        sections=sections,  # type: ignore[arg-type]
    )
    assert "Кто стороны договора?" in prompt
    assert "[dogovor#storony]" in prompt
    assert "Тело про стороны." in prompt
    # Структурные теги-обёртки сохранились.
    assert "<sections>" in prompt and "</sections>" in prompt


def test_build_answer_prompt_without_sections_marks_empty():
    prompt = build_answer_user_prompt(question="что-то", sections=[])
    assert "вопрос" not in prompt.lower() or "что-то" in prompt
    assert "Доступных секций" in prompt
    assert "<sections>" not in prompt  # для пустого случая не плодим пустые теги


# --- ответ LLM: coercion ---

def test_coerce_answer_extracts_string():
    assert _coerce_answer({"answer": "  готовый ответ  "}) == "готовый ответ"


def test_coerce_answer_handles_garbage():
    assert _coerce_answer(None) == ""
    assert _coerce_answer({}) == ""
    assert _coerce_answer({"answer": 42}) == ""
    assert _coerce_answer({"answer": ["a", "b"]}) == ""


def test_coerce_citations_dedup_and_filter():
    raw = {
        "citations": [
            {"slug": "dogovor", "anchor": "storony"},
            {"slug": "dogovor", "anchor": "storony"},  # дубль
            {"slug": " ", "anchor": "x"},  # пустой slug → отбрасываем
            {"slug": "kontragent", "anchor": ""},  # пустой anchor → отбрасываем
            "not-a-dict",
            {"slug": 5, "anchor": "x"},  # неверный тип
            {"slug": "k", "anchor": "y"},
        ]
    }
    out = _coerce_citations(raw)
    assert out == [("dogovor", "storony"), ("k", "y")]


def test_coerce_citations_handles_garbage():
    assert _coerce_citations(None) == []
    assert _coerce_citations({}) == []
    assert _coerce_citations({"citations": "string"}) == []


# --- цитаты ↔ retrieved секции ---

def test_match_citations_drops_hallucinations():
    sec_a = _mk_section("dogovor", "storony")
    sec_b = _mk_section("kontragent", "overview")
    retrieved = [sec_a, sec_b]
    pairs = [
        ("dogovor", "storony"),
        ("dogovor", "NONEXISTENT"),  # галлюцинация — нет такого anchor'а
        ("kontragent", "overview"),
        ("ghost-page", "overview"),  # галлюцинация — нет такой страницы
    ]
    matched = _match_citations(pairs, retrieved)
    assert [m.slug for m in matched] == ["dogovor", "kontragent"]


def test_match_citations_preserves_order_and_dedups_by_section_id():
    sec_a = _mk_section("dogovor", "storony")
    retrieved = [sec_a]
    pairs = [("dogovor", "storony"), ("dogovor", "storony")]
    matched = _match_citations(pairs, retrieved)
    assert len(matched) == 1


# --- dataclasses sanity ---

def test_query_result_defaults():
    r = QueryResult(question="?", answer="ok")
    assert r.citations == ()
    assert r.retrieved == ()
    assert r.gap is False


def test_citation_holds_section():
    sec = _mk_section("x", "y")
    cit = Citation(section=sec)
    assert cit.chunks == ()
    assert cit.section.slug == "x"


# --- prompt sanity ---

def test_answer_tool_name_stable():
    # Имя tool'а часть контракта с LLMClient.call_json — менять без обновления тестов нельзя.
    assert ANSWER_TOOL_NAME == "save_answer"
