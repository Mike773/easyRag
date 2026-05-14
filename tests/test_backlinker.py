"""Шаг 5 (back-linking): схемы relink-промпта и pure-helpers backlinker'а.

DB-инегрейшен (полный flow: ingest → backfill_links → wiki_link обновлён) живёт
в e2e-проверке против Postgres и проверяется руками — здесь только
детерминированная «вокруг pgvector» часть, как и для резолвера.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from easyrag.query.prompts import (
    WIKI_RELINK_SCHEMA,
    WIKI_RELINK_SYSTEM,
    WIKI_RELINK_TOOL_DESCRIPTION,
    WIKI_RELINK_TOOL_NAME,
    build_relink_user_prompt,
)
from easyrag.wiki.backlinker import (
    _coerce_aliases,
    _coerce_body,
    _has_link_change,
    _relevant_fresh_for_page,
    _slugs_overlap,
    _strings_for,
)


# --- schema ---

def test_relink_schema_is_strict_json_schema():
    assert WIKI_RELINK_SCHEMA["type"] == "object"
    assert set(WIKI_RELINK_SCHEMA["required"]) == {"body_md", "aliases"}
    assert WIKI_RELINK_SCHEMA["properties"]["aliases"]["items"]["type"] == "string"
    assert WIKI_RELINK_SCHEMA["additionalProperties"] is False
    json.dumps(WIKI_RELINK_SCHEMA, ensure_ascii=False)


def test_relink_tool_name_stable():
    assert WIKI_RELINK_TOOL_NAME == "relink_wiki_page"
    assert "[[" not in WIKI_RELINK_TOOL_NAME
    assert WIKI_RELINK_TOOL_DESCRIPTION  # не пустое


def test_relink_system_forbids_content_changes():
    # Регресс-якорь: relink НЕ должен ни менять формулировки, ни добавлять факты.
    assert "НЕ меняй формулировки" in WIKI_RELINK_SYSTEM
    assert "НЕ добавляй новых фактов" in WIKI_RELINK_SYSTEM
    assert "НЕ выдумывай" in WIKI_RELINK_SYSTEM
    # И запрет самореференции должен остаться — иначе self-loop вернутся.
    assert "Самореференция" in WIKI_RELINK_SYSTEM or "самой этой страницы" in WIKI_RELINK_SYSTEM.lower()


# --- user prompt ---

def test_build_relink_prompt_lists_catalog_entries():
    prompt = build_relink_user_prompt(
        title="Колобок",
        current_body="## Overview\n\nКолобок встретил медведя и лису.",
        current_aliases=[],
        catalog=[
            ("медведь", ["косолапый"]),
            ("лиса", []),
        ],
    )
    assert "Колобок" in prompt
    assert "- медведь (псевдонимы: косолапый)" in prompt
    assert "- лиса" in prompt
    assert "<current_body>" in prompt and "</current_body>" in prompt


def test_build_relink_prompt_handles_empty_catalog():
    prompt = build_relink_user_prompt(
        title="Колобок",
        current_body="тело",
        current_aliases=[],
        catalog=[],
    )
    assert "Каталог пуст" in prompt


def test_build_relink_prompt_handles_empty_body():
    prompt = build_relink_user_prompt(
        title="X",
        current_body="",
        current_aliases=[],
        catalog=[("медведь", [])],
    )
    assert "Текущего тела страницы нет" in prompt


def test_build_relink_prompt_preserves_aliases_instruction():
    prompt = build_relink_user_prompt(
        title="Колобок",
        current_body="тело",
        current_aliases=["колобочек"],
        catalog=[("медведь", [])],
    )
    assert "колобочек" in prompt
    assert "без изменений" in prompt


# --- _slugs_overlap ---

def test_slugs_overlap_detects_prefix_relation():
    # «медведь» как prefix «медведь-косолапый» — пересечение.
    assert _slugs_overlap("медведь", "медведь-косолапый") is True
    assert _slugs_overlap("медведь-косолапый", "медведь") is True


def test_slugs_overlap_false_on_unrelated_slugs():
    assert _slugs_overlap("медведь", "лиса") is False
    assert _slugs_overlap("колобок", "теремок") is False


def test_slugs_overlap_false_on_self_or_empty():
    assert _slugs_overlap("медведь", "медведь") is False
    assert _slugs_overlap("", "медведь") is False
    assert _slugs_overlap("медведь", "") is False


# --- _strings_for ---

def test_strings_for_drops_short_strings():
    out = _strings_for("ИП", ["компания", "X"])
    # «ИП» (2 символа) и «X» (1 символ) — короткие, фильтруются.
    assert out == ["компания"]


def test_strings_for_strips_whitespace_and_empties():
    out = _strings_for("  медведь  ", ["", "  ", "косолапый"])
    assert out == ["медведь", "косолапый"]


def test_strings_for_handles_none():
    assert _strings_for(None, None) == []


# --- _has_link_change ---

def test_has_link_change_detects_added_link():
    old = "## Overview\n\nЛиса встретила Колобка."
    new = "## Overview\n\n[[Лиса]] встретила Колобка."
    assert _has_link_change(old, new) is True


def test_has_link_change_ignores_whitespace_when_links_same():
    old = "## Overview\n\n[[Лиса]] видит."
    new = "## Overview\n\n[[Лиса]] видит.  "
    assert _has_link_change(old, new) is False


def test_has_link_change_empty_to_empty():
    assert _has_link_change("", "") is False


# --- _coerce_body ---

def test_coerce_body_returns_llm_value_when_non_empty():
    out = _coerce_body({"body_md": "  ## H\n\nтело  "}, fallback="старое")
    assert out == "## H\n\nтело"


def test_coerce_body_falls_back_when_llm_empty():
    assert _coerce_body({"body_md": ""}, fallback="старое") == "старое"


def test_coerce_body_falls_back_on_garbage():
    assert _coerce_body(None, fallback="старое") == "старое"
    assert _coerce_body({"body_md": 5}, fallback="старое") == "старое"
    assert _coerce_body({}, fallback="старое") == "старое"


# --- _coerce_aliases ---

def test_coerce_aliases_returns_llm_list_when_valid():
    out = _coerce_aliases(
        {"aliases": ["Один", "Два"]}, fallback=["fallback"]
    )
    assert out == ["Один", "Два"]


def test_coerce_aliases_falls_back_when_llm_empty():
    assert _coerce_aliases({"aliases": []}, fallback=["x"]) == ["x"]
    assert _coerce_aliases(None, fallback=["x", "y"]) == ["x", "y"]


def test_coerce_aliases_filters_garbage_items():
    out = _coerce_aliases(
        {"aliases": ["valid", 5, "", "   "]}, fallback=["fallback"]
    )
    assert out == ["valid"]


# --- _relevant_fresh_for_page ---

@dataclass
class _PageStub:
    body_md: str


def test_relevant_fresh_filters_to_substrings_present_in_body():
    page = _PageStub(body_md="## Overview\n\nКолобок и медведь идут вместе.")
    fresh = {
        "медведь": ["медведь", "косолапый"],
        "лиса": ["лиса"],
        "теремок": ["теремок"],
    }
    out = _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=True
    )
    # «медведь» — substring body, «лиса»/«теремок» — нет.
    assert out == {"медведь"}


def test_relevant_fresh_blind_to_russian_inflection():
    # Фиксируем известное ограничение: substring-prefilter не ловит склонения.
    # «медведя» (винит. падеж) не содержит «медведь» как substring и поэтому
    # будет пропущен. Для языков с богатой морфологией это аргумент в пользу
    # выключения prefilter'а (backlink_prefilter=False).
    page = _PageStub(body_md="## Overview\n\nКолобок встретил медведя.")
    fresh = {"медведь": ["медведь"]}
    out = _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=True
    )
    assert out == set()


def test_relevant_fresh_skips_slugs_already_linked():
    page = _PageStub(
        body_md="## Overview\n\n[[медведь]] и лиса встретились."
    )
    fresh = {"медведь": ["медведь"], "лиса": ["лиса"]}
    out = _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=True
    )
    # «медведь» уже залинкован — пропустить; «лиса» осталась.
    assert out == {"лиса"}


def test_relevant_fresh_empty_when_no_match():
    page = _PageStub(body_md="## Overview\n\nКолобок встретил кота.")
    fresh = {"медведь": ["медведь"]}
    out = _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=True
    )
    assert out == set()


def test_relevant_fresh_returns_empty_when_prefilter_disabled():
    # Контракт: при выключенном prefilter возвращаем пустое множество — это
    # сигнал вызывающему «отдай весь fresh-каталог как есть».
    page = _PageStub(body_md="## Overview\n\nЛиса.")
    fresh = {"лиса": ["лиса"], "медведь": ["медведь"]}
    out = _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=False
    )
    assert out == set()


def test_relevant_fresh_handles_empty_body():
    page = _PageStub(body_md="")
    fresh = {"лиса": ["лиса"]}
    assert _relevant_fresh_for_page(
        page, fresh_strings_by_slug=fresh, prefilter=True
    ) == set()
