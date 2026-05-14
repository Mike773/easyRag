"""Извлечение сущностей-кандидатов из текста через LLM.

Содержит две операции поверх :class:`LLMClient`:

* :func:`analyze_document` — один вызов на документ; собирает короткий профиль
  (см. :class:`DocumentBrief`), который потом подаётся как контекст в
  extraction. Это нужно, чтобы сама модель определила, какие классы объектов
  в данном документе являются носителями смысла — без зашитых в код доменных
  ограничений.
* :func:`extract_entities` — один вызов на чанк; зовёт tool ``save_entities``
  и приводит ответ к списку :class:`ExtractedEntity`.

Валидация (а не доверие модели на слово) важна потому, что:

* GigaChat иногда возвращает statements как строку вместо массива;
* у обоих провайдеров встречаются пустые ``name`` / дубликаты — их фильтруем,
  чтобы не создавать мусорные строки в ``entity_candidate``;
* brief может прийти с пустым ``summary`` / пустым ``entity_types`` — в этом
  случае возвращаем ``None``, и extraction идёт без подсказок.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from easyrag.ingest.prompts import (
    DOC_BRIEF_SCHEMA,
    DOC_BRIEF_SYSTEM,
    DOC_BRIEF_TOOL_DESCRIPTION,
    DOC_BRIEF_TOOL_NAME,
    ENTITY_EXTRACTION_SCHEMA,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_TOOL_DESCRIPTION,
    ENTITY_EXTRACTION_TOOL_NAME,
    build_brief_user_prompt,
    build_extraction_user_prompt,
)
from easyrag.llm.client import LLMClient, get_llm

_MAX_STATEMENTS = 5


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    descriptor: str
    statements: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DocumentBrief:
    """Профиль документа, построенный LLM по его началу.

    ``entity_types`` — короткие классы (во множественном числе), которые модель
    сама предложила исходя из содержимого. ``summary`` — 1-2 предложения,
    что это за документ.
    """

    summary: str
    entity_types: tuple[str, ...] = field(default_factory=tuple)


async def analyze_document(
    text: str,
    *,
    source_hint: str | None = None,
    llm: LLMClient | None = None,
) -> DocumentBrief | None:
    """Построить :class:`DocumentBrief` по началу документа.

    Возвращает ``None``, если текст пустой или модель вернула неинформативный
    ответ (пусто/без ``summary``/без ``entity_types``). Исключений не кидает —
    вызывающий код просто продолжит ingest без brief'а.
    """
    if not text or not text.strip():
        return None
    client = llm or get_llm()
    try:
        raw = await client.call_json(
            system=DOC_BRIEF_SYSTEM,
            user=build_brief_user_prompt(text, source_hint=source_hint),
            tool_name=DOC_BRIEF_TOOL_NAME,
            tool_description=DOC_BRIEF_TOOL_DESCRIPTION,
            input_schema=DOC_BRIEF_SCHEMA,
        )
    except Exception:
        return None
    return _coerce_brief(raw)


async def extract_entities(
    text: str,
    *,
    source_hint: str | None = None,
    domain_brief: DocumentBrief | None = None,
    llm: LLMClient | None = None,
) -> list[ExtractedEntity]:
    """Извлечь сущности-кандидаты из ``text``.

    ``domain_brief`` — опциональный профиль документа, инжектится в user-prompt
    как подсказка о том, что считать сущностью в этом конкретном документе.

    Возвращает дедуплицированный список (по ``name`` без учёта регистра/пробелов).
    Если модель вернула пустой список или мусор — возвращает пустой список,
    исключение НЕ кидает: дальнейший пайплайн просто запишет чанк без кандидатов.
    """
    if not text or not text.strip():
        return []
    client = llm or get_llm()
    raw = await client.call_json(
        system=ENTITY_EXTRACTION_SYSTEM,
        user=build_extraction_user_prompt(
            text, source_hint=source_hint, domain_brief=domain_brief
        ),
        tool_name=ENTITY_EXTRACTION_TOOL_NAME,
        tool_description=ENTITY_EXTRACTION_TOOL_DESCRIPTION,
        input_schema=ENTITY_EXTRACTION_SCHEMA,
    )
    return _coerce_entities(raw)


def _coerce_brief(raw: dict[str, Any]) -> DocumentBrief | None:
    if not isinstance(raw, dict):
        return None
    summary = _clean_str(raw.get("summary"))
    entity_types = _coerce_str_list(raw.get("entity_types"))
    # Считаем brief полезным, только если есть хотя бы summary ИЛИ
    # непустой entity_types — иначе extraction всё равно ничем не подкрепить.
    if not summary and not entity_types:
        return None
    return DocumentBrief(summary=summary, entity_types=entity_types)


def _coerce_entities(raw: dict[str, Any]) -> list[ExtractedEntity]:
    items = raw.get("entities") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    out: list[ExtractedEntity] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name"))
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        descriptor = _clean_str(item.get("descriptor"))
        statements = _coerce_statements(item.get("statements"))
        out.append(
            ExtractedEntity(
                name=name,
                descriptor=descriptor,
                statements=statements,
            )
        )
    return out


def _coerce_statements(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        # GigaChat иногда отдаёт всё одной строкой — оставляем как один statement.
        cleaned = _clean_str(value)
        return (cleaned,) if cleaned else ()
    if not isinstance(value, list):
        return ()
    cleaned: list[str] = []
    for s in value:
        if not isinstance(s, str):
            continue
        s = _clean_str(s)
        if s:
            cleaned.append(s)
        if len(cleaned) >= _MAX_STATEMENTS:
            break
    return tuple(cleaned)


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for s in value:
        if not isinstance(s, str):
            continue
        s = _clean_str(s)
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)


def _clean_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


__all__ = [
    "DocumentBrief",
    "ExtractedEntity",
    "analyze_document",
    "extract_entities",
]
