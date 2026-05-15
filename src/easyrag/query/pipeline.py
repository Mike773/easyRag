"""Оркестратор query (шаг 4).

:func:`answer_query` принимает вопрос пользователя и:

1. Эмбеддит его (тот же провайдер, что использует ingest).
2. Делает retrieval по wiki — vector top-K + graph expansion
   (см. :mod:`easyrag.query.retrieval`).
3. Если ничего не нашлось, пишет ``query_gap`` без секций и возвращает
   ответ-заглушку. Это даёт enrichment-loop'у (шаг 5) сигнал «вопрос есть,
   данных нет».
4. Зовёт LLM с tool ``save_answer``: модель обязана отвечать только по
   контексту и выдать список цитат ``(slug, anchor)``. Цитаты разрешаются
   обратно в загруженные ``RetrievedSection``; если LLM ссылается на
   несуществующую пару — она отбрасывается.
5. По каждой цитате тянет ``section_provenance → source_chunk → source_doc``
   и возвращает caller'у структуру с ``ChunkProvenance``: uri документа,
   char-offset'ы, id чанка.
6. Пишет ``query_gap`` всегда — но если хотя бы одна цитата прошла, ставит
   ``resolved_at`` + ``resolved_section_ids``. Это превращает таблицу
   `query_gap` в полноценный журнал запросов с маркером успеха.

Транзакцию ведёт caller (через ``session_scope``). Embedder/LLM можно
подменить в тестах через kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from easyrag.config import get_settings
from easyrag.db.models import (
    QueryGap,
    SectionProvenance,
    SourceChunk,
    SourceDoc,
)
from easyrag.llm.client import LLMClient, get_llm
from easyrag.llm.embeddings import EmbeddingClient, get_embeddings
from easyrag.query.prompts import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM,
    ANSWER_TOOL_DESCRIPTION,
    ANSWER_TOOL_NAME,
    build_answer_user_prompt,
)
from easyrag.query.retrieval import RetrievedSection, retrieve_sections

DEFAULT_TOP_K = 8
_NO_DATA_ANSWER = "Нет данных в wiki по этому вопросу."


@dataclass(frozen=True)
class ChunkProvenance:
    chunk_id: UUID
    doc_id: UUID
    uri: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Citation:
    section: RetrievedSection
    chunks: tuple[ChunkProvenance, ...] = ()


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    retrieved: tuple[RetrievedSection, ...] = field(default_factory=tuple)
    gap: bool = False  # True, если ни одной валидной цитаты не получено


async def answer_query(
    session: AsyncSession,
    *,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    graph_expand: bool = True,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
) -> QueryResult:
    """Ответить на вопрос пользователя по wiki."""
    q = (question or "").strip()
    if not q:
        raise ValueError("answer_query: empty question")

    llm_client = llm or get_llm()
    embedder = embeddings or get_embeddings()
    settings = get_settings()

    q_vec = await embedder.embed_one(q)

    retrieved = await retrieve_sections(
        session,
        query_vec=q_vec,
        top_k=top_k,
        graph_expand=graph_expand,
        graph_expand_thresh=settings.graph_expand_thresh,
    )

    if not retrieved:
        await _record_gap(session, q, q_vec, resolved_section_ids=())
        return QueryResult(
            question=q,
            answer=_NO_DATA_ANSWER,
            retrieved=(),
            citations=(),
            gap=True,
        )

    raw = await llm_client.call_json(
        system=ANSWER_SYSTEM,
        user=build_answer_user_prompt(question=q, sections=retrieved),  # type: ignore[arg-type]
        tool_name=ANSWER_TOOL_NAME,
        tool_description=ANSWER_TOOL_DESCRIPTION,
        input_schema=ANSWER_SCHEMA,
    )

    answer = _coerce_answer(raw)
    citations_pairs = _coerce_citations(raw)

    matched_sections = _match_citations(citations_pairs, retrieved)

    citations: list[Citation] = []
    if matched_sections:
        prov_map = await _load_chunk_provenance(
            session, [s.section_id for s in matched_sections]
        )
        for sec in matched_sections:
            citations.append(Citation(section=sec, chunks=prov_map.get(sec.section_id, ())))

    gap = not citations
    if not answer:
        # Модель отказалась отвечать.
        answer = _NO_DATA_ANSWER
        gap = True

    await _record_gap(
        session,
        q,
        q_vec,
        resolved_section_ids=tuple(c.section.section_id for c in citations),
    )

    return QueryResult(
        question=q,
        answer=answer,
        citations=tuple(citations),
        retrieved=tuple(retrieved),
        gap=gap,
    )


# ---------------------------------------------------------------------------
# Парсинг ответа LLM
# ---------------------------------------------------------------------------

def _coerce_answer(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    value = raw.get("answer")
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_citations(raw: Any) -> list[tuple[str, str]]:
    """Достать список (slug, anchor) из ответа LLM, отфильтровать мусор."""
    if not isinstance(raw, dict):
        return []
    items = raw.get("citations")
    if not isinstance(items, list):
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        anchor = item.get("anchor")
        if not isinstance(slug, str) or not isinstance(anchor, str):
            continue
        slug = slug.strip()
        anchor = anchor.strip()
        # Защита от LLM, склеивающих заголовок "slug#anchor" в поле anchor:
        # сам anchor никогда не содержит '#', так что хвост после последнего
        # '#' — это настоящий якорь, а голова — slug (если поле slug пустое).
        if "#" in anchor:
            head, _, tail = anchor.rpartition("#")
            anchor = tail.strip()
            if not slug:
                slug = head.strip()
        if not slug or not anchor:
            continue
        key = (slug, anchor)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _match_citations(
    pairs: list[tuple[str, str]],
    retrieved: list[RetrievedSection],
) -> list[RetrievedSection]:
    """Сопоставить (slug, anchor) из ответа LLM с реально загруженными секциями.

    Защита от галлюцинаций: если модель назвала пару, которой нет в retrieved —
    она просто отбрасывается, в выводе не появится.
    """
    by_key = {(s.slug, s.anchor): s for s in retrieved}
    out: list[RetrievedSection] = []
    seen: set[UUID] = set()
    for slug, anchor in pairs:
        sec = by_key.get((slug, anchor))
        if sec is None:
            continue
        if sec.section_id in seen:
            continue
        seen.add(sec.section_id)
        out.append(sec)
    return out


# ---------------------------------------------------------------------------
# Провенанс и журнал запросов
# ---------------------------------------------------------------------------

async def _load_chunk_provenance(
    session: AsyncSession, section_ids: list[UUID]
) -> dict[UUID, tuple[ChunkProvenance, ...]]:
    """Загрузить ``section_provenance`` для набора секций.

    Возвращает map ``section_id → tuple[ChunkProvenance, ...]``. Если у секции
    нет провенанса (например, страница создана вне ingest-пайплайна) — её ключ
    в map'е отсутствует.
    """
    if not section_ids:
        return {}
    stmt = (
        select(
            SectionProvenance.section_id,
            SourceChunk.id,
            SourceChunk.doc_id,
            SourceDoc.uri,
            SourceChunk.char_start,
            SourceChunk.char_end,
        )
        .join(SourceChunk, SectionProvenance.source_chunk_id == SourceChunk.id)
        .join(SourceDoc, SourceChunk.doc_id == SourceDoc.id)
        .where(SectionProvenance.section_id.in_(section_ids))
    )
    rows = (await session.execute(stmt)).all()
    grouped: dict[UUID, list[ChunkProvenance]] = {}
    for sec_id, chunk_id, doc_id, uri, c_start, c_end in rows:
        grouped.setdefault(sec_id, []).append(
            ChunkProvenance(
                chunk_id=chunk_id,
                doc_id=doc_id,
                uri=uri,
                char_start=int(c_start),
                char_end=int(c_end),
            )
        )
    return {k: tuple(v) for k, v in grouped.items()}


async def _record_gap(
    session: AsyncSession,
    question: str,
    embedding: list[float],
    *,
    resolved_section_ids: tuple[UUID, ...],
) -> None:
    """Записать запрос в ``query_gap``.

    Поле ``resolved_section_ids`` означает «вот эти секции реально дали ответ» —
    запись с пустым массивом и без ``resolved_at`` тогда означает «вопрос задан,
    ответа в wiki нет» (это и есть gap для enrichment-loop'а).
    """
    gap = QueryGap(
        query=question,
        embedding=embedding,
        resolved_section_ids=list(resolved_section_ids),
    )
    if resolved_section_ids:
        gap.resolved_at = datetime.now(timezone.utc)
    session.add(gap)
    await session.flush()


__all__ = [
    "Citation",
    "ChunkProvenance",
    "DEFAULT_TOP_K",
    "QueryResult",
    "answer_query",
]
