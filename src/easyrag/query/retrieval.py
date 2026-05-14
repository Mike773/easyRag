"""Retrieval поверх ``wiki_section`` — vector + graph expansion.

Контракт публичной функции :func:`retrieve_sections`:

1. Берёт уже посчитанный вектор запроса (эмбеддинг делает caller — он же знает,
   из какого провайдера он пришёл).
2. Делает cosine-top-K по ``wiki_section.embedding`` через pgvector
   (``<=>`` оператор; в SQLAlchemy — ``Vector.cosine_distance``).
3. Расширяет результат по ``wiki_link``: подтягивает секции страниц, на которые
   ссылаются страницы из top-K, и оставляет те, у которых similarity к запросу
   ≥ ``graph_expand_thresh``.

Возвращает упорядоченный по убыванию similarity список ``RetrievedSection`` —
без дубликатов по ``section.id``. Initial-top-K идёт первым (даже если их
similarity ниже добавленных по графу — это эвристика «семантика важнее
структуры»; expand-секции просто добавляются ниже).

Подключённая к ``wiki_section.page`` ``WikiPage`` подгружена через ``joinedload``
— значит, caller может читать ``section.page.slug`` / ``section.page.title``
без N+1 запросов.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from easyrag.config import get_settings
from easyrag.db.models import WikiLink, WikiPage, WikiSection


@dataclass(frozen=True)
class RetrievedSection:
    section_id: UUID
    page_id: UUID
    slug: str
    anchor: str
    page_title: str
    section_title: str
    body_md: str
    similarity: float
    source: str  # "vector" | "graph"


async def retrieve_sections(
    session: AsyncSession,
    *,
    query_vec: list[float],
    top_k: int = 8,
    graph_expand: bool = True,
    graph_expand_thresh: float | None = None,
) -> list[RetrievedSection]:
    """Найти секции для ответа на запрос."""
    if top_k <= 0:
        return []

    settings = get_settings()
    expand_thresh = (
        settings.graph_expand_thresh if graph_expand_thresh is None else graph_expand_thresh
    )

    base = await _vector_top_k(session, query_vec, top_k)
    if not base:
        return []

    if not graph_expand:
        return base

    seen_section_ids = {r.section_id for r in base}
    seen_page_ids = {r.page_id for r in base}
    extra = await _graph_expand(
        session,
        query_vec=query_vec,
        seed_page_ids=seen_page_ids,
        skip_section_ids=seen_section_ids,
        min_similarity=expand_thresh,
    )

    return base + extra


async def _vector_top_k(
    session: AsyncSession, query_vec: list[float], top_k: int
) -> list[RetrievedSection]:
    distance = WikiSection.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(WikiSection, distance)
        .options(joinedload(WikiSection.page))
        .where(WikiSection.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).unique().all()
    return [_row_to_retrieved(sec, float(dist), "vector") for sec, dist in rows]


async def _graph_expand(
    session: AsyncSession,
    *,
    query_vec: list[float],
    seed_page_ids: set[UUID],
    skip_section_ids: set[UUID],
    min_similarity: float,
) -> list[RetrievedSection]:
    if not seed_page_ids:
        return []

    # Шаг 1: собрать целевые страницы — куда указывают ссылки из seed-страниц.
    link_stmt = select(WikiLink.to_page_id).where(
        WikiLink.from_page_id.in_(seed_page_ids),
        WikiLink.to_page_id.is_not(None),
    )
    target_ids_raw = (await session.execute(link_stmt)).scalars().all()
    target_ids = {t for t in target_ids_raw if t is not None} - seed_page_ids
    if not target_ids:
        return []

    # Шаг 2: достать секции этих страниц + сразу посчитать distance к запросу.
    distance = WikiSection.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(WikiSection, distance)
        .options(joinedload(WikiSection.page))
        .where(
            WikiSection.page_id.in_(target_ids),
            WikiSection.embedding.is_not(None),
        )
        .order_by(distance.asc())
    )
    rows = (await session.execute(stmt)).unique().all()

    out: list[RetrievedSection] = []
    for sec, dist in rows:
        if sec.id in skip_section_ids:
            continue
        sim = 1.0 - float(dist)
        if sim < min_similarity:
            continue
        out.append(_row_to_retrieved(sec, float(dist), "graph"))
    return out


def _row_to_retrieved(sec: WikiSection, distance: float, source: str) -> RetrievedSection:
    page: WikiPage = sec.page
    return RetrievedSection(
        section_id=sec.id,
        page_id=sec.page_id,
        slug=page.slug,
        anchor=sec.anchor,
        page_title=page.title,
        section_title=sec.title,
        body_md=sec.body_md,
        similarity=1.0 - distance,
        source=source,
    )


__all__ = ["RetrievedSection", "retrieve_sections"]
