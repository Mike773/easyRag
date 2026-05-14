"""Резолвер кандидатов в wiki-страницы (шаг 4).

Берёт пачку ``entity_candidate`` (обычно — только что записанных ingest'ом)
и материализует их в ``wiki_page`` / ``wiki_section`` / ``section_provenance``.

Алгоритм:

1. Для каждого нерезолвенного кандидата выбираем целевой slug:
   * сначала по точному совпадению со slug существующей страницы
     (``make_slug(candidate.name)``);
   * иначе — ближайшая страница по вектору ``name. descriptor`` против
     ``wiki_section.embedding`` (берём максимум по странице). Если
     similarity ≥ ``resolve_thresh_high`` — то она; если между low и high —
     помечаем кандидата как ambiguous и пропускаем; ниже low — новая страница
     со slug'ом из имени кандидата.
2. Группируем кандидатов по целевому slug'у — все, кто попал в одну группу,
   сольются одним merge-вызовом.
3. Для каждой группы:
   * загружаем (или создаём в памяти) текущую страницу;
   * собираем descriptor'ы, statements, source uri's;
   * LLM-merge → новый ``body_md`` и список алиасов;
   * ``upsert_page`` — пересинхронизирует секции и ``wiki_link``;
   * заново эмбеддим все секции страницы (после rewrite старые embedding'и
     удалены каскадом);
   * для каждой секции пишем ``section_provenance`` против всех чанков-
     источников этого раунда — точечную секционную атрибуцию мы здесь не
     знаем (LLM перерасставил секции), поэтому атрибуцируем все чанки группы
     ко всем секциям страницы;
   * выставляем кандидатам ``resolved_page_id`` + ``resolved_at``.
4. Возвращаем сводку: сколько слилось, сколько создано, сколько ambiguous.

В тестах резолвер можно дёргать с заранее подготовленными ``llm`` и
``embeddings`` стабами — синглтоны из ``llm/`` подменяются опционально.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from easyrag.config import get_settings
from easyrag.db.models import (
    EntityCandidate,
    SectionProvenance,
    SourceChunk,
    SourceDoc,
    WikiPage,
    WikiSection,
)
from easyrag.llm.client import LLMClient, get_llm
from easyrag.llm.embeddings import EmbeddingClient, get_embeddings
from easyrag.query.prompts import (
    WIKI_MERGE_SCHEMA,
    WIKI_MERGE_SYSTEM,
    WIKI_MERGE_TOOL_DESCRIPTION,
    WIKI_MERGE_TOOL_NAME,
    build_merge_user_prompt,
)
from easyrag.wiki.markdown import make_slug, strip_self_links
from easyrag.wiki.merge_utils import (
    load_existing_catalog,
    reembed_sections,
)
from easyrag.wiki.repository import upsert_page

# Аннотация целевой страницы при резолве кандидата.
#   target == "new"        → создаём новую страницу с slug = candidate_slug
#   target == "existing"   → сливаем в уже существующую (page_id выставлен)
#   target == "ambiguous"  → similarity между low и high, оставляем кандидата
#                            нерезолвенным, без merge
_TARGET_NEW = "new"
_TARGET_EXISTING = "existing"
_TARGET_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ResolveOutcome:
    candidate_id: UUID
    target: str  # _TARGET_NEW | _TARGET_EXISTING | _TARGET_AMBIGUOUS
    page_slug: str | None
    page_id: UUID | None
    similarity: float | None


@dataclass(frozen=True)
class ResolveResult:
    created_pages: tuple[str, ...] = field(default_factory=tuple)
    merged_pages: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_candidate_ids: tuple[UUID, ...] = field(default_factory=tuple)
    resolved_candidate_count: int = 0

    @property
    def page_count(self) -> int:
        return len(self.created_pages) + len(self.merged_pages)


async def resolve_candidates(
    session: AsyncSession,
    candidate_ids: list[UUID],
    *,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
) -> ResolveResult:
    """Резолвит ``candidate_ids`` в wiki-страницы.

    Транзакцию ведёт caller. Возвращает сводку: какие страницы созданы /
    обновлены, какие кандидаты остались ambiguous.
    """
    if not candidate_ids:
        return ResolveResult()

    llm_client = llm or get_llm()
    embedder = embeddings or get_embeddings()
    settings = get_settings()

    candidates = await _load_candidates(session, candidate_ids)
    if not candidates:
        return ResolveResult()

    # Шаг 1: вычисляем целевую страницу для каждого кандидата.
    outcomes: list[ResolveOutcome] = []
    for c in candidates:
        outcome = await _resolve_target(
            session,
            candidate=c,
            thresh_high=settings.resolve_thresh_high,
            thresh_low=settings.resolve_thresh_low,
        )
        outcomes.append(outcome)

    # Шаг 2: группируем кандидатов по целевому slug'у.
    groups: dict[str, list[tuple[EntityCandidate, ResolveOutcome]]] = defaultdict(list)
    ambiguous_ids: list[UUID] = []
    for c, out in zip(candidates, outcomes):
        if out.target == _TARGET_AMBIGUOUS or out.page_slug is None:
            ambiguous_ids.append(c.id)
            continue
        groups[out.page_slug].append((c, out))

    created_pages: list[str] = []
    merged_pages: list[str] = []
    resolved_total = 0

    # Шаг 3: сливаем каждую группу одним merge-вызовом.
    for slug, items in groups.items():
        was_new = all(out.target == _TARGET_NEW for _, out in items)
        page = await _merge_group(
            session,
            slug=slug,
            items=items,
            llm=llm_client,
            embedder=embedder,
            forced_new=was_new,
        )
        if was_new:
            created_pages.append(slug)
        else:
            merged_pages.append(slug)

        # section_provenance: новые секции × все вкладывающие чанки группы
        section_ids = await _section_ids(session, page.id)
        chunk_ids = {c.chunk_id for c, _ in items}
        await _write_provenance(session, section_ids, chunk_ids)

        # Помечаем кандидатов как resolved.
        now = datetime.now(timezone.utc)
        for c, _ in items:
            c.resolved_page_id = page.id
            c.resolved_at = now
            resolved_total += 1

    await session.flush()

    return ResolveResult(
        created_pages=tuple(created_pages),
        merged_pages=tuple(merged_pages),
        ambiguous_candidate_ids=tuple(ambiguous_ids),
        resolved_candidate_count=resolved_total,
    )


async def _find_pages_by_alias(
    session: AsyncSession, name: str
) -> list[tuple[UUID, str]]:
    """Найти страницы, у которых ``name`` встречается среди ``aliases``.

    Сравнение case-insensitive (``lower(alias) = lower(name)``). PostgreSQL-only:
    использует ``unnest`` поверх ``ARRAY(Text)``. Ограничиваем 2 строками —
    нам важно лишь различить 0 / 1 / «несколько» совпадений.
    """
    name_lower = (name or "").strip().casefold()
    if not name_lower:
        return []
    rows = (
        await session.execute(
            select(WikiPage.id, WikiPage.slug)
            .where(
                text(
                    "EXISTS (SELECT 1 FROM unnest(wiki_page.aliases) AS _alias "
                    "WHERE lower(_alias) = :_alias_name)"
                ).bindparams(_alias_name=name_lower)
            )
            .limit(2)
        )
    ).all()
    return [(r[0], r[1]) for r in rows]


async def _load_candidates(
    session: AsyncSession, candidate_ids: list[UUID]
) -> list[EntityCandidate]:
    if not candidate_ids:
        return []
    stmt = select(EntityCandidate).where(
        EntityCandidate.id.in_(candidate_ids),
        EntityCandidate.resolved_page_id.is_(None),
    )
    return list((await session.execute(stmt)).scalars().all())


async def _resolve_target(
    session: AsyncSession,
    *,
    candidate: EntityCandidate,
    thresh_high: float,
    thresh_low: float,
) -> ResolveOutcome:
    """Подобрать целевой slug для одного кандидата."""
    candidate_slug = make_slug(candidate.name)

    # 1. Точное совпадение slug'а — самое сильное доказательство, что это та же сущность.
    existing = (
        await session.execute(select(WikiPage).where(WikiPage.slug == candidate_slug))
    ).scalar_one_or_none()
    if existing is not None:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_EXISTING,
            page_slug=existing.slug,
            page_id=existing.id,
            similarity=1.0,
        )

    # 1.5. Точное (case-insensitive) совпадение с одним из алиасов существующей страницы.
    # Помогает склеить cross-chunk варианты имени, которые дают разные slug'и
    # ("Acme" vs "Acme Corp"): если LLM в предыдущем merge положил вариант в
    # wiki_page.aliases, новый кандидат с этим вариантом теперь сольётся в ту же
    # страницу, а не породит дубль. При >1 совпадении (страницы с пересекающимися
    # алиасами — сигнал неконсистентности) уходим в обычный embedding-flow.
    alias_match = await _find_pages_by_alias(session, candidate.name)
    if len(alias_match) == 1:
        page_id, slug = alias_match[0]
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_EXISTING,
            page_slug=slug,
            page_id=page_id,
            similarity=1.0,
        )

    # 2. Vector top-1 по wiki_section: берём страницу с минимальной distance.
    if candidate.embedding is None:
        # Без эмбеддинга сравнить не с чем — создаём новую страницу.
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW,
            page_slug=candidate_slug,
            page_id=None,
            similarity=None,
        )

    distance = WikiSection.embedding.cosine_distance(candidate.embedding)
    stmt = (
        select(WikiSection.page_id, distance.label("dist"))
        .where(WikiSection.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW,
            page_slug=candidate_slug,
            page_id=None,
            similarity=None,
        )

    best_page_id, best_distance = row
    similarity = 1.0 - float(best_distance)

    if similarity >= thresh_high:
        page = await session.get(WikiPage, best_page_id)
        if page is None:
            # Гонка / битая ссылка — fallback на новую страницу.
            return ResolveOutcome(
                candidate_id=candidate.id,
                target=_TARGET_NEW,
                page_slug=candidate_slug,
                page_id=None,
                similarity=similarity,
            )
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_EXISTING,
            page_slug=page.slug,
            page_id=page.id,
            similarity=similarity,
        )

    if similarity >= thresh_low:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_AMBIGUOUS,
            page_slug=None,
            page_id=None,
            similarity=similarity,
        )

    return ResolveOutcome(
        candidate_id=candidate.id,
        target=_TARGET_NEW,
        page_slug=candidate_slug,
        page_id=None,
        similarity=similarity,
    )


async def _merge_group(
    session: AsyncSession,
    *,
    slug: str,
    items: list[tuple[EntityCandidate, ResolveOutcome]],
    llm: LLMClient,
    embedder: EmbeddingClient,
    forced_new: bool,
) -> WikiPage:
    """Слить группу кандидатов в одну страницу и пересинхронизировать секции."""
    # Выбираем заголовок: для существующей — оставляем текущий, для новой —
    # самое полное имя из кандидатов.
    existing = (
        await session.execute(select(WikiPage).where(WikiPage.slug == slug))
    ).scalar_one_or_none()

    if existing is not None:
        title = existing.title
        current_body = existing.body_md or ""
        current_aliases = list(existing.aliases or [])
    else:
        title = _pick_title(items)
        current_body = ""
        current_aliases = []

    descriptors = [c.descriptor for c, _ in items if c.descriptor]
    statements: list[str] = []
    chunk_ids = list({c.chunk_id for c, _ in items})
    source_uris = await _chunk_source_uris(session, chunk_ids)

    for c, _ in items:
        for s in c.statements or []:
            if s and s.strip():
                statements.append(s.strip())
    # Дедуп с сохранением порядка — модель не любит шум.
    statements = list(dict.fromkeys(statements))

    settings = get_settings()
    existing_entities = await load_existing_catalog(
        session,
        exclude_slug=slug,
        limit=settings.merge_catalog_limit,
    )

    user_prompt = build_merge_user_prompt(
        title=title,
        current_body=current_body,
        current_aliases=current_aliases,
        new_descriptors=descriptors,
        new_statements=statements,
        source_uris=source_uris,
        existing_entities=existing_entities,
    )
    raw = await llm.call_json(
        system=WIKI_MERGE_SYSTEM,
        user=user_prompt,
        tool_name=WIKI_MERGE_TOOL_NAME,
        tool_description=WIKI_MERGE_TOOL_DESCRIPTION,
        input_schema=WIKI_MERGE_SCHEMA,
    )

    body_md = _coerce_body(raw, statements_fallback=statements, title=title)
    # Подстраховка: LLM иногда оборачивает заголовок страницы в [[…]] даже при
    # прямом запрете в prompt'е. Чистим self-references до записи, чтобы граф
    # ссылок не содержал self-loop'ов.
    body_md = strip_self_links(body_md, page_slug=slug)
    aliases = _coerce_aliases(raw, current_aliases, [c.name for c, _ in items], title)

    page = await upsert_page(
        session,
        slug=slug,
        title=title,
        body_md=body_md,
        aliases=aliases,
    )
    # upsert_page внутри сделал flush — секции уже в БД, можно эмбеддить.
    await reembed_sections(session, page, embedder)
    if forced_new:
        # Заметка: ``forced_new`` для логирования; upsert_page сам разбирается с
        # INSERT vs UPDATE. Тут ничего особенного делать не нужно.
        pass
    return page


def _pick_title(items: list[tuple[EntityCandidate, ResolveOutcome]]) -> str:
    """Самое длинное / полное имя среди кандидатов группы — обычно лучшее заглавие."""
    names = sorted({c.name.strip() for c, _ in items if c.name and c.name.strip()}, key=len, reverse=True)
    return names[0] if names else "Без названия"


async def _chunk_source_uris(session: AsyncSession, chunk_ids: list[UUID]) -> list[str]:
    if not chunk_ids:
        return []
    stmt = (
        select(SourceDoc.uri)
        .join(SourceChunk, SourceChunk.doc_id == SourceDoc.id)
        .where(SourceChunk.id.in_(chunk_ids))
        .distinct()
    )
    return list((await session.execute(stmt)).scalars().all())


def _coerce_body(raw: dict, *, statements_fallback: list[str], title: str) -> str:
    """Достать ``body_md`` из ответа LLM; на пустой ответ — детерминированный fallback.

    Без fallback'а bug-у в модели мы бы потеряли извлечённые из источников факты.
    Минимальный осмысленный markdown: одна H2-секция с буллетами по statements.
    """
    body = raw.get("body_md") if isinstance(raw, dict) else None
    if isinstance(body, str) and body.strip():
        return body.strip()

    if not statements_fallback:
        return ""

    bullets = "\n".join(f"- {s}" for s in statements_fallback)
    return f"## Overview\n\n{bullets}"


def _coerce_aliases(
    raw: dict,
    current: list[str],
    new_names: list[str],
    title: str,
) -> list[str]:
    """Собрать итоговый набор алиасов: LLM + текущие + новые имена кандидатов.

    Удаляем дубликаты (case-insensitive) и заголовок страницы из алиасов —
    он там не нужен.
    """
    items: list[str] = []
    if isinstance(raw, dict):
        raw_aliases = raw.get("aliases")
        if isinstance(raw_aliases, list):
            items.extend(a for a in raw_aliases if isinstance(a, str))
    items.extend(current)
    items.extend(new_names)

    title_key = title.strip().casefold()
    seen: set[str] = set()
    out: list[str] = []
    for a in items:
        cleaned = (a or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key == title_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


async def _section_ids(session: AsyncSession, page_id: UUID) -> list[UUID]:
    rows = await session.execute(
        select(WikiSection.id).where(WikiSection.page_id == page_id)
    )
    return list(rows.scalars().all())


async def _write_provenance(
    session: AsyncSession,
    section_ids: list[UUID],
    chunk_ids: set[UUID],
) -> None:
    """Записать ``section_provenance`` для пар (section, chunk) этого раунда.

    Если запись уже существует (повторный merge той же группы) — пропускаем.
    """
    if not section_ids or not chunk_ids:
        return
    pairs = {(sid, cid) for sid in section_ids for cid in chunk_ids}
    # Загружаем уже существующие пары, чтобы не нарваться на PK-конфликт.
    existing = await session.execute(
        select(SectionProvenance.section_id, SectionProvenance.source_chunk_id).where(
            SectionProvenance.section_id.in_(section_ids),
            SectionProvenance.source_chunk_id.in_(chunk_ids),
        )
    )
    existing_pairs = {(sid, cid) for sid, cid in existing.all()}
    for sid, cid in pairs - existing_pairs:
        session.add(SectionProvenance(section_id=sid, source_chunk_id=cid))


__all__ = [
    "ResolveOutcome",
    "ResolveResult",
    "resolve_candidates",
]
