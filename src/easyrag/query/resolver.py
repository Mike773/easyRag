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

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
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
    RESOLVE_JUDGE_SCHEMA,
    RESOLVE_JUDGE_SYSTEM,
    RESOLVE_JUDGE_TOOL_DESCRIPTION,
    RESOLVE_JUDGE_TOOL_NAME,
    WIKI_MERGE_SCHEMA,
    WIKI_MERGE_SYSTEM,
    WIKI_MERGE_TOOL_DESCRIPTION,
    WIKI_MERGE_TOOL_NAME,
    build_merge_user_prompt,
    build_resolve_judge_user_prompt,
)
from easyrag.wiki.markdown import make_slug, strip_self_links
from easyrag.wiki.merge_utils import (
    load_existing_catalog,
    reembed_sections,
)
from easyrag.wiki.repository import upsert_page

logger = logging.getLogger(__name__)

# Аннотация целевой страницы при резолве кандидата.
#   target == "new"        → создаём новую страницу с slug = candidate_slug
#   target == "existing"   → сливаем в уже существующую (page_id выставлен)
#   target == "ambiguous"  → similarity между low и high, оставляем кандидата
#                            нерезолвенным, без merge
_TARGET_NEW = "new"
_TARGET_EXISTING = "existing"
_TARGET_AMBIGUOUS = "ambiguous"

# Векторный матчинг: сколько секций тянем из БД, чтобы агрегировать в top-N
# страниц-кандидатов; сколько страниц показываем LLM-судье в ambiguous-зоне.
_JUDGE_SECTION_LIMIT = 30
_JUDGE_PAGE_LIMIT = 5
# Сколько лексически-похожих страниц добавляем поверх векторных (по difflib
# ratio имени кандидата против page.title / aliases). Это закрывает кейсы,
# где векторное сходство name-only низкое, но имя страницы лексически очень
# близко: «Мышка-норушка» vs «Мышка», «Медведь косолапый» vs «Медведь».
_JUDGE_LEXICAL_LIMIT = 3
_JUDGE_LEXICAL_THRESH = 0.35
# Общий потолок опций для судьи — больше уже раздувает prompt.
_JUDGE_OPTIONS_TOTAL = 7
# Сколько символов из body_md показывать судье — достаточно, чтобы понять
# суть, но не раздувать prompt.
_JUDGE_BODY_EXCERPT_CHARS = 400


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
            llm=llm_client,
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
    llm: LLMClient,
) -> ResolveOutcome:
    """Подобрать целевой slug для одного кандидата.

    Маршрутизация:

    1. exact slug match — самое сильное доказательство.
    2. alias match (одно совпадение) — следующий уровень.
    3. векторный top-K по wiki_section, агрегированный в top-N страниц:
       * top-1 sim ≥ thresh_high → EXISTING (уверенный merge);
       * top-1 sim < thresh_low → NEW (уверенно новая сущность);
       * thresh_low ≤ top-1 sim < thresh_high → LLM-судья смотрит на top-N
         кандидатов и решает: existing/new/ambiguous. Если решить не удалось,
         откладываем как ambiguous (лучше отложить, чем слить разные сущности).
    """
    candidate_slug = make_slug(candidate.name)

    # 1. Точное совпадение slug'а.
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

    # 1.5. Точное (case-insensitive) совпадение с одним из алиасов.
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

    # 2. Вектор: без эмбеддинга — создаём новую страницу.
    if candidate.embedding is None:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW,
            page_slug=candidate_slug,
            page_id=None,
            similarity=None,
        )

    # 2.1. Векторный top-K секций → агрегируем по максимальной similarity на
    #      страницу → top-N страниц-претендентов.
    distance = WikiSection.embedding.cosine_distance(candidate.embedding)
    stmt = (
        select(WikiSection.page_id, distance.label("dist"))
        .where(WikiSection.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(_JUDGE_SECTION_LIMIT)
    )
    rows = (await session.execute(stmt)).all()
    best_by_page: dict[UUID, float] = {}
    for page_id, dist in rows:
        d = float(dist)
        prev = best_by_page.get(page_id)
        if prev is None or d < prev:
            best_by_page[page_id] = d
    sorted_vector = sorted(best_by_page.items(), key=lambda kv: kv[1])[:_JUDGE_PAGE_LIMIT]
    top_sim = (1.0 - sorted_vector[0][1]) if sorted_vector else 0.0
    top_page_id = sorted_vector[0][0] if sorted_vector else None

    # 2.2. Уверенный merge: top-1 уже выше HIGH порога — не зовём судью.
    if top_page_id is not None and top_sim >= thresh_high:
        page = await session.get(WikiPage, top_page_id)
        if page is None:
            return ResolveOutcome(
                candidate_id=candidate.id,
                target=_TARGET_NEW,
                page_slug=candidate_slug,
                page_id=None,
                similarity=top_sim,
            )
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_EXISTING,
            page_slug=page.slug,
            page_id=page.id,
            similarity=top_sim,
        )

    # 2.3. Лексический pre-step: страницы с высоким difflib-ratio имени против
    #      title/aliases. Закрывает кейсы, где синонимы лексически близки, но
    #      vector top-N их не подтянул (напр., «Мышка-норушка» ↔ «Мышка»).
    lexical_matches = await _lexical_candidates(session, candidate.name)
    has_strong_lexical = bool(lexical_matches)

    # 2.4. Если нет ни вектор-кандидатов с sim ≥ LOW, ни лексических — уверенно
    #      новая сущность.
    if top_sim < thresh_low and not has_strong_lexical:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW,
            page_slug=candidate_slug,
            page_id=None,
            similarity=top_sim if sorted_vector else None,
        )

    # 2.5. Собираем опции для судьи: vector top-N + лексические, дедуплицируем
    #      по page_id, обрезаем по общему лимиту.
    page_ids_ordered: list[UUID] = []
    seen: set[UUID] = set()
    for pid, _ in sorted_vector:
        if pid not in seen:
            seen.add(pid)
            page_ids_ordered.append(pid)
    for pid, _ in lexical_matches:
        if pid not in seen and len(page_ids_ordered) < _JUDGE_OPTIONS_TOTAL:
            seen.add(pid)
            page_ids_ordered.append(pid)
    page_ids_ordered = page_ids_ordered[:_JUDGE_OPTIONS_TOTAL]

    pages_by_id = await _load_pages_by_ids(session, page_ids_ordered)
    options: list[tuple[str, str, list[str], str]] = []
    for pid in page_ids_ordered:
        page = pages_by_id.get(pid)
        if page is None:
            continue
        body_excerpt = (page.body_md or "")[:_JUDGE_BODY_EXCERPT_CHARS]
        options.append(
            (page.slug, page.title, list(page.aliases or []), body_excerpt)
        )
    if not options:
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW if top_sim < thresh_low else _TARGET_AMBIGUOUS,
            page_slug=candidate_slug if top_sim < thresh_low else None,
            page_id=None,
            similarity=top_sim if sorted_vector else None,
        )

    decision, decided_slug = await _judge_target(
        llm=llm,
        candidate=candidate,
        options=options,
    )
    logger.info(
        "resolver judge candidate=%r top_sim=%.3f decision=%s slug=%s options=%s",
        candidate.name, top_sim, decision, decided_slug,
        [s for s, _, _, _ in options],
    )
    if decision == "existing" and decided_slug:
        match = next((p for p in pages_by_id.values() if p.slug == decided_slug), None)
        if match is not None:
            return ResolveOutcome(
                candidate_id=candidate.id,
                target=_TARGET_EXISTING,
                page_slug=match.slug,
                page_id=match.id,
                similarity=top_sim,
            )
        # Судья выдал slug, которого не было в опциях — защита от галлюцинации.
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_AMBIGUOUS,
            page_slug=None,
            page_id=None,
            similarity=top_sim,
        )
    if decision == "new":
        return ResolveOutcome(
            candidate_id=candidate.id,
            target=_TARGET_NEW,
            page_slug=candidate_slug,
            page_id=None,
            similarity=top_sim,
        )
    # decision == "ambiguous" или ошибка/мусор.
    return ResolveOutcome(
        candidate_id=candidate.id,
        target=_TARGET_AMBIGUOUS,
        page_slug=None,
        page_id=None,
        similarity=top_sim,
    )


async def _load_pages_by_ids(
    session: AsyncSession, page_ids: list[UUID]
) -> dict[UUID, WikiPage]:
    if not page_ids:
        return {}
    rows = (
        await session.execute(select(WikiPage).where(WikiPage.id.in_(page_ids)))
    ).scalars().all()
    return {p.id: p for p in rows}


async def _lexical_candidates(
    session: AsyncSession, name: str
) -> list[tuple[UUID, float]]:
    """Найти страницы, лексически близкие к ``name`` (по difflib-ratio).

    Сканирует все страницы, считает максимум по ratio(title) и ratio(alias),
    оставляет те, у кого ratio ≥ ``_JUDGE_LEXICAL_THRESH``, и возвращает
    top-``_JUDGE_LEXICAL_LIMIT`` по убыванию.

    Дешёвый O(N) скан по wiki_page — без вектора, без LLM. На больших wiki
    стоит заменить trigram-индексом, но при текущем масштабе достаточно.
    """
    cand_lower = (name or "").strip().casefold()
    if not cand_lower:
        return []
    rows = (
        await session.execute(select(WikiPage.id, WikiPage.title, WikiPage.aliases))
    ).all()
    scored: list[tuple[UUID, float]] = []
    for pid, title, aliases in rows:
        title_lower = (title or "").strip().casefold()
        ratio = SequenceMatcher(None, cand_lower, title_lower).ratio() if title_lower else 0.0
        for alias in (aliases or ()):
            alias_lower = (alias or "").strip().casefold()
            if not alias_lower:
                continue
            r = SequenceMatcher(None, cand_lower, alias_lower).ratio()
            if r > ratio:
                ratio = r
        if ratio >= _JUDGE_LEXICAL_THRESH:
            scored.append((pid, ratio))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:_JUDGE_LEXICAL_LIMIT]


async def _judge_target(
    *,
    llm: LLMClient,
    candidate: EntityCandidate,
    options: list[tuple[str, str, list[str], str]],
) -> tuple[str, str | None]:
    """Спросить LLM, является ли кандидат одной из ``options`` или это новая.

    Возвращает кортеж ``(decision, slug)``. ``decision`` — одно из
    ``{"existing","new","ambiguous"}``. ``slug`` валиден только при ``existing``.
    На любую ошибку или мусорный ответ возвращает ``("ambiguous", None)`` —
    сейф: лучше отложить кандидата, чем создать дубль.
    """
    valid_slugs = {slug for slug, _, _, _ in options}
    user_prompt = build_resolve_judge_user_prompt(
        candidate_name=candidate.name or "",
        candidate_descriptor=candidate.descriptor or "",
        candidate_statements=list(candidate.statements or []),
        options=options,
    )
    try:
        raw = await llm.call_json(
            system=RESOLVE_JUDGE_SYSTEM,
            user=user_prompt,
            tool_name=RESOLVE_JUDGE_TOOL_NAME,
            tool_description=RESOLVE_JUDGE_TOOL_DESCRIPTION,
            input_schema=RESOLVE_JUDGE_SCHEMA,
        )
    except Exception:
        return ("ambiguous", None)
    if not isinstance(raw, dict):
        return ("ambiguous", None)
    decision = raw.get("decision")
    if decision not in ("existing", "new", "ambiguous"):
        return ("ambiguous", None)
    if decision == "existing":
        slug = raw.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            return ("ambiguous", None)
        slug = slug.strip()
        if slug not in valid_slugs:
            return ("ambiguous", None)
        return ("existing", slug)
    return (decision, None)


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
