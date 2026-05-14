"""Оркестратор ingest-пайплайна.

Делает за один проход:
1. Считает sha256 от текста — для дедупликации (идемпотентность по содержимому).
2. Создаёт строку ``source_doc``.
3. Режет текст на чанки (:func:`chunk_text`), эмбеддит их батчем,
   пишет ``source_chunk``.
4. Для каждого чанка вызывает LLM (:func:`extract_entities`), эмбеддит каждого
   кандидата по строке ``name. descriptor``, пишет ``entity_candidate``.
5. На шаге 4 — резолвит свежих кандидатов в ``wiki_page`` / ``wiki_section`` +
   пишет ``section_provenance`` (см. :func:`easyrag.query.resolve_candidates`).
   Можно выключить через ``resolve=False`` — полезно для отладочного ingest
   без LLM-merge'а.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from easyrag.config import get_settings
from easyrag.db.models import EntityCandidate, SourceChunk, SourceDoc
from easyrag.ingest.chunker import Chunk, chunk_text
from easyrag.ingest.extractor import (
    DocumentBrief,
    ExtractedEntity,
    analyze_document,
    extract_entities,
)
from easyrag.llm.client import LLMClient
from easyrag.llm.embeddings import EmbeddingClient, get_embeddings
from easyrag.wiki.merge_utils import embed_batched


@dataclass(frozen=True)
class IngestResult:
    doc_id: UUID
    chunk_count: int
    entity_count: int
    deduplicated: bool = False
    chunk_ids: tuple[UUID, ...] = field(default_factory=tuple)
    created_pages: tuple[str, ...] = field(default_factory=tuple)
    merged_pages: tuple[str, ...] = field(default_factory=tuple)
    ambiguous_candidate_count: int = 0
    resolved_candidate_count: int = 0
    relinked_pages: tuple[str, ...] = field(default_factory=tuple)
    domain_brief: DocumentBrief | None = None


async def ingest_text(
    session: AsyncSession,
    *,
    uri: str,
    text: str,
    mime: str | None = None,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
    resolve: bool = True,
    brief_window: int | None = None,
    chunk_target_size: int | None = None,
    chunk_max_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Прогнать ``text`` через пайплайн и записать всё в БД.

    Транзакцию (commit/rollback) ведёт вызывающий код — здесь только
    добавление и flush'и.

    ``resolve=True`` (по умолчанию) дополнительно вызывает
    :func:`easyrag.query.resolve_candidates` для свежих кандидатов: создаёт
    или дополняет wiki-страницы и пишет провенанс. Передайте ``resolve=False``,
    если нужно положить кандидатов в БД без LLM-merge'а (для отладки).
    """
    if not text or not text.strip():
        raise ValueError("ingest_text: empty text")

    embedder = embeddings or get_embeddings()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    existing = (
        await session.execute(select(SourceDoc).where(SourceDoc.sha256 == sha))
    ).scalar_one_or_none()
    if existing is not None:
        chunk_rows = (
            await session.execute(
                select(SourceChunk.id).where(SourceChunk.doc_id == existing.id)
            )
        ).scalars().all()
        ent_count = await _count_entities(session, existing.id)
        return IngestResult(
            doc_id=existing.id,
            chunk_count=len(chunk_rows),
            entity_count=ent_count,
            deduplicated=True,
            chunk_ids=tuple(chunk_rows),
        )

    doc = SourceDoc(uri=uri, mime=mime, sha256=sha)
    session.add(doc)
    await session.flush()

    # Шаг 2.5: построить domain brief — один LLM-вызов по началу документа.
    # Brief подаётся как контекст в каждый последующий extract_entities, чтобы
    # модель калибровалась под фактический жанр документа. None означает, что
    # brief построить не удалось; extraction всё равно работает, но без подсказок.
    settings = get_settings()
    window = brief_window if brief_window is not None else settings.doc_brief_window
    target_size = (
        chunk_target_size if chunk_target_size is not None else settings.chunk_target_size
    )
    max_size = chunk_max_size if chunk_max_size is not None else settings.chunk_max_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    domain_brief: DocumentBrief | None = None
    if window > 0:
        domain_brief = await analyze_document(
            text[:window], source_hint=uri, llm=llm
        )
        if domain_brief is not None:
            doc.domain_brief = _serialize_brief(domain_brief)

    chunks = chunk_text(text, target_size=target_size, max_size=max_size, overlap=overlap)
    chunk_rows = await _persist_chunks(session, doc.id, chunks, embedder)

    entity_total = 0
    candidate_ids: list[UUID] = []
    for chunk_row, parsed in zip(chunk_rows, chunks):
        extracted = await extract_entities(
            parsed.text,
            source_hint=uri,
            domain_brief=domain_brief,
            llm=llm,
        )
        if not extracted:
            continue
        new_ids = await _persist_entities(
            session,
            doc_id=doc.id,
            chunk_id=chunk_row.id,
            extracted=extracted,
            embedder=embedder,
        )
        candidate_ids.extend(new_ids)
        entity_total += len(new_ids)

    await session.flush()

    created_pages: tuple[str, ...] = ()
    merged_pages: tuple[str, ...] = ()
    ambiguous_count = 0
    resolved_count = 0
    relinked_pages: tuple[str, ...] = ()
    if resolve and candidate_ids:
        # Импорт локально, чтобы избежать циклов: easyrag.query тянет wiki/repository,
        # который зависит от тех же db-моделей.
        from easyrag.query import resolve_candidates
        from easyrag.wiki.backlinker import backfill_links

        summary = await resolve_candidates(
            session,
            candidate_ids,
            llm=llm,
            embeddings=embedder,
        )
        created_pages = summary.created_pages
        merged_pages = summary.merged_pages
        ambiguous_count = len(summary.ambiguous_candidate_ids)
        resolved_count = summary.resolved_candidate_count

        # Back-link: проставить [[…]] в старых страницах, которые упоминают
        # созданные/обновлённые в этом раунде сущности. Без этого граф
        # связей знает только односторонние ссылки «новые → старые».
        if settings.backlink_enabled:
            backfill = await backfill_links(
                session,
                exclude_slugs=set(created_pages) | set(merged_pages),
                llm=llm,
                embeddings=embedder,
            )
            relinked_pages = backfill.relinked

    return IngestResult(
        doc_id=doc.id,
        chunk_count=len(chunk_rows),
        entity_count=entity_total,
        deduplicated=False,
        chunk_ids=tuple(c.id for c in chunk_rows),
        created_pages=created_pages,
        merged_pages=merged_pages,
        ambiguous_candidate_count=ambiguous_count,
        resolved_candidate_count=resolved_count,
        relinked_pages=relinked_pages,
        domain_brief=domain_brief,
    )


async def _persist_chunks(
    session: AsyncSession,
    doc_id: UUID,
    chunks: list[Chunk],
    embedder: EmbeddingClient,
) -> list[SourceChunk]:
    if not chunks:
        return []
    vectors = await embed_batched(embedder, [c.text for c in chunks])
    rows: list[SourceChunk] = []
    for chunk, vec in zip(chunks, vectors):
        row = SourceChunk(
            doc_id=doc_id,
            ord=chunk.ord,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            embedding=vec,
        )
        session.add(row)
        rows.append(row)
    await session.flush()
    return rows


async def _persist_entities(
    session: AsyncSession,
    *,
    doc_id: UUID,
    chunk_id: UUID,
    extracted: list[ExtractedEntity],
    embedder: EmbeddingClient,
) -> list[UUID]:
    texts = [_embed_text(e) for e in extracted]
    vectors = await embed_batched(embedder, texts)
    rows: list[EntityCandidate] = []
    for ent, vec in zip(extracted, vectors):
        row = EntityCandidate(
            doc_id=doc_id,
            chunk_id=chunk_id,
            name=ent.name,
            descriptor=ent.descriptor,
            statements=list(ent.statements),
            embedding=vec,
        )
        session.add(row)
        rows.append(row)
    if rows:
        # Нужны id для последующего resolve_candidates(...) — flush заранее.
        await session.flush()
    return [r.id for r in rows]


def _serialize_brief(brief: DocumentBrief) -> str:
    """Сохраняем brief в source_doc.domain_brief как JSON-строку.

    Структура совпадает с DOC_BRIEF_SCHEMA, чтобы при необходимости можно было
    восстановить :class:`DocumentBrief` обратным проходом без дополнительной
    транскрипции.
    """
    return json.dumps(
        {
            "summary": brief.summary,
            "entity_types": list(brief.entity_types),
        },
        ensure_ascii=False,
    )


def _embed_text(entity: ExtractedEntity) -> str:
    """Текст для эмбеддинга кандидата — имя + дескриптор.

    Statements в эмбеддинг не включаем намеренно: иначе сходство будет
    «тянуть» близкие по фактам, но разные сущности. Шаг 4 резолвит кандидата
    по name+descriptor, statements нужны позже как сырьё для wiki-страницы.
    """
    if entity.descriptor:
        return f"{entity.name}. {entity.descriptor}"
    return entity.name


async def _count_entities(session: AsyncSession, doc_id: UUID) -> int:
    res = await session.execute(
        select(func.count())
        .select_from(EntityCandidate)
        .where(EntityCandidate.doc_id == doc_id)
    )
    return int(res.scalar_one() or 0)


__all__ = ["IngestResult", "ingest_text"]
