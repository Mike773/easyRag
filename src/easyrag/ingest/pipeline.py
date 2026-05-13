"""Оркестратор ingest-пайплайна.

Делает за один проход:
1. Считает sha256 от текста — для дедупликации (идемпотентность по содержимому).
2. Создаёт строку ``source_doc``.
3. Режет текст на чанки (:func:`chunk_text`), эмбеддит их батчем,
   пишет ``source_chunk``.
4. Для каждого чанка вызывает LLM (:func:`extract_entities`), эмбеддит каждого
   кандидата по строке ``name. descriptor``, пишет ``entity_candidate``.

Что НЕ делает на этом шаге:
* не резолвит кандидатов в существующие wiki-страницы (это шаг 4);
* не создаёт wiki-страницы (это шаг 4);
* не делает provenance-привязки ``section_provenance``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Безопасный размер батча эмбеддингов:
# OpenAI text-embedding-3-* принимает до 2048 input'ов за запрос; у GigaChat
# лимит существенно ниже. 128 — компромисс, при котором обе модели работают
# стабильно даже на длинных документах.
_EMBED_BATCH_SIZE = 128

from easyrag.db.models import EntityCandidate, SourceChunk, SourceDoc
from easyrag.ingest.chunker import Chunk, chunk_text
from easyrag.ingest.extractor import ExtractedEntity, extract_entities
from easyrag.llm.client import LLMClient
from easyrag.llm.embeddings import EmbeddingClient, get_embeddings


@dataclass(frozen=True)
class IngestResult:
    doc_id: UUID
    chunk_count: int
    entity_count: int
    deduplicated: bool = False
    chunk_ids: tuple[UUID, ...] = field(default_factory=tuple)


async def ingest_text(
    session: AsyncSession,
    *,
    uri: str,
    text: str,
    mime: str | None = None,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
) -> IngestResult:
    """Прогнать ``text`` через пайплайн и записать всё в БД.

    Транзакцию (commit/rollback) ведёт вызывающий код — здесь только
    добавление и flush'и.
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

    chunks = chunk_text(text)
    chunk_rows = await _persist_chunks(session, doc.id, chunks, embedder)

    entity_total = 0
    for chunk_row, parsed in zip(chunk_rows, chunks):
        extracted = await extract_entities(parsed.text, source_hint=uri, llm=llm)
        if not extracted:
            continue
        entity_total += await _persist_entities(
            session,
            doc_id=doc.id,
            chunk_id=chunk_row.id,
            extracted=extracted,
            embedder=embedder,
        )

    await session.flush()
    return IngestResult(
        doc_id=doc.id,
        chunk_count=len(chunk_rows),
        entity_count=entity_total,
        deduplicated=False,
        chunk_ids=tuple(c.id for c in chunk_rows),
    )


async def _embed_batched(
    embedder: EmbeddingClient, texts: list[str]
) -> list[list[float]]:
    """Эмбеддим длинные списки по частям — иначе провайдер ругнётся на лимит batch."""
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        out.extend(await embedder.embed_many(texts[i : i + _EMBED_BATCH_SIZE]))
    return out


async def _persist_chunks(
    session: AsyncSession,
    doc_id: UUID,
    chunks: list[Chunk],
    embedder: EmbeddingClient,
) -> list[SourceChunk]:
    if not chunks:
        return []
    vectors = await _embed_batched(embedder, [c.text for c in chunks])
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
) -> int:
    texts = [_embed_text(e) for e in extracted]
    vectors = await _embed_batched(embedder, texts)
    for ent, vec in zip(extracted, vectors):
        session.add(
            EntityCandidate(
                doc_id=doc_id,
                chunk_id=chunk_id,
                name=ent.name,
                descriptor=ent.descriptor,
                statements=list(ent.statements),
                embedding=vec,
            )
        )
    return len(extracted)


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
