from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from easyrag.config import get_settings

EMBED_DIM = get_settings().embed_dim


class Base(DeclarativeBase):
    pass


# --- Источники (только для провенанса) ---

class SourceDoc(Base):
    __tablename__ = "source_doc"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str | None] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)


class SourceChunk(Base):
    __tablename__ = "source_chunk"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("source_doc.id", ondelete="CASCADE"), nullable=False
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # Эмбеддинг чанка нужен ТОЛЬКО для enrichment-loop (gap → re-extract).
    # Query pipeline эту таблицу НЕ читает.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))

    __table_args__ = (
        UniqueConstraint("doc_id", "ord", name="uq_source_chunk_doc_ord"),
    )


# --- Wiki ---

class WikiPage(Base):
    __tablename__ = "wiki_page"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str | None] = mapped_column(String(64))
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    body_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    sections: Mapped[list["WikiSection"]] = relationship(
        back_populates="page", cascade="all, delete-orphan", order_by="WikiSection.ord"
    )

    __table_args__ = (
        Index("ix_wiki_page_type", "type"),
    )


class WikiSection(Base):
    __tablename__ = "wiki_section"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("wiki_page.id", ondelete="CASCADE"), nullable=False
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    anchor: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body_md: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))

    page: Mapped[WikiPage] = relationship(back_populates="sections")

    __table_args__ = (
        UniqueConstraint("page_id", "ord", name="uq_wiki_section_page_ord"),
        UniqueConstraint("page_id", "anchor", name="uq_wiki_section_page_anchor"),
    )


class WikiLink(Base):
    """Производный индекс рёбер. Пересобирается из wiki_page.body_md.
    НИКОГДА не редактируется руками."""

    __tablename__ = "wiki_link"

    from_page_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("wiki_page.id", ondelete="CASCADE"),
        primary_key=True,
    )
    from_section_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("wiki_section.id", ondelete="CASCADE"),
        primary_key=True,
    )
    to_slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    # Может быть NULL, если страница назначения ещё не существует (висячий [[link]]).
    to_page_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("wiki_page.id", ondelete="SET NULL")
    )

    __table_args__ = (
        Index("ix_wiki_link_to_slug", "to_slug"),
        Index("ix_wiki_link_to_page", "to_page_id"),
    )


class SectionProvenance(Base):
    __tablename__ = "section_provenance"

    section_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("wiki_section.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_chunk_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("source_chunk.id", ondelete="CASCADE"),
        primary_key=True,
    )
    contributed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


# --- Кандидаты сущностей ---

class EntityCandidate(Base):
    __tablename__ = "entity_candidate"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("source_doc.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("source_chunk.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    descriptor: Mapped[str] = mapped_column(Text, nullable=False, default="")
    statements: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    resolved_page_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("wiki_page.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))

    __table_args__ = (
        Index("ix_entity_candidate_chunk", "chunk_id"),
        Index("ix_entity_candidate_resolved_page", "resolved_page_id"),
    )


# --- Аббревиатуры ---

class Abbreviation(Base):
    __tablename__ = "abbreviation"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    short: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    expansion: Mapped[str | None] = mapped_column(Text)
    resolved_page_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("wiki_page.id", ondelete="SET NULL")
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_chunk_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="unresolved")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    __table_args__ = (
        UniqueConstraint("short", "expansion", name="uq_abbreviation_short_expansion"),
        CheckConstraint(
            "source IN ('inline','inferred','manual','seed')",
            name="ck_abbreviation_source",
        ),
        CheckConstraint(
            "status IN ('unresolved','candidate','confirmed')",
            name="ck_abbreviation_status",
        ),
        Index("ix_abbreviation_status", "status"),
    )


# --- Цикл обратной связи ---

class QueryGap(Base):
    __tablename__ = "query_gap"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    asked_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    resolved_section_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)), nullable=False, default=list
    )
    # Признак, что в запросе была неразрешённая аббревиатура.
    unresolved_abbr: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)), nullable=False, default=list
    )

    __table_args__ = (
        Index("ix_query_gap_resolved_at", "resolved_at"),
    )
