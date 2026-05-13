"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-13

ВСЯ схема одним файлом — пока стабилизируется. Без ANN-индексов на vector-полях
(seq-scan + cosine достаточно при текущем объёме).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

from easyrag.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = get_settings().embed_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # source_doc
    op.create_table(
        "source_doc",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("uri", sa.Text, nullable=False),
        sa.Column("mime", sa.String(64)),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("sha256", sa.String(64)),
    )
    op.create_index("ix_source_doc_sha256", "source_doc", ["sha256"])

    # source_chunk
    op.create_table(
        "source_chunk",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("doc_id", pg.UUID(as_uuid=True), sa.ForeignKey("source_doc.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM)),
        sa.UniqueConstraint("doc_id", "ord", name="uq_source_chunk_doc_ord"),
    )

    # wiki_page
    op.create_table(
        "wiki_page",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("type", sa.String(64)),
        sa.Column("aliases", pg.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("body_md", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
    )
    op.create_index("ix_wiki_page_type", "wiki_page", ["type"])

    # wiki_section
    op.create_table(
        "wiki_section",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("page_id", pg.UUID(as_uuid=True), sa.ForeignKey("wiki_page.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("anchor", sa.String(255), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("body_md", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM)),
        sa.UniqueConstraint("page_id", "ord", name="uq_wiki_section_page_ord"),
        sa.UniqueConstraint("page_id", "anchor", name="uq_wiki_section_page_anchor"),
    )

    # wiki_link (производный индекс)
    op.create_table(
        "wiki_link",
        sa.Column(
            "from_page_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_page.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "from_section_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_section.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("to_slug", sa.String(255), primary_key=True),
        sa.Column(
            "to_page_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_page.id", ondelete="SET NULL"),
        ),
    )
    op.create_index("ix_wiki_link_to_slug", "wiki_link", ["to_slug"])
    op.create_index("ix_wiki_link_to_page", "wiki_link", ["to_page_id"])

    # section_provenance
    op.create_table(
        "section_provenance",
        sa.Column(
            "section_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_section.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "source_chunk_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("source_chunk.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("contributed_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # entity_candidate
    op.create_table(
        "entity_candidate",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("doc_id", pg.UUID(as_uuid=True), sa.ForeignKey("source_doc.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", pg.UUID(as_uuid=True), sa.ForeignKey("source_chunk.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("descriptor", sa.Text, nullable=False, server_default=""),
        sa.Column("statements", pg.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column(
            "resolved_page_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_page.id", ondelete="SET NULL"),
        ),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("embedding", Vector(EMBED_DIM)),
    )
    op.create_index("ix_entity_candidate_chunk", "entity_candidate", ["chunk_id"])
    op.create_index("ix_entity_candidate_resolved_page", "entity_candidate", ["resolved_page_id"])

    # abbreviation
    op.create_table(
        "abbreviation",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("short", sa.String(32), nullable=False),
        sa.Column("expansion", sa.Text),
        sa.Column(
            "resolved_page_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("wiki_page.id", ondelete="SET NULL"),
        ),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("evidence_chunk_ids", pg.ARRAY(pg.UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="unresolved"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("confirmed_at", sa.TIMESTAMP(timezone=True)),
        sa.UniqueConstraint("short", "expansion", name="uq_abbreviation_short_expansion"),
        sa.CheckConstraint(
            "source IN ('inline','inferred','manual','seed')", name="ck_abbreviation_source"
        ),
        sa.CheckConstraint(
            "status IN ('unresolved','candidate','confirmed')", name="ck_abbreviation_status"
        ),
    )
    op.create_index("ix_abbreviation_short", "abbreviation", ["short"])
    op.create_index("ix_abbreviation_status", "abbreviation", ["status"])

    # query_gap
    op.create_table(
        "query_gap",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM)),
        sa.Column("asked_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("resolved_section_ids", pg.ARRAY(pg.UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("unresolved_abbr", pg.ARRAY(sa.String(32)), nullable=False, server_default="{}"),
    )
    op.create_index("ix_query_gap_resolved_at", "query_gap", ["resolved_at"])


def downgrade() -> None:
    for table in (
        "query_gap",
        "abbreviation",
        "entity_candidate",
        "section_provenance",
        "wiki_link",
        "wiki_section",
        "wiki_page",
        "source_chunk",
        "source_doc",
    ):
        op.drop_table(table)
