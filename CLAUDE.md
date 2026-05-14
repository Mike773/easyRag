# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

`easyrag` is a **CLI-only** wiki-first RAG system. There is no HTTP/API layer — the only entrypoint is the Click CLI (`easyrag.cli:cli`, exposed as the `easyrag` script). Do not add FastAPI / uvicorn / web handlers.

Stack: Python 3.11, async SQLAlchemy 2.0, PostgreSQL + pgvector, LangChain (OpenAI **or** GigaChat — chosen independently for LLM and embeddings via env vars), Alembic for migrations.

## Commands

```bash
# Bring up Postgres + pgvector (listens on host port 5435)
docker compose up -d

# Install deps (editable) — uses uv.lock but `pip install -e '.[dev]'` works
pip install -e '.[dev]'

# Apply migrations (uses synchronous DSN; alembic env reads easyrag.config)
alembic upgrade head

# Run all tests (asyncio_mode=auto)
pytest

# Smoke tests only — no Postgres, no network (uses mock LLM + embeddings)
pytest tests/test_smoke.py -v

# Single test
pytest tests/test_ingest.py::test_name -v

# Lint
ruff check .

# Run the CLI
easyrag status
easyrag ingest --uri contracts/7 --file path/to/text.txt [--no-resolve] [--brief-window 4000]
easyrag query "вопрос" [--top-k 8] [--no-graph] [--no-show-sources]
```

Mock mode for tests is enabled via `EASYRAG_LLM_MOCK=1` and `EASYRAG_EMBED_MOCK=1` (set in `tests/conftest.py`); under mocks the LLM returns deterministic schema-shaped zeros and embeddings are deterministic.

## Architecture

The system is wiki-first: source documents are raw material, **wiki pages are the unit of retrieval**. Source chunks are kept only for provenance (and as fuel for the future enrichment loop). The `wiki_link` table is a *derived* index — never edit it by hand; it is rebuilt from `wiki_page.body_md` by `easyrag.wiki.repository.upsert_page`.

### Two pipelines

**Ingest** (`easyrag.ingest.pipeline.ingest_text`, invoked by `easyrag ingest`):
1. `sha256` the input text → dedup against `source_doc.sha256` (idempotent by content).
2. `analyze_document` (one LLM call on the first `EASYRAG_DOC_BRIEF_WINDOW` chars) builds a `DocumentBrief` (summary + entity types) that is then injected into every per-chunk extraction prompt. **The brief is the only adaptation lever** — extraction prompts must stay domain-agnostic (see `memory/feedback_extraction_domain_agnostic.md`).
3. `chunk_text` → embeddings (batched, 128/req) → `source_chunk`.
4. Per-chunk `extract_entities` → `EntityCandidate` rows (embedded on `name. descriptor`; statements deliberately excluded from the embedding).
5. If `resolve=True` (the default; `--no-resolve` skips this), `easyrag.query.resolver.resolve_candidates` materializes candidates into `wiki_page` / `wiki_section` / `section_provenance` via LLM-merge.

**Query** (`easyrag.query.pipeline.answer_query`, invoked by `easyrag query`):
1. Embed the question (same provider as ingest).
2. `retrieve_sections`: pgvector top-K cosine over `wiki_section.embedding`, optionally expanded via `wiki_link` (sections of linked pages, filtered by `EASYRAG_GRAPH_EXPAND_THRESH`).
3. Tool-call the LLM with `save_answer` (schema-constrained). Answer must cite `(slug, anchor)` pairs; citations that don't match retrieved sections are dropped (anti-hallucination).
4. For each matched citation, load `section_provenance → source_chunk → source_doc` and return uri + char offsets.
5. Always write a `query_gap` row; set `resolved_at` only when ≥1 valid citation came back. Unresolved gaps are fuel for the future enrichment step.

### Resolver — three-way candidate routing

`resolve_candidates` decides per-candidate target by, in order: (1) exact slug match against existing `wiki_page.slug`; (2) case-insensitive alias match (exactly one hit); (3) vector top-1 over `wiki_section.embedding`. The vector similarity hits one of three buckets:

- `≥ EASYRAG_RESOLVE_THRESH_HIGH` (0.85) → merge into existing page.
- between `RESOLVE_THRESH_LOW` (0.65) and HIGH → **ambiguous** — candidate stays in `entity_candidate` without materialization (NOT a failure; awaits future resolution).
- `< LOW` → create a new page with `make_slug(name)`.

Candidates are grouped by target slug and merged **one LLM call per slug**. After merge, `upsert_page` re-parses the markdown into sections and rebuilds `wiki_link`. The resolver then re-embeds **all** sections of the page (old embeddings were cascade-deleted) and writes `section_provenance` as the cross-product of (new sections × chunks that contributed in this round) — section-level attribution isn't preserved because the LLM rewrote section boundaries.

### LLM client — tool-binding pattern

All structured output goes through `LLMClient.call_json` (`easyrag.llm.client`). It uses LangChain's `bind_tools([...], tool_choice=name)` — the model "calls" a tool with the schema we pass, and we return its `args`. This is uniform across OpenAI and GigaChat. There's a GigaChat-specific fallback path that reads `additional_kwargs.function_call.arguments`. **Do not** parse free-form JSON from LLM text.

When defining a new structured call, follow the pattern in `easyrag/ingest/prompts.py` or `easyrag/query/prompts.py`: define `*_SYSTEM`, `*_TOOL_NAME`, `*_TOOL_DESCRIPTION`, `*_SCHEMA` (a raw JSON-schema dict, **not** a pydantic model — `convert_to_openai_tool` silently strips custom schemas off pydantic models), and a `build_*_user_prompt` helper.

### DB notes

- `pgvector.sqlalchemy.Vector` is used with the **async** engine. **Do not** register `pgvector.asyncpg.register_vector` on the engine connections — `Vector.bind_processor` ships vectors as `'[..]'` strings, and the binary asyncpg codec then fails on INSERT. See the warning in `easyrag/db/session.py`.
- Alembic uses the **sync** DSN (`db_dsn_sync`, `postgresql+psycopg://…`); the runtime uses the **async** DSN (`db_dsn`, `postgresql+asyncpg://…`). Both default to `localhost:5435` (the docker-compose mapping).
- Transactions are managed by the caller via `session_scope()` (`db/session.py`); pipeline functions only `add` / `flush`, never commit.
- `EMBED_DIM` is read from settings at import time of `db/models.py` and frozen into the `Vector(EMBED_DIM)` column. Changing `EASYRAG_EMBED_DIM` requires a fresh migration.

### Config

All settings live in `easyrag.config.Settings` (pydantic-settings, prefix `EASYRAG_`, reads `.env`). Some keys (provider credentials) accept both prefixed and unprefixed names via `AliasChoices`: `OPENAI_API_KEY` / `EASYRAG_OPENAI_API_KEY`, `EMBED_API_KEY` / `EASYRAG_EMBED_API_KEY`, `GIGACHAT_CREDENTIALS` / `EASYRAG_GIGACHAT_CREDENTIALS`. LLM and embedding providers are independent (`EASYRAG_LLM_PROVIDER` vs `EASYRAG_EMBED_PROVIDER`).

## Project status

Per the README: step 4 done (ingest with auto-resolve + query with provenance). Steps 3 (abbreviations), 5 (enrichment loop driven by `query_gap`), and 6 (emergent typing) are planned but not implemented. The packages `abbreviations/`, `enrichment/`, `typing_/` referenced in the README don't exist yet. The DB models for these (`Abbreviation`, `QueryGap`) are already in place.
