"""Шаг 4: резолв кандидатов в wiki + query-пайплайн.

Публичное API:

* :func:`resolve_candidates` — материализует ``entity_candidate`` в
  ``wiki_page`` / ``wiki_section`` / ``section_provenance``. Зовётся
  автоматически в конце :func:`easyrag.ingest.ingest_text`.
* :func:`answer_query` — ответ на вопрос пользователя по wiki с провенансом.
* :func:`retrieve_sections` — низкоуровневый retrieval (vector + graph),
  пригодится в шаге 5 для поиска gap'ов в исходных чанках.
"""
from easyrag.query.pipeline import (
    Citation,
    ChunkProvenance,
    DEFAULT_TOP_K,
    QueryResult,
    answer_query,
)
from easyrag.query.resolver import ResolveOutcome, ResolveResult, resolve_candidates
from easyrag.query.retrieval import RetrievedSection, retrieve_sections

__all__ = [
    "Citation",
    "ChunkProvenance",
    "DEFAULT_TOP_K",
    "QueryResult",
    "ResolveOutcome",
    "ResolveResult",
    "RetrievedSection",
    "answer_query",
    "resolve_candidates",
    "retrieve_sections",
]
