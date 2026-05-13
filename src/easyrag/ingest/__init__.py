"""Шаг 2: ingest-пайплайн.

Из исходного документа делает:
1. чанки (``source_chunk``) с char-offset'ами и эмбеддингами — для enrichment-loop;
2. кандидатов сущностей (``entity_candidate``) с эмбеддингами — для последующего
   резолва в wiki-страницы (шаг 4+).

Публичное API:
* :func:`chunk_text` — детерминированное разбиение на чанки.
* :func:`extract_entities` — извлечь сущности из текста через LLM.
* :func:`ingest_text` — оркестратор: пишет ``source_doc`` + чанки + кандидатов в БД.
"""
from easyrag.ingest.chunker import Chunk, chunk_text
from easyrag.ingest.extractor import ExtractedEntity, extract_entities
from easyrag.ingest.pipeline import IngestResult, ingest_text

__all__ = [
    "Chunk",
    "ExtractedEntity",
    "IngestResult",
    "chunk_text",
    "extract_entities",
    "ingest_text",
]
