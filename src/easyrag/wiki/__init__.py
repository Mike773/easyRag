"""Шаг 1: wiki-markdown и link-индекс.

Публичное API:
* :func:`make_slug` — стабильный slug из произвольного текста.
* :func:`parse_page` — разбить body_md на H2-секции и извлечь wiki-ссылки.
* :func:`extract_links` — найти все ``[[link]]`` в произвольной строке.
* :func:`upsert_page` / :func:`rebuild_page_links` / :func:`rebuild_all_links` —
  репозиторные операции, синхронизирующие markdown с таблицами
  ``wiki_page`` / ``wiki_section`` / ``wiki_link``.
"""
from easyrag.wiki.markdown import (
    ExtractedLink,
    ParsedPage,
    ParsedSection,
    extract_links,
    parse_page,
)
from easyrag.wiki.repository import (
    rebuild_all_links,
    rebuild_page_links,
    upsert_page,
)
from easyrag.wiki.slug import make_slug

__all__ = [
    "ExtractedLink",
    "ParsedPage",
    "ParsedSection",
    "extract_links",
    "make_slug",
    "parse_page",
    "rebuild_all_links",
    "rebuild_page_links",
    "upsert_page",
]
