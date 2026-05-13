"""Стабильный slug для wiki — ASCII, lowercase, dash-separated.

Используется и для ``wiki_page.slug`` (первичный ключ страницы для линковки),
и для ``wiki_section.anchor`` (уникален в пределах страницы). Slug всегда
получается не пустым: если ``python-slugify`` вернул пустую строку
(например, на чисто-символьном вводе), берётся детерминированный хеш.
"""
from __future__ import annotations

import hashlib

from slugify import slugify

_MAX_LEN = 200


def make_slug(text: str, *, max_length: int = _MAX_LEN) -> str:
    s = slugify(text or "", lowercase=True, max_length=max_length)
    if s:
        return s
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    return f"page-{digest}"


__all__ = ["make_slug"]
