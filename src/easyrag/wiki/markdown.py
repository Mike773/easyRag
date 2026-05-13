"""Парсинг wiki-страницы.

``body_md`` страницы — markdown, разбитый на H2-секции (``## Заголовок``).
Каждая секция получает anchor — slug заголовка, уникальный в пределах страницы
(коллизии разрешаются суффиксом ``-2``, ``-3``...). Контент до первого H2,
если он не пустой, превращается в синтетическую секцию ``Overview``. Из тела
каждой секции извлекаются wiki-ссылки ``[[target]]`` или ``[[target|display]]``.

Парсер чистый (без БД), детерминированный — удобно тестировать в smoke-тестах
без Postgres.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from easyrag.wiki.slug import make_slug

# [[target]] или [[target|display]] — без вложенных скобок и переносов строк.
_LINK_RE = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|([^\[\]\n]+?))?]]")
# H2 в начале строки. Markdown допускает до 3 ведущих пробелов перед `#`,
# мы строже — заголовок должен начинаться с самого начала строки.
_H2_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# Fenced code (``` или ~~~) и inline-code (`...`). Ссылки/заголовки внутри
# code-блоков парсить не нужно: маскируем их пробелами, сохраняя позиции.
_FENCE_RE = re.compile(r"(?ms)^([ \t]{0,3})(```|~~~)[^\n]*\n.*?^\1\2[ \t]*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_OVERVIEW_TITLE = "Overview"


def _mask_code(text: str) -> str:
    """Заменить тело code-блоков пробелами, сохраняя длину строки.

    Парсер ищет H2-заголовки и ``[[ссылки]]`` по маскированной версии — это
    гасит ложные срабатывания на ``## fake`` внутри fenced-блока или
    ``[[Target]]`` внутри inline-кода. Срезы для ``section.body_md`` берутся
    из оригинала, поэтому контент code-блоков сохраняется в БД как есть.
    """

    def _blank(m: re.Match[str]) -> str:
        # Сохраняем переносы строк, чтобы построчные regex'ы (H2 ^##) не «склеивали» строки.
        return "".join(ch if ch == "\n" else " " for ch in m.group(0))

    masked = _FENCE_RE.sub(_blank, text)
    return _INLINE_CODE_RE.sub(_blank, masked)


@dataclass(frozen=True)
class ExtractedLink:
    target: str
    display: str
    to_slug: str


@dataclass(frozen=True)
class ParsedSection:
    ord: int
    title: str
    anchor: str
    body_md: str
    links: tuple[ExtractedLink, ...]


@dataclass(frozen=True)
class ParsedPage:
    sections: tuple[ParsedSection, ...]
    links: tuple[ExtractedLink, ...]


def extract_links(text: str) -> list[ExtractedLink]:
    """Найти все ``[[...]]`` в тексте в порядке появления.

    Содержимое fenced- и inline-кода игнорируется — это сахар для редкого
    случая, когда в narrative-тексте встречается ``разметка `[[X]]` ``.
    """
    if not text:
        return []
    masked = _mask_code(text)
    out: list[ExtractedLink] = []
    for m in _LINK_RE.finditer(masked):
        target = m.group(1).strip()
        if not target:
            continue
        display = (m.group(2) or target).strip()
        out.append(ExtractedLink(target=target, display=display, to_slug=make_slug(target)))
    return out


def parse_page(body_md: str) -> ParsedPage:
    body = (body_md or "").strip()
    # Маска используется только для поиска H2 / ссылок; срезы тел секций
    # берём из оригинала, чтобы code-блоки попадали в БД без искажений.
    masked = _mask_code(body)
    matches = list(_H2_RE.finditer(masked))

    raw: list[tuple[str, str]] = []
    if not matches:
        if body:
            raw.append((_OVERVIEW_TITLE, body))
    else:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            raw.append((_OVERVIEW_TITLE, preamble))
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_body = body[m.end():end].strip()
            raw.append((title, section_body))

    used: dict[str, int] = {}
    sections: list[ParsedSection] = []
    for ord_idx, (title, sbody) in enumerate(raw):
        base = make_slug(title) or f"section-{ord_idx + 1}"
        used[base] = used.get(base, 0) + 1
        anchor = base if used[base] == 1 else f"{base}-{used[base]}"
        sections.append(
            ParsedSection(
                ord=ord_idx,
                title=title,
                anchor=anchor,
                body_md=sbody,
                links=tuple(extract_links(sbody)),
            )
        )

    all_links: list[ExtractedLink] = []
    for s in sections:
        all_links.extend(s.links)

    return ParsedPage(sections=tuple(sections), links=tuple(all_links))


__all__ = [
    "ExtractedLink",
    "ParsedPage",
    "ParsedSection",
    "extract_links",
    "parse_page",
]
