"""Запись wiki-страниц в БД и пересборка link-индекса.

``wiki_link`` — производный индекс рёбер: всегда строится из ``wiki_page.body_md``
и никогда не редактируется руками (см. ``db/models.py``). Любая операция,
меняющая markdown страницы, обязана пересобрать её рёбра здесь же.

Контракт:
* :func:`upsert_page` — единая точка записи страницы. Парсит markdown в секции,
  переписывает их в БД, пересобирает ``wiki_link`` для страницы и подтягивает
  «висячие» ссылки с других страниц на новосозданный slug.
* :func:`rebuild_page_links` — пересобрать рёбра только для одной страницы
  (например, после внешней миграции body_md).
* :func:`rebuild_all_links` — полностью переинициализировать таблицу
  ``wiki_link`` из текущего состояния markdown. Чинит индекс после ручных
  правок БД.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from easyrag.db.models import WikiLink, WikiPage, WikiSection
from easyrag.wiki.markdown import ParsedPage, parse_page


async def upsert_page(
    session: AsyncSession,
    *,
    slug: str,
    title: str,
    body_md: str,
    type_: str | None = None,
    aliases: Sequence[str] = (),
) -> WikiPage:
    """Создать или обновить страницу и пересинхронизировать секции + рёбра.

    Возвращает persistent ``WikiPage`` (already in session). Транзакцию вызывающий
    код управляет сам (commit/rollback).

    Семантика опциональных полей на UPDATE-пути:
    * ``aliases`` всегда перезаписывается (передавайте полный набор; ``()``
      очищает).
    * ``type_=None`` означает «не трогать существующее значение». Чтобы очистить
      тип, передайте пустую строку (а не ``None``).
    """
    aliases_list = list(aliases)
    parsed = parse_page(body_md)

    page = (
        await session.execute(select(WikiPage).where(WikiPage.slug == slug))
    ).scalar_one_or_none()

    if page is None:
        page = WikiPage(
            slug=slug,
            title=title,
            body_md=body_md,
            type=type_,
            aliases=aliases_list,
        )
        session.add(page)
        await session.flush()  # получить page.id для секций
    else:
        page.title = title
        page.body_md = body_md
        page.aliases = aliases_list
        if type_ is not None:
            page.type = type_
        page.version = (page.version or 0) + 1
        # Сносим старые секции — CASCADE удалит привязанные wiki_link.
        await session.execute(
            delete(WikiSection).where(WikiSection.page_id == page.id)
        )
        await session.flush()

    # Этот блок выполняется на обоих путях (INSERT/UPDATE): без flush ниже
    # `_insert_links_for_page` не увидит свежие секции при SELECT.
    for ps in parsed.sections:
        session.add(
            WikiSection(
                page_id=page.id,
                ord=ps.ord,
                anchor=ps.anchor,
                title=ps.title,
                body_md=ps.body_md,
            )
        )
    await session.flush()

    await _insert_links_for_page(session, page, parsed)
    await _resolve_dangling_links_for_slug(session, page.slug, page.id)
    return page


async def rebuild_page_links(session: AsyncSession, page: WikiPage) -> None:
    """Пересобрать ``wiki_link`` для одной страницы по её текущему ``body_md``.

    Также подтягивает висячие ссылки чужих страниц, указывающие на этот slug —
    после ручного добавления страницы вызов ``rebuild_page_links(...)`` должен
    приводить wiki к консистентному состоянию без последующих исправлений.
    """
    parsed = parse_page(page.body_md)
    await session.execute(
        delete(WikiLink).where(WikiLink.from_page_id == page.id)
    )
    await session.flush()
    await _insert_links_for_page(session, page, parsed)
    await _resolve_dangling_links_for_slug(session, page.slug, page.id)


async def rebuild_all_links(session: AsyncSession) -> None:
    """Полная переинициализация ``wiki_link`` из markdown всех страниц.

    Используется как «починка» после ручных правок БД или миграций. Слегка
    дорого (O(N) страниц), но идемпотентно. Висячие ссылки разрешаются
    автоматически: каждая вставка проходит через ``_insert_links_for_page``,
    который заполняет ``to_page_id`` из карты slug→id по существующим страницам.
    """
    await session.execute(delete(WikiLink))
    await session.flush()
    pages = (await session.execute(select(WikiPage))).scalars().all()
    for page in pages:
        parsed = parse_page(page.body_md)
        await _insert_links_for_page(session, page, parsed)


async def _resolve_dangling_links_for_slug(
    session: AsyncSession, slug: str, page_id: object
) -> None:
    """Заполнить ``to_page_id`` у строк ``wiki_link``, ссылающихся на ``slug``.

    Условие `to_page_id IS NULL` гарантирует, что мы не затрём уже разрешённые
    ссылки (на случай гонок / повторных вызовов).
    """
    await session.execute(
        update(WikiLink)
        .where(WikiLink.to_slug == slug, WikiLink.to_page_id.is_(None))
        .values(to_page_id=page_id)
    )


async def _insert_links_for_page(
    session: AsyncSession, page: WikiPage, parsed: ParsedPage
) -> None:
    """Вставить ``wiki_link`` строки для распарсенных секций страницы.

    Предполагает, что секции уже записаны (есть id). ``to_page_id`` подставляется
    по существующим страницам; неизвестные slug'и остаются NULL (висячие ссылки).
    """
    if not parsed.sections:
        return

    sections = (
        await session.execute(
            select(WikiSection).where(WikiSection.page_id == page.id)
        )
    ).scalars().all()
    section_by_anchor = {s.anchor: s for s in sections}

    referenced = {link.to_slug for ps in parsed.sections for link in ps.links}
    slug_to_id: dict[str, object] = {}
    if referenced:
        rows = (
            await session.execute(
                select(WikiPage.slug, WikiPage.id).where(WikiPage.slug.in_(referenced))
            )
        ).all()
        slug_to_id = {s: i for s, i in rows}

    for ps in parsed.sections:
        sec = section_by_anchor.get(ps.anchor)
        if sec is None:  # invariant: section just inserted
            continue
        # PK = (from_page_id, from_section_id, to_slug) — дедуплицируем в секции.
        # Две ссылки на один slug в одной секции схлопываются; ссылки на один
        # slug из разных секций сохраняются как отдельные строки.
        seen: set[str] = set()
        for link in ps.links:
            if link.to_slug in seen:
                continue
            seen.add(link.to_slug)
            session.add(
                WikiLink(
                    from_page_id=page.id,
                    from_section_id=sec.id,
                    to_slug=link.to_slug,
                    to_page_id=cast("object", slug_to_id.get(link.to_slug)),
                )
            )
    await session.flush()


__all__ = [
    "rebuild_all_links",
    "rebuild_page_links",
    "upsert_page",
]
