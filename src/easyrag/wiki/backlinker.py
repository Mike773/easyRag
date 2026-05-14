"""Back-link: дописать [[…]] ссылки в старые страницы после появления новых сущностей.

``wiki_link`` — производный индекс из ``body_md``. Если страница X была создана
до того, как в wiki появилась страница Y, и упоминает Y голым текстом, ссылки
``X → Y`` не существует, пока кто-нибудь не перепишет ``X.body_md``.

Эта функция запускается в конце каждого ingest'а: для всех существующих
страниц, кроме только что merged/created, она просит LLM проставить ``[[…]]``
ссылки на сущности, появившиеся в этом раунде. LLM работает по узкому
промпту :data:`WIKI_RELINK_SYSTEM` — он не имеет права менять формулировки и
добавлять факты.

Опциональный pre-filter (``settings.backlink_prefilter``) пропускает страницу,
если её ``body_md`` не содержит ни одного substring-совпадения с фрешими
title/alias — экономит «холостые» LLM-вызовы.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from easyrag.config import get_settings
from easyrag.db.models import WikiPage
from easyrag.llm.client import LLMClient, get_llm
from easyrag.llm.embeddings import EmbeddingClient, get_embeddings
from easyrag.query.prompts import (
    WIKI_RELINK_SCHEMA,
    WIKI_RELINK_SYSTEM,
    WIKI_RELINK_TOOL_DESCRIPTION,
    WIKI_RELINK_TOOL_NAME,
    build_relink_user_prompt,
)
from easyrag.wiki.markdown import parse_page, strip_self_links
from easyrag.wiki.merge_utils import (
    reembed_sections,
    restore_provenance_by_anchor,
    snapshot_provenance,
)
from easyrag.wiki.repository import upsert_page

# Алиасы/title короче этой длины игнорируются prefilter'ом — слишком высок
# риск ложных substring-совпадений ("ИП" внутри "график").
_MIN_FRESH_TOKEN_LEN = 3


@dataclass(frozen=True)
class BackfillResult:
    relinked: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def relinked_count(self) -> int:
        return len(self.relinked)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


async def backfill_links(
    session: AsyncSession,
    *,
    exclude_slugs: Iterable[str] = (),
    force: bool = False,
    llm: LLMClient | None = None,
    embeddings: EmbeddingClient | None = None,
) -> BackfillResult:
    """Прогнать relink-LLM по всем страницам, кроме ``exclude_slugs``.

    ``exclude_slugs`` — slug'и страниц, которые только что прошли
    create/merge в этом раунде resolve_candidates: они уже видели актуальный
    каталог и в обновлении не нуждаются.

    ``force=True`` — пропустить страховку «нет триггера → не работаем».
    Используется для standalone-вызова (``easyrag relink``) по всей wiki без
    свежего ingest'а.

    Каталог сущностей, отдаваемый LLM, — **полный** (вся wiki кроме самой
    страницы, с лимитом ``merge_catalog_limit``). То есть relink восстанавливает
    не только ссылки на свежие сущности этого раунда, но и любые недолинкованные
    связи на старые соседние страницы.

    Транзакцию ведёт caller — здесь только ``add`` / ``flush``.
    """
    settings = get_settings()
    if not settings.backlink_enabled:
        return BackfillResult()

    exclude_set = {s for s in exclude_slugs if s}
    if not exclude_set and not force:
        # Без триггера (ничего не создано/не merged) — не делаем лишних проходов.
        return BackfillResult()

    # Полный каталог wiki (с лимитом, как у merge): кандидаты-страницы тоже
    # видят остальные слаги, чтобы LLM мог дозалинковать давние пропуски.
    catalog_rows = (
        await session.execute(
            select(WikiPage.slug, WikiPage.title, WikiPage.aliases)
            .order_by(WikiPage.updated_at.desc())
            .limit(settings.merge_catalog_limit)
        )
    ).all()
    if not catalog_rows:
        return BackfillResult()

    catalog_by_slug: dict[str, tuple[str, list[str]]] = {
        slug: (title, list(aliases or [])) for slug, title, aliases in catalog_rows
    }
    catalog_strings_by_slug: dict[str, list[str]] = {
        slug: _strings_for(title, aliases)
        for slug, (title, aliases) in catalog_by_slug.items()
    }

    candidate_rows = (
        await session.execute(
            select(WikiPage).where(WikiPage.slug.notin_(exclude_set))
        )
    ).scalars().all()
    if not candidate_rows:
        return BackfillResult()

    llm_client = llm or get_llm()
    embedder = embeddings or get_embeddings()

    relinked: list[str] = []
    skipped: list[str] = []

    for page in candidate_rows:
        # Каталог для этой страницы — все слаги, кроме её самой. exclude_set
        # ограничивает только то, КОГО мы дёргаем backfill'ом (fresh-страницы
        # уже знают актуальный граф через свой свежий merge), но в каталог
        # для других страниц fresh-сущности обязательно должны входить —
        # ради них всё затевалось. Дополнительно отсекаем «похожие» slug'и:
        # если slug пересекается substring'ом со slug'ом самой страницы
        # (например, «медведь» vs «медведь-косолапый») — LLM любит обернуть
        # упоминание текущей сущности в ссылку на «похожую», что переписывает
        # narrative. Безопаснее не давать ему этого делать.
        page_catalog_slugs = {
            s for s in catalog_by_slug
            if s != page.slug and not _slugs_overlap(s, page.slug)
        }
        if not page_catalog_slugs:
            skipped.append(page.slug)
            continue

        relevant_slugs = _relevant_fresh_for_page(
            page,
            fresh_strings_by_slug={
                s: catalog_strings_by_slug[s] for s in page_catalog_slugs
            },
            prefilter=settings.backlink_prefilter,
        )
        if settings.backlink_prefilter and not relevant_slugs:
            skipped.append(page.slug)
            continue

        # Каталог, который мы отдадим LLM: либо отфильтрованный по
        # реальным упоминаниям, либо весь page_catalog_slugs.
        target_slugs = relevant_slugs if relevant_slugs else page_catalog_slugs
        catalog = [catalog_by_slug[slug] for slug in target_slugs]
        if not catalog:
            skipped.append(page.slug)
            continue

        prov_snapshot = await snapshot_provenance(session, page.id)
        current_body = page.body_md or ""
        current_aliases = list(page.aliases or [])

        user_prompt = build_relink_user_prompt(
            title=page.title,
            current_body=current_body,
            current_aliases=current_aliases,
            catalog=catalog,
        )
        raw = await llm_client.call_json(
            system=WIKI_RELINK_SYSTEM,
            user=user_prompt,
            tool_name=WIKI_RELINK_TOOL_NAME,
            tool_description=WIKI_RELINK_TOOL_DESCRIPTION,
            input_schema=WIKI_RELINK_SCHEMA,
        )

        new_body = _coerce_body(raw, fallback=current_body)
        new_body = strip_self_links(new_body, page_slug=page.slug)
        new_aliases = _coerce_aliases(raw, fallback=current_aliases)

        if not _has_link_change(current_body, new_body) and new_aliases == current_aliases:
            # Граф ссылок не изменился — текст мог дрогнуть на пробелах, не пишем.
            skipped.append(page.slug)
            continue

        updated = await upsert_page(
            session,
            slug=page.slug,
            title=page.title,
            body_md=new_body,
            aliases=new_aliases,
        )
        await reembed_sections(session, updated, embedder)
        await restore_provenance_by_anchor(session, updated.id, prov_snapshot)
        relinked.append(page.slug)

    await session.flush()
    return BackfillResult(relinked=tuple(relinked), skipped=tuple(skipped))


def _slugs_overlap(a: str, b: str) -> bool:
    """True если один slug является substring другого (например,
    ``медведь`` ↔ ``медведь-косолапый``).

    Используется, чтобы не отдавать в relink-каталог сущности с слишком
    близкими slug'ами к самой странице — LLM в таких случаях склонен
    перепутать «себя» с «соседом» и переписать narrative.
    """
    if not a or not b or a == b:
        return False
    return a in b or b in a


def _strings_for(title: str | None, aliases: list[str] | None) -> list[str]:
    """Сформировать набор «достаточно длинных» имён для substring-prefilter'а."""
    out: list[str] = []
    if title:
        t = title.strip()
        if len(t) >= _MIN_FRESH_TOKEN_LEN:
            out.append(t)
    for a in aliases or []:
        if not a:
            continue
        a = a.strip()
        if len(a) >= _MIN_FRESH_TOKEN_LEN:
            out.append(a)
    return out


def _relevant_fresh_for_page(
    page: WikiPage,
    *,
    fresh_strings_by_slug: dict[str, list[str]],
    prefilter: bool,
) -> set[str]:
    """Какие fresh-slug'и реально имеет смысл сватать этой странице.

    При выключенном prefilter возвращаем пустое множество (вызывающий код
    интерпретирует это как «отдай весь fresh-каталог»). При включённом —
    оставляем только те fresh-slug'и, чьи строки встречаются в body_md и
    которые ещё не залинкованы из этой страницы.
    """
    if not prefilter:
        return set()

    body = page.body_md or ""
    if not body:
        return set()
    body_lower = body.lower()

    parsed = parse_page(body)
    already_linked = {
        link.to_slug for section in parsed.sections for link in section.links
    }

    relevant: set[str] = set()
    for slug, strings in fresh_strings_by_slug.items():
        if slug in already_linked:
            continue
        for s in strings:
            if s.lower() in body_lower:
                relevant.add(slug)
                break
    return relevant


def _has_link_change(old_body: str, new_body: str) -> bool:
    """Проверить, изменился ли граф ссылок (a не только пробелы/пунктуация).

    Сравниваем множества ``(anchor, to_slug)`` обеих версий — это то, что в
    итоге окажется в ``wiki_link`` после ``upsert_page``.
    """
    old = {
        (s.anchor, link.to_slug)
        for s in parse_page(old_body).sections
        for link in s.links
    }
    new = {
        (s.anchor, link.to_slug)
        for s in parse_page(new_body).sections
        for link in s.links
    }
    return old != new


def _coerce_body(raw: dict | None, *, fallback: str) -> str:
    """Достать ``body_md`` из ответа LLM; на мусор — fallback на текущее тело.

    В отличие от resolver'а, у нас НЕТ права построить body из statements'ов —
    relink не должен вообще менять контент. Если LLM вернул пустоту/мусор,
    бережно возвращаем то, что было.
    """
    if isinstance(raw, dict):
        body = raw.get("body_md")
        if isinstance(body, str) and body.strip():
            return body.strip()
    return (fallback or "").strip()


def _coerce_aliases(raw: dict | None, *, fallback: list[str]) -> list[str]:
    """Достать ``aliases`` из ответа LLM; на мусор — отдать те, что были."""
    if isinstance(raw, dict):
        raw_aliases = raw.get("aliases")
        if isinstance(raw_aliases, list):
            cleaned = [
                a.strip()
                for a in raw_aliases
                if isinstance(a, str) and a and a.strip()
            ]
            if cleaned:
                return cleaned
    return list(fallback)


__all__ = ["BackfillResult", "backfill_links"]
