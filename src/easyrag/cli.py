"""CLI entrypoint (Click).

На шаге 4 доступны команды ``ingest`` (с авто-резолвом в wiki) и ``query``
(ответ по wiki с цитатами). Команды enrich / retype / resolve-abbr добавятся
на шагах 5/6.
"""
import asyncio
import mimetypes
from pathlib import Path

import click

from easyrag import __version__
from easyrag.config import get_settings
from easyrag.db.session import session_scope
from easyrag.ingest import ingest_text
from easyrag.query import DEFAULT_TOP_K, answer_query


@click.group()
@click.version_option(__version__, prog_name="easyrag")
def cli() -> None:
    """easyrag — wiki-first RAG (work in progress)."""


@cli.command()
def status() -> None:
    """Показать статус CLI."""
    click.echo(
        f"easyrag {__version__} — step 4. "
        "Доступно: ingest (с авто-резолвом), query. "
        "Команды enrich/retype/resolve-abbr ещё не реализованы."
    )


@cli.command("ingest")
@click.option("--uri", required=True, help="Логическое имя источника (любая строка).")
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Путь к текстовому файлу для загрузки.",
)
@click.option(
    "--mime",
    default=None,
    help="MIME-тип источника. Если не задан — определяется по расширению.",
)
@click.option(
    "--no-resolve",
    is_flag=True,
    default=False,
    help="Не запускать LLM-merge кандидатов в wiki (оставить ingest сырым).",
)
@click.option(
    "--brief-window",
    type=int,
    default=None,
    help=(
        "Сколько символов от начала документа отдать в analyze_document. "
        "По умолчанию — значение из настроек (EASYRAG_DOC_BRIEF_WINDOW, обычно 4000). "
        "Поставьте 0, чтобы пропустить шаг brief (extraction пойдёт без подсказок)."
    ),
)
def ingest_cmd(
    uri: str,
    file_path: Path,
    mime: str | None,
    no_resolve: bool,
    brief_window: int | None,
) -> None:
    """Прочитать файл, нарезать на чанки, извлечь сущности и материализовать wiki."""
    text = file_path.read_text(encoding="utf-8")
    resolved_mime = mime or (mimetypes.guess_type(str(file_path))[0])
    settings = get_settings()
    window = brief_window if brief_window is not None else settings.doc_brief_window
    result = asyncio.run(
        _run_ingest(
            uri=uri,
            text=text,
            mime=resolved_mime,
            resolve=not no_resolve,
            brief_window=window,
        )
    )
    prefix = "already ingested" if result.deduplicated else "ingested"
    click.echo(
        f"{prefix}: doc_id={result.doc_id} "
        f"chunks={result.chunk_count} entities={result.entity_count}"
    )
    if not result.deduplicated and not no_resolve:
        click.echo(
            f"resolved: candidates={result.resolved_candidate_count} "
            f"created={len(result.created_pages)} merged={len(result.merged_pages)} "
            f"ambiguous={result.ambiguous_candidate_count}"
        )
        if result.created_pages:
            click.echo("  new pages: " + ", ".join(result.created_pages))
        if result.merged_pages:
            click.echo("  merged into: " + ", ".join(result.merged_pages))


@cli.command("query")
@click.argument("question")
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Сколько wiki-секций тянуть вектор-поиском.",
)
@click.option(
    "--no-graph",
    is_flag=True,
    default=False,
    help="Отключить расширение по wiki_link.",
)
@click.option(
    "--show-sources/--no-show-sources",
    default=True,
    help="Печатать список цитированных секций и их провенанс.",
)
def query_cmd(question: str, top_k: int, no_graph: bool, show_sources: bool) -> None:
    """Ответить на вопрос пользователя по wiki."""
    result = asyncio.run(
        _run_query(question=question, top_k=top_k, graph_expand=not no_graph)
    )
    click.echo(result.answer)

    if show_sources and result.citations:
        click.echo("\nИсточники:")
        for c in result.citations:
            sec = c.section
            click.echo(
                f"  • [[{sec.slug}#{sec.anchor}]] {sec.page_title} → {sec.section_title} "
                f"(sim={sec.similarity:.2f}, {sec.source})"
            )
            for cp in c.chunks:
                click.echo(
                    f"     ↳ {cp.uri} [chars {cp.char_start}–{cp.char_end}]"
                )

    if result.gap:
        click.echo("\n(в wiki не нашлось данных для уверенного ответа)")


async def _run_ingest(
    *,
    uri: str,
    text: str,
    mime: str | None,
    resolve: bool,
    brief_window: int,
):
    async with session_scope() as session:
        return await ingest_text(
            session,
            uri=uri,
            text=text,
            mime=mime,
            resolve=resolve,
            brief_window=brief_window,
        )


async def _run_query(*, question: str, top_k: int, graph_expand: bool):
    async with session_scope() as session:
        return await answer_query(
            session,
            question=question,
            top_k=top_k,
            graph_expand=graph_expand,
        )


if __name__ == "__main__":
    cli()
