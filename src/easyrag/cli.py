"""CLI entrypoint (Click).

На шаге 2 появилась команда ``easyrag ingest``. Остальные (query / enrich /
retype / resolve-abbr) добавятся на шагах 4/5/6.
"""
import asyncio
import mimetypes
from pathlib import Path

import click

from easyrag import __version__
from easyrag.db.session import session_scope
from easyrag.ingest import ingest_text


@click.group()
@click.version_option(__version__, prog_name="easyrag")
def cli() -> None:
    """easyrag — wiki-first RAG (work in progress)."""


@cli.command()
def status() -> None:
    """Показать статус CLI."""
    click.echo(
        f"easyrag {__version__} — step 2. "
        "Доступно: ingest. Команды query/enrich/retype ещё не реализованы."
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
def ingest_cmd(uri: str, file_path: Path, mime: str | None) -> None:
    """Прочитать файл, нарезать на чанки и извлечь сущности."""
    text = file_path.read_text(encoding="utf-8")
    resolved_mime = mime or (mimetypes.guess_type(str(file_path))[0])
    result = asyncio.run(_run_ingest(uri=uri, text=text, mime=resolved_mime))
    if result.deduplicated:
        click.echo(
            f"already ingested: doc_id={result.doc_id} "
            f"chunks={result.chunk_count} entities={result.entity_count}"
        )
    else:
        click.echo(
            f"ingested: doc_id={result.doc_id} "
            f"chunks={result.chunk_count} entities={result.entity_count}"
        )


async def _run_ingest(*, uri: str, text: str, mime: str | None):
    async with session_scope() as session:
        return await ingest_text(session, uri=uri, text=text, mime=mime)


if __name__ == "__main__":
    cli()
