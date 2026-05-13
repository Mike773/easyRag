"""CLI entrypoint (Click).

Шаг 0: только `easyrag status`. Полноценные команды (ingest / query /
enrich / retype / resolve-abbr) появятся на шагах 2/4/5/6.
"""
import click

from easyrag import __version__


@click.group()
@click.version_option(__version__, prog_name="easyrag")
def cli() -> None:
    """easyrag — wiki-first RAG (work in progress)."""


@cli.command()
def status() -> None:
    """Показать статус CLI."""
    click.echo(
        f"easyrag {__version__} — step 0 scaffold. "
        "Команды ingest/query/enrich/retype ещё не реализованы. См. README."
    )


if __name__ == "__main__":
    cli()
