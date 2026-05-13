from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from easyrag.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    engine = create_async_engine(get_settings().db_dsn, pool_pre_ping=True)

    # Регистрируем pgvector-кодек на каждом новом asyncpg-соединении пула.
    # Без этого SELECT поля Vector упадёт: asyncpg не знает тип `vector`.
    @event.listens_for(engine.sync_engine, "connect")
    def _register_pgvector(dbapi_conn, _connection_record) -> None:
        dbapi_conn.run_async(lambda c: register_vector(c))

    return engine


@lru_cache(maxsize=1)
def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Транзакционный scope. Коммит при успехе, rollback при исключении."""
    session = _session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
