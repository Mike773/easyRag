from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from easyrag.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    # ВАЖНО: НЕ регистрируем `pgvector.asyncpg.register_vector` на соединениях
    # этого engine. `pgvector.sqlalchemy.Vector.bind_processor` всегда отдаёт
    # вектор строкой `'[..]'`, а бинарный кодек register_vector ждёт массив —
    # вместе они дают `DataError: could not convert string to float` при INSERT.
    # SQLAlchemy-результаты для Vector проходят через `_from_db`, который сам
    # разбирает строковую форму, так что отдельный asyncpg-кодек не нужен.
    return create_async_engine(get_settings().db_dsn, pool_pre_ping=True)


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
