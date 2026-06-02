from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool


pool: AsyncConnectionPool | None = None


async def init_pool(database_url: str) -> None:
    global pool
    if pool is None:
        pool = AsyncConnectionPool(conninfo=database_url, open=False, min_size=1, max_size=10)
        await pool.open()


async def close_pool() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None


@asynccontextmanager
async def get_conn():
    if pool is None:
        raise RuntimeError("Database pool is not initialized")
    async with pool.connection() as conn:
        yield conn
