import asyncio

import asyncpg

from app.core.config import settings


async def test_postgres_connection() -> None:
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        await conn.execute("SELECT 1;")
    finally:
        await conn.close()


def test_postgres_connection_sync() -> None:
    asyncio.get_event_loop().run_until_complete(test_postgres_connection())

