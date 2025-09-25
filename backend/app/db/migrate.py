from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

import asyncpg

from app.core.config import settings


MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def apply_migrations() -> None:
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )

        applied = {
            r[0]
            for r in await conn.fetch("SELECT filename FROM schema_migrations ORDER BY id")
        }

        for sql_file in _iter_migration_files(MIGRATIONS_DIR):
            if sql_file.name in applied:
                continue
            sql = sql_file.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", sql_file.name
                )
    finally:
        await conn.close()


def _iter_migration_files(path: Path) -> Iterable[Path]:
    files = sorted(path.glob("*.sql"), key=lambda p: p.name)
    return files


def apply_migrations_sync() -> None:
    asyncio.get_event_loop().run_until_complete(apply_migrations())
