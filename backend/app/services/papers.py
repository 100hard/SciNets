from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from app.db.pool import get_pool
from app.models.paper import Paper, PaperCreate


async def list_papers(limit: int = 50, offset: int = 0, q: Optional[str] = None) -> List[Paper]:
    pool = get_pool()
    query = "SELECT id, title, authors, venue, year, status, created_at, updated_at FROM papers"
    params: list[Any] = []
    if q:
        query += " WHERE title ILIKE $1"
        params.append(f"%{q}%")
    query += " ORDER BY created_at DESC LIMIT $2 OFFSET $3" if q else " ORDER BY created_at DESC LIMIT $1 OFFSET $2"
    if q:
        params.extend([limit, offset])
    else:
        params.extend([limit, offset])
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [Paper(**dict(row)) for row in rows]


async def create_paper(data: PaperCreate) -> Paper:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO papers (title, authors, venue, year, status)
            VALUES ($1, $2, $3, $4, COALESCE($5, 'uploaded'))
            RETURNING id, title, authors, venue, year, status, created_at, updated_at
            """,
            data.title,
            data.authors,
            data.venue,
            data.year,
            data.status,
        )
    return Paper(**dict(row))


async def get_paper(paper_id: UUID) -> Optional[Paper]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, title, authors, venue, year, status, created_at, updated_at FROM papers WHERE id=$1",
            paper_id,
        )
    return Paper(**dict(row)) if row else None

