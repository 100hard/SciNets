from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from app.db.pool import get_pool
from app.models.paper import Paper, PaperCreate


PAPER_COLUMNS = (
    "id, title, authors, venue, year, status, file_path, file_name, file_size, "
    "file_content_type, created_at, updated_at"
)


async def list_papers(limit: int = 50, offset: int = 0, q: Optional[str] = None) -> List[Paper]:
    pool = get_pool()
    query = f"SELECT {PAPER_COLUMNS} FROM papers"
    params: list[Any] = []
    param_idx = 1
    if q:
        query += f" WHERE title ILIKE ${param_idx}"
        params.append(f"%{q}%")
        param_idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [Paper(**dict(row)) for row in rows]


async def create_paper(data: PaperCreate) -> Paper:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO papers (
                title,
                authors,
                venue,
                year,
                status,
                file_path,
                file_name,
                file_size,
                file_content_type
            )
            VALUES (
                $1,
                $2,
                $3,
                $4,
                COALESCE($5, 'uploaded'),
                $6,
                $7,
                $8,
                $9
            )
            RETURNING
                id,
                title,
                authors,
                venue,
                year,
                status,
                file_path,
                file_name,
                file_size,
                file_content_type,
                created_at,
                updated_at
            """,
            data.title,
            data.authors,
            data.venue,
            data.year,
            data.status,
            data.file_path,
            data.file_name,
            data.file_size,
            data.file_content_type,
        )
    return Paper(**dict(row))


async def get_paper(paper_id: UUID) -> Optional[Paper]:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {PAPER_COLUMNS} FROM papers WHERE id=$1",
            paper_id,
        )
    return Paper(**dict(row)) if row else None

