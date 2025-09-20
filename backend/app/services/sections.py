from __future__ import annotations

from typing import List, Optional, Sequence
from uuid import UUID

from app.db.pool import get_pool
from app.models.section import Section, SectionCreate


async def list_sections(
    paper_id: UUID,
    *,
    limit: int = 100,
    offset: int = 0,
) -> List[Section]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, title, content, char_start, char_end, page_number, snippet, created_at, updated_at
        FROM sections
        WHERE paper_id = $1
        ORDER BY page_number NULLS LAST, char_start NULLS LAST, created_at
        LIMIT $2 OFFSET $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, paper_id, limit, offset)
    return [Section(**dict(row)) for row in rows]


async def create_section(data: SectionCreate) -> Section:
    pool = get_pool()
    query = """
        INSERT INTO sections (paper_id, title, content, char_start, char_end, page_number, snippet)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, paper_id, title, content, char_start, char_end, page_number, snippet, created_at, updated_at
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            data.paper_id,
            data.title,
            data.content,
            data.char_start,
            data.char_end,
            data.page_number,
            data.snippet,
        )
    return Section(**dict(row))


async def get_section(section_id: UUID) -> Optional[Section]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, title, content, char_start, char_end, page_number, snippet, created_at, updated_at
        FROM sections
        WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, section_id)
    return Section(**dict(row)) if row else None


async def replace_sections(paper_id: UUID, sections: Sequence[SectionCreate]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sections WHERE paper_id = $1", paper_id)
            if sections:
                values = [
                    (
                        section.paper_id,
                        section.title,
                        section.content,
                        section.char_start,
                        section.char_end,
                        section.page_number,
                        section.snippet,
                    )
                    for section in sections
                ]
                await conn.executemany(
                    """
                    INSERT INTO sections (
                        paper_id,
                        title,
                        content,
                        char_start,
                        char_end,
                        page_number,
                        snippet
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    values,
                )

