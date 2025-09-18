from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from app.db.pool import get_pool
from app.models.concept import Concept, ConceptCreate


async def list_concepts(
    paper_id: UUID,
    *,
    limit: int = 100,
    offset: int = 0,
) -> List[Concept]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, name, type, description, created_at, updated_at
        FROM concepts
        WHERE paper_id = $1
        ORDER BY name
        LIMIT $2 OFFSET $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, paper_id, limit, offset)
    return [Concept(**dict(row)) for row in rows]


async def create_concept(data: ConceptCreate) -> Concept:
    pool = get_pool()
    query = """
        INSERT INTO concepts (paper_id, name, type, description)
        VALUES ($1, $2, $3, $4)
        RETURNING id, paper_id, name, type, description, created_at, updated_at
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            data.paper_id,
            data.name,
            data.type,
            data.description,
        )
    return Concept(**dict(row))


async def get_concept(concept_id: UUID) -> Optional[Concept]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, name, type, description, created_at, updated_at
        FROM concepts
        WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, concept_id)
    return Concept(**dict(row)) if row else None

