from __future__ import annotations

from typing import List, Optional, Sequence
from uuid import UUID

from app.db.pool import get_pool
from app.models.concept import Concept, ConceptCreate


from typing import Optional
INSERT_CONCEPT_QUERY = """
    INSERT INTO concepts (paper_id, name, type, description)
    VALUES ($1, $2, $3, $4)
    RETURNING id, paper_id, name, type, description, created_at, updated_at
"""


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
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            INSERT_CONCEPT_QUERY,
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


async def delete_concepts_for_paper(paper_id: UUID) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM concepts WHERE paper_id = $1", paper_id)


async def replace_concepts(
    paper_id: UUID, concepts: Sequence[ConceptCreate]
) -> List[Concept]:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM concepts WHERE paper_id = $1", paper_id)
            if not concepts:
                return []

            inserted: List[Concept] = []
            for concept in concepts:
                row = await conn.fetchrow(
                    INSERT_CONCEPT_QUERY,
                    concept.paper_id,
                    concept.name,
                    concept.type,
                    concept.description,
                )
                inserted.append(Concept(**dict(row)))
    return inserted