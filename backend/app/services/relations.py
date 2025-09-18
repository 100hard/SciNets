from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from app.db.pool import get_pool
from app.models.relation import Relation, RelationCreate


async def list_relations(
    paper_id: UUID,
    *,
    concept_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Relation]:
    pool = get_pool()
    base_query = """
        SELECT id, paper_id, concept_id, related_concept_id, section_id, relation_type, description, created_at, updated_at
        FROM relations
        WHERE paper_id = $1
    """
    params: List[object] = [paper_id]
    if concept_id:
        base_query += " AND concept_id = $2"
        params.append(concept_id)
    base_query += " ORDER BY created_at DESC LIMIT $%d OFFSET $%d" % (len(params) + 1, len(params) + 2)
    params.extend([limit, offset])
    async with pool.acquire() as conn:
        rows = await conn.fetch(base_query, *params)
    return [Relation(**dict(row)) for row in rows]


async def create_relation(data: RelationCreate) -> Relation:
    pool = get_pool()
    query = """
        INSERT INTO relations (paper_id, concept_id, related_concept_id, section_id, relation_type, description)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, paper_id, concept_id, related_concept_id, section_id, relation_type, description, created_at, updated_at
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            data.paper_id,
            data.concept_id,
            data.related_concept_id,
            data.section_id,
            data.relation_type,
            data.description,
        )
    return Relation(**dict(row))


async def get_relation(relation_id: UUID) -> Optional[Relation]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, concept_id, related_concept_id, section_id, relation_type, description, created_at, updated_at
        FROM relations
        WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, relation_id)
    return Relation(**dict(row)) if row else None

