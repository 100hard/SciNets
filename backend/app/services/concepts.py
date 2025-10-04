from __future__ import annotations

import json
from typing import Any, List, Sequence
from uuid import UUID

from app.db.pool import get_pool
from app.models.concept import Concept, ConceptCreate


INSERT_CONCEPT_QUERY = """
    INSERT INTO concepts (paper_id, name, type, description, aliases, metadata)
    VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
    RETURNING id, paper_id, name, type, description, aliases, metadata, created_at, updated_at
"""


def _clean_aliases(raw_aliases: Any) -> list[str]:
    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, list):
        return [alias.strip() for alias in raw_aliases if isinstance(alias, str) and alias.strip()]
    if isinstance(raw_aliases, str):
        try:
            decoded = json.loads(raw_aliases)
        except json.JSONDecodeError:
            candidate = raw_aliases.strip()
            return [candidate] if candidate else []
        if isinstance(decoded, list):
            return [alias.strip() for alias in decoded if isinstance(alias, str) and alias.strip()]
        if isinstance(decoded, str):
            candidate = decoded.strip()
            return [candidate] if candidate else []
    return []


def _clean_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, str):
        try:
            decoded = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return dict(decoded)
    return {}


def _concept_from_row(row: Any) -> Concept:
    payload = dict(row)
    payload["aliases"] = _clean_aliases(payload.get("aliases"))
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    return Concept(**payload)


async def list_concepts(
    paper_id: UUID,
    *,
    limit: int = 100,
    offset: int = 0,
) -> List[Concept]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, name, type, description, aliases, metadata, created_at, updated_at
        FROM concepts
        WHERE paper_id = $1
        ORDER BY name
        LIMIT $2 OFFSET $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, paper_id, limit, offset)
    return [_concept_from_row(row) for row in rows]


async def create_concept(data: ConceptCreate) -> Concept:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            INSERT_CONCEPT_QUERY,
            data.paper_id,
            data.name,
            data.type,
            data.description,
            json.dumps(data.aliases),
            json.dumps(data.metadata),
        )
    return _concept_from_row(row)


async def get_concept(concept_id: UUID) -> Optional[Concept]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, name, type, description, aliases, metadata, created_at, updated_at
        FROM concepts
        WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, concept_id)
    return _concept_from_row(row) if row else None


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
                    json.dumps(concept.aliases),
                    json.dumps(concept.metadata),
                )
                inserted.append(_concept_from_row(row))
    return inserted