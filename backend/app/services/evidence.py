from __future__ import annotations

import json
from typing import Any, List, Optional
from uuid import UUID

from app.db.pool import get_pool
from app.models.evidence import Evidence, EvidenceCreate
from app.utils.text_sanitize import sanitize_text


def _encode_metadata(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False)


def _decode_metadata(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded
        return None
    return None


async def list_evidence(
    paper_id: UUID,
    *,
    section_id: Optional[UUID] = None,
    concept_id: Optional[UUID] = None,
    relation_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Evidence]:
    pool = get_pool()
    conditions = ["paper_id = $1"]
    params: List[object] = [paper_id]
    placeholder_index = 2

    if section_id:
        conditions.append(f"section_id = ${placeholder_index}")
        params.append(section_id)
        placeholder_index += 1
    if concept_id:
        conditions.append(f"concept_id = ${placeholder_index}")
        params.append(concept_id)
        placeholder_index += 1
    if relation_id:
        conditions.append(f"relation_id = ${placeholder_index}")
        params.append(relation_id)
        placeholder_index += 1

    query = """
        SELECT id, paper_id, section_id, concept_id, relation_id, snippet, vector_id, embedding_model, score, metadata, created_at, updated_at
        FROM evidence
        WHERE {conditions}
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
    """.format(
        conditions=" AND ".join(conditions),
        limit_idx=placeholder_index,
        offset_idx=placeholder_index + 1,
    )
    params.extend([limit, offset])
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    normalized = []
    for row in rows:
        payload = dict(row)
        payload["metadata"] = _decode_metadata(payload.get("metadata"))
        normalized.append(Evidence(**payload))
    return normalized


async def create_evidence(data: EvidenceCreate) -> Evidence:
    pool = get_pool()
    query = """
        INSERT INTO evidence (paper_id, section_id, concept_id, relation_id, snippet, vector_id, embedding_model, score, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id, paper_id, section_id, concept_id, relation_id, snippet, vector_id, embedding_model, score, metadata, created_at, updated_at
    """
    metadata = _encode_metadata(data.metadata)
    snippet = sanitize_text(data.snippet)
    if not snippet:
        raise ValueError("Evidence snippet cannot be empty after sanitization")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            data.paper_id,
            data.section_id,
            data.concept_id,
            data.relation_id,
            snippet,
            data.vector_id,
            data.embedding_model,
            data.score,
            metadata,
        )
    payload = dict(row)
    payload["metadata"] = _decode_metadata(payload.get("metadata"))
    return Evidence(**payload)


async def delete_evidence_for_paper(paper_id: UUID) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM evidence WHERE paper_id = $1", paper_id)


async def delete_evidence_for_paper(paper_id: UUID) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM evidence WHERE paper_id = $1", paper_id)


async def get_evidence(evidence_id: UUID) -> Optional[Evidence]:
    pool = get_pool()
    query = """
        SELECT id, paper_id, section_id, concept_id, relation_id, snippet, vector_id, embedding_model, score, metadata, created_at, updated_at
        FROM evidence
        WHERE id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, evidence_id)
    if not row:
        return None
    payload = dict(row)
    payload["metadata"] = _decode_metadata(payload.get("metadata"))
    return Evidence(**payload)

