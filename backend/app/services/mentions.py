from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from uuid import UUID

from app.db.pool import get_pool
from app.models.ontology import ConceptResolutionType


@dataclass
class MentionObservation:
    resolution_type: ConceptResolutionType
    entity_id: UUID
    paper_id: UUID
    section_id: Optional[UUID]
    surface: str
    mention_type: Optional[str]
    snippet: Optional[str]
    start: Optional[int]
    end: Optional[int]
    source: Optional[str]
    first_seen_year: Optional[int]


def _normalize_surface(value: str) -> str:
    normalized = " ".join(value.strip().casefold().split())
    return normalized


def _is_acronym(surface: str) -> bool:
    cleaned = surface.strip()
    if len(cleaned) < 2 or len(cleaned) > 16:
        return False
    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if not alpha_chars:
        return False
    return cleaned.upper() == cleaned and all(ch.isalpha() for ch in cleaned if ch.isalpha())


def _has_digit(surface: str) -> bool:
    return any(ch.isdigit() for ch in surface)


def _build_context_embedding(snippet: Optional[str]) -> Optional[list[float]]:
    if not snippet:
        return None
    length = float(len(snippet))
    if length == 0:
        return [0.0, 0.0, 0.0]
    uppercase = sum(1 for ch in snippet if ch.isupper())
    digits = sum(1 for ch in snippet if ch.isdigit())
    spaces = sum(1 for ch in snippet if ch.isspace())
    return [
        length,
        float(uppercase) / length,
        float(digits + spaces) / length,
    ]


def _mark_shared_flags(records: Sequence[dict[str, object]]) -> None:
    surface_index: dict[tuple[str, str], set[UUID]] = {}
    for record in records:
        normalized_surface = record.get("normalized_surface")
        resolution_type = record.get("resolution_type")
        entity_id = record.get("entity_id")
        if not isinstance(normalized_surface, str) or not normalized_surface:
            continue
        if not isinstance(resolution_type, str) or not isinstance(entity_id, UUID):
            continue
        key = (resolution_type, normalized_surface)
        surface_index.setdefault(key, set()).add(entity_id)

    for record in records:
        normalized_surface = record.get("normalized_surface")
        resolution_type = record.get("resolution_type")
        if not isinstance(normalized_surface, str) or not normalized_surface:
            record["is_shared"] = False
            continue
        if not isinstance(resolution_type, str):
            record["is_shared"] = False
            continue
        key = (resolution_type, normalized_surface)
        record["is_shared"] = len(surface_index.get(key, set())) > 1


async def replace_mentions_for_paper(
    paper_id: UUID, mentions: Iterable[MentionObservation]
) -> None:
    prepared: list[dict[str, object]] = []
    for mention in mentions:
        surface = (mention.surface or "").strip()
        if not surface:
            continue
        normalized_surface = _normalize_surface(surface)
        record = {
            "resolution_type": mention.resolution_type.value,
            "entity_id": mention.entity_id,
            "paper_id": mention.paper_id,
            "section_id": mention.section_id,
            "surface": surface,
            "normalized_surface": normalized_surface,
            "mention_type": mention.mention_type,
            "context_snippet": mention.snippet,
            "evidence_start": mention.start,
            "evidence_end": mention.end,
            "context_embedding": _build_context_embedding(mention.snippet),
            "first_seen_year": mention.first_seen_year,
            "is_acronym": _is_acronym(surface),
            "has_digit": _has_digit(surface),
            "is_shared": False,  # set below
            "source": mention.source,
        }
        prepared.append(record)

    if not prepared:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM ontology_mentions WHERE paper_id = $1", paper_id)
        return

    _mark_shared_flags(prepared)

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM ontology_mentions WHERE paper_id = $1", paper_id)
            await conn.executemany(
                """
                INSERT INTO ontology_mentions (
                    resolution_type,
                    entity_id,
                    paper_id,
                    section_id,
                    surface,
                    normalized_surface,
                    mention_type,
                    context_snippet,
                    evidence_start,
                    evidence_end,
                    context_embedding,
                    first_seen_year,
                    is_acronym,
                    has_digit,
                    is_shared,
                    source
                )
                VALUES (
                    $1::concept_resolution_type,
                    $2,
                    $3,
                    $4,
                    $5,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16
                )
                """,
                [
                    (
                        record["resolution_type"],
                        record["entity_id"],
                        record["paper_id"],
                        record["section_id"],
                        record["surface"],
                        record["normalized_surface"],
                        record.get("mention_type"),
                        record.get("context_snippet"),
                        record.get("evidence_start"),
                        record.get("evidence_end"),
                        record.get("context_embedding"),
                        record.get("first_seen_year"),
                        record.get("is_acronym", False),
                        record.get("has_digit", False),
                        record.get("is_shared", False),
                        record.get("source"),
                    )
                    for record in prepared
                ],
            )
