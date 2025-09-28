from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence
from uuid import UUID

from app.core.config import settings
from app.db.pool import get_pool
from app.models.ontology import (
    CanonicalizationExample,
    CanonicalizationMergedItem,
    CanonicalizationReport,
    CanonicalizationTypeReport,
    ConceptResolutionType,
)
from app.services.embeddings import EmbeddingBackend, build_embedding_backend
@dataclass
class _MentionSignal:
    surface: str
    normalized_surface: str
    mention_type: Optional[str]
    paper_id: Optional[UUID]
    section_id: Optional[UUID]
    start: Optional[int]
    end: Optional[int]
    first_seen_year: Optional[int]
    is_acronym: bool
    has_digit: bool
    is_shared: bool
    context_embedding: Optional[Sequence[float]]


@dataclass
class _OntologyRecord:
    id: UUID
    name: str
    aliases: list[str]
    created_at: datetime
    mentions: list[_MentionSignal]


@dataclass(frozen=True)
class _TypeConfig:
    table: str
    fk_column: str


_TYPE_CONFIG: Dict[ConceptResolutionType, _TypeConfig] = {
    ConceptResolutionType.METHOD: _TypeConfig("methods", "method_id"),
    ConceptResolutionType.DATASET: _TypeConfig("datasets", "dataset_id"),
    ConceptResolutionType.METRIC: _TypeConfig("metrics", "metric_id"),
    ConceptResolutionType.TASK: _TypeConfig("tasks", "task_id"),
}


_SIMILARITY_THRESHOLDS: Dict[ConceptResolutionType, float] = {
    ConceptResolutionType.METHOD: 0.85,
    ConceptResolutionType.DATASET: 0.82,
    ConceptResolutionType.METRIC: 0.90,
    ConceptResolutionType.TASK: 0.80,
}


_MAX_EXAMPLES = 3


class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[UUID, UUID] = {}
        self._rank: Dict[UUID, int] = {}

    def add(self, item: UUID) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0

    def find(self, item: UUID) -> UUID:
        parent = self._parent.get(item)
        if parent is None:
            raise KeyError(item)
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: UUID, right: UUID) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        rank_left = self._rank[root_left]
        rank_right = self._rank[root_right]
        if rank_left < rank_right:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if rank_left == rank_right:
            self._rank[root_left] += 1
        return True


@dataclass
class _CanonicalizationComputation:
    before: int
    after: int
    merges: int
    alias_map: Dict[UUID, list[tuple[str, float]]]
    aliases_by_record: Dict[UUID, list[str]]
    id_to_canonical: Dict[UUID, UUID]
    examples: list[CanonicalizationExample]


_embedding_backend: Optional[EmbeddingBackend] = None


def _prepare_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().split())


def _normalise_key(value: str) -> str:
    return _prepare_text(value).casefold()


def _select_canonical(group_ids: Sequence[UUID], records: Mapping[UUID, _OntologyRecord]) -> UUID:
    return min(
        group_ids,
        key=lambda record_id: (
            records[record_id].created_at,
            records[record_id].name.casefold(),
            str(records[record_id].id),
        ),
    )


def _generate_candidate_pairs(records: Sequence[_OntologyRecord]) -> set[tuple[UUID, UUID]]:
    pairs: set[tuple[UUID, UUID]] = set()

    def _add_pairs(ids: Sequence[UUID]) -> None:
        unique = list(dict.fromkeys(ids))
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                ordered = tuple(sorted((left, right), key=str))
                if ordered[0] != ordered[1]:
                    pairs.add(ordered)

    alias_blocks: Dict[str, list[UUID]] = defaultdict(list)
    for record in records:
        for variant in [record.name, *record.aliases]:
            key = _normalise_key(variant)
            if key:
                alias_blocks[key].append(record.id)
    for ids in alias_blocks.values():
        if len(ids) > 1:
            _add_pairs(ids)

    mention_blocks: Dict[str, list[UUID]] = defaultdict(list)
    paper_blocks: Dict[UUID, list[UUID]] = defaultdict(list)
    for record in records:
        for mention in record.mentions:
            if mention.normalized_surface:
                mention_blocks[mention.normalized_surface].append(record.id)
            if mention.paper_id:
                paper_blocks[mention.paper_id].append(record.id)

    for ids in mention_blocks.values():
        if len(ids) > 1:
            _add_pairs(ids)

    for ids in paper_blocks.values():
        if len(ids) > 1:
            _add_pairs(ids)

    if not pairs:
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                ordered = tuple(sorted((left.id, right.id), key=str))
                if ordered[0] != ordered[1]:
                    pairs.add(ordered)

    return pairs


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left))
    right_norm = math.sqrt(sum(r * r for r in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, numerator / (left_norm * right_norm))))


def _compute_mention_similarity(
    left: _OntologyRecord,
    right: _OntologyRecord,
) -> float:
    if not left.mentions or not right.mentions:
        return 0.0

    left_by_surface: Dict[str, list[_MentionSignal]] = defaultdict(list)
    right_by_surface: Dict[str, list[_MentionSignal]] = defaultdict(list)
    for mention in left.mentions:
        if mention.normalized_surface:
            left_by_surface[mention.normalized_surface].append(mention)
    for mention in right.mentions:
        if mention.normalized_surface:
            right_by_surface[mention.normalized_surface].append(mention)

    best = 0.0
    shared_surfaces = set(left_by_surface.keys()).intersection(right_by_surface.keys())
    for surface in shared_surfaces:
        combined = left_by_surface[surface] + right_by_surface[surface]
        candidate = 0.88
        if not any(mention.is_shared for mention in combined):
            candidate += 0.05
        if any(mention.is_acronym for mention in combined):
            candidate += 0.02
        if any(mention.has_digit for mention in combined):
            candidate += 0.01

        years = [mention.first_seen_year for mention in combined if mention.first_seen_year]
        if years:
            diff = max(years) - min(years)
            if diff <= 1:
                candidate += 0.03
            elif diff > 5:
                candidate -= 0.05

        context_scores: list[float] = []
        for left_signal in left_by_surface[surface]:
            if not left_signal.context_embedding:
                continue
            for right_signal in right_by_surface[surface]:
                if not right_signal.context_embedding:
                    continue
                context_scores.append(
                    _cosine_similarity(left_signal.context_embedding, right_signal.context_embedding)
                )
        if context_scores:
            candidate = max(candidate, 0.75 + 0.2 * max(context_scores))

        best = max(best, max(0.0, min(1.0, candidate)))

    if best >= 0.9:
        return best

    shared_papers = {
        mention.paper_id
        for mention in left.mentions
        if mention.paper_id is not None
    }.intersection(
        {
            mention.paper_id
            for mention in right.mentions
            if mention.paper_id is not None
        }
    )
    if shared_papers:
        best = max(best, 0.82)

    return max(0.0, min(1.0, best))


def _score_pair(
    left: _OntologyRecord,
    right: _OntologyRecord,
    embeddings: Dict[str, Sequence[float]],
) -> float:
    base = _compute_similarity(left.name, right.name, embeddings)
    mention_score = _compute_mention_similarity(left, right)
    return max(base, mention_score)


def get_embedding_backend() -> EmbeddingBackend:
    global _embedding_backend
    if _embedding_backend is None:
        _embedding_backend = build_embedding_backend(
            settings.embedding_model_name, settings.embedding_dimension
        )
    return _embedding_backend


async def canonicalize(
    types: Optional[Sequence[ConceptResolutionType]] = None,
) -> CanonicalizationReport:
    selected = list(types) if types else list(_TYPE_CONFIG.keys())
    pool = get_pool()
    summary: list[CanonicalizationTypeReport] = []
    async with pool.acquire() as conn:
        for resolution_type in selected:
            config = _TYPE_CONFIG[resolution_type]
            records = await _load_records(conn, config, resolution_type)
            if not records:
                summary.append(
                    CanonicalizationTypeReport(
                        resolution_type=resolution_type,
                        before=0,
                        after=0,
                        merges=0,
                        examples=[],
                    )
                )
                continue

            computation = await _compute_canonicalization(records, resolution_type)
            async with conn.transaction():
                await _persist_concept_resolutions(conn, resolution_type, computation)
                await _update_aliases(conn, config, computation)
                await _update_results(conn, config, computation)

            summary.append(
                CanonicalizationTypeReport(
                    resolution_type=resolution_type,
                    before=computation.before,
                    after=computation.after,
                    merges=computation.merges,
                    examples=computation.examples,
                )
            )
    return CanonicalizationReport(summary=summary)


async def _load_records(
    conn,
    config: _TypeConfig,
    resolution_type: ConceptResolutionType,
) -> list[_OntologyRecord]:
    query = f"""
        SELECT
            base.id AS entity_id,
            base.name AS entity_name,
            base.aliases AS entity_aliases,
            base.created_at AS entity_created_at,
            mention.surface AS mention_surface,
            mention.normalized_surface AS mention_normalized_surface,
            mention.mention_type AS mention_type,
            mention.paper_id AS mention_paper_id,
            mention.section_id AS mention_section_id,
            mention.evidence_start AS mention_start,
            mention.evidence_end AS mention_end,
            mention.first_seen_year AS mention_first_seen_year,
            mention.is_acronym AS mention_is_acronym,
            mention.has_digit AS mention_has_digit,
            mention.is_shared AS mention_is_shared,
            mention.context_embedding AS mention_context_embedding
        FROM {config.table} AS base
        LEFT JOIN ontology_mentions AS mention
            ON mention.entity_id = base.id
           AND mention.resolution_type = $1::concept_resolution_type
        ORDER BY base.id
    """
    rows = await conn.fetch(query, resolution_type.value)
    records: Dict[UUID, _OntologyRecord] = {}
    for row in rows:
        payload = dict(row)
        record_id: UUID = payload["entity_id"]
        record = records.get(record_id)
        if record is None:
            name = _prepare_text(str(payload["entity_name"]))
            alias_values = payload.get("entity_aliases") or []
            aliases = [
                alias
                for alias in (_prepare_text(str(value)) for value in alias_values)
                if alias
            ]
            record = _OntologyRecord(
                id=record_id,
                name=name,
                aliases=aliases,
                created_at=payload["entity_created_at"],
                mentions=[],
            )
            records[record_id] = record

        surface = payload.get("mention_surface")
        if surface:
            normalized_surface = payload.get("mention_normalized_surface") or _normalise_key(
                str(surface)
            )
            raw_embedding = payload.get("mention_context_embedding")
            vector: Optional[list[float]] = None
            if isinstance(raw_embedding, (list, tuple)):
                vector = [float(value) for value in raw_embedding]
            record.mentions.append(
                _MentionSignal(
                    surface=_prepare_text(str(surface)),
                    normalized_surface=normalized_surface,
                    mention_type=payload.get("mention_type"),
                    paper_id=payload.get("mention_paper_id"),
                    section_id=payload.get("mention_section_id"),
                    start=payload.get("mention_start"),
                    end=payload.get("mention_end"),
                    first_seen_year=payload.get("mention_first_seen_year"),
                    is_acronym=bool(payload.get("mention_is_acronym")),
                    has_digit=bool(payload.get("mention_has_digit")),
                    is_shared=bool(payload.get("mention_is_shared")),
                    context_embedding=vector,
                )
            )

    return list(records.values())


async def _compute_canonicalization(
    records: Sequence[_OntologyRecord],
    resolution_type: ConceptResolutionType,
) -> _CanonicalizationComputation:
    record_by_id: Dict[UUID, _OntologyRecord] = {record.id: record for record in records}
    texts: list[str] = []
    seen_texts: set[str] = set()
    for record in records:
        for variant in [record.name, *record.aliases]:
            if not variant:
                continue
            if variant not in seen_texts:
                seen_texts.add(variant)
                texts.append(variant)
    embeddings = await _embed_texts(texts)

    uf = _UnionFind()
    for record in records:
        uf.add(record.id)

    # Exact matches via aliases / names
    alias_index: Dict[str, list[UUID]] = defaultdict(list)
    for record in records:
        for variant in [record.name, *record.aliases]:
            if not variant:
                continue
            alias_index[_normalise_key(variant)].append(record.id)
    for ids in alias_index.values():
        if len(ids) <= 1:
            continue
        base = ids[0]
        for other in ids[1:]:
            uf.union(base, other)

    threshold = _SIMILARITY_THRESHOLDS[resolution_type]
    candidate_pairs = _generate_candidate_pairs(records)
    for left_id, right_id in candidate_pairs:
        if uf.find(left_id) == uf.find(right_id):
            continue
        left_record = record_by_id[left_id]
        right_record = record_by_id[right_id]
        score = _score_pair(left_record, right_record, embeddings)
        if score >= threshold:
            uf.union(left_id, right_id)

    groups: Dict[UUID, list[UUID]] = defaultdict(list)
    for record in records:
        groups[uf.find(record.id)].append(record.id)

    canonical_groups: Dict[UUID, list[UUID]] = {}
    id_to_canonical: Dict[UUID, UUID] = {}
    for member_ids in groups.values():
        canonical_id = _select_canonical(member_ids, record_by_id)
        canonical_groups[canonical_id] = member_ids
        for member_id in member_ids:
            id_to_canonical[member_id] = canonical_id

    alias_map: Dict[UUID, list[tuple[str, float]]] = {}
    aliases_by_record: Dict[UUID, list[str]] = {}
    for canonical_id, member_ids in canonical_groups.items():
        canonical_record = record_by_id[canonical_id]
        collected: Dict[str, str] = {}
        for member_id in member_ids:
            member = record_by_id[member_id]
            for variant in [member.name, *member.aliases]:
                if not variant:
                    continue
                key = _normalise_key(variant)
                collected.setdefault(key, variant)

        alias_entries: list[tuple[str, float]] = []
        for variant in collected.values():
            if _normalise_key(variant) == _normalise_key(canonical_record.name):
                score = 1.0
            else:
                score = _compute_similarity(variant, canonical_record.name, embeddings)
            alias_entries.append((variant, float(max(0.0, min(1.0, score)))))

        alias_entries.sort(key=lambda item: (-item[1], item[0].casefold()))
        alias_map[canonical_id] = alias_entries

        alias_texts_sorted = [text for text, _ in alias_entries]
        for member_id in member_ids:
            member = record_by_id[member_id]
            filtered = [
                alias
                for alias in alias_texts_sorted
                if _normalise_key(alias) != _normalise_key(member.name)
            ]
            aliases_by_record[member_id] = filtered

    before = len(records)
    after = len(canonical_groups)
    merges = max(0, before - after)

    examples: list[CanonicalizationExample] = []
    for canonical_id, member_ids in canonical_groups.items():
        if len(member_ids) <= 1:
            continue
        canonical_record = record_by_id[canonical_id]
        merged_items: list[CanonicalizationMergedItem] = []
        for member_id in member_ids:
            if member_id == canonical_id:
                continue
            member = record_by_id[member_id]
            score = _compute_similarity(member.name, canonical_record.name, embeddings)
            merged_items.append(
                CanonicalizationMergedItem(
                    id=member.id,
                    name=member.name,
                    score=float(max(0.0, min(1.0, score))),
                )
            )
        merged_items.sort(key=lambda item: (-item.score, item.name.casefold()))
        examples.append(
            CanonicalizationExample(
                canonical_id=canonical_id,
                canonical_name=canonical_record.name,
                merged=merged_items,
            )
        )
        if len(examples) >= _MAX_EXAMPLES:
            break

    return _CanonicalizationComputation(
        before=before,
        after=after,
        merges=merges,
        alias_map=alias_map,
        aliases_by_record=aliases_by_record,
        id_to_canonical=id_to_canonical,
        examples=examples,
    )


async def _persist_concept_resolutions(
    conn,
    resolution_type: ConceptResolutionType,
    computation: _CanonicalizationComputation,
) -> None:
    await conn.execute(
        "DELETE FROM concept_resolutions WHERE resolution_type = ANY($1::concept_resolution_type[])",
        [resolution_type.value],
    )
    records: list[tuple[str, UUID, str, float]] = []
    for canonical_id, alias_entries in computation.alias_map.items():
        for alias_text, score in alias_entries:
            records.append((resolution_type.value, canonical_id, alias_text, score))
    if records:
        await conn.executemany(
            """
            INSERT INTO concept_resolutions (resolution_type, canonical_id, alias_text, score)
            VALUES ($1, $2, $3, $4)
            """,
            records,
        )


async def _update_aliases(
    conn,
    config: _TypeConfig,
    computation: _CanonicalizationComputation,
) -> None:
    updates = [
        (record_id, computation.aliases_by_record.get(record_id, []))
        for record_id in computation.id_to_canonical.keys()
    ]
    if not updates:
        return
    await conn.executemany(
        f"UPDATE {config.table} SET aliases = $2, updated_at = NOW() WHERE id = $1",
        updates,
    )


async def _update_results(
    conn,
    config: _TypeConfig,
    computation: _CanonicalizationComputation,
) -> None:
    updates = [
        (record_id, canonical_id)
        for record_id, canonical_id in computation.id_to_canonical.items()
        if record_id != canonical_id
    ]
    if not updates:
        return
    await conn.executemany(
        f"UPDATE results SET {config.fk_column} = $2 WHERE {config.fk_column} = $1",
        updates,
    )


async def _embed_texts(texts: Sequence[str]) -> Dict[str, Sequence[float]]:
    if not texts:
        return {}
    backend = get_embedding_backend()
    vectors = await backend.embed(texts, settings.embedding_batch_size)
    return {
        text: [float(value) for value in vector]
        for text, vector in zip(texts, vectors)
    }


def _compute_similarity(
    left: str,
    right: str,
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    vector_left = embeddings.get(left)
    vector_right = embeddings.get(right)
    cosine = _cosine_similarity(vector_left, vector_right)
    jaro = _jaro_winkler(left.casefold(), right.casefold())
    score = 0.6 * cosine + 0.4 * jaro
    return float(max(0.0, min(1.0, score)))


def _cosine_similarity(
    left: Optional[Sequence[float]],
    right: Optional[Sequence[float]],
) -> float:
    if left is None or right is None:
        return 0.0
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    value = dot / (norm_left * norm_right)
    return float(max(-1.0, min(1.0, value)))


def _jaro_winkler(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    jaro = _jaro_similarity(left, right)
    prefix = 0
    for l_char, r_char in zip(left, right):
        if l_char != r_char or prefix == 4:
            break
        prefix += 1
    return jaro + 0.1 * prefix * (1 - jaro)


def _jaro_similarity(left: str, right: str) -> float:
    max_distance = max(len(left), len(right)) // 2 - 1
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)

    matches = 0
    for i, l_char in enumerate(left):
        start = max(0, i - max_distance)
        end = min(i + max_distance + 1, len(right))
        for j in range(start, end):
            if right_matches[j]:
                continue
            if l_char != right[j]:
                continue
            left_matches[i] = True
            right_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    j_index = 0
    for i, match in enumerate(left_matches):
        if not match:
            continue
        while not right_matches[j_index]:
            j_index += 1
        if left[i] != right[j_index]:
            transpositions += 1
        j_index += 1

    transpositions //= 2
    return (
        matches / len(left)
        + matches / len(right)
        + (matches - transpositions) / matches
    ) / 3