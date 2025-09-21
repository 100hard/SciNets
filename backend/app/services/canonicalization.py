from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Dict, Iterable, Mapping, Sequence
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
class _OntologyRecord:
    id: UUID
    name: str
    aliases: list[str]
    created_at: datetime


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


_embedding_backend: EmbeddingBackend | None = None


def _prepare_text(value: str | None) -> str:
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


def get_embedding_backend() -> EmbeddingBackend:
    global _embedding_backend
    if _embedding_backend is None:
        _embedding_backend = build_embedding_backend(
            settings.embedding_model_name, settings.embedding_dimension
        )
    return _embedding_backend


async def canonicalize(
    types: Sequence[ConceptResolutionType] | None = None,
) -> CanonicalizationReport:
    selected = list(types) if types else list(_TYPE_CONFIG.keys())
    pool = get_pool()
    summary: list[CanonicalizationTypeReport] = []
    async with pool.acquire() as conn:
        for resolution_type in selected:
            config = _TYPE_CONFIG[resolution_type]
            records = await _load_records(conn, config)
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


async def _load_records(conn, config: _TypeConfig) -> list[_OntologyRecord]:
    query = f"SELECT id, name, aliases, created_at FROM {config.table}"
    rows = await conn.fetch(query)
    records: list[_OntologyRecord] = []
    for row in rows:
        name = _prepare_text(str(row["name"]))
        alias_values = row.get("aliases", []) if isinstance(row, dict) else row["aliases"]
        aliases = [
            alias
            for alias in (_prepare_text(str(value)) for value in alias_values or [])
            if alias
        ]
        records.append(
            _OntologyRecord(
                id=row["id"],
                name=name,
                aliases=aliases,
                created_at=row["created_at"],
            )
        )
    return records


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
    for idx, left in enumerate(records):
        for right in records[idx + 1 :]:
            if uf.find(left.id) == uf.find(right.id):
                continue
            score = _compute_similarity(left.name, right.name, embeddings)
            if score >= threshold:
                uf.union(left.id, right.id)

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
    left: Sequence[float] | None,
    right: Sequence[float] | None,
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
