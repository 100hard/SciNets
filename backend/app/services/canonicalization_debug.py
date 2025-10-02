"""Utilities for logging canonicalization merge decisions during debugging.

This helper mirrors the verbose logging that used to live inline in
:mod:`backend.app.services.canonicalization`.  Having it here keeps the
core service lean while still providing an easy way to inspect how
candidate pairs are evaluated when diagnosing canonicalisation issues.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, MutableMapping, Sequence
from uuid import UUID

from .canonicalization import (
    ConceptResolutionType,
    _OntologyRecord,
    _SIMILARITY_THRESHOLDS,
    _UnionFind,
    _generate_candidate_pairs,
    _score_pair,
)


def log_merge_evaluation(
    records: Sequence[_OntologyRecord],
    resolution_type: ConceptResolutionType,
    uf: _UnionFind,
    record_by_id: MutableMapping[UUID, _OntologyRecord],
    embeddings: Mapping[str, Sequence[float]],
    print_fn: Callable[[str], None] = print,
) -> None:
    """Emit detailed diagnostics for the canonicalisation clustering loop.

    Args:
        records: The ontology records under consideration for the current
            resolution type.
        resolution_type: Which concept resolution bucket is being processed.
        uf: The union-find structure tracking already-merged concept IDs.
        record_by_id: Mapping of record IDs to their corresponding ontology
            records.
        embeddings: Cached embeddings keyed by the normalised text that was
            embedded.
        print_fn: Hook used to emit log lines, defaults to :func:`print`.
    """

    threshold = _SIMILARITY_THRESHOLDS[resolution_type]
    candidate_pairs = _generate_candidate_pairs(records)

    print_fn(
        f"[DEBUG] Processing {len(candidate_pairs)} candidate pairs for {resolution_type}"
    )
    print_fn(f"[DEBUG] Similarity threshold: {threshold}")
    print_fn(f"[DEBUG] Total records: {len(records)}")

    merges_count = 0
    for left_id, right_id in candidate_pairs:
        if uf.find(left_id) == uf.find(right_id):
            continue
        left = record_by_id[left_id]
        right = record_by_id[right_id]
        score = _score_pair(left, right, embeddings)

        # Debug similar concepts that share the same surface or aliases.
        if (
            left.name.lower() == right.name.lower()
            or any(alias.lower() == right.name.lower() for alias in left.aliases)
            or any(alias.lower() == left.name.lower() for alias in right.aliases)
        ):
            print_fn(
                f"[DEBUG] EXACT MATCH: '{left.name}' vs '{right.name}': score={score:.3f}"
            )

        if score >= threshold:
            print_fn(
                f"[DEBUG] MERGING: '{left.name}' + '{right.name}' (score={score:.3f})"
            )
            uf.union(left_id, right_id)
            merges_count += 1
        elif score > 0.5:  # Show high-scoring pairs that didn't merge
            print_fn(
                "[DEBUG] HIGH SCORE (no merge): "
                f"'{left.name}' vs '{right.name}': score={score:.3f}"
            )

    print_fn(f"[DEBUG] Total merges: {merges_count}")

    groups: dict[UUID, list[UUID]] = defaultdict(list)
    for record in records:
        root = uf.find(record.id)
        groups[root].append(record.id)

    print_fn(
        f"[DEBUG] Final groups: {len(groups)} (started with {len(records)})"
    )
    for group_id, group_members in groups.items():
        if len(group_members) > 1:
            members = [record_by_id[member_id].name for member_id in group_members]
            print_fn(
                f"[DEBUG] Group with {len(group_members)} members: {members}"
            )
