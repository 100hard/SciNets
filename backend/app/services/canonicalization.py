from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast
import re
from uuid import UUID

try:  # pragma: no cover - optional dependency in tests
    from asyncpg.pgproto import pgproto  # type: ignore
except Exception:  # pragma: no cover - asyncpg may be absent in tests
    pgproto = None  # type: ignore[assignment]

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


@dataclass
class CanonicalizationAdjudicationRequest:
    resolution_type: ConceptResolutionType
    left_id: UUID
    right_id: UUID
    verdict: str
    rationale: Optional[str] = None
    score: Optional[float] = None
    decision_source: str = "llm"
    adjudicator_metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.decision_source not in {"hard", "llm"}:
            raise ValueError(
                "decision_source must be either 'hard' or 'llm' to match the audit enum"
            )

    def to_merge_decision(self) -> "_MergeDecision":
        score = 0.0 if self.score is None else float(max(0.0, min(1.0, self.score)))
        rationale = self.rationale or "Manual adjudication decision applied."
        return _MergeDecision(
            canonical_id=self.left_id,
            merged_id=self.right_id,
            score=score,
            decision_source=self.decision_source,
            verdict=self.verdict,
            rationale=rationale,
            adjudicator_metadata=self.adjudicator_metadata,
        )


@dataclass
class CanonicalizationAdjudicationResult:
    canonical_id: UUID
    merged_id: UUID
    verdict: str
    score: float
    decision_source: str
    rationale: str
    adjudicator_metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_merge_decision(
        cls, decision: "_MergeDecision"
    ) -> "CanonicalizationAdjudicationResult":
        return cls(
            canonical_id=decision.canonical_id,
            merged_id=decision.merged_id,
            verdict=decision.verdict,
            score=decision.score,
            decision_source=decision.decision_source,
            rationale=decision.rationale,
            adjudicator_metadata=decision.adjudicator_metadata,
        )


@dataclass
class _MergeDecision:
    canonical_id: UUID
    merged_id: UUID
    score: float
    decision_source: str
    verdict: str
    rationale: str
    adjudicator_metadata: Optional[dict[str, Any]] = None


@dataclass
class _CanonicalizationComputation:
    before: int
    after: int
    merges: int
    alias_map: Dict[UUID, list[tuple[str, float]]]
    aliases_by_record: Dict[UUID, list[str]]
    id_to_canonical: Dict[UUID, UUID]
    decisions: list[_MergeDecision]
    examples: list[CanonicalizationExample]


_TYPE_CONFIG: Dict[ConceptResolutionType, _TypeConfig] = {
    ConceptResolutionType.METHOD: _TypeConfig("methods", "method_id"),
    ConceptResolutionType.DATASET: _TypeConfig("datasets", "dataset_id"),
    ConceptResolutionType.METRIC: _TypeConfig("metrics", "metric_id"),
    ConceptResolutionType.TASK: _TypeConfig("tasks", "task_id"),
}


_SIMILARITY_THRESHOLDS: Dict[ConceptResolutionType, float] = {
    ConceptResolutionType.METHOD: 0.70,
    ConceptResolutionType.DATASET: 0.65,
    ConceptResolutionType.METRIC: 0.75,
    ConceptResolutionType.TASK: 0.65,
}


_SIMILARITY_BORDERLINE_WINDOWS: Dict[ConceptResolutionType, Tuple[float, float]] = {
    ConceptResolutionType.METHOD: (0.80, 0.85),
    ConceptResolutionType.DATASET: (0.78, 0.82),
    ConceptResolutionType.METRIC: (0.86, 0.90),
    ConceptResolutionType.TASK: (0.76, 0.80),
}


_MAX_EXAMPLES = 3

# Simple stop/noise phrases that should not become aliases or names
_NOISE_ALIASES: set[str] = {
    "in table",
    "table",
    "at less than 1/",
    "less than 1/",
    "big model",
    "big transformer model",
}

_NOISE_PREFIXES: tuple[str, ...] = (
    "see ",
    "our ",
    "their ",
    "its ",
    "this ",
    "that ",
    "these ",
    "those ",
)

_NUMERAL_WORDS: set[str] = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}


def _is_noise_alias(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    lower = text.lower()
    if lower in _NOISE_ALIASES:
        return True
    if lower.startswith(_NOISE_PREFIXES):
        return True
    if "appendix" in lower:
        return True
    if re.search(r"\btable\s+\d+", lower):
        return True
    if re.search(r"\bfigure\s+\d+", lower):
        return True
    if "results" in lower and lower.startswith(("these ", "those ", "our ", "the ")):
        return True
    if " or " in lower:
        return True
    words = [word for word in re.split(r"\s+", lower) if word]
    if not words:
        return True
    if len(words) <= 3 and text == lower and all(word.isalpha() for word in words):
        if words[0] in _NUMERAL_WORDS:
            return True
    return False

# Regex to detect WMT year + language pair in flexible forms
_WMT_PATTERN = re.compile(
    r"\bWMT\s*([0-9]{4})\b.*?(english|en)[\s\-→–—]*to[\s\-→–—]*?(german|de|french|fr)"
    r"|\bWMT\s*([0-9]{4})\b.*?(german|de|french|fr)[\s\-→–—]*to[\s\-→–—]*?(english|en)",
    re.IGNORECASE,
)


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


_embedding_backend: Optional[EmbeddingBackend] = None


def _prepare_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = " ".join(value.strip().split())
    # Normalize unicode arrows/dashes and common WMT phrasing
    text = (
        text.replace("→", "-")
        .replace("↔", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("En–", "En-")
        .replace("En—", "En-")
        .replace("De–", "De-")
        .replace("De—", "De-")
        .replace("English–", "English-")
        .replace("English—", "English-")
    )
    # Unify WMT year tokens
    text = text.replace("WMT14", "WMT 2014").replace("WMT 14", "WMT 2014")
    # Remove generic suffixes
    lower = text.lower()
    for noise in (" translation task", " task"):
        if lower.endswith(noise):
            text = text[: -len(noise)]
            break
    # Collapse multiple spaces again
    return " ".join(text.split())


def _extract_alias_values(raw_aliases: Any) -> list[str]:
    if raw_aliases is None:
        return []

    queue: deque[Any] = deque()
    if isinstance(raw_aliases, str):
        stripped = raw_aliases.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                queue.append(raw_aliases)
            else:
                queue.append(parsed)
        else:
            queue.append(raw_aliases)
    else:
        queue.append(raw_aliases)

    flattened: list[str] = []
    while queue:
        item = queue.popleft()
        if item is None:
            continue
        if isinstance(item, str):
            prepared = _prepare_text(item)
            if prepared and not _is_noise_alias(prepared):
                flattened.append(prepared)
            continue
        if isinstance(item, dict):
            queue.extend(item.values())
            continue
        if isinstance(item, (list, tuple, set)):
            text_items: list[str] = []
            other_items: list[Any] = []
            for value in item:
                if isinstance(value, str):
                    text_items.append(value)
                else:
                    other_items.append(value)

            if text_items:
                prepared_items = [_prepare_text(value) for value in text_items]
                prepared_items = [
                    value for value in prepared_items if value and not _is_noise_alias(value)
                ]
                if prepared_items:

                    if all(len(value) <= 1 for value in prepared_items):
                        combined = _prepare_text("".join(prepared_items))
                        if combined and not _is_noise_alias(combined):
                            flattened.append(combined)
                    else:
                        flattened.extend(prepared_items)
            queue.extend(other_items)
            continue
        prepared = _prepare_text(str(item))
        if prepared and not _is_noise_alias(prepared):
            flattened.append(prepared)

    return flattened



def _normalise_alias_candidates(values: Sequence[Any]) -> list[str]:
    normalised: list[str] = []
    seen: set[str] = set()
    for value in values:
        for candidate in _extract_alias_values(value):
            prepared = _prepare_text(candidate)
            if not prepared:
                continue
            key = prepared.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            normalised.append(prepared)
    return normalised

def _normalise_key(value: str) -> str:
    return _prepare_text(value).casefold()


def _collect_normalised_variants(record: _OntologyRecord) -> set[str]:
    return {
        _normalise_key(value)
        for value in _normalise_alias_candidates([record.name, *record.aliases])
        if value
    }


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
                    ordered_pair: tuple[UUID, UUID] = (ordered[0], ordered[1])
                    pairs.add(ordered_pair)

    alias_blocks: Dict[str, list[UUID]] = defaultdict(list)
    for record in records:
        for variant in _normalise_alias_candidates([record.name, *record.aliases]):
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
                    ordered_pair: tuple[UUID, UUID] = (ordered[0], ordered[1])
                    pairs.add(ordered_pair)

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
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    base = _compute_similarity(left.name, right.name, embeddings)
    mention_score = _compute_mention_similarity(left, right)
    score = max(base, mention_score)
    # Heuristic boost: WMT datasets/tasks with same year and language pair
    left_text = f"{left.name} {' '.join(left.aliases)}".lower()
    right_text = f"{right.name} {' '.join(right.aliases)}".lower()
    left_match = _WMT_PATTERN.search(left_text)
    right_match = _WMT_PATTERN.search(right_text)
    if left_match and right_match:
        score = max(score, 0.85)
    return score


def get_embedding_backend() -> EmbeddingBackend:
    global _embedding_backend
    if _embedding_backend is None:
        _embedding_backend = build_embedding_backend(
            settings.embedding_model_name, settings.embedding_dimension
        )
    return _embedding_backend


async def canonicalize(
    types: Optional[Sequence[ConceptResolutionType]] = None,
    adjudications: Optional[Sequence[CanonicalizationAdjudicationRequest]] = None,
) -> CanonicalizationReport:
    selected = list(types) if types else list(_TYPE_CONFIG.keys())
    pool = get_pool()
    summary: list[CanonicalizationTypeReport] = []
    adjudications_by_type = _group_adjudications(adjudications or [])
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

            computation = await _compute_canonicalization(
                records,
                resolution_type,
            )
            async with conn.transaction():
                await _persist_concept_resolutions(conn, resolution_type, computation)
                await _update_aliases(conn, config, computation)
                await _update_results(conn, config, computation)
                manual_decisions = adjudications_by_type.get(resolution_type, [])
                await _record_merge_audit(
                    conn,
                    resolution_type,
                    computation,
                    manual_decisions,
                )

            summary.append(
                CanonicalizationTypeReport(
                    resolution_type=resolution_type,
                    before=computation.before,
                    after=computation.after,
                    merges=computation.merges,
                    examples=computation.examples[:_MAX_EXAMPLES],
                )
            )
    return CanonicalizationReport(summary=summary)


def _group_adjudications(
    adjudications: Sequence[CanonicalizationAdjudicationRequest],
) -> Dict[ConceptResolutionType, list[_MergeDecision]]:
    grouped: Dict[ConceptResolutionType, list[_MergeDecision]] = defaultdict(list)
    for request in adjudications:
        grouped[request.resolution_type].append(request.to_merge_decision())
    return grouped


async def _load_records(
    conn,
    config: _TypeConfig,
    resolution_type: ConceptResolutionType,
) -> list[_OntologyRecord]:
    query = f"""
        SELECT
            entity.id AS entity_id,
            entity.name AS entity_name,
            entity.aliases AS entity_aliases,
            entity.created_at AS entity_created_at,
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
        FROM {config.table} AS entity
        LEFT JOIN ontology_mentions AS mention
            ON mention.entity_id = entity.id
            AND mention.resolution_type = $1
    """
    rows = await conn.fetch(query, resolution_type.value)
    records_by_id: Dict[UUID, _OntologyRecord] = {}
    for row in rows:
        payload = dict(row)
        record_id: UUID = payload["entity_id"]
        record = records_by_id.get(record_id)
        if record is None:
            name = _prepare_text(str(payload.get("entity_name") or ""))
            raw_aliases = payload.get("entity_aliases")
            aliases: list[str] = []
            seen_aliases: set[str] = set()
            for candidate in _extract_alias_values(raw_aliases):
                prepared = _prepare_text(candidate)
                if not prepared:
                    continue
                if len(prepared) <= 1:
                    continue
                key = prepared.casefold()
                if key in seen_aliases:
                    continue
                seen_aliases.add(key)
                aliases.append(prepared)
            record = _OntologyRecord(
                id=record_id,
                name=name,
                aliases=aliases,
                created_at=payload["entity_created_at"],
                mentions=[],
            )
            records_by_id[record_id] = record

        surface_raw = payload.get("mention_surface")
        normalized_surface = payload.get("mention_normalized_surface")
        prepared_surface = _prepare_text(surface_raw)
        prepared_normalized = (
            _prepare_text(normalized_surface)
            if normalized_surface
            else _normalise_key(prepared_surface)
        )
        has_mention = any(
            payload.get(key) is not None
            for key in (
                "mention_surface",
                "mention_normalized_surface",
                "mention_type",
                "mention_paper_id",
                "mention_section_id",
                "mention_start",
                "mention_end",
                "mention_first_seen_year",
                "mention_is_acronym",
                "mention_has_digit",
                "mention_is_shared",
                "mention_context_embedding",
            )
        )
        if has_mention and (prepared_surface or prepared_normalized):
            raw_embedding = payload.get("mention_context_embedding")
            vector: Optional[list[float]] = None
            if isinstance(raw_embedding, (list, tuple)):
                vector = [float(value) for value in raw_embedding]
            record.mentions.append(
                _MentionSignal(
                    surface=prepared_surface,
                    normalized_surface=prepared_normalized,
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

    return list(records_by_id.values())


async def _compute_canonicalization(
    records: Sequence[_OntologyRecord],
    resolution_type: ConceptResolutionType,
) -> _CanonicalizationComputation:
    record_by_id: Dict[UUID, _OntologyRecord] = {record.id: record for record in records}
    texts: list[str] = []
    seen_texts: set[str] = set()
    for record in records:
        for variant in _normalise_alias_candidates([record.name, *record.aliases]):
            if variant not in seen_texts:
                seen_texts.add(variant)
                texts.append(variant)
    embeddings = await _embed_texts(texts)

    uf = _UnionFind()
    for record in records:
        uf.add(record.id)

    alias_index: Dict[str, list[UUID]] = defaultdict(list)
    for record in records:
        for variant in _normalise_alias_candidates([record.name, *record.aliases]):
            alias_index[_normalise_key(variant)].append(record.id)
    for ids in alias_index.values():
        if len(ids) <= 1:
            continue
        base = ids[0]
        for other in ids[1:]:
            uf.union(base, other)

    mention_index: Dict[str, list[UUID]] = defaultdict(list)
    for record in records:
        for mention in record.mentions:
            if mention.normalized_surface:
                mention_index[mention.normalized_surface].append(record.id)
    for ids in mention_index.values():
        if len(ids) <= 1:
            continue
        base = ids[0]
        for other in ids[1:]:
            uf.union(base, other)

    threshold = _SIMILARITY_THRESHOLDS[resolution_type]
    candidate_pairs = _generate_candidate_pairs(records)
    
    print(f"[DEBUG] Processing {len(candidate_pairs)} candidate pairs for {resolution_type}")
    print(f"[DEBUG] Similarity threshold: {threshold}")
    print(f"[DEBUG] Total records: {len(records)}")
    
    merges_count = 0
    for left_id, right_id in candidate_pairs:
        if uf.find(left_id) == uf.find(right_id):
            continue
        left = record_by_id[left_id]
        right = record_by_id[right_id]
        score = _score_pair(left, right, embeddings)
        
        # Debug similar concepts
        if left.name.lower() == right.name.lower() or any(alias.lower() == right.name.lower() for alias in left.aliases) or any(alias.lower() == left.name.lower() for alias in right.aliases):
            print(f"[DEBUG] EXACT MATCH: '{left.name}' vs '{right.name}': score={score:.3f}")
        
        if score >= threshold:
            print(f"[DEBUG] MERGING: '{left.name}' + '{right.name}' (score={score:.3f})")
            uf.union(left_id, right_id)
            merges_count += 1
        elif score > 0.5:  # Show high-scoring pairs that didn't merge
            print(f"[DEBUG] HIGH SCORE (no merge): '{left.name}' vs '{right.name}': score={score:.3f}")
    
    print(f"[DEBUG] Total merges: {merges_count}")

    groups: Dict[UUID, list[UUID]] = defaultdict(list)
    for record in records:
        root = uf.find(record.id)
        groups[root].append(record.id)
    
    print(f"[DEBUG] Final groups: {len(groups)} (started with {len(records)})")
    for group_id, group_members in groups.items():
        if len(group_members) > 1:
            print(f"[DEBUG] Group with {len(group_members)} members: {[record_by_id[member_id].name for member_id in group_members]}")

    alias_map: Dict[UUID, list[tuple[str, float]]] = {}
    aliases_by_record: Dict[UUID, list[str]] = {}
    id_to_canonical: Dict[UUID, UUID] = {}
    decisions: list[_MergeDecision] = []
    examples: list[CanonicalizationExample] = []

    for group_ids in groups.values():
        canonical_id = _select_canonical(group_ids, record_by_id)
        canonical_record = record_by_id[canonical_id]
        id_to_canonical[canonical_id] = canonical_id

        per_record_variants: Dict[UUID, list[str]] = {}
        for record_id in group_ids:
            record = record_by_id[record_id]
            variants = _normalise_alias_candidates([record.name, *record.aliases])
            per_record_variants[record_id] = variants

        variant_scores: Dict[str, tuple[str, float]] = {}
        for record_id, variants in per_record_variants.items():
            if record_id == canonical_id:
                pair_score = 1.0
            else:
                pair_score = _score_pair(
                    canonical_record,
                    record_by_id[record_id],
                    embeddings,
                )
            for variant in variants:
                key = variant.casefold()
                existing = variant_scores.get(key)
                candidate = (variant, pair_score)
                if (
                    existing is None
                    or pair_score > existing[1]
                    or (pair_score == existing[1] and variant < existing[0])
                ):
                    variant_scores[key] = candidate

        alias_entries = sorted(
            variant_scores.values(), key=lambda item: (-item[1], item[0])
        )
        deduped_alias_entries: list[tuple[str, float]] = []
        seen_aliases: set[str] = set()
        for alias_text, score in alias_entries:
            key = alias_text.casefold()
            if not key or key in seen_aliases:
                continue
            seen_aliases.add(key)
            deduped_alias_entries.append((alias_text, score))
        alias_map[canonical_id] = deduped_alias_entries
        canonical_variants = {
            key: variant for key, (variant, _) in variant_scores.items()
        }

        for record_id, variants in per_record_variants.items():
            record_variant_keys = {variant.casefold() for variant in variants}
            other_variants = sorted(
                variant
                for key, variant in canonical_variants.items()
                if key not in record_variant_keys
            )
            record = record_by_id[record_id]
            existing_aliases: Dict[str, str] = {}
            for alias in _normalise_alias_candidates(record.aliases):
                if len(alias) > 1:
                    existing_aliases.setdefault(alias.casefold(), alias)

            combined_alias_map = dict(existing_aliases)
            for variant in other_variants:
                if len(variant) > 1:
                    combined_alias_map.setdefault(variant.casefold(), variant)

            combined_aliases = sorted(
                combined_alias_map.values(), key=lambda value: (value.casefold(), value)
            )
            aliases_by_record[record_id] = combined_aliases
            id_to_canonical[record_id] = canonical_id

        if len(group_ids) > 1:
            merged_items: list[CanonicalizationMergedItem] = []
            for record_id in group_ids:
                if record_id == canonical_id:
                    continue
                record = record_by_id[record_id]
                score = _score_pair(canonical_record, record, embeddings)
                decisions.append(
                    _MergeDecision(
                        canonical_id=canonical_id,
                        merged_id=record_id,
                        score=score,
                        decision_source="llm",
                        verdict="accepted",
                        rationale=(
                            "Automatic merge based on similarity score "
                            f"{score:.2f} between '{canonical_record.name}' and '{record.name}'."
                        ),
                        adjudicator_metadata=None,
                    )
                )
                merged_items.append(
                    CanonicalizationMergedItem(
                        id=record_id,
                        name=record.name,
                        score=score,
                    )
                )
            merged_items.sort(key=lambda item: item.score, reverse=True)
            examples.append(
                CanonicalizationExample(
                    canonical_id=canonical_id,
                    canonical_name=canonical_record.name,
                    merged=merged_items,
                )
            )

    before = len(records)
    after = len(groups)
    merges = sum(len(group) - 1 for group in groups.values())

    return _CanonicalizationComputation(
        before=before,
        after=after,
        merges=merges,
        alias_map=alias_map,
        aliases_by_record=aliases_by_record,
        id_to_canonical=id_to_canonical,
        decisions=decisions,
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
        seen: set[str] = set()
        for alias_text, score in alias_entries:

            for prepared in _normalise_alias_candidates([alias_text]):
                if len(prepared) <= 1:
                    continue
                key = prepared.casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                records.append((resolution_type.value, canonical_id, prepared, score))
    if records:
        try:
            await conn.executemany(
                """
                INSERT INTO concept_resolutions (resolution_type, canonical_id, alias_text, score)
                VALUES ($1, $2, $3, $4)
                """,
                records,
            )
        except Exception as exc:
            print(f"[DEBUG] _persist_concept_resolutions failed")
            print(f"[DEBUG] Error: {exc}")
            print(f"[DEBUG] First record: {records[0] if records else 'None'}")
            raise


async def _update_aliases(
    conn,
    config: _TypeConfig,
    computation: _CanonicalizationComputation,
) -> None:
    updates: list[tuple[UUID, Any]] = []
    for record_id in computation.id_to_canonical.keys():
        aliases_list = computation.aliases_by_record.get(record_id, [])
        normalized_aliases = [alias for alias in aliases_list if isinstance(alias, str) and alias]
        payload: Any
        jsonb_factory = getattr(pgproto, "Jsonb", None) if pgproto is not None else None
        if jsonb_factory is not None:
            payload = jsonb_factory(normalized_aliases)
        else:
            payload = normalized_aliases
        updates.append((record_id, payload))
    if not updates:
        return
    try:
        await conn.executemany(
            f"UPDATE {config.table} SET aliases = $2::jsonb, updated_at = NOW() WHERE id = $1",
            updates,
        )
    except Exception as exc:
        print(f"[DEBUG] _update_aliases failed for table {config.table}")
        print(f"[DEBUG] Error: {exc}")
        print(f"[DEBUG] First update: {updates[0] if updates else 'None'}")
        raise


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
    try:
        await conn.executemany(
            f"UPDATE results SET {config.fk_column} = $2 WHERE {config.fk_column} = $1",
            updates,
        )
    except Exception as exc:
        print(f"[DEBUG] _update_results failed for column {config.fk_column}")
        print(f"[DEBUG] Error: {exc}")
        print(f"[DEBUG] First update: {updates[0] if updates else 'None'}")
        raise


async def _record_merge_audit(
    conn,
    resolution_type: ConceptResolutionType,
    computation: _CanonicalizationComputation,
    manual_decisions: Sequence[_MergeDecision] | None = None,
) -> None:
    mapping_version = settings.canonicalization_mapping_version
    await conn.execute(
        """
        DELETE FROM canonicalization_merge_decisions
        WHERE resolution_type = ANY($1::concept_resolution_type[])
          AND mapping_version = $2
        """,
        [resolution_type.value],
        mapping_version,
    )
    decisions: list[_MergeDecision] = list(computation.decisions)
    if manual_decisions:
        decisions.extend(manual_decisions)
    if not decisions:
        return
    records = [
        (
            resolution_type.value,
            decision.canonical_id,
            decision.merged_id,
            decision.score,
            decision.decision_source,
            decision.verdict,
            decision.rationale,
            mapping_version,
            decision.adjudicator_metadata,
        )
        for decision in decisions
    ]
    try:
        await conn.executemany(
            """
            INSERT INTO canonicalization_merge_decisions (
                resolution_type,
                left_id,
                right_id,
                score,
                decision_source,
                verdict,
                rationale,
                mapping_version,
                adjudicator_metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            records,
        )
    except Exception as exc:
        print(f"[DEBUG] _record_merge_audit failed")
        print(f"[DEBUG] Error: {exc}")
        print(f"[DEBUG] First record: {records[0] if records else 'None'}")
        raise


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
    cosine = _cosine_similarity(vector_left or [], vector_right or [])
    jaro = _jaro_winkler(left.casefold(), right.casefold())
    score = 0.6 * cosine + 0.4 * jaro
    return float(max(0.0, min(1.0, score)))


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
