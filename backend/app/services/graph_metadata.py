from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.models.graph import NodeType, RelationType


class GraphOntologyMetadata(BaseModel):
    default_node_types: Tuple[NodeType, ...]
    allowed_node_types: Tuple[NodeType, ...]
    allowed_relations: Tuple[RelationType, ...]
    ordered_relations: Tuple[RelationType, ...]
    concept_types: Tuple[NodeType, ...]


_DEFAULT_METADATA = GraphOntologyMetadata(
    default_node_types=(
        NodeType.METHOD,
        NodeType.DATASET,
        NodeType.METRIC,
        NodeType.TASK,
    ),
    allowed_node_types=(
        NodeType.METHOD,
        NodeType.DATASET,
        NodeType.METRIC,
        NodeType.TASK,
        NodeType.CONCEPT,
        NodeType.MATERIAL,
        NodeType.ORGANISM,
        NodeType.FINDING,
        NodeType.PROCESS,
        NodeType.MODEL,
    ),
    allowed_relations=(
        RelationType.PROPOSES,
        RelationType.EVALUATES_ON,
        RelationType.REPORTS,
        RelationType.COMPARES,
        RelationType.USES,
        RelationType.CAUSES,
        RelationType.PART_OF,
        RelationType.IS_A,
        RelationType.OUTPERFORMS,
        RelationType.ASSUMES,
    ),
    ordered_relations=(
        RelationType.PROPOSES,
        RelationType.EVALUATES_ON,
        RelationType.REPORTS,
        RelationType.COMPARES,
        RelationType.USES,
        RelationType.CAUSES,
        RelationType.PART_OF,
        RelationType.IS_A,
        RelationType.OUTPERFORMS,
        RelationType.ASSUMES,
    ),
    concept_types=(
        NodeType.METHOD,
        NodeType.DATASET,
        NodeType.METRIC,
        NodeType.TASK,
        NodeType.CONCEPT,
        NodeType.MATERIAL,
        NodeType.ORGANISM,
        NodeType.FINDING,
        NodeType.PROCESS,
        NodeType.MODEL,
    ),
)


def _coerce_node_types(values: Iterable[str]) -> tuple[NodeType, ...]:
    coerced: list[NodeType] = []
    for value in values:
        try:
            coerced.append(NodeType(str(value)))
        except ValueError:
            continue
    return tuple(coerced)


def _coerce_relation_types(values: Iterable[str]) -> tuple[RelationType, ...]:
    coerced: list[RelationType] = []
    for value in values:
        try:
            coerced.append(RelationType(str(value)))
        except ValueError:
            continue
    return tuple(coerced)


def _load_metadata_from_path(path: Path) -> GraphOntologyMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return GraphOntologyMetadata(
            default_node_types=_coerce_node_types(payload.get("default_node_types", ())),
            allowed_node_types=_coerce_node_types(payload.get("allowed_node_types", ())),
            allowed_relations=_coerce_relation_types(payload.get("allowed_relations", ())),
            ordered_relations=_coerce_relation_types(payload.get("ordered_relations", ())),
            concept_types=_coerce_node_types(payload.get("concept_types", ())),
        )
    except (OSError, json.JSONDecodeError, ValidationError):
        return None


def _resolve_metadata_path() -> Path | None:
    raw_path = settings.graph_metadata_path
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


@lru_cache(maxsize=1)
def get_graph_metadata() -> GraphOntologyMetadata:
    path = _resolve_metadata_path()
    if path:
        loaded = _load_metadata_from_path(path)
        if loaded is not None:
            return loaded
    return _DEFAULT_METADATA


def iter_node_type_values(values: Iterable[NodeType]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)


def iter_relation_values(values: Iterable[RelationType]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)
