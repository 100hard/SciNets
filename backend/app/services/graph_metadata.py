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
    ),
    allowed_relations=(
        RelationType.PROPOSES,
        RelationType.EVALUATES_ON,
        RelationType.REPORTS,
        RelationType.COMPARES,
    ),
    ordered_relations=(
        RelationType.PROPOSES,
        RelationType.EVALUATES_ON,
        RelationType.REPORTS,
        RelationType.COMPARES,
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
    ),
)


def _load_metadata_from_path(path: Path) -> GraphOntologyMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return GraphOntologyMetadata.model_validate(payload)
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
