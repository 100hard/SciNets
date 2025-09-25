from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, cast
from uuid import UUID, uuid4, uuid5

from app.db.pool import get_pool
from app.models.graph import (
    GraphEdge,
    GraphEdgeData,
    GraphEvidenceItem,
    GraphMeta,
    GraphNode,
    GraphNodeData,
    GraphNodeLink,
    GraphResponse,
    NodeType,
    RelationType,
)


from typing import Optional
MAX_GRAPH_LIMIT = 500
_DEFAULT_TYPES: tuple[NodeType, ...] = ("method", "dataset", "metric", "task")
_ALLOWED_TYPES = set(_DEFAULT_TYPES)
_ALLOWED_RELATIONS: set[RelationType] = {"proposes", "evaluates_on", "reports", "compares"}
_DEFAULT_MIN_CONFIDENCE = 0.6
_MAX_NODE_EVIDENCE = 8
_MAX_EDGE_EVIDENCE = 12
_MAX_TOP_LINKS = 6
_ORDERED_RELATIONS: tuple[RelationType, ...] = ("proposes", "evaluates_on", "reports", "compares")
_CONCEPT_TYPES: tuple[str, ...] = ("method", "dataset", "metric", "task")
_FALLBACK_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")

_RESULT_SELECT = """
    SELECT
        r.id AS result_id,
        r.paper_id,
        r.method_id,
        r.dataset_id,
        r.metric_id,
        r.task_id,
        r.confidence,
        r.evidence,
        p.title AS paper_title,
        m.name AS method_name,
        m.aliases AS method_aliases,
        m.description AS method_description,
        d.name AS dataset_name,
        d.aliases AS dataset_aliases,
        d.description AS dataset_description,
        mt.name AS metric_name,
        mt.aliases AS metric_aliases,
        mt.description AS metric_description,
        mt.unit AS metric_unit,
        t.name AS task_name,
        t.aliases AS task_aliases,
        t.description AS task_description
    FROM results r
    JOIN papers p ON p.id = r.paper_id
    LEFT JOIN methods m ON r.method_id = m.id
    LEFT JOIN datasets d ON r.dataset_id = d.id
    LEFT JOIN metrics mt ON r.metric_id = mt.id
    LEFT JOIN tasks t ON r.task_id = t.id
"""


class GraphEntityNotFoundError(RuntimeError):
    """Raised when the requested graph node cannot be located."""


@dataclass
class NodeDetail:
    id: UUID
    type: NodeType
    label: str
    aliases: tuple[str, ...]
    description: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class EdgeInstance:
    paper_id: UUID
    paper_title: Optional[str]
    confidence: float
    evidence: list[dict[str, Any]]
    dataset_id: Optional[UUID]
    dataset_label: Optional[str]
    metric_id: Optional[UUID]
    metric_label: Optional[str]
    task_id: Optional[UUID]
    task_label: Optional[str]


@dataclass
class MethodContext:
    method_id: UUID
    method_label: str
    confidence: float
    evidence: list[dict[str, Any]]
    paper_id: UUID
    paper_title: Optional[str]
    dataset_id: Optional[UUID]
    dataset_label: Optional[str]
    metric_id: Optional[UUID]
    metric_label: Optional[str]
    task_id: Optional[UUID]
    task_label: Optional[str]


@dataclass
class AggregatedEdge:
    relation: RelationType
    source_key: tuple[NodeType, UUID]
    target_key: tuple[NodeType, UUID]
    paper_details: list[dict[str, Any]]
    average_confidence: float
    weight: float
    evidence: list[dict[str, Any]]
    metadata: Dict[str, Any]

    @property
    def paper_count(self) -> int:
        return len(self.paper_details)


def _normalize_limit(limit: int, *, minimum: int = 1, maximum: int = MAX_GRAPH_LIMIT) -> int:
    if limit < minimum:
        return minimum
    if limit > maximum:
        return maximum
    return limit


def _node_key(node_type: NodeType, entity_id: UUID) -> tuple[NodeType, UUID]:
    return node_type, entity_id


def _node_id_str(node_type: NodeType, entity_id: UUID) -> str:
    return f"{node_type}:{entity_id}"


def _parse_selection(
    values: Optional[Sequence[str]],
    allowed: Iterable[str],
    default: Iterable[str],
) -> set[str]:
    allowed_set = {item.lower() for item in allowed}
    if not values:
        return {item.lower() for item in default}

    selection: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        for token in str(raw).split(","):
            normalized = token.strip().lower()
            if not normalized:
                continue
            if normalized not in allowed_set:
                raise ValueError(f"Unsupported selection value '{token}'")
            selection.add(normalized)
    return selection or {item.lower() for item in default}


def _confidence_value(value: Any) -> float:
    try:
        if value is None:
            return float(_DEFAULT_MIN_CONFIDENCE)
        numeric = float(value)
        if numeric < 0:
            return 0.0
        if numeric > 1:
            return 1.0
        return numeric
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return float(_DEFAULT_MIN_CONFIDENCE)


def _coerce_aliases(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return tuple(dict.fromkeys(cleaned))
    if isinstance(value, str):
        return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    return ()


def _safe_label(name: Any, fallback_prefix: str, entity_id: UUID) -> str:
    cleaned = str(name).strip() if name is not None else ""
    if cleaned:
        return cleaned
    return f"{fallback_prefix} {entity_id}"


def _clean_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "")}


async def _fetch_results(
    conn: Any,
    *,
    paper_ids: Optional[Sequence[UUID]] = None,
) -> Sequence[Mapping[str, Any]]:
    if paper_ids:
        return await conn.fetch(
            f"{_RESULT_SELECT} WHERE r.paper_id = ANY($1::uuid[])",
            list(paper_ids),
        )
    return await conn.fetch(_RESULT_SELECT)


async def _fetch_concept_fallback_rows(
    conn: Any,
    *,
    paper_ids: Optional[Sequence[UUID]] = None,
) -> list[Mapping[str, Any]]:
    concept_types = _CONCEPT_TYPES
    if not concept_types:
        return []

    params: list[Any] = [list(concept_types)]
    where_clause = "WHERE c.type = ANY($1::text[])"
    if paper_ids:
        params.append(list(paper_ids))
        where_clause += " AND c.paper_id = ANY($2::uuid[])"

    records = await conn.fetch(
        f"""
        SELECT
            c.id,
            c.paper_id,
            c.name,
            c.type,
            p.title AS paper_title,
            p.created_at,
            c.created_at AS concept_created_at
        FROM concepts c
        JOIN papers p ON p.id = c.paper_id
        {where_clause}
        ORDER BY p.created_at DESC, concept_created_at DESC
        LIMIT 2000
        """,
        *params,
    )

    grouped: dict[UUID, dict[str, Any]] = {}
    for record in records:
        concept_type = (record.get("type") or "").lower()
        if concept_type not in _CONCEPT_TYPES:
            continue
        paper_id: UUID = record["paper_id"]
        bucket = grouped.setdefault(
            paper_id,
            {
                "paper_title": record.get("paper_title"),
                "method": [],
                "dataset": [],
                "metric": [],
                "task": [],
            },
        )
        bucket[concept_type].append(
            {
                "id": record["id"],
                "name": str(record.get("name") or "").strip(),
                "type": concept_type,
                "metadata": {"concept": True},
            }
        )

    fallback_rows: list[Mapping[str, Any]] = []
    for paper_id, payload in grouped.items():
        methods = payload.get("method", [])[:4]
        if not methods:
            continue

        datasets = payload.get("dataset", [])[:3]
        metrics = payload.get("metric", [])[:3]
        tasks = payload.get("task", [])[:3]
        paper_title = payload.get("paper_title")

        placeholder_dataset: Optional[Mapping[str, Any]] = None
        placeholder_metric: Optional[Mapping[str, Any]] = None

        if not datasets:
            placeholder_dataset = {
                "id": uuid5(_FALLBACK_NAMESPACE, f"{paper_id}:dataset"),
                "name": "Dataset (unspecified)",
                "type": "dataset",
                "metadata": {"placeholder": True},
            }
            datasets = [placeholder_dataset]

        if not metrics:
            placeholder_metric = {
                "id": uuid5(_FALLBACK_NAMESPACE, f"{paper_id}:metric"),
                "name": "Metric (unspecified)",
                "type": "metric",
                "metadata": {"placeholder": True},
            }
            metrics = [placeholder_metric]

        def _append_row(
            *,
            method: Mapping[str, Any],
            dataset: Optional[Mapping[str, Any]],
            metric: Optional[Mapping[str, Any]],
            task: Optional[Mapping[str, Any]],
        ) -> None:
            fallback_rows.append(
                {
                    "result_id": uuid4(),
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "confidence": 0.8,
                    "evidence": [],
                    "method_id": method["id"],
                    "method_name": method.get("name"),
                    "method_aliases": [],
                    "method_description": None,
                    "method_metadata": method.get("metadata"),
                    "dataset_id": dataset.get("id") if dataset else None,
                    "dataset_name": dataset.get("name") if dataset else None,
                    "dataset_aliases": [],
                    "dataset_description": None,
                    "dataset_metadata": dataset.get("metadata") if dataset else None,
                    "metric_id": metric.get("id") if metric else None,
                    "metric_name": metric.get("name") if metric else None,
                    "metric_aliases": [],
                    "metric_description": None,
                    "metric_unit": None,
                    "metric_metadata": metric.get("metadata") if metric else None,
                    "task_id": task.get("id") if task else None,
                    "task_name": task.get("name") if task else None,
                    "task_aliases": [],
                    "task_description": None,
                    "task_metadata": task.get("metadata") if task else None,
                }
            )

        for method in methods:
            if datasets or metrics:
                dataset_candidates = datasets or [None]
                metric_candidates = metrics or [None]
                for dataset in dataset_candidates:
                    for metric in metric_candidates:
                        if not dataset and not metric:
                            continue
                        _append_row(method=method, dataset=dataset, metric=metric, task=None)

            if tasks:
                for task in tasks:
                    _append_row(method=method, dataset=None, metric=None, task=task)

        # Avoid creating placeholder-only rows when there was a single method and no
        # additional context available.
        if (
            placeholder_dataset
            and placeholder_metric
            and not tasks
            and len(methods) == 1
        ):
            # For lone methods fall back to a single generic task edge so the
            # node appears in the graph.
            fallback_rows.append(
                {
                    "result_id": uuid4(),
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "confidence": 0.8,
                    "evidence": [],
                    "method_id": methods[0]["id"],
                    "method_name": methods[0].get("name"),
                    "method_aliases": [],
                    "method_description": None,
                    "method_metadata": methods[0].get("metadata"),
                    "dataset_id": None,
                    "dataset_name": None,
                    "dataset_aliases": [],
                    "dataset_description": None,
                    "metric_id": None,
                    "metric_name": None,
                    "metric_aliases": [],
                    "metric_description": None,
                    "metric_unit": None,
                    "task_id": uuid5(_FALLBACK_NAMESPACE, f"{paper_id}:task"),
                    "task_name": "Task (unspecified)",
                    "task_aliases": [],
                    "task_description": None,
                    "task_metadata": {"placeholder": True},
                }
            )

    return fallback_rows


async def _resolve_entity(conn: Any, entity_id: UUID) ->Optional[NodeDetail]:
    row = await conn.fetchrow(
        "SELECT id, name, aliases, description FROM methods WHERE id = $1",
        entity_id,
    )
    if row:
        return NodeDetail(
            id=row["id"],
            type="method",
            label=_safe_label(row.get("name"), "Method", row["id"]),
            aliases=_coerce_aliases(row.get("aliases")),
            description=row.get("description"),
            metadata={},
        )

    row = await conn.fetchrow(
        "SELECT id, name, aliases, description FROM datasets WHERE id = $1",
        entity_id,
    )
    if row:
        return NodeDetail(
            id=row["id"],
            type="dataset",
            label=_safe_label(row.get("name"), "Dataset", row["id"]),
            aliases=_coerce_aliases(row.get("aliases")),
            description=row.get("description"),
            metadata={},
        )

    row = await conn.fetchrow(
        "SELECT id, name, aliases, description, unit FROM metrics WHERE id = $1",
        entity_id,
    )
    if row:
        metadata = _clean_metadata({"unit": row.get("unit")})
        return NodeDetail(
            id=row["id"],
            type="metric",
            label=_safe_label(row.get("name"), "Metric", row["id"]),
            aliases=_coerce_aliases(row.get("aliases")),
            description=row.get("description"),
            metadata=metadata,
        )

    row = await conn.fetchrow(
        "SELECT id, name, aliases, description FROM tasks WHERE id = $1",
        entity_id,
    )
    if row:
        return NodeDetail(
            id=row["id"],
            type="task",
            label=_safe_label(row.get("name"), "Task", row["id"]),
            aliases=_coerce_aliases(row.get("aliases")),
            description=row.get("description"),
            metadata={},
        )

    row = await conn.fetchrow(
        "SELECT id, name, type, paper_id, description FROM concepts WHERE id = $1",
        entity_id,
    )
    if row:
        concept_type = str(row.get("type") or "").lower()
        if concept_type in _CONCEPT_TYPES:
            node_type = cast(NodeType, concept_type)
            metadata = {
                "concept": True,
                "paper_id": str(row.get("paper_id")) if row.get("paper_id") else None,
            }
            metadata = _clean_metadata(metadata)
            return NodeDetail(
                id=row["id"],
                type=node_type,
                label=_safe_label(row.get("name"), node_type.title(), row["id"]),
                aliases=(),
                description=row.get("description"),
                metadata=metadata,
            )

    return None


async def _fetch_related_papers(conn: Any, center: NodeDetail) -> list[UUID]:
    column = {
        "method": "method_id",
        "dataset": "dataset_id",
        "metric": "metric_id",
        "task": "task_id",
    }[center.type]
    rows = await conn.fetch(
        f"SELECT DISTINCT paper_id FROM results WHERE {column} = $1",
        center.id,
    )
    paper_ids = [row["paper_id"] for row in rows]
    if paper_ids:
        return paper_ids

    concept_row = await conn.fetchrow(
        "SELECT paper_id FROM concepts WHERE id = $1",
        center.id,
    )
    if concept_row and concept_row.get("paper_id"):
        return [concept_row["paper_id"]]
    return []


def _build_node_detail(
    node_details: dict[tuple[NodeType, UUID], NodeDetail],
    *,
    node_type: NodeType,
    entity_id: UUID,
    name: Any,
    aliases: Any,
    description: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    key = _node_key(node_type, entity_id)
    if key in node_details:
        return
    detail = NodeDetail(
        id=entity_id,
        type=node_type,
        label=_safe_label(name, node_type.title(), entity_id),
        aliases=_coerce_aliases(aliases),
        description=str(description).strip() if description else None,
        metadata=_clean_metadata(dict(metadata or {})),
    )
    node_details[key] = detail


def _aggregate_edges(
    rows: Sequence[Mapping[str, Any]],
    allowed_types: set[str],
    allowed_relations: set[str],
    min_conf: float,
    node_details: dict[tuple[NodeType, UUID], NodeDetail],
) -> list[AggregatedEdge]:
    edges: dict[
        tuple[RelationType, tuple[NodeType, UUID], tuple[NodeType, UUID]],
        list[EdgeInstance],
    ] = defaultdict(list)
    compare_contexts: dict[
        tuple[UUID, UUID, UUID],
        list[MethodContext],
    ] = defaultdict(list)

    for row in rows:
        paper_id: UUID = row["paper_id"]
        paper_title = row.get("paper_title")
        confidence = _confidence_value(row.get("confidence"))
        evidence = list(row.get("evidence") or [])

        method_id: Optional[UUID] = row.get("method_id")
        dataset_id: Optional[UUID] = row.get("dataset_id")
        metric_id: Optional[UUID] = row.get("metric_id")
        task_id: Optional[UUID] = row.get("task_id")

        method_metadata = row.get("method_metadata")
        if method_id:
            _build_node_detail(
                node_details,
                node_type="method",
                entity_id=method_id,
                name=row.get("method_name"),
                aliases=row.get("method_aliases"),
                description=row.get("method_description"),
                metadata=method_metadata,
            )
        dataset_metadata = row.get("dataset_metadata")
        if dataset_id:
            _build_node_detail(
                node_details,
                node_type="dataset",
                entity_id=dataset_id,
                name=row.get("dataset_name"),
                aliases=row.get("dataset_aliases"),
                description=row.get("dataset_description"),
                metadata=dataset_metadata,
            )
        metric_metadata = row.get("metric_metadata")
        if metric_id:
            _build_node_detail(
                node_details,
                node_type="metric",
                entity_id=metric_id,
                name=row.get("metric_name"),
                aliases=row.get("metric_aliases"),
                description=row.get("metric_description"),
                metadata={"unit": row.get("metric_unit")} | (metric_metadata or {}),
            )
        task_metadata = row.get("task_metadata")
        if task_id:
            _build_node_detail(
                node_details,
                node_type="task",
                entity_id=task_id,
                name=row.get("task_name"),
                aliases=row.get("task_aliases"),
                description=row.get("task_description"),
                metadata=task_metadata,
            )

        dataset_label = node_details.get(("dataset", dataset_id)).label if dataset_id and ("dataset", dataset_id) in node_details else None
        metric_label = node_details.get(("metric", metric_id)).label if metric_id and ("metric", metric_id) in node_details else None
        task_label = node_details.get(("task", task_id)).label if task_id and ("task", task_id) in node_details else None

        if method_id and dataset_id:
            key = ("evaluates_on", _node_key("method", method_id), _node_key("dataset", dataset_id))
            edges[key].append(
                EdgeInstance(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    confidence=confidence,
                    evidence=evidence,
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    metric_id=metric_id,
                    metric_label=metric_label,
                    task_id=task_id,
                    task_label=task_label,
                )
            )

        if method_id and metric_id:
            key = ("reports", _node_key("method", method_id), _node_key("metric", metric_id))
            edges[key].append(
                EdgeInstance(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    confidence=confidence,
                    evidence=evidence,
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    metric_id=metric_id,
                    metric_label=metric_label,
                    task_id=task_id,
                    task_label=task_label,
                )
            )

        if method_id and task_id:
            key = ("proposes", _node_key("method", method_id), _node_key("task", task_id))
            edges[key].append(
                EdgeInstance(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    confidence=confidence,
                    evidence=evidence,
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    metric_id=metric_id,
                    metric_label=metric_label,
                    task_id=task_id,
                    task_label=task_label,
                )
            )

        if method_id and dataset_id and metric_id:
            compare_contexts[(paper_id, dataset_id, metric_id)].append(
                MethodContext(
                    method_id=method_id,
                    method_label=node_details[("method", method_id)].label,
                    confidence=confidence,
                    evidence=evidence,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    metric_id=metric_id,
                    metric_label=metric_label,
                    task_id=task_id,
                    task_label=task_label,
                )
            )

    for contexts in compare_contexts.values():
        if len(contexts) < 2:
            continue
        contexts = sorted(contexts, key=lambda ctx: ctx.method_id)
        for index, primary in enumerate(contexts):
            for secondary in contexts[index + 1 :]:
                if primary.method_id == secondary.method_id:
                    continue
                source_key = _node_key("method", primary.method_id)
                target_key = _node_key("method", secondary.method_id)
                key = ("compares", source_key, target_key)
                combined_confidence = (primary.confidence + secondary.confidence) / 2
                combined_evidence = list(primary.evidence) + list(secondary.evidence)
                edges[key].append(
                    EdgeInstance(
                        paper_id=primary.paper_id,
                        paper_title=primary.paper_title,
                        confidence=combined_confidence,
                        evidence=combined_evidence,
                        dataset_id=primary.dataset_id,
                        dataset_label=primary.dataset_label,
                        metric_id=primary.metric_id,
                        metric_label=primary.metric_label,
                        task_id=primary.task_id,
                        task_label=primary.task_label,
                    )
                )

    aggregated: list[AggregatedEdge] = []
    for (relation, source_key, target_key), instances in edges.items():
        if relation not in allowed_relations:
            continue
        if source_key[0] not in allowed_types or target_key[0] not in allowed_types:
            continue

        paper_confidences: dict[UUID, list[float]] = defaultdict(list)
        paper_titles: dict[UUID, str] = {}
        contexts_map: dict[tuple[UUID, Optional[UUID], Optional[UUID], Optional[UUID]], dict[str, Any]] = {}
        evidence_items: list[dict[str, Any]] = []

        for instance in instances:
            paper_confidences[instance.paper_id].append(instance.confidence)
            paper_titles.setdefault(instance.paper_id, instance.paper_title)
            context_key = (
                instance.paper_id,
                instance.dataset_id,
                instance.metric_id,
                instance.task_id,
            )
            if context_key not in contexts_map:
                context: dict[str, Any] = {
                    "paper_id": instance.paper_id,
                    "paper_title": instance.paper_title,
                }
                if instance.dataset_label:
                    context["dataset"] = instance.dataset_label
                if instance.metric_label:
                    context["metric"] = instance.metric_label
                if instance.task_label:
                    context["task"] = instance.task_label
                contexts_map[context_key] = context
            for snippet in instance.evidence:
                snippet_text = None
                if isinstance(snippet, Mapping):
                    snippet_text = snippet.get("snippet")
                elif isinstance(snippet, str):
                    snippet_text = snippet
                evidence_items.append(
                    {
                        "paper_id": instance.paper_id,
                        "paper_title": instance.paper_title,
                        "snippet": snippet_text,
                        "confidence": instance.confidence,
                        "relation": relation,
                    }
                )

        if not paper_confidences:
            continue

        per_paper_avg = [sum(values) / len(values) for values in paper_confidences.values()]
        average_confidence = sum(per_paper_avg) / len(per_paper_avg)
        if average_confidence < min_conf:
            continue

        paper_details = [
            {
                "paper_id": paper_id,
                "paper_title": paper_titles.get(paper_id),
                "average_confidence": sum(values) / len(values),
            }
            for paper_id, values in paper_confidences.items()
        ]
        weight = len(paper_details) * average_confidence
        metadata: dict[str, Any] = {"papers": paper_details}
        if contexts_map:
            metadata["contexts"] = list(contexts_map.values())

        evidence_items = evidence_items[:_MAX_EDGE_EVIDENCE]
        if evidence_items:
            metadata["evidence"] = evidence_items

        aggregated.append(
            AggregatedEdge(
                relation=relation,
                source_key=source_key,
                target_key=target_key,
                paper_details=paper_details,
                average_confidence=average_confidence,
                weight=weight,
                evidence=evidence_items,
                metadata=metadata,
            )
        )

    return aggregated


def _build_graph_response(
    aggregated_edges: list[AggregatedEdge],
    node_details: dict[tuple[NodeType, UUID], NodeDetail],
    *,
    limit: int,
    allowed_types: set[str],
    allowed_relations: set[str],
    min_conf: float,
    center_key: Optional[tuple[NodeType, UUID]] = None,
) -> GraphResponse:
    node_papers: dict[tuple[NodeType, UUID], set[UUID]] = defaultdict(set)
    node_edges: dict[tuple[NodeType, UUID], list[AggregatedEdge]] = defaultdict(list)

    for edge in aggregated_edges:
        source_key = edge.source_key
        target_key = edge.target_key
        papers = {detail["paper_id"] for detail in edge.paper_details}
        node_papers[source_key].update(papers)
        node_papers[target_key].update(papers)
        node_edges[source_key].append(edge)
        node_edges[target_key].append(edge)

    if center_key and center_key not in node_papers:
        node_papers[center_key] = set()
        node_edges.setdefault(center_key, [])

    all_node_keys = list(node_papers.keys())
    total_nodes = len(all_node_keys)

    def sort_key(item: tuple[NodeType, UUID]) -> tuple[int, str]:
        paper_count = len(node_papers[item])
        label = node_details.get(item)
        return (-paper_count, (label.label if label else str(item[1])).lower())

    sorted_nodes = sorted(all_node_keys, key=sort_key)

    selected: list[tuple[NodeType, UUID]] = []
    if center_key:
        selected.append(center_key)
    for key in sorted_nodes:
        if key == center_key:
            continue
        selected.append(key)
        if len(selected) >= limit:
            break

    allowed_node_keys = set(selected)
    filtered_edges = [
        edge
        for edge in aggregated_edges
        if edge.source_key in allowed_node_keys and edge.target_key in allowed_node_keys
    ]

    node_links: dict[tuple[NodeType, UUID], list[GraphNodeLink]] = defaultdict(list)
    node_evidence: dict[tuple[NodeType, UUID], list[GraphEvidenceItem]] = defaultdict(list)

    for edge in filtered_edges:
        source_key = edge.source_key
        target_key = edge.target_key
        source_detail = node_details.get(source_key)
        target_detail = node_details.get(target_key)
        if source_detail and target_detail:
            node_links[source_key].append(
                GraphNodeLink(
                    id=_node_id_str(target_key[0], target_key[1]),
                    label=target_detail.label,
                    type=target_detail.type,
                    relation=edge.relation,
                    weight=edge.weight,
                )
            )
            node_links[target_key].append(
                GraphNodeLink(
                    id=_node_id_str(source_key[0], source_key[1]),
                    label=source_detail.label,
                    type=source_detail.type,
                    relation=edge.relation,
                    weight=edge.weight,
                )
            )

        for evidence in edge.evidence:
            item = GraphEvidenceItem(
                paper_id=evidence["paper_id"],
                paper_title=evidence.get("paper_title"),
                snippet=evidence.get("snippet"),
                confidence=evidence.get("confidence", edge.average_confidence),
                relation=edge.relation,
            )
            node_evidence[source_key].append(item)
            node_evidence[target_key].append(item)

    nodes: list[GraphNode] = []
    for key in selected:
        detail = node_details.get(key)
        if not detail:
            detail = NodeDetail(
                id=key[1],
                type=key[0],
                label=_safe_label(None, key[0].title(), key[1]),
                aliases=(),
                description=None,
                metadata={},
            )

        papers = node_papers.get(key, set())
        links = sorted(
            node_links.get(key, []),
            key=lambda link: (-link.weight, link.label.lower()),
        )[:_MAX_TOP_LINKS]

        evidences = node_evidence.get(key, [])[:_MAX_NODE_EVIDENCE]

        nodes.append(
            GraphNode(
                data=GraphNodeData(
                    id=_node_id_str(key[0], key[1]),
                    type=detail.type,
                    label=detail.label,
                    entity_id=detail.id,
                    paper_count=len(papers),
                    aliases=list(detail.aliases),
                    description=detail.description,
                    top_links=links,
                    evidence=evidences,
                    metadata=detail.metadata or None,
                )
            )
        )

    edges: list[GraphEdge] = []
    for edge in sorted(filtered_edges, key=lambda item: (-item.weight, item.relation, item.source_key[0])):
        metadata = dict(edge.metadata)
        if not metadata.get("papers"):
            metadata.pop("papers", None)
        if not metadata.get("contexts"):
            metadata.pop("contexts", None)
        if not metadata.get("evidence"):
            metadata.pop("evidence", None)

        edges.append(
            GraphEdge(
                data=GraphEdgeData(
                    id=f"edge:{edge.relation}:{_node_id_str(edge.source_key[0], edge.source_key[1])}->{_node_id_str(edge.target_key[0], edge.target_key[1])}",
                    source=_node_id_str(edge.source_key[0], edge.source_key[1]),
                    target=_node_id_str(edge.target_key[0], edge.target_key[1]),
                    type=edge.relation,
                    weight=edge.weight,
                    paper_count=edge.paper_count,
                    average_confidence=edge.average_confidence,
                    metadata=metadata or None,
                )
            )
        )

    paper_ids = set()
    for key in allowed_node_keys:
        paper_ids.update(node_papers.get(key, set()))

    has_more = total_nodes > len(selected)
    ordered_types = [item for item in _DEFAULT_TYPES if item in allowed_types]
    extra_types = sorted(allowed_types - set(_DEFAULT_TYPES))
    ordered_relations = [item for item in _ORDERED_RELATIONS if item in allowed_relations]
    extra_relations = sorted(allowed_relations - set(_ORDERED_RELATIONS))

    meta = GraphMeta(
        limit=limit,
        node_count=len(nodes),
        edge_count=len(edges),
        concept_count=total_nodes,
        paper_count=len(paper_ids),
        has_more=has_more if has_more else None,
        center_id=_node_id_str(center_key[0], center_key[1]) if center_key else None,
        center_type=center_key[0] if center_key else None,
        filters={
            "types": ordered_types + extra_types,
            "relations": ordered_relations + extra_relations,
            "min_conf": min_conf,
        },
    )

    return GraphResponse(nodes=nodes, edges=edges, meta=meta)


async def get_graph_overview(
    limit: int = 100,
    *,
    types: Optional[Sequence[str]] = None,
    relations: Optional[Sequence[str]] = None,
    min_conf: float = _DEFAULT_MIN_CONFIDENCE,
) -> GraphResponse:
    normalized_limit = _normalize_limit(limit, maximum=MAX_GRAPH_LIMIT)
    allowed_types = _parse_selection(types, _ALLOWED_TYPES, _DEFAULT_TYPES)
    allowed_relations = _parse_selection(relations, _ALLOWED_RELATIONS, _ALLOWED_RELATIONS)

    pool = get_pool()
    async with pool.acquire() as conn:
        records = await _fetch_results(conn)
        rows = [dict(record) for record in records]
        node_details: dict[tuple[NodeType, UUID], NodeDetail] = {}
        aggregated_edges = _aggregate_edges(rows, allowed_types, allowed_relations, min_conf, node_details)
    response = _build_graph_response(
        aggregated_edges,
        node_details,
        limit=normalized_limit,
        allowed_types=allowed_types,
        allowed_relations=allowed_relations,
        min_conf=min_conf,
    )

    if response.nodes or response.edges:
        return response

    async with pool.acquire() as conn:
        fallback_rows = await _fetch_concept_fallback_rows(conn)

    if not fallback_rows:
        return response

    combined_rows = fallback_rows + rows
    node_details = {}
    fallback_edges = _aggregate_edges(combined_rows, allowed_types, allowed_relations, min_conf, node_details)
    fallback_response = _build_graph_response(
        fallback_edges,
        node_details,
        limit=normalized_limit,
        allowed_types=allowed_types,
        allowed_relations=allowed_relations,
        min_conf=min_conf,
    )

    if fallback_response.nodes or fallback_response.edges:
        return fallback_response
    return response


async def get_graph_neighborhood(
    node_id: UUID,
    *,
    limit: int = 50,
    types: Optional[Sequence[str]] = None,
    relations: Optional[Sequence[str]] = None,
    min_conf: float = _DEFAULT_MIN_CONFIDENCE,
) -> GraphResponse:
    normalized_limit = _normalize_limit(limit, maximum=MAX_GRAPH_LIMIT)
    allowed_types = _parse_selection(types, _ALLOWED_TYPES, _DEFAULT_TYPES)
    allowed_relations = _parse_selection(relations, _ALLOWED_RELATIONS, _ALLOWED_RELATIONS)

    pool = get_pool()
    async with pool.acquire() as conn:
        center_detail = await _resolve_entity(conn, node_id)
        if center_detail is None:
            raise GraphEntityNotFoundError(f"Graph node {node_id} was not found")

        paper_ids = await _fetch_related_papers(conn, center_detail)
        records = await _fetch_results(conn, paper_ids=paper_ids)
        rows = [dict(record) for record in records]

    allowed_types.add(center_detail.type)

    node_details: dict[tuple[NodeType, UUID], NodeDetail] = {(_node_key(center_detail.type, center_detail.id)): center_detail}
    aggregated_edges = _aggregate_edges(rows, allowed_types, allowed_relations, min_conf, node_details)
    response = _build_graph_response(
        aggregated_edges,
        node_details,
        limit=normalized_limit,
        allowed_types=allowed_types,
        allowed_relations=allowed_relations,
        min_conf=min_conf,
        center_key=_node_key(center_detail.type, center_detail.id),
    )

    if response.nodes or response.edges:
        return response

    if not paper_ids:
        return response

    pool = get_pool()
    async with pool.acquire() as conn:
        fallback_rows = await _fetch_concept_fallback_rows(conn, paper_ids=paper_ids)

    if not fallback_rows:
        return response

    combined_rows = fallback_rows + rows
    node_details = {(_node_key(center_detail.type, center_detail.id)): center_detail}
    fallback_edges = _aggregate_edges(combined_rows, allowed_types, allowed_relations, min_conf, node_details)
    fallback_response = _build_graph_response(
        fallback_edges,
        node_details,
        limit=normalized_limit,
        allowed_types=allowed_types,
        allowed_relations=allowed_relations,
        min_conf=min_conf,
        center_key=_node_key(center_detail.type, center_detail.id),
    )

    if fallback_response.nodes or fallback_response.edges:
        return fallback_response
    return response