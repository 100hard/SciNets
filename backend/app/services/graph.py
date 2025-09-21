from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Dict, Optional, Set
from uuid import UUID

from app.db.pool import get_pool
from app.models.graph import (
    GraphEdge,
    GraphEdgeData,
    GraphMeta,
    GraphNode,
    GraphNodeData,
    GraphResponse,
)


MAX_GRAPH_LIMIT = 500
_RELATION_LIMIT_MULTIPLIER = 3

_CONCEPT_SELECT_BASE = """
    SELECT c.id AS concept_id,
           c.name AS concept_name,
           c.type AS concept_type,
           c.description AS concept_description,
           c.paper_id AS paper_id,
           p.title AS paper_title,
           p.authors AS paper_authors,
           p.venue AS paper_venue,
           p.year AS paper_year
    FROM concepts c
    JOIN papers p ON p.id = c.paper_id
"""


class GraphEntityNotFoundError(RuntimeError):
    """Raised when the requested graph node cannot be located."""


def _normalize_limit(limit: int, minimum: int = 1, maximum: int = MAX_GRAPH_LIMIT) -> int:
    if limit < minimum:
        return minimum
    if limit > maximum:
        return maximum
    return limit


def _concept_node_id(concept_id: UUID) -> str:
    return f"concept:{concept_id}"


def _paper_node_id(paper_id: UUID) -> str:
    return f"paper:{paper_id}"


def _paper_concept_edge_id(paper_node_id: str, concept_node_id: str) -> str:
    return f"edge:{paper_node_id}->{concept_node_id}"


def _clean_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cleaned = {key: value for key, value in metadata.items() if value is not None}
    return cleaned or None


def _build_graph_elements(
    concept_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    *,
    nodes: MutableMapping[str, GraphNode] | None = None,
    edges: MutableMapping[str, GraphEdge] | None = None,
    include_paper_concept_edges: bool = True,
) -> tuple[MutableMapping[str, GraphNode], MutableMapping[str, GraphEdge]]:
    node_map: MutableMapping[str, GraphNode] = nodes if nodes is not None else OrderedDict()
    edge_map: MutableMapping[str, GraphEdge] = edges if edges is not None else OrderedDict()

    for row in concept_rows:
        concept_id: UUID = row["concept_id"]
        paper_id: UUID = row["paper_id"]
        concept_node_key = _concept_node_id(concept_id)
        if concept_node_key not in node_map:
            concept_metadata = _clean_metadata(
                {
                    "type": row.get("concept_type"),
                    "description": row.get("concept_description"),
                }
            )
            node_map[concept_node_key] = GraphNode(
                data=GraphNodeData(
                    id=concept_node_key,
                    type="concept",
                    label=row.get("concept_name") or str(concept_id),
                    concept_id=concept_id,
                    paper_id=paper_id,
                    metadata=concept_metadata,
                )
            )

        if not include_paper_concept_edges:
            continue

        paper_node_key = _paper_node_id(paper_id)
        if paper_node_key not in node_map:
            paper_metadata = _clean_metadata(
                {
                    "authors": row.get("paper_authors"),
                    "venue": row.get("paper_venue"),
                    "year": row.get("paper_year"),
                }
            )
            node_map[paper_node_key] = GraphNode(
                data=GraphNodeData(
                    id=paper_node_key,
                    type="paper",
                    label=row.get("paper_title") or str(paper_id),
                    paper_id=paper_id,
                    metadata=paper_metadata,
                )
            )

        edge_key = _paper_concept_edge_id(paper_node_key, concept_node_key)
        if edge_key not in edge_map:
            edge_map[edge_key] = GraphEdge(
                data=GraphEdgeData(
                    id=edge_key,
                    source=paper_node_key,
                    target=concept_node_key,
                    type="mentions",
                    paper_id=paper_id,
                    concept_id=concept_id,
                )
            )

    for row in relation_rows:
        relation_id: Optional[UUID] = row.get("id")
        source_concept_id: Optional[UUID] = row.get("concept_id")
        target_concept_id: Optional[UUID] = row.get("related_concept_id")
        if source_concept_id is None or target_concept_id is None:
            continue

        source_key = _concept_node_id(source_concept_id)
        target_key = _concept_node_id(target_concept_id)
        if source_key not in node_map or target_key not in node_map:
            continue

        edge_key = f"relation:{relation_id}" if relation_id is not None else f"relation:{source_key}->{target_key}"
        if edge_key in edge_map:
            continue

        relation_metadata = _clean_metadata({"description": row.get("description")})
        edge_map[edge_key] = GraphEdge(
            data=GraphEdgeData(
                id=edge_key,
                source=source_key,
                target=target_key,
                type=row.get("relation_type") or "related",
                paper_id=row.get("paper_id"),
                concept_id=source_concept_id,
                related_concept_id=target_concept_id,
                relation_id=relation_id,
                metadata=relation_metadata,
            )
        )

    return node_map, edge_map


async def get_graph_overview(
    limit: int = 100,
    *,
    paper_id: Optional[UUID] = None,
    concept_type: Optional[str] = None,
) -> GraphResponse:
    normalized_limit = _normalize_limit(limit)
    pool = get_pool()

    filters: Dict[str, Any] = {}
    params: list[Any] = []
    where_clauses: list[str] = []

    if paper_id:
        where_clauses.append(f"c.paper_id = ${len(params) + 1}")
        params.append(paper_id)
        filters["paper_id"] = str(paper_id)

    if concept_type:
        where_clauses.append(f"c.type = ${len(params) + 1}")
        params.append(concept_type)
        filters["concept_type"] = concept_type

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_placeholder = len(params) + 1
    concept_query = f"""{_CONCEPT_SELECT_BASE}
        {where_sql}
        ORDER BY c.created_at DESC
        LIMIT ${limit_placeholder}
    """

    async with pool.acquire() as conn:
        concept_rows = await conn.fetch(concept_query, *params, normalized_limit)
        count_query = f"SELECT COUNT(*) FROM concepts c {where_sql}"
        total_concepts = await conn.fetchval(count_query, *params)

        concept_ids = [row["concept_id"] for row in concept_rows]
        relation_rows: Sequence[Mapping[str, Any]] = []
        if concept_ids:
            relation_params: list[Any] = [concept_ids]
            relation_limit_placeholder = 2
            relation_where = "concept_id = ANY($1::uuid[]) AND related_concept_id = ANY($1::uuid[])"
            if paper_id:
                relation_where += f" AND paper_id = ${relation_limit_placeholder}"
                relation_params.append(paper_id)
                relation_limit_placeholder += 1
            relation_params.append(min(normalized_limit * _RELATION_LIMIT_MULTIPLIER, MAX_GRAPH_LIMIT))
            relations_query = f"""
                SELECT id, paper_id, concept_id, related_concept_id, relation_type, description
                FROM relations
                WHERE {relation_where}
                ORDER BY created_at DESC
                LIMIT ${relation_limit_placeholder}
            """
            relation_rows = await conn.fetch(relations_query, *relation_params)

    nodes, edges = _build_graph_elements(concept_rows, relation_rows)
    paper_ids: Set[UUID] = {
        node.data.paper_id
        for node in nodes.values()
        if node.data.type == "paper" and node.data.paper_id is not None
    }
    concept_node_count = sum(1 for node in nodes.values() if node.data.type == "concept")
    meta = GraphMeta(
        limit=normalized_limit,
        node_count=len(nodes),
        edge_count=len(edges),
        concept_count=total_concepts,
        paper_count=len(paper_ids),
        has_more=bool(total_concepts and total_concepts > concept_node_count),
        filters=filters or None,
    )
    return GraphResponse(nodes=list(nodes.values()), edges=list(edges.values()), meta=meta)


async def get_graph_neighborhood(node_id: UUID, *, limit: int = 50) -> GraphResponse:
    normalized_limit = _normalize_limit(limit)
    pool = get_pool()

    concept_query = f"""{_CONCEPT_SELECT_BASE}
        WHERE c.id = $1
    """

    async with pool.acquire() as conn:
        concept_row = await conn.fetchrow(concept_query, node_id)
        if concept_row:
            return await _build_concept_neighborhood(conn, concept_row, normalized_limit)

        paper_row = await conn.fetchrow(
            """
            SELECT p.id AS paper_id,
                   p.title AS paper_title,
                   p.authors AS paper_authors,
                   p.venue AS paper_venue,
                   p.year AS paper_year
            FROM papers p
            WHERE p.id = $1
            """,
            node_id,
        )
        if paper_row:
            return await _build_paper_neighborhood(conn, paper_row, normalized_limit)

    raise GraphEntityNotFoundError(f"Graph node {node_id} was not found")


async def _build_concept_neighborhood(
    conn: Any,
    concept_row: Mapping[str, Any],
    limit: int,
) -> GraphResponse:
    nodes, edges = _build_graph_elements([concept_row], [])
    paper_id: UUID = concept_row["paper_id"]
    concept_id: UUID = concept_row["concept_id"]

    neighbor_rows = await conn.fetch(
        f"""{_CONCEPT_SELECT_BASE}
            WHERE c.paper_id = $1
            ORDER BY c.created_at DESC
            LIMIT $2
        """,
        paper_id,
        limit,
    )
    nodes, edges = _build_graph_elements(neighbor_rows, [], nodes=nodes, edges=edges)

    total_concepts_for_paper = await conn.fetchval(
        "SELECT COUNT(*) FROM concepts WHERE paper_id = $1",
        paper_id,
    )

    relation_rows = await conn.fetch(
        """
        SELECT id, paper_id, concept_id, related_concept_id, relation_type, description
        FROM relations
        WHERE (concept_id = $1 OR related_concept_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        """,
        concept_id,
        min(limit * _RELATION_LIMIT_MULTIPLIER, MAX_GRAPH_LIMIT),
    )

    existing_concept_ids: Set[UUID] = {
        node.data.concept_id
        for node in nodes.values()
        if node.data.type == "concept" and node.data.concept_id is not None
    }
    missing_concept_ids: Set[UUID] = set()
    for row in relation_rows:
        source_id = row.get("concept_id")
        target_id = row.get("related_concept_id")
        if source_id and source_id not in existing_concept_ids:
            missing_concept_ids.add(source_id)
        if target_id and target_id not in existing_concept_ids:
            missing_concept_ids.add(target_id)

    if missing_concept_ids:
        extra_rows = await conn.fetch(
            f"""{_CONCEPT_SELECT_BASE}
                WHERE c.id = ANY($1::uuid[])
            """,
            list(missing_concept_ids),
        )
        nodes, edges = _build_graph_elements(extra_rows, [], nodes=nodes, edges=edges)

    nodes, edges = _build_graph_elements(
        [],
        relation_rows,
        nodes=nodes,
        edges=edges,
        include_paper_concept_edges=False,
    )

    same_paper_concepts = [
        node
        for node in nodes.values()
        if node.data.type == "concept" and node.data.paper_id == paper_id
    ]
    has_more = bool(total_concepts_for_paper and total_concepts_for_paper > len(same_paper_concepts))
    paper_ids: Set[UUID] = {
        node.data.paper_id
        for node in nodes.values()
        if node.data.type == "paper" and node.data.paper_id is not None
    }

    meta = GraphMeta(
        limit=limit,
        node_count=len(nodes),
        edge_count=len(edges),
        concept_count=total_concepts_for_paper,
        paper_count=len(paper_ids),
        has_more=has_more,
        center_id=_concept_node_id(concept_id),
        center_type="concept",
    )
    return GraphResponse(nodes=list(nodes.values()), edges=list(edges.values()), meta=meta)


async def _build_paper_neighborhood(
    conn: Any,
    paper_row: Mapping[str, Any],
    limit: int,
) -> GraphResponse:
    paper_id: UUID = paper_row["paper_id"]
    nodes: MutableMapping[str, GraphNode] = OrderedDict()
    edges: MutableMapping[str, GraphEdge] = OrderedDict()

    paper_metadata = _clean_metadata(
        {
            "authors": paper_row.get("paper_authors"),
            "venue": paper_row.get("paper_venue"),
            "year": paper_row.get("paper_year"),
        }
    )
    paper_key = _paper_node_id(paper_id)
    nodes[paper_key] = GraphNode(
        data=GraphNodeData(
            id=paper_key,
            type="paper",
            label=paper_row.get("paper_title") or str(paper_id),
            paper_id=paper_id,
            metadata=paper_metadata,
        )
    )

    concept_rows = await conn.fetch(
        f"""{_CONCEPT_SELECT_BASE}
            WHERE c.paper_id = $1
            ORDER BY c.created_at DESC
            LIMIT $2
        """,
        paper_id,
        limit,
    )
    nodes, edges = _build_graph_elements(concept_rows, [], nodes=nodes, edges=edges)

    total_concepts_for_paper = await conn.fetchval(
        "SELECT COUNT(*) FROM concepts WHERE paper_id = $1",
        paper_id,
    )

    concept_ids: Set[UUID] = {
        node.data.concept_id
        for node in nodes.values()
        if node.data.type == "concept" and node.data.concept_id is not None and node.data.paper_id == paper_id
    }

    relation_rows: Sequence[Mapping[str, Any]] = []
    if concept_ids:
        relation_rows = await conn.fetch(
            """
            SELECT id, paper_id, concept_id, related_concept_id, relation_type, description
            FROM relations
            WHERE paper_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            paper_id,
            min(limit * _RELATION_LIMIT_MULTIPLIER, MAX_GRAPH_LIMIT),
        )

    nodes, edges = _build_graph_elements(
        [],
        [
            row
            for row in relation_rows
            if row.get("concept_id") in concept_ids
            and row.get("related_concept_id") in concept_ids
        ],
        nodes=nodes,
        edges=edges,
        include_paper_concept_edges=False,
    )

    paper_ids: Set[UUID] = {
        node.data.paper_id
        for node in nodes.values()
        if node.data.type == "paper" and node.data.paper_id is not None
    }
    has_more = bool(total_concepts_for_paper and total_concepts_for_paper > len(concept_ids))

    meta = GraphMeta(
        limit=limit,
        node_count=len(nodes),
        edge_count=len(edges),
        concept_count=total_concepts_for_paper,
        paper_count=len(paper_ids),
        has_more=has_more,
        center_id=paper_key,
        center_type="paper",
    )
    return GraphResponse(nodes=list(nodes.values()), edges=list(edges.values()), meta=meta)
