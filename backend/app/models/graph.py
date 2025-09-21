from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel


NodeType = Literal["paper", "concept"]


class GraphNodeData(BaseModel):
    id: str
    type: NodeType
    label: str
    paper_id: Optional[UUID] = None
    concept_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class GraphNode(BaseModel):
    data: GraphNodeData

    class Config:
        from_attributes = True


class GraphEdgeData(BaseModel):
    id: str
    source: str
    target: str
    type: str
    paper_id: Optional[UUID] = None
    concept_id: Optional[UUID] = None
    related_concept_id: Optional[UUID] = None
    relation_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class GraphEdge(BaseModel):
    data: GraphEdgeData

    class Config:
        from_attributes = True


class GraphMeta(BaseModel):
    limit: int
    node_count: int
    edge_count: int
    concept_count: Optional[int] = None
    paper_count: Optional[int] = None
    has_more: Optional[bool] = None
    center_id: Optional[str] = None
    center_type: Optional[NodeType] = None
    filters: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    meta: GraphMeta

    class Config:
        from_attributes = True
