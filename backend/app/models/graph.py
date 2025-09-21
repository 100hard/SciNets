from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


NodeType = Literal["method", "dataset", "metric", "task"]
RelationType = Literal["proposes", "evaluates_on", "reports", "compares"]


class GraphNodeLink(BaseModel):
    id: str
    label: str
    type: NodeType
    relation: RelationType
    weight: float


class GraphEvidenceItem(BaseModel):
    paper_id: UUID
    paper_title: Optional[str] = None
    snippet: Optional[str] = None
    confidence: float
    relation: RelationType


class GraphNodeData(BaseModel):
    id: str
    type: NodeType
    label: str
    entity_id: UUID
    paper_count: int
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    top_links: List[GraphNodeLink] = Field(default_factory=list)
    evidence: List[GraphEvidenceItem] = Field(default_factory=list)
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
    type: RelationType
    weight: float
    paper_count: int
    average_confidence: float
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
