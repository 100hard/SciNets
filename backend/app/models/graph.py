from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    CONCEPT = "concept"
    MATERIAL = "material"
    ORGANISM = "organism"
    FINDING = "finding"
    PROCESS = "process"
    MODEL = "model"

    def __str__(self) -> str:  # pragma: no cover - trivial behaviour
        return str(self.value)


class RelationType(str, Enum):
    PROPOSES = "proposes"
    EVALUATES_ON = "evaluates_on"
    REPORTS = "reports"
    COMPARES = "compares"
    USES = "uses"
    CAUSES = "causes"
    PART_OF = "part_of"
    IS_A = "is_a"
    OUTPERFORMS = "outperforms"
    ASSUMES = "assumes"

    def __str__(self) -> str:  # pragma: no cover - trivial behaviour
        return str(self.value)


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