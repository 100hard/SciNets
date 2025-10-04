from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

EvidencePayload = list[dict[str, Any]]


class _AliasMixin(BaseModel):
    name: str = Field(..., min_length=1)
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodBase(_AliasMixin):
    pass


class MethodCreate(MethodBase):
    pass


class Method(MethodBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskBase(_AliasMixin):
    pass


class TaskCreate(TaskBase):
    pass


class Task(TaskBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApplicationBase(_AliasMixin):
    pass


class ApplicationCreate(ApplicationBase):
    pass


class Application(ApplicationBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetBase(_AliasMixin):
    pass


class DatasetCreate(DatasetBase):
    pass


class Dataset(DatasetBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MetricBase(_AliasMixin):
    unit: Optional[str] = None


class MetricCreate(MetricBase):
    pass


class Metric(MetricBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResearchAreaBase(_AliasMixin):
    pass


class ResearchAreaCreate(ResearchAreaBase):
    pass


class ResearchArea(ResearchAreaBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ClaimCategory(str, Enum):
    CONTRIBUTION = "contribution"
    LIMITATION = "limitation"
    ABLATION = "ablation"
    FUTURE_WORK = "future_work"
    OTHER = "other"


class PaperRelationType(str, Enum):
    CITES = "cites"
    EXTENDS = "extends"
    COMPARES = "compares"


class ConceptResolutionType(str, Enum):
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    APPLICATION = "application"
    RESEARCH_AREA = "research_area"


class MethodRelationType(str, Enum):
    EVALUATES_ON = "evaluates_on"
    PROPOSES = "proposes"


class ResultBase(BaseModel):
    method_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    metric_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    split: Optional[str] = None
    value_numeric: Optional[Decimal] = None
    value_text: Optional[str] = None
    is_sota: bool = False
    confidence: Optional[float] = None
    evidence: EvidencePayload = Field(default_factory=list)
    verified: Optional[bool] = None
    verifier_notes: Optional[str] = None


class ResultCreate(ResultBase):
    paper_id: UUID


class Result(ResultBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ClaimBase(BaseModel):
    category: ClaimCategory
    text: str = Field(..., min_length=1)
    confidence: Optional[float] = None
    evidence: EvidencePayload = Field(default_factory=list)


class ClaimCreate(ClaimBase):
    paper_id: UUID


class Claim(ClaimBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaperRelationBase(BaseModel):
    dst_paper_id: UUID
    relation_type: PaperRelationType
    confidence: Optional[float] = None
    evidence: EvidencePayload = Field(default_factory=list)


class PaperRelationCreate(PaperRelationBase):
    src_paper_id: UUID


class PaperRelation(PaperRelationBase):
    id: UUID
    src_paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MethodRelationBase(BaseModel):
    method_id: UUID
    relation_type: MethodRelationType
    dataset_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    confidence: Optional[float] = None
    evidence: EvidencePayload = Field(default_factory=list)


class MethodRelationCreate(MethodRelationBase):
    paper_id: UUID


class MethodRelation(MethodRelationBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConceptResolutionBase(BaseModel):
    resolution_type: ConceptResolutionType
    canonical_id: UUID
    alias_text: str = Field(..., min_length=1)
    score: Optional[float] = None


class ConceptResolutionCreate(ConceptResolutionBase):
    pass


class ConceptResolution(ConceptResolutionBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CanonicalizationMergedItem(BaseModel):
    id: UUID
    name: str
    score: float


class CanonicalizationExample(BaseModel):
    canonical_id: UUID
    canonical_name: str
    merged: list[CanonicalizationMergedItem] = Field(default_factory=list)


class CanonicalizationTypeReport(BaseModel):
    resolution_type: ConceptResolutionType
    before: int
    after: int
    merges: int
    examples: list[CanonicalizationExample] = Field(default_factory=list)


class CanonicalizationReport(BaseModel):
    summary: list[CanonicalizationTypeReport] = Field(default_factory=list)
