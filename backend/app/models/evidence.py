from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EvidenceBase(BaseModel):
    snippet: str = Field(..., min_length=1)
    section_id: Optional[UUID] = None
    concept_id: Optional[UUID] = None
    relation_id: Optional[UUID] = None
    vector_id: Optional[str] = Field(default=None, max_length=255)
    embedding_model: Optional[str] = Field(default=None, max_length=255)
    score: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    provenance: Optional[dict[str, Any]] = None


class EvidenceCreate(EvidenceBase):
    paper_id: UUID


class Evidence(EvidenceBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
