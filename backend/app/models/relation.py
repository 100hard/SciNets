from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


from typing import Optional
class RelationBase(BaseModel):
    relation_type: str = Field(..., min_length=1)
    description: Optional[str] = None
    concept_id: Optional[UUID] = None
    related_concept_id: Optional[UUID] = None
    section_id: Optional[UUID] = None


class RelationCreate(RelationBase):
    paper_id: UUID


class Relation(RelationBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True