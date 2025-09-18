from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ConceptBase(BaseModel):
    name: str = Field(..., min_length=1)
    type: Optional[str] = Field(default=None, max_length=128)
    description: Optional[str] = None


class ConceptCreate(ConceptBase):
    paper_id: UUID


class Concept(ConceptBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

