from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from typing import Optional


class SectionBase(BaseModel):
    id: Optional[UUID] = None
    title: Optional[str] = Field(default=None, max_length=512)
    content: str = Field(..., min_length=1)
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=0)
    page_number: Optional[int] = Field(default=None, ge=0)
    snippet: Optional[str] = None


class SectionCreate(SectionBase):
    paper_id: UUID


class Section(SectionBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True