from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PaperBase(BaseModel):
    title: str = Field(..., min_length=1)
    authors: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2100)


class PaperCreate(PaperBase):
    status: Optional[str] = None


class Paper(PaperBase):
    id: UUID
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

