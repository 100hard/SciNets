from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


from typing import Optional
class SearchResult(BaseModel):
    paper_id: UUID
    section_id: Optional[UUID] = None
    section_title: Optional[str] = Field(default=None, max_length=500)
    snippet: str = Field(..., min_length=1)
    score: float
    page_number: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    class Config:
        from_attributes = True