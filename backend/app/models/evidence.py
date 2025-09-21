from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer, field_validator


class EvidenceBase(BaseModel):
    snippet: str = Field(..., min_length=1)
    section_id: Optional[UUID] = None
    concept_id: Optional[UUID] = None
    relation_id: Optional[UUID] = None
    vector_id: Optional[str] = Field(default=None, max_length=255)
    embedding_model: Optional[str] = Field(default=None, max_length=255)
    score: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


class EvidenceCreate(EvidenceBase):
    paper_id: UUID

    # This serializer automatically converts the dictionary to a JSON string before validation
    @field_serializer("metadata")
    def serialize_metadata(self, v: Optional[dict[str, Any]]) -> Optional[str]:
        if v is None:
            return None
        return json.dumps(v)


class Evidence(EvidenceBase):
    id: UUID
    paper_id: UUID
    created_at: datetime
    updated_at: datetime

    # This validator automatically converts the JSON string from the DB back to a dictionary
    @field_validator("metadata", mode="before")
    @classmethod
    def deserialize_metadata(cls, v: Any) -> Optional[dict[str, Any]]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        from_attributes = True

