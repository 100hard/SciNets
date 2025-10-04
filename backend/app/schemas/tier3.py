"""Tier-3 relation extraction schemas and JSON helpers."""

from __future__ import annotations

import copy
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings
from app.schemas.tier2 import RelationGuess, TypeGuess


class RelationEvidenceSpan(BaseModel):
    """Evidence span pointing to a section sentence."""

    model_config = ConfigDict(extra="ignore")

    section_id: str = Field(..., min_length=1, max_length=64)
    sentence_index: int = Field(..., ge=0, le=5000)
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)


class RelationTriplePayload(BaseModel):
    """Triple returned by the Tier-3 LLM fallback."""

    model_config = ConfigDict(extra="ignore")

    subject: str = Field(..., min_length=2, max_length=120)
    relation: str = Field(..., min_length=2, max_length=60)
    object: str = Field(..., min_length=1, max_length=160)
    evidence: str = Field(..., min_length=10, max_length=600)
    subject_span: list[int] = Field(default_factory=list, max_length=2)
    object_span: list[int] = Field(default_factory=list, max_length=2)
    evidence_spans: list[RelationEvidenceSpan] = Field(default_factory=list, max_length=8)
    subject_type_guess: TypeGuess = Field(default="Unknown")
    object_type_guess: TypeGuess = Field(default="Unknown")
    relation_type_guess: RelationGuess = Field(default="OTHER")
    triple_conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    schema_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("subject_span", "object_span", mode="before")
    @classmethod
    def _clamp_span(cls, value: Optional[list[int]]) -> list[int]:
        if not isinstance(value, list) or len(value) != 2:
            return []
        try:
            start = max(0, int(value[0]))
            end = max(start, int(value[1]))
        except (TypeError, ValueError):
            return []
        return [start, end]


class RelationExtractionResponse(BaseModel):
    """Top-level container for Tier-3 LLM outputs."""

    model_config = ConfigDict(extra="ignore")

    triples: list[RelationTriplePayload] = Field(
        default_factory=list, max_length=settings.tier3_llm_max_triples
    )
    warnings: list[str] = Field(default_factory=list)


RELATION_JSON_SCHEMA_TEMPLATE = RelationExtractionResponse.model_json_schema()


def _build_relation_json_schema(max_triples: int) -> dict[str, object]:
    schema = copy.deepcopy(RELATION_JSON_SCHEMA_TEMPLATE)
    properties = schema.get("properties", {})
    triples_schema = properties.get("triples", {})
    if isinstance(triples_schema, dict):
        triples_schema["maxItems"] = max_triples
        properties["triples"] = triples_schema
        schema["properties"] = properties
    return schema


def get_relation_json_schema() -> dict[str, object]:
    """Return the JSON schema for Tier-3 LLM responses."""

    return _build_relation_json_schema(settings.tier3_llm_max_triples)

