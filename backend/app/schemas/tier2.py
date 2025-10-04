from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings

TYPE_GUESS_VALUES: tuple[str, ...] = (
    "Method",
    "Task",
    "Dataset",
    "Metric",
    "Concept",
    "Material",
    "Organism",
    "Model",
    "Unknown",
)
RELATION_GUESS_VALUES: tuple[str, ...] = (
    "IS_A",
    "USES",
    "EVALUATED_ON",
    "COMPARED_TO",
    "OUTPERFORMS",
    "MEASURES",
    "REPORTS",
    "PROPOSES",
    "PART_OF",
    "CAUSES",
    "ASSUMES",
    "OTHER",
)

TypeGuess = Literal[
    "Method",
    "Task",
    "Dataset",
    "Metric",
    "Concept",
    "Material",
    "Organism",
    "Model",
    "Unknown",
]
RelationGuess = Literal[
    "IS_A",
    "USES",
    "EVALUATED_ON",
    "COMPARED_TO",
    "OUTPERFORMS",
    "MEASURES",
    "REPORTS",
    "PROPOSES",
    "PART_OF",
    "CAUSES",
    "ASSUMES",
    "OTHER",
]


class TriplePayload(BaseModel):
    """Triple extracted by Tier-2 LLM following the strict JSON schema."""

    model_config = ConfigDict(extra="ignore")

    subject: str = Field(..., min_length=2, max_length=120)
    relation: str = Field(..., min_length=2, max_length=60)
    object: str = Field(..., min_length=1, max_length=120)
    evidence: str = Field(..., min_length=10, max_length=400)
    subject_span: list[int] = Field(default_factory=list, min_length=2, max_length=2)
    object_span: list[int] = Field(default_factory=list, min_length=2, max_length=2)
    subject_type_guess: TypeGuess = Field(default="Unknown")
    object_type_guess: TypeGuess = Field(default="Unknown")
    relation_type_guess: RelationGuess = Field(default="OTHER")
    triple_conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    schema_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    section_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    chunk_id: Optional[str] = Field(default=None, min_length=1, max_length=64)

    @field_validator("subject_type_guess", "object_type_guess", mode="before")
    @classmethod
    def _coerce_type_guess(cls, value: Optional[str]) -> str:
        """Map unknown type guesses to the ``Unknown`` sentinel value."""

        if not isinstance(value, str):
            return "Unknown"

        normalized = value.strip()
        if normalized in TYPE_GUESS_VALUES:
            return normalized

        return "Unknown"

    @field_validator("relation_type_guess", mode="before")
    @classmethod
    def _coerce_relation_guess(cls, value: Optional[str]) -> str:
        """Map unknown relation guesses to the ``OTHER`` sentinel value."""

        if not isinstance(value, str):
            return "OTHER"

        normalized = value.strip()
        if normalized in RELATION_GUESS_VALUES:
            return normalized

        return "OTHER"


class TripleExtractionResponse(BaseModel):
    """Top-level response returned by the Tier-2 LLM."""

    model_config = ConfigDict(extra="ignore")

    triples: list[TriplePayload] = Field(
        default_factory=list, max_length=settings.tier2_llm_max_triples
    )
    discarded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
