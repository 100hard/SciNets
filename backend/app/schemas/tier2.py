from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

class EvidenceSpan(BaseModel):
    """Span of evidence supporting a result or claim."""

    model_config = ConfigDict(extra="ignore")

    section_id: Optional[str] = Field(default=None, min_length=1)
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)


class MethodPayload(BaseModel):
    """Method description extracted by the Tier-2 LLM."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    is_new: Optional[bool] = None
    aliases: List[str] = Field(default_factory=list)


class ResultPayload(BaseModel):
    """Quantitative result extracted by the Tier-2 LLM."""

    model_config = ConfigDict(extra="ignore")

    method: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    value: Optional[Union[float, int, str]] = None
    split: Optional[str] = None
    task: Optional[str] = None
    evidence_span: Optional[EvidenceSpan] = None


class ClaimPayload(BaseModel):
    """Natural-language claim extracted by the Tier-2 LLM."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    evidence_span: Optional[EvidenceSpan] = None


class Tier2LLMPayload(BaseModel):
    """Top-level response returned by the Tier-2 LLM."""

    model_config = ConfigDict(extra="ignore")

    paper_title: str = ""
    methods: List[MethodPayload] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    results: List[ResultPayload] = Field(default_factory=list)
    claims: List[ClaimPayload] = Field(default_factory=list)