from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from app.models.ontology import ClaimCategory, ClaimCreate, ResultCreate
from app.models.section import Section
from app.services.ontology_store import (
    ensure_dataset,
    ensure_method,
    ensure_metric,
    ensure_task,
    replace_claims,
    replace_results,
)
from app.services.papers import get_paper
from app.services.sections import list_sections


TIER_NAME = "llm_structurer"
BASE_CONFIDENCE = 0.7
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class Tier2ValidationError(RuntimeError):
    """Raised when the tier-2 LLM payload cannot be validated."""


class EvidenceSpan(BaseModel):
    section_id: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class MethodPayload(BaseModel):
    name: str = Field(..., min_length=1)
    is_new: bool | None = None
    aliases: list[str] = Field(default_factory=list)


class ResultPayload(BaseModel):
    method: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    value: float | int | str | None = None
    split: str | None = None
    task: str | None = None
    evidence_span: EvidenceSpan | None = None


class ClaimPayload(BaseModel):
    category: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    evidence_span: EvidenceSpan | None = None


class Tier2LLMPayload(BaseModel):
    paper_title: str = Field(..., min_length=1)
    methods: list[MethodPayload] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: list[ResultPayload] = Field(default_factory=list)
    claims: list[ClaimPayload] = Field(default_factory=list)


@dataclass
class _Caches:
    methods: dict[str, Any]
    datasets: dict[str, Any]
    metrics: dict[str, Any]
    tasks: dict[str, Any]


async def run_tier2_structurer(
    paper_id: UUID,
    *,
    base_summary: dict[str, Any] | None = None,
    llm_response: str | None = None,
    sections: Sequence[Section] | None = None,
) -> dict[str, Any]:
    """Execute the Tier-2 LLM structurer for the given paper.

    Parameters
    ----------
    paper_id:
        Identifier of the paper to enrich with Tier-2 extractions.
    base_summary:
        Optional summary produced by earlier tiers. When provided the Tier-2
        output augments the existing results and claims.
    llm_response:
        Optional JSON payload returned by the LLM. When omitted the
        ``call_structurer_llm`` helper is invoked.
    sections:
        Optional override for the sections considered by the LLM. This is
        primarily useful for tests.
    """

    paper = await get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    if sections is None:
        sections = await list_sections(paper_id=paper_id, limit=500, offset=0)

    if llm_response is None:
        section_payloads = [
            {
                "id": str(section.id),
                "title": section.title,
                "content": section.content,
                "char_start": section.char_start,
                "char_end": section.char_end,
            }
            for section in sections
        ]
        llm_response = await call_structurer_llm(
            paper_title=paper.title,
            sections=section_payloads,
        )

    try:
        payload = Tier2LLMPayload.model_validate_json(llm_response)
    except (ValidationError, ValueError, TypeError) as exc:
        raise Tier2ValidationError(f"Tier-2 structurer returned invalid payload: {exc}") from exc

    caches = _Caches(methods={}, datasets={}, metrics={}, tasks={})

    summary = base_summary or {
        "paper_id": str(paper_id),
        "tiers": [],
        "methods": [],
        "datasets": [],
        "metrics": [],
        "tasks": [],
        "results": [],
        "claims": [],
    }

    existing_result_models = await _convert_summary_results(paper_id, summary, caches)
    existing_claim_models = await _convert_summary_claims(paper_id, summary)

    await _ensure_catalog_from_summary(summary, caches)
    await _ensure_catalog_from_payload(payload, caches)

    tier2_result_models = await _convert_payload_results(paper_id, payload, caches)
    tier2_claim_models = _convert_payload_claims(paper_id, payload)

    combined_results = [*existing_result_models, *tier2_result_models]
    combined_claims = [*existing_claim_models, *tier2_claim_models]

    stored_results = await replace_results(paper_id, combined_results)
    stored_claims = await replace_claims(paper_id, combined_claims)

    method_models = {model.id: model for model in caches.methods.values()}
    dataset_models = {model.id: model for model in caches.datasets.values()}
    metric_models = {model.id: model for model in caches.metrics.values()}
    task_models = {model.id: model for model in caches.tasks.values()}

    method_list = list(method_models.values())
    dataset_list = list(dataset_models.values())
    metric_list = list(metric_models.values())
    task_list = list(task_models.values())

    summary_payload = {
        "paper_id": str(paper_id),
        "tiers": _merge_tiers(summary.get("tiers", []), [2]),
        "methods": [_serialize_method(model) for model in method_list],
        "datasets": [_serialize_dataset(model) for model in dataset_list],
        "metrics": [_serialize_metric(model) for model in metric_list],
        "tasks": [_serialize_task(model) for model in task_list],
        "results": [],
        "claims": [],
    }

    tier2_start_index = len(existing_result_models)
    for index, result in enumerate(stored_results):
        serialized = _serialize_result(
            result,
            method_by_id=method_models,
            dataset_by_id=dataset_models,
            metric_by_id=metric_models,
            task_by_id=task_models,
        )
        if index >= tier2_start_index:
            serialized["tier"] = TIER_NAME
        summary_payload["results"].append(serialized)

    tier2_claim_start = len(existing_claim_models)
    for index, claim in enumerate(stored_claims):
        serialized = _serialize_claim(claim)
        if index >= tier2_claim_start:
            serialized["tier"] = TIER_NAME
        summary_payload["claims"].append(serialized)

    return summary_payload


async def call_structurer_llm(*, paper_title: str, sections: Sequence[dict[str, Any]]) -> str:
    """Invoke the Tier-2 structuring LLM.

    This project does not bundle a concrete LLM integration. The function is
    intended to be monkeypatched in tests or overridden by downstream
    deployments. The default implementation raises ``RuntimeError`` to make the
    missing configuration explicit.
    """

    raise RuntimeError("Tier-2 structurer LLM is not configured")


async def _ensure_catalog_from_summary(summary: dict[str, Any], caches: _Caches) -> None:
    for method in summary.get("methods", []):
        name = method.get("name")
        if name:
            await _ensure_method(name, caches, aliases=method.get("aliases"), description=method.get("description"))

    for dataset in summary.get("datasets", []):
        name = dataset.get("name")
        if name:
            await _ensure_dataset(name, caches, aliases=dataset.get("aliases"), description=dataset.get("description"))

    for metric in summary.get("metrics", []):
        name = metric.get("name")
        if name:
            await _ensure_metric(
                name,
                caches,
                unit=metric.get("unit"),
                aliases=metric.get("aliases"),
                description=metric.get("description"),
            )

    for task in summary.get("tasks", []):
        name = task.get("name")
        if name:
            await _ensure_task(name, caches, aliases=task.get("aliases"), description=task.get("description"))


async def _ensure_catalog_from_payload(payload: Tier2LLMPayload, caches: _Caches) -> None:
    for method in payload.methods:
        await _ensure_method(method.name, caches, aliases=method.aliases)

    for dataset in payload.datasets:
        await _ensure_dataset(dataset, caches)

    for metric in payload.metrics:
        await _ensure_metric(metric, caches)

    for task in payload.tasks:
        await _ensure_task(task, caches)


async def _convert_summary_results(
    paper_id: UUID,
    summary: dict[str, Any],
    caches: _Caches,
) -> list[ResultCreate]:
    converted: list[ResultCreate] = []
    for result in summary.get("results", []):
        converted_model = await _summary_result_to_model(paper_id, result, caches)
        if converted_model is not None:
            converted.append(converted_model)
    return converted


async def _summary_result_to_model(
    paper_id: UUID,
    payload: dict[str, Any],
    caches: _Caches,
) -> ResultCreate | None:
    method_model = await _extract_summary_method(payload.get("method"), caches)
    dataset_model = await _extract_summary_dataset(payload.get("dataset"), caches)
    metric_model = await _extract_summary_metric(payload.get("metric"), caches)
    task_model = await _extract_summary_task(payload.get("task"), caches)

    value_numeric = payload.get("value_numeric")
    numeric_decimal = None
    if value_numeric is not None:
        numeric_decimal = Decimal(str(value_numeric))

    return ResultCreate(
        paper_id=paper_id,
        method_id=method_model.id if method_model else None,
        dataset_id=dataset_model.id if dataset_model else None,
        metric_id=metric_model.id if metric_model else None,
        task_id=task_model.id if task_model else None,
        split=payload.get("split"),
        value_numeric=numeric_decimal,
        value_text=payload.get("value_text"),
        is_sota=bool(payload.get("is_sota")),
        confidence=payload.get("confidence"),
        evidence=list(payload.get("evidence") or []),
        verified=payload.get("verified"),
        verifier_notes=payload.get("verifier_notes"),
    )


async def _extract_summary_method(payload: dict[str, Any] | None, caches: _Caches):
    if not payload:
        return None
    name = payload.get("name")
    if not name:
        return None
    return await _ensure_method(
        name,
        caches,
        aliases=payload.get("aliases"),
        description=payload.get("description"),
    )


async def _extract_summary_dataset(payload: dict[str, Any] | None, caches: _Caches):
    if not payload:
        return None
    name = payload.get("name")
    if not name:
        return None
    return await _ensure_dataset(
        name,
        caches,
        aliases=payload.get("aliases"),
        description=payload.get("description"),
    )


async def _extract_summary_metric(payload: dict[str, Any] | None, caches: _Caches):
    if not payload:
        return None
    name = payload.get("name")
    if not name:
        return None
    return await _ensure_metric(
        name,
        caches,
        unit=payload.get("unit"),
        aliases=payload.get("aliases"),
        description=payload.get("description"),
    )


async def _extract_summary_task(payload: dict[str, Any] | None, caches: _Caches):
    if not payload:
        return None
    name = payload.get("name")
    if not name:
        return None
    return await _ensure_task(
        name,
        caches,
        aliases=payload.get("aliases"),
        description=payload.get("description"),
    )


async def _convert_payload_results(
    paper_id: UUID,
    payload: Tier2LLMPayload,
    caches: _Caches,
) -> list[ResultCreate]:
    converted: list[ResultCreate] = []
    method_lookup = { _normalize_text(method.name): method for method in payload.methods }
    for result in payload.results:
        method_aliases: Iterable[str] | None = None
        method_meta = method_lookup.get(_normalize_text(result.method))
        if method_meta is not None:
            method_aliases = method_meta.aliases

        method_model = await _ensure_method(result.method, caches, aliases=method_aliases)
        dataset_model = await _ensure_dataset(result.dataset, caches)
        metric_model = await _ensure_metric(result.metric, caches)

        task_model = None
        task_name = result.task or (payload.tasks[0] if payload.tasks else None)
        if task_name:
            task_model = await _ensure_task(task_name, caches)

        numeric_decimal = None
        value_text = None
        if result.value is not None:
            value_text = str(result.value)
            try:
                numeric_decimal = Decimal(str(result.value))
            except Exception:
                numeric_decimal = None

        evidence_payload: list[dict[str, Any]] = []
        if result.evidence_span is not None:
            evidence_payload.append(
                {
                    "tier": TIER_NAME,
                    "evidence_span": result.evidence_span.model_dump(),
                }
            )

        converted.append(
            ResultCreate(
                paper_id=paper_id,
                method_id=method_model.id,
                dataset_id=dataset_model.id,
                metric_id=metric_model.id,
                task_id=task_model.id if task_model else None,
                split=result.split,
                value_numeric=numeric_decimal,
                value_text=value_text,
                is_sota=False,
                confidence=BASE_CONFIDENCE,
                evidence=evidence_payload,
            )
        )

    return converted


def _convert_payload_claims(paper_id: UUID, payload: Tier2LLMPayload) -> list[ClaimCreate]:
    converted: list[ClaimCreate] = []
    for claim in payload.claims:
        category_value = claim.category.strip().lower()
        try:
            category = ClaimCategory(category_value)
        except ValueError:
            category = ClaimCategory.OTHER

        evidence_payload: list[dict[str, Any]] = []
        if claim.evidence_span is not None:
            evidence_payload.append(
                {
                    "tier": TIER_NAME,
                    "evidence_span": claim.evidence_span.model_dump(),
                }
            )

        converted.append(
            ClaimCreate(
                paper_id=paper_id,
                category=category,
                text=claim.text,
                confidence=BASE_CONFIDENCE,
                evidence=evidence_payload,
            )
        )

    return converted


async def _convert_summary_claims(paper_id: UUID, summary: dict[str, Any]) -> list[ClaimCreate]:
    converted: list[ClaimCreate] = []
    for claim in summary.get("claims", []):
        category_raw = claim.get("category", "").strip().lower()
        try:
            category = ClaimCategory(category_raw)
        except ValueError:
            category = ClaimCategory.OTHER

        text = claim.get("text")
        if not text:
            continue

        converted.append(
            ClaimCreate(
                paper_id=paper_id,
                category=category,
                text=text,
                confidence=claim.get("confidence"),
                evidence=list(claim.get("evidence") or []),
            )
        )
    return converted


async def _ensure_method(
    name: str,
    caches: _Caches,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Method name cannot be empty")
    model = caches.methods.get(normalized)
    if model is None:
        model = await ensure_method(name, aliases=aliases, description=description)
        caches.methods[normalized] = model
    return model


async def _ensure_dataset(
    name: str,
    caches: _Caches,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Dataset name cannot be empty")
    model = caches.datasets.get(normalized)
    if model is None:
        model = await ensure_dataset(name, aliases=aliases, description=description)
        caches.datasets[normalized] = model
    return model


async def _ensure_metric(
    name: str,
    caches: _Caches,
    *,
    unit: str | None = None,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Metric name cannot be empty")
    model = caches.metrics.get(normalized)
    if model is None:
        model = await ensure_metric(name, unit=unit, aliases=aliases, description=description)
        caches.metrics[normalized] = model
    return model


async def _ensure_task(
    name: str,
    caches: _Caches,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Task name cannot be empty")
    model = caches.tasks.get(normalized)
    if model is None:
        model = await ensure_task(name, aliases=aliases, description=description)
        caches.tasks[normalized] = model
    return model


def _normalize_text(value: str) -> str:
    normalized = _NON_ALNUM.sub(" ", value.lower())
    return " ".join(normalized.split())


def _merge_tiers(existing: Sequence[int] | Sequence[str], new: Sequence[int]) -> list[int]:
    tier_set: set[int] = set()
    for tier in existing:
        try:
            tier_set.add(int(tier))
        except (TypeError, ValueError):
            continue
    tier_set.update(new)
    return sorted(tier_set)


def _serialize_method(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": list(model.aliases),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_dataset(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": list(model.aliases),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_metric(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "unit": model.unit,
        "aliases": list(model.aliases),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_task(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": list(model.aliases),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_result(
    result,
    *,
    method_by_id,
    dataset_by_id,
    metric_by_id,
    task_by_id,
) -> dict[str, Any]:
    return {
        "id": str(result.id),
        "paper_id": str(result.paper_id),
        "method": _serialize_method(method_by_id[result.method_id]) if result.method_id else None,
        "dataset": _serialize_dataset(dataset_by_id[result.dataset_id]) if result.dataset_id else None,
        "metric": _serialize_metric(metric_by_id[result.metric_id]) if result.metric_id else None,
        "task": _serialize_task(task_by_id[result.task_id]) if result.task_id else None,
        "split": result.split,
        "value_numeric": float(result.value_numeric) if result.value_numeric is not None else None,
        "value_text": result.value_text,
        "is_sota": result.is_sota,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "verified": result.verified,
        "verifier_notes": result.verifier_notes,
        "created_at": result.created_at.isoformat(),
        "updated_at": result.updated_at.isoformat(),
    }


def _serialize_claim(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "paper_id": str(model.paper_id),
        "category": model.category.value,
        "text": model.text,
        "confidence": model.confidence,
        "evidence": model.evidence,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


