from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from textwrap import dedent
from typing import Any, Iterable, Sequence, Optional
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.core.config import settings
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
from app.schemas.tier2 import Tier2LLMPayload

logger = logging.getLogger(__name__)

TIER_NAME = "llm_structurer"
BASE_CONFIDENCE = 0.7
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
LLM_RETRY_DELAYS = (0.5, 1.0, 2.0)


class Tier2ValidationError(RuntimeError):
    """Raised when the tier-2 LLM payload cannot be validated."""

class TransientLLMError(RuntimeError):
    """Raised when the LLM request fails with a retriable error."""

class ParseLLMError(RuntimeError):
    """Raised when the LLM response cannot be parsed."""


@dataclass
class _Caches:
    methods: dict[str, Any]
    datasets: dict[str, Any]
    metrics: dict[str, Any]
    tasks: dict[str, Any]


def _summary_list(summary: dict[str, Any], key: str) -> list[Any]:
    """Return a safe list for the given summary key."""
    value = summary.get(key)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _truncate_for_log(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}…"


def _stringify_for_log(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload)
    except Exception:  # pragma: no cover - defensive
        return repr(payload)


def _describe_sections(sections: Optional[Sequence[dict[str, Any]]]) -> str:
    """Return a compact description of the provided sections for logging."""
    section_list = list(sections or [])
    count = len(section_list)
    
    identifiers: list[str] = []
    for section in section_list:
        if section_id := section.get("id"):
            identifiers.append(str(section_id))

    if not identifiers:
        return f"count={count}"

    preview = ", ".join(identifiers[:5])
    if len(identifiers) > 5:
        preview = f"{preview}, …"
    return f"count={count} ids=[{preview}]"


def _coerce_tier2_payload(
    raw_payload: Any,
    *,
    paper_id: UUID,
    paper_title: str,
    sections: Optional[Sequence[dict[str, Any]]],
) -> Tier2LLMPayload:
    """Coerce the raw LLM payload into the validated Tier-2 schema."""
    section_records = list(sections or [])
    section_descriptor = _describe_sections(section_records)

    try:
        normalised = _normalise_llm_payload(
            raw_payload,
            paper_title=paper_title,
            sections=section_records,
        )
    except RuntimeError as exc:
        raw_repr = _truncate_for_log(_stringify_for_log(raw_payload))
        detail = f"{exc} ({exc.__cause__})" if exc.__cause__ else str(exc)
        logger.error(
            "[tier2] failed to normalise payload paper=%s sections=%s error=%s raw=%s",
            paper_id, section_descriptor, detail, raw_repr,
        )
        return Tier2LLMPayload()

    try:
        return Tier2LLMPayload.model_validate(normalised)
    except ValidationError as exc:
        serialised = _stringify_for_log(normalised)
        logger.error(
            "[tier2] schema validation failed paper=%s sections=%s error=%s data=%s",
            paper_id, section_descriptor, exc, _truncate_for_log(serialised),
        )
        return Tier2LLMPayload()


async def run_tier2_structurer(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]] = None,
    llm_response: Optional[str] = None,
    sections: Optional[Sequence[Section]] = None,
) -> dict[str, Any]:
    """Execute the Tier-2 LLM structurer for the given paper."""
    paper = await get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    if sections is None:
        sections = await list_sections(paper_id=paper_id, limit=500, offset=0)

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

    if llm_response is None:
        payload = await call_structurer_llm(
            paper_id=paper_id,
            paper_title=paper.title,
            sections=section_payloads,
        )
    else:
        payload = _coerce_tier2_payload(
            llm_response,
            paper_id=paper_id,
            paper_title=paper.title,
            sections=section_payloads,
        )

    caches = _Caches(methods={}, datasets={}, metrics={}, tasks={})

    summary = base_summary or {"paper_id": str(paper_id)}
    
    existing_result_models = await _convert_summary_results(paper_id, summary, caches)
    existing_claim_models = await _convert_summary_claims(paper_id, summary)

    await _ensure_catalog_from_summary(summary, caches)
    await _ensure_catalog_from_payload(payload, caches)

    tier2_result_models = await _convert_payload_results(paper_id, payload, caches)
    tier2_claim_models = _convert_payload_claims(paper_id, payload)

    combined_results = existing_result_models + tier2_result_models
    combined_claims = existing_claim_models + tier2_claim_models

    stored_results = await replace_results(paper_id, combined_results)
    stored_claims = await replace_claims(paper_id, combined_claims)
    
    # Final summary construction
    method_models = {model.id: model for model in caches.methods.values()}
    dataset_models = {model.id: model for model in caches.datasets.values()}
    metric_models = {model.id: model for model in caches.metrics.values()}
    task_models = {model.id: model for model in caches.tasks.values()}

    summary_payload = {
        "paper_id": str(paper_id),
        "tiers": _merge_tiers(_summary_list(summary, "tiers"), [2]),
        "methods": [_serialize_method(model) for model in method_models.values()],
        "datasets": [_serialize_dataset(model) for model in dataset_models.values()],
        "metrics": [_serialize_metric(model) for model in metric_models.values()],
        "tasks": [_serialize_task(model) for model in task_models.values()],
        "results": [],
        "claims": [],
    }

    tier2_start_index = len(existing_result_models)
    for index, result in enumerate(stored_results):
        serialized = _serialize_result(
            result, method_by_id=method_models, dataset_by_id=dataset_models,
            metric_by_id=metric_models, task_by_id=task_models,
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


async def call_structurer_llm(
    *,
    paper_id: UUID,
    paper_title: str,
    sections: Optional[Sequence[dict[str, Any]]],
) -> Tier2LLMPayload:
    """Invoke the Tier-2 structuring LLM and return a validated payload."""
    section_descriptor = _describe_sections(sections)
    section_records = list(sections or [])

    attempts = len(LLM_RETRY_DELAYS) + 1
    last_transient: Optional[TransientLLMError] = None

    for index, delay in enumerate((0.0, *LLM_RETRY_DELAYS), start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            raw_content = await _call_structurer_llm_once(
                paper_title=paper_title,
                sections=section_records,
            )
        except TransientLLMError as exc:
            last_transient = exc
            logger.warning(
                "[tier2] transient LLM error attempt=%s/%s paper=%s sections=%s error=%s",
                index, attempts, paper_id, section_descriptor, exc,
            )
            continue
        except ParseLLMError as exc:
            logger.error(
                "[tier2] failed to acquire LLM payload paper=%s sections=%s error=%s",
                paper_id, section_descriptor, exc,
            )
            return Tier2LLMPayload()

        return _coerce_tier2_payload(
            raw_content,
            paper_id=paper_id,
            paper_title=paper_title,
            sections=section_records,
        )

    if last_transient is not None:
        logger.error(
            "[tier2] transient LLM error after retries paper=%s sections=%s error=%s",
            paper_id, section_descriptor, last_transient,
        )

    return Tier2LLMPayload()


async def _call_structurer_llm_once(
    *, paper_title: str, sections: Sequence[dict[str, Any]]
) -> str:
    """Call the Tier-2 LLM once and return the raw response content."""
    api_key = settings.openai_api_key
    model = settings.tier2_llm_model

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    if not model:
        raise RuntimeError("TIER2_LLM_MODEL environment variable is not set")

    rendered_sections = _render_sections_for_prompt(
        sections,
        max_sections=settings.tier2_llm_max_sections,
        max_chars=settings.tier2_llm_max_section_chars,
    )
    if not rendered_sections:
        raise RuntimeError("No sections available for Tier-2 structurer prompt")

    system_message = dedent("""
        You are an expert scientific information extraction system...
        """).strip() # Prompt content truncated for brevity

    extraction_instructions = dedent("""
        Follow these guidelines when constructing the JSON response...
        JSON schema: { ... }
        """).strip() # Prompt content truncated for brevity

    user_prompt = dedent(f"""
        Paper title: {paper_title}

        Sections:
        {rendered_sections}

        Return only the JSON object that follows the schema.
        """).strip()

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": extraction_instructions},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": settings.tier2_llm_temperature,
        "top_p": settings.tier2_llm_top_p,
        "max_tokens": settings.tier2_llm_max_output_tokens,
    }
    if settings.tier2_llm_force_json:
        request_payload["response_format"] = {"type": "json_object"}

    timeout = httpx.Timeout(settings.tier2_llm_timeout_seconds)
    base_url = settings.tier2_llm_base_url or "https://api.openai.com/v1"
    path = settings.tier2_llm_completion_path or "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if settings.openai_organization:
        headers["OpenAI-Organization"] = settings.openai_organization

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        try:
            response = await client.post(path, headers=headers, json=request_payload)
        except httpx.RequestError as exc:
            raise TransientLLMError(f"network error: {exc}") from exc

    status = response.status_code
    if status in TRANSIENT_STATUS_CODES:
        raise TransientLLMError(f"http {status}")
    if status >= 400:
        raise ParseLLMError(f"http {status}: {_truncate_for_log(response.text or '')}")

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ParseLLMError("invalid JSON or unexpected structure in LLM response") from exc

    if not isinstance(content, str) or not content.strip():
        raise ParseLLMError("empty response content")
    
    return content


def _render_sections_for_prompt(
    sections: Sequence[dict[str, Any]], *, max_sections: int, max_chars: int,
) -> str:
    if not sections or max_sections <= 0 or max_chars <= 0:
        return ""

    rendered_parts: list[str] = []
    for index, section in enumerate(sections):
        if index >= max_sections:
            break
        
        section_id = str(section.get("id", ""))
        title = (section.get("title") or "").strip() or "Untitled"
        content = section.get("content") or ""
        
        span_description = ""
        if isinstance(section.get("char_start"), int) and isinstance(section.get("char_end"), int):
            span_description = f"[{section['char_start']}, {section['char_end']}]"

        truncated_content = _truncate_text(content, max_chars)
        rendered_parts.append(
            dedent(f"""
                Section ID: {section_id}
                Title: {title}
                Span: {span_description}
                Content:
                {truncated_content}
            """).strip()
        )
    return "\n\n".join(rendered_parts)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    truncated = value[:max(0, max_chars - 1)].rstrip()
    return f"{truncated}…"


def _normalise_llm_payload(
    raw_content: Any, *, paper_title: str, sections: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(raw_content, str):
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Tier-2 structurer returned non-JSON response") from exc
    elif isinstance(raw_content, dict):
        parsed = raw_content
    else:
        raise RuntimeError("Tier-2 structurer returned unexpected payload type")

    for key in ["methods", "tasks", "datasets", "metrics", "results", "claims"]:
        parsed.setdefault(key, [])
    
    parsed["paper_title"] = str(parsed.get("paper_title", paper_title))

    parsed["methods"] = _normalise_method_entries(parsed.get("methods"))
    parsed["tasks"] = _normalise_string_list(parsed.get("tasks"))
    parsed["datasets"] = _normalise_string_list(parsed.get("datasets"))
    parsed["metrics"] = _normalise_string_list(parsed.get("metrics"))

    section_lookup = {str(s["id"]): s for s in sections if s.get("id")}
    fallback_section_id = next(iter(section_lookup), None)

    parsed["results"] = _normalise_result_entries(
        parsed.get("results"), section_lookup=section_lookup, fallback_section_id=fallback_section_id
    )
    parsed["claims"] = _normalise_claim_entries(
        parsed.get("claims"), section_lookup=section_lookup, fallback_section_id=fallback_section_id
    )
    return parsed


def _normalise_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalised: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and (stripped := value.strip()) and stripped.lower() not in seen:
            normalised.append(stripped)
            seen.add(stripped.lower())
    return normalised


def _normalise_method_entries(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict) and (name := _coerce_string(value.get("name"))) and name.lower() not in seen:
            seen.add(name.lower())
            aliases = _normalise_string_list(value.get("aliases", []))
            entry = {"name": name, "aliases": aliases}
            if (is_new := value.get("is_new")) is not None:
                entry["is_new"] = bool(is_new)
            normalised.append(entry)
    return normalised


def _normalise_result_entries(
    values: Any, *, section_lookup: dict[str, dict[str, Any]], fallback_section_id: Optional[str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    
    normalised: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        
        method = _coerce_string(value.get("method"))
        dataset = _coerce_string(value.get("dataset"))
        metric = _coerce_string(value.get("metric"))

        if not (method and dataset and metric):
            continue

        result_entry: dict[str, Any] = {
            "method": method, "dataset": dataset, "metric": metric,
            "value": _coerce_result_value(value.get("value")),
            "split": _coerce_optional_string(value.get("split")),
            "task": _coerce_optional_string(value.get("task")),
            "evidence_span": _normalise_evidence_span(
                value.get("evidence_span"), section_lookup=section_lookup,
                fallback_section_id=fallback_section_id
            ),
        }
        normalised.append(result_entry)
    return normalised


def _normalise_claim_entries(
    values: Any, *, section_lookup: dict[str, dict[str, Any]], fallback_section_id: Optional[str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    normalised: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict) and (text := _coerce_string(value.get("text"))):
            entry: dict[str, Any] = {
                "text": text,
                "category": _coerce_string(value.get("category")) or "other",
                "evidence_span": _normalise_evidence_span(
                    value.get("evidence_span"), section_lookup=section_lookup,
                    fallback_section_id=fallback_section_id
                ),
            }
            normalised.append(entry)
    return normalised


def _normalise_evidence_span(
    value: Any, *, section_lookup: dict[str, dict[str, Any]], fallback_section_id: Optional[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    section_id = _coerce_string(value.get("section_id"))
    if not section_id or section_id not in section_lookup:
        section_id = fallback_section_id
    if section_id is None:
        return None

    section = section_lookup.get(section_id) or {}
    section_length = len(section.get("content") or "")

    start = _coerce_int(value.get("start"), minimum=0) or 0
    end = _coerce_int(value.get("end"), minimum=start) or section_length
    
    start = min(start, section_length)
    end = min(end, section_length)
    if end < start:
        start, end = 0, min(section_length, max(end, 0))

    return {"section_id": section_id, "start": start, "end": end}


def _coerce_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip() or None
    return None

_coerce_optional_string = _coerce_string

def _coerce_result_value(value: Any) -> Optional[float | int | str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _coerce_int(value: Any, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(minimum, int(value))
    if isinstance(value, str) and (stripped := value.strip()):
        try:
            return max(minimum, int(float(stripped)))
        except (TypeError, ValueError):
            return None
    return None


async def _ensure_catalog_from_summary(summary: dict[str, Any], caches: _Caches) -> None:
    for item in _summary_list(summary, "methods"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_method(name, caches, aliases=item.get("aliases"), description=item.get("description"))
    for item in _summary_list(summary, "datasets"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_dataset(name, caches, aliases=item.get("aliases"), description=item.get("description"))
    for item in _summary_list(summary, "metrics"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_metric(name, caches, unit=item.get("unit"), aliases=item.get("aliases"), description=item.get("description"))
    for item in _summary_list(summary, "tasks"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_task(name, caches, aliases=item.get("aliases"), description=item.get("description"))


async def _ensure_catalog_from_payload(payload: Tier2LLMPayload, caches: _Caches) -> None:
    for method in (payload.methods or []):
        await _ensure_method(method.name, caches, aliases=method.aliases)
    for dataset in (payload.datasets or []):
        await _ensure_dataset(dataset, caches)
    for metric in (payload.metrics or []):
        await _ensure_metric(metric, caches)
    for task in (payload.tasks or []):
        await _ensure_task(task, caches)


async def _convert_summary_results(
    paper_id: UUID, summary: dict[str, Any], caches: _Caches,
) -> list[ResultCreate]:
    converted: list[ResultCreate] = []
    for result in _summary_list(summary, "results"):
        if isinstance(result, dict):
            if converted_model := await _summary_result_to_model(paper_id, result, caches):
                converted.append(converted_model)
    return converted


async def _summary_result_to_model(
    paper_id: UUID, payload: dict[str, Any], caches: _Caches,
) -> Optional[ResultCreate]:
    method_model = await _extract_summary_method(payload.get("method"), caches)
    dataset_model = await _extract_summary_dataset(payload.get("dataset"), caches)
    metric_model = await _extract_summary_metric(payload.get("metric"), caches)
    task_model = await _extract_summary_task(payload.get("task"), caches)

    numeric_decimal = None
    if (value_numeric := payload.get("value_numeric")) is not None:
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


async def _extract_summary_ontology(payload: Optional[dict[str, Any]], ensure_func, caches: _Caches):
    if payload and (name := payload.get("name")):
        return await ensure_func(
            name, caches, aliases=payload.get("aliases"), description=payload.get("description")
        )
    return None

async def _extract_summary_method(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_method, caches)

async def _extract_summary_dataset(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_dataset, caches)

async def _extract_summary_metric(payload: Optional[dict[str, Any]], caches: _Caches):
    if payload and (name := payload.get("name")):
        return await _ensure_metric(
            name, caches, unit=payload.get("unit"), aliases=payload.get("aliases"), 
            description=payload.get("description")
        )
    return None

async def _extract_summary_task(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_task, caches)


async def _convert_payload_results(
    paper_id: UUID, payload: Tier2LLMPayload, caches: _Caches,
) -> list[ResultCreate]:
    converted: list[ResultCreate] = []
    method_lookup = { _normalize_text(m.name): m for m in (payload.methods or []) }
    
    for result in (payload.results or []):
        method_meta = method_lookup.get(_normalize_text(result.method))
        method_aliases = method_meta.aliases if method_meta else None
        
        method_model = await _ensure_method(result.method, caches, aliases=method_aliases)
        dataset_model = await _ensure_dataset(result.dataset, caches)
        metric_model = await _ensure_metric(result.metric, caches)

        task_name = result.task or (payload.tasks[0] if payload.tasks else None)
        task_model = await _ensure_task(task_name, caches) if task_name else None

        value_text, numeric_decimal = None, None
        if result.value is not None:
            value_text = str(result.value)
            try:
                numeric_decimal = Decimal(value_text)
            except Exception:
                numeric_decimal = None

        evidence_payload: list[dict[str, Any]] = []
        if result.evidence_span:
            evidence_payload.append({
                "tier": TIER_NAME,
                "evidence_span": result.evidence_span.model_dump(),
            })

        converted.append(ResultCreate(
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
        ))
    return converted


def _convert_payload_claims(paper_id: UUID, payload: Tier2LLMPayload) -> list[ClaimCreate]:
    converted: list[ClaimCreate] = []
    for claim in (payload.claims or []):
        try:
            category = ClaimCategory(claim.category.strip().lower())
        except ValueError:
            category = ClaimCategory.OTHER

        evidence_payload: list[dict[str, Any]] = []
        if claim.evidence_span:
            evidence_payload.append({
                "tier": TIER_NAME,
                "evidence_span": claim.evidence_span.model_dump(),
            })

        converted.append(ClaimCreate(
            paper_id=paper_id,
            category=category,
            text=claim.text,
            confidence=BASE_CONFIDENCE,
            evidence=evidence_payload,
        ))
    return converted


async def _convert_summary_claims(paper_id: UUID, summary: dict[str, Any]) -> list[ClaimCreate]:
    converted: list[ClaimCreate] = []
    for claim in _summary_list(summary, "claims"):
        if isinstance(claim, dict) and (text := claim.get("text")):
            try:
                category = ClaimCategory(claim.get("category", "").strip().lower())
            except ValueError:
                category = ClaimCategory.OTHER
            
            converted.append(ClaimCreate(
                paper_id=paper_id,
                category=category,
                text=text,
                confidence=claim.get("confidence"),
                evidence=list(claim.get("evidence") or []),
            ))
    return converted


async def _ensure_method(
    name: str, caches: _Caches, *, aliases: Optional[Iterable[str]] = None, description: Optional[str] = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Method name cannot be empty")
    if (model := caches.methods.get(normalized)) is None:
        model = await ensure_method(name, aliases=aliases, description=description)
        caches.methods[normalized] = model
    return model


async def _ensure_dataset(
    name: str, caches: _Caches, *, aliases: Optional[Iterable[str]] = None, description: Optional[str] = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Dataset name cannot be empty")
    if (model := caches.datasets.get(normalized)) is None:
        model = await ensure_dataset(name, aliases=aliases, description=description)
        caches.datasets[normalized] = model
    return model


async def _ensure_metric(
    name: str, caches: _Caches, *, unit: Optional[str] = None, aliases: Optional[Iterable[str]] = None, description: Optional[str] = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Metric name cannot be empty")
    if (model := caches.metrics.get(normalized)) is None:
        model = await ensure_metric(name, unit=unit, aliases=aliases, description=description)
        caches.metrics[normalized] = model
    return model


async def _ensure_task(
    name: str, caches: _Caches, *, aliases: Optional[Iterable[str]] = None, description: Optional[str] = None,
):
    normalized = _normalize_text(name)
    if not normalized:
        raise ValueError("Task name cannot be empty")
    if (model := caches.tasks.get(normalized)) is None:
        model = await ensure_task(name, aliases=aliases, description=description)
        caches.tasks[normalized] = model
    return model


def _normalize_text(value: str) -> str:
    normalized = _NON_ALNUM.sub(" ", value.lower())
    return " ".join(normalized.split())


def _merge_tiers(existing: Optional[Sequence[int | str]], new: Sequence[int]) -> list[int]:
    tier_set: set[int] = set(new)
    for tier in existing or []:
        try:
            tier_set.add(int(tier))
        except (TypeError, ValueError):
            continue
    return sorted(tier_set)


def _coerce_aliases_for_serialization(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, Iterable) and not isinstance(values, bytes):
        return [alias.strip() for alias in values if isinstance(alias, str) and alias.strip()]
    return []


def _serialize_ontology(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": _coerce_aliases_for_serialization(getattr(model, "aliases", None)),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }

_serialize_method = _serialize_ontology
_serialize_dataset = _serialize_ontology
_serialize_task = _serialize_ontology

def _serialize_metric(model) -> dict[str, Any]:
    payload = _serialize_ontology(model)
    payload["unit"] = model.unit
    return payload


def _serialize_result(
    result, *, method_by_id, dataset_by_id, metric_by_id, task_by_id,
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