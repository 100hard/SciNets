from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from textwrap import dedent
from typing import Any, Iterable, Sequence
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
    """Return a safe list for the given summary key.

    Earlier tiers occasionally produce ``None`` for collection fields which
    causes ``TypeError`` when iterated over. To ensure Tier-2 remains robust we
    coerce ``None`` (or other unexpected scalar values) into empty lists. Tuples
    are converted to lists to preserve order while allowing iteration.
    """

    value = summary.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
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


def _describe_sections(sections: Sequence[dict[str, Any]]) -> str:
    """Return a compact description of the provided sections for logging."""

    identifiers: list[str] = []
    for section in sections:
        section_id = section.get("id")
        if not section_id:
            continue
        identifiers.append(str(section_id))

    count = len(sections)
    if not identifiers:
        return f"count={count}"

    preview = ", ".join(identifiers[:5])
    if len(identifiers) > 5:
        preview = f"{preview}, …"
    return f"count={count} ids=[{preview}]"



def _format_section_ids(
    sections: Sequence[dict[str, Any]] | None,
) -> str:
    """Return a sanitised list of section identifiers for logging purposes."""

    section_list = list(sections or [])

    identifiers: list[str] = []
    for section in section_list:

def _format_section_ids(sections: Sequence[dict[str, Any]]) -> str:
    """Return a sanitised list of section identifiers for logging purposes."""

    identifiers: list[str] = []
    for section in sections:

        section_id = section.get("id")
        if section_id is None:
            continue
        if isinstance(section_id, str):
            cleaned = section_id.strip()
            if not cleaned:
                continue
            identifiers.append(cleaned)
            continue
        try:
            identifiers.append(str(section_id))
        except Exception:  # pragma: no cover - defensive
            continue


    count = len(section_list)

    count = len(sections)

    if not identifiers:
        return f"count={count}"

    preview = identifiers[:5]
    if len(identifiers) > 5:
        preview.append("…")
    return f"count={count} ids=[{', '.join(preview)}]"


def _coerce_tier2_payload(
    raw_payload: Any,
    *,
    paper_id: UUID,
    paper_title: str,
    sections: Sequence[dict[str, Any]] | None,
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
        detail = str(exc)
        if exc.__cause__ is not None:
            detail = f"{detail} ({exc.__cause__})"
        logger.error(
            "[tier2] failed to normalise payload paper=%s sections=%s error=%s raw=%s",
            paper_id,
            section_descriptor,
            detail,
            raw_repr,
        )
        return Tier2LLMPayload()

    try:
        return Tier2LLMPayload.model_validate(normalised)
    except ValidationError as exc:
        try:
            serialised = json.dumps(normalised)
        except Exception:  # pragma: no cover - defensive
            serialised = repr(normalised)
        logger.error(
            "[tier2] schema validation failed paper=%s sections=%s error=%s data=%s",
            paper_id,
            section_descriptor,
            exc,
            _truncate_for_log(serialised),
        )
        return Tier2LLMPayload()



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
    else:
        sections = list(sections)

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
        "tiers": _merge_tiers(_summary_list(summary, "tiers"), [2]),
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


async def call_structurer_llm(
    *,
    paper_id: UUID,
    paper_title: str,
    sections: Sequence[dict[str, Any]] | None,
) -> Tier2LLMPayload:
    """Invoke the Tier-2 structuring LLM and return a validated payload."""

    section_records = list(sections or [])
    section_descriptor = _format_section_ids(section_records)

    attempts = len(LLM_RETRY_DELAYS) + 1
    last_transient: TransientLLMError | None = None

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
                index,
                attempts,
                paper_id,
                section_descriptor,
                exc,
            )
            continue
        except ParseLLMError as exc:
            logger.error(
                "[tier2] failed to acquire LLM payload paper=%s sections=%s error=%s",
                paper_id,
                section_descriptor,
                exc,
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
            paper_id,
            section_descriptor,
            last_transient,
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

    system_message = dedent(
        """
        You are an expert scientific information extraction system. Given a research
        paper you must populate a JSON object that lists the unique methods, tasks,
        datasets, metrics, quantitative results, and natural-language claims that
        appear in the paper. Carefully read the provided sections before producing
        the JSON output. Only include information that is explicitly supported by
        the text. If an item is not present in the paper, leave the relevant array
        empty.
        """
    ).strip()

    extraction_instructions = dedent(
        """
        Follow these guidelines when constructing the JSON response:

        1. Return a single JSON object that matches the schema described below. Do
           not wrap the JSON in markdown fences or include any commentary.
        2. The `paper_title` field must echo the provided paper title string.
        3. Populate the `methods`, `datasets`, `metrics`, and `tasks` arrays with
           distinct strings encountered in the text. If you are unsure, omit the
           item.
        4. For each quantitative result create an entry with `method`, `dataset`,
           `metric`, `value`, and, when available, `split` or `task`. Use the
           canonical names surfaced in the earlier arrays. If a numeric value is
           present express it as a number; otherwise fall back to a short string.
        5. Populate `claims` with important findings or conclusions from the paper
           and classify them into one of: `sota`, `ablations`, `limitations`,
           `future_work`, `data`, or `other`.
        6. Whenever a result or claim is grounded in a specific section, provide
           an `evidence_span` object containing the `section_id` along with
           character offsets `start` and `end` relative to the provided section
           content. If precise offsets are unclear you may use `0` and the
           section length as a reasonable bound.

        JSON schema:

        {
          "paper_title": string,
          "methods": [
            {
              "name": string,
              "is_new": boolean (optional),
              "aliases": array of strings (optional)
            }
          ],
          "tasks": [string],
          "datasets": [string],
          "metrics": [string],
          "results": [
            {
              "method": string,
              "dataset": string,
              "metric": string,
              "value": number|string|null,
              "split": string|null,
              "task": string|null,
              "evidence_span": {
                "section_id": string,
                "start": integer,
                "end": integer
              } | null
            }
          ],
          "claims": [
            {
              "category": string,
              "text": string,
              "evidence_span": {
                "section_id": string,
                "start": integer,
                "end": integer
              } | null
            }
          ]
        }
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Paper title: {paper_title}

        Sections:
        {rendered_sections}

        Return only the JSON object that follows the schema.
        """
    ).strip()

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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
        raise ParseLLMError(
            f"http {status}: {_truncate_for_log(response.text or '')}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ParseLLMError(f"invalid JSON response: {exc}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ParseLLMError("missing message content") from exc


    if not isinstance(content, str) or not content.strip():
        raise ParseLLMError("empty response content")



    if not isinstance(content, str) or not content.strip():
        raise ParseLLMError("empty response content")


    return content


def _render_sections_for_prompt(
    sections: Sequence[dict[str, Any]],
    *,
    max_sections: int,
    max_chars: int,
) -> str:
    if max_sections <= 0 or max_chars <= 0:
        return ""

    rendered_parts: list[str] = []
    for index, section in enumerate(sections):
        if index >= max_sections:
            break

        section_id = str(section.get("id", ""))
        title = (section.get("title") or "").strip() or "Untitled"
        content = section.get("content") or ""
        char_start = section.get("char_start")
        char_end = section.get("char_end")
        span_description = ""
        if isinstance(char_start, int) and isinstance(char_end, int):
            span_description = f"[{char_start}, {char_end}]"

        truncated_content = _truncate_text(content, max_chars)
        rendered_parts.append(
            dedent(
                f"""
                Section ID: {section_id}
                Title: {title}
                Span: {span_description}
                Content:
                {truncated_content}
                """
            ).strip()
        )

    return "\n\n".join(rendered_parts)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    truncated = value[: max(0, max_chars - 1)].rstrip()
    return f"{truncated}…"


def _normalise_llm_payload(
    raw_content: Any,
    *,
    paper_title: str,
    sections: Sequence[dict[str, Any]],
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

    parsed.setdefault("paper_title", paper_title)
    if not isinstance(parsed["paper_title"], str):
        parsed["paper_title"] = str(parsed["paper_title"])

    parsed.setdefault("methods", [])
    parsed.setdefault("tasks", [])
    parsed.setdefault("datasets", [])
    parsed.setdefault("metrics", [])
    parsed.setdefault("results", [])
    parsed.setdefault("claims", [])

    parsed["methods"] = _normalise_method_entries(parsed.get("methods"))
    parsed["tasks"] = _normalise_string_list(parsed.get("tasks"))
    parsed["datasets"] = _normalise_string_list(parsed.get("datasets"))
    parsed["metrics"] = _normalise_string_list(parsed.get("metrics"))

    section_lookup = {
        str(section.get("id")): section for section in sections if section.get("id")
    }
    fallback_section_id = next(iter(section_lookup), None)

    parsed["results"] = _normalise_result_entries(
        parsed.get("results"),
        section_lookup=section_lookup,
        fallback_section_id=fallback_section_id,
    )
    parsed["claims"] = _normalise_claim_entries(
        parsed.get("claims"),
        section_lookup=section_lookup,
        fallback_section_id=fallback_section_id,
    )

    return parsed


def _normalise_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalised: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        key = stripped.lower()
        if stripped and key not in seen:
            normalised.append(stripped)
            seen.add(key)
    return normalised


def _normalise_method_entries(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        name_raw = value.get("name")
        if not isinstance(name_raw, str):
            continue
        name = name_raw.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases = value.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        alias_list = [alias.strip() for alias in aliases if isinstance(alias, str) and alias.strip()]
        is_new = value.get("is_new")
        if is_new is None:
            normalised_is_new = None
        else:
            normalised_is_new = bool(is_new)
        entry = {"name": name, "aliases": alias_list}
        if normalised_is_new is not None:
            entry["is_new"] = normalised_is_new
        normalised.append(entry)
    return normalised


def _normalise_result_entries(
    values: Any,
    *,
    section_lookup: dict[str, dict[str, Any]],
    fallback_section_id: str | None,
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
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "value": _coerce_result_value(value.get("value")),
            "split": _coerce_optional_string(value.get("split")),
            "task": _coerce_optional_string(value.get("task")),
        }

        evidence = _normalise_evidence_span(
            value.get("evidence_span"),
            section_lookup=section_lookup,
            fallback_section_id=fallback_section_id,
        )
        result_entry["evidence_span"] = evidence

        normalised.append(result_entry)

    return normalised


def _normalise_claim_entries(
    values: Any,
    *,
    section_lookup: dict[str, dict[str, Any]],
    fallback_section_id: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    normalised: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue

        text = _coerce_string(value.get("text"))
        category = _coerce_string(value.get("category")) or "other"
        if not text:
            continue

        entry: dict[str, Any] = {
            "text": text,
            "category": category,
        }

        evidence = _normalise_evidence_span(
            value.get("evidence_span"),
            section_lookup=section_lookup,
            fallback_section_id=fallback_section_id,
        )
        entry["evidence_span"] = evidence

        normalised.append(entry)

    return normalised


def _normalise_evidence_span(
    value: Any,
    *,
    section_lookup: dict[str, dict[str, Any]],
    fallback_section_id: str | None,
) -> dict[str, Any] | None:
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
        start = 0
        end = min(section_length, max(end, 0))

    return {
        "section_id": section_id,
        "start": start,
        "end": end,
    }


def _coerce_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _coerce_optional_string(value: Any) -> str | None:
    result = _coerce_string(value)
    return result


def _coerce_result_value(value: Any) -> float | int | str | None:
    if isinstance(value, bool):
        # Prefer textual representation for booleans to avoid confusing them with
        # numeric scores.
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _coerce_int(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            integer_value = int(value)
        except (TypeError, ValueError):
            return None
        return max(minimum, integer_value)
    if isinstance(value, str):
        value_stripped = value.strip()
        if not value_stripped:
            return None
        try:
            integer_value = int(float(value_stripped))
        except (TypeError, ValueError):
            return None
        return max(minimum, integer_value)
    return None


async def _ensure_catalog_from_summary(summary: dict[str, Any], caches: _Caches) -> None:
    for method in _summary_list(summary, "methods"):
        if not isinstance(method, dict):
            continue
        name = method.get("name")
        if name:
            await _ensure_method(name, caches, aliases=method.get("aliases"), description=method.get("description"))

    for dataset in _summary_list(summary, "datasets"):
        if not isinstance(dataset, dict):
            continue
        name = dataset.get("name")
        if name:
            await _ensure_dataset(name, caches, aliases=dataset.get("aliases"), description=dataset.get("description"))

    for metric in _summary_list(summary, "metrics"):
        if not isinstance(metric, dict):
            continue
        name = metric.get("name")
        if name:
            await _ensure_metric(
                name,
                caches,
                unit=metric.get("unit"),
                aliases=metric.get("aliases"),
                description=metric.get("description"),
            )

    for task in _summary_list(summary, "tasks"):
        if not isinstance(task, dict):
            continue
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
    for result in _summary_list(summary, "results"):
        if not isinstance(result, dict):
            continue
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
    for claim in _summary_list(summary, "claims"):
        if not isinstance(claim, dict):
            continue
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


def _merge_tiers(
    existing: Sequence[int] | Sequence[str] | None, new: Sequence[int]
) -> list[int]:
    tier_set: set[int] = set()
    for tier in existing or []:
        try:
            tier_set.add(int(tier))
        except (TypeError, ValueError):
            continue
    tier_set.update(new)
    return sorted(tier_set)


def _coerce_aliases_for_serialization(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, (str, bytes)):
        stripped = values.strip()
        return [stripped] if stripped else []
    if isinstance(values, Iterable):
        aliases: list[str] = []
        for alias in values:
            if not isinstance(alias, str):
                continue
            stripped = alias.strip()
            if stripped:
                aliases.append(stripped)
        return aliases
    return []


def _serialize_method(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": _coerce_aliases_for_serialization(getattr(model, "aliases", None)),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_dataset(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": _coerce_aliases_for_serialization(getattr(model, "aliases", None)),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_metric(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "unit": model.unit,
        "aliases": _coerce_aliases_for_serialization(getattr(model, "aliases", None)),
        "description": model.description,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


def _serialize_task(model) -> dict[str, Any]:
    return {
        "id": str(model.id),
        "name": model.name,
        "aliases": _coerce_aliases_for_serialization(getattr(model, "aliases", None)),
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


