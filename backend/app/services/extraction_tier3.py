from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Any, Iterable, Sequence
from uuid import UUID

from app.models.ontology import ResultCreate
from app.models.section import Section
from app.services.ontology_store import replace_results
from app.services.papers import get_paper
from app.services.sections import list_sections


from typing import Optional
TIER_NAME = "tier3_verifier"
_VALUE_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_SPAN_WINDOW = 80


@dataclass
class _VerificationOutcome:
    result: ResultCreate
    notes: list[str]
    matched_span: bool


async def run_tier3_verifier(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]] = None,
    sections: Optional[Sequence[Section]] = None,
) -> dict[str, Any]:
    """Cross-check stored results against the paper text and annotate verification."""

    if base_summary is None:
        raise ValueError("Tier-3 verifier requires an existing summary to audit")

    paper = await get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    if sections is None:
        sections = await list_sections(paper_id=paper_id, limit=500, offset=0)

    section_lookup = {str(section.id): section for section in sections}

    method_payloads = _build_entity_lookup(base_summary.get("methods", []))
    dataset_payloads = _build_entity_lookup(base_summary.get("datasets", []))
    metric_payloads = _build_entity_lookup(base_summary.get("metrics", []))
    task_payloads = _build_entity_lookup(base_summary.get("tasks", []))

    results_payload = list(base_summary.get("results", []))
    result_models: list[ResultCreate] = []
    audit_entries: list[dict[str, Any]] = []

    for index, payload in enumerate(results_payload):
        model = _summary_result_to_model(paper_id, payload)
        metric_payload = metric_payloads.get(model.metric_id)

        outcome = _verify_result(
            model,
            metric_payload=metric_payload,
            sections=section_lookup,
        )
        result_models.append(outcome.result)

        audit_entries.append(
            {
                "index": index,
                "method": _lookup_name(method_payloads, model.method_id),
                "dataset": _lookup_name(dataset_payloads, model.dataset_id),
                "metric": _lookup_name(metric_payloads, model.metric_id),
                "verified": outcome.result.verified,
                "confidence": outcome.result.confidence,
                "matched_span": outcome.matched_span,
                "notes": outcome.notes or None,
            }
        )

    stored_results = await replace_results(paper_id, result_models)

    tiers = _merge_tiers(base_summary.get("tiers", []), [3])

    summary_payload = {
        "paper_id": str(paper_id),
        "tiers": tiers,
        "methods": copy.deepcopy(base_summary.get("methods", [])),
        "datasets": copy.deepcopy(base_summary.get("datasets", [])),
        "metrics": copy.deepcopy(base_summary.get("metrics", [])),
        "tasks": copy.deepcopy(base_summary.get("tasks", [])),
        "results": [],
        "claims": copy.deepcopy(base_summary.get("claims", [])),
        "audit_log": audit_entries,
    }

    tier_by_index = [payload.get("tier") for payload in results_payload]

    for index, result in enumerate(stored_results):
        serialized = _serialize_result(
            result,
            method_payloads=method_payloads,
            dataset_payloads=dataset_payloads,
            metric_payloads=metric_payloads,
            task_payloads=task_payloads,
        )
        tier_value = tier_by_index[index] if index < len(tier_by_index) else None
        if tier_value is not None:
            serialized["tier"] = tier_value
        serialized["verifier"] = TIER_NAME
        summary_payload["results"].append(serialized)

    return summary_payload




def _confidence_from_evidence(evidence: Optional[Iterable[dict[str, Any]]]) -> tuple[float, bool, str]:
    base = 0.5
    has_ner = False
    provenance = "baseline heuristics"
    for entry in evidence or []:
        tier = str(entry.get("tier") or "")
        source = str(entry.get("source") or "")
        lowered_source = source.lower()
        lowered_tier = tier.lower()
        if lowered_source == "table":
            if base < 0.7:
                base = 0.7
                provenance = "table evidence"
        elif lowered_source:
            if base < 0.6:
                base = 0.6
                provenance = f"{lowered_source} evidence"
        if lowered_tier:
            if "tier1" in lowered_tier and base < 0.6:
                base = 0.6
                provenance = f"{tier} evidence"
            if lowered_tier == "spacy_structurer":
                if base < 0.55:
                    base = 0.55
                    provenance = "spaCy/SciSpaCy evidence"
                has_ner = True
    return base, has_ner, provenance
def _verify_result(
    model: ResultCreate,
    *,
    metric_payload: Optional[dict[str, Any]],
    sections: dict[str, Section],
) -> _VerificationOutcome:
    notes: list[str] = []
    matched_span = False

    base_confidence, has_ner, provenance = _confidence_from_evidence(model.evidence)
    confidence = base_confidence
    if model.confidence is not None:
        try:
            confidence = max(confidence, float(model.confidence))
        except (TypeError, ValueError):
            pass

    notes.append(f"confidence baseline derived from {provenance}")

    if has_ner:
        confidence = min(confidence + 0.05, 1.0)
        notes.append("spaCy/SciSpaCy evidence corroborates extraction")

    numeric_value = model.value_numeric
    text_value = model.value_text.strip() if model.value_text else None

    normalized_numeric, normalized_text, normalization_notes = _normalize_value(
        numeric_value,
        text_value,
        metric_payload,
    )
    if normalization_notes:
        notes.extend(normalization_notes)

    numeric_value = normalized_numeric
    text_value = normalized_text

    if numeric_value is None:
        notes.append("unable to determine numeric value for verification")

    is_outlier, outlier_note = _detect_outlier(numeric_value, metric_payload)
    verified = numeric_value is not None and not is_outlier

    if outlier_note:
        notes.append(outlier_note)

    if verified:
        matched_span, span_note = _match_value_in_evidence(
            numeric_value,
            text_value,
            model.evidence,
            sections,
            metric_payload,
        )
        if span_note:
            notes.append(span_note)
        if matched_span:
            confidence = min(confidence + 0.05, 1.0)
        else:
            confidence = max(confidence - 0.05, 0.0)
            verified = False
            notes.append("metric value not confirmed near evidence span")

    verifier_notes = _combine_notes(notes)

    updated_model = ResultCreate(
        paper_id=model.paper_id,
        method_id=model.method_id,
        dataset_id=model.dataset_id,
        metric_id=model.metric_id,
        task_id=model.task_id,
        split=model.split,
        value_numeric=numeric_value,
        value_text=text_value,
        is_sota=model.is_sota,
        confidence=confidence,
        evidence=model.evidence,
        verified=verified,
        verifier_notes=verifier_notes,
    )

    return _VerificationOutcome(result=updated_model, notes=notes, matched_span=matched_span)


def _normalize_value(
    numeric_value: Optional[Decimal],
    text_value: Optional[str],
    metric_payload: Optional[dict[str, Any]],
) -> tuple[Optional[Decimal], Optional[str], list[str]]:
    notes: list[str] = []
    sanitized_text = text_value.strip() if text_value else None

    extracted = _extract_decimal(sanitized_text) if sanitized_text else None
    if numeric_value is None and extracted is not None:
        numeric_value = extracted
        notes.append("numeric value inferred from text")

    is_percent_metric = _is_percent_metric(metric_payload, sanitized_text)

    if numeric_value is not None and is_percent_metric:
        if numeric_value <= 1:
            numeric_value = numeric_value * Decimal("100")
            notes.append("scaled fractional percent to 0-100 range")
        if numeric_value < 0:
            numeric_value = None
            notes.append("discarded negative percent value")
        elif numeric_value > 100:
            notes.append("percent metric exceeds 100")
        if sanitized_text:
            sanitized_text = sanitized_text.replace("%", "").strip()
    elif sanitized_text and extracted is not None and numeric_value is None:
        numeric_value = extracted

    if numeric_value is None and sanitized_text:
        notes.append("text did not contain a parsable number")

    normalized_text = sanitized_text
    if numeric_value is not None and normalized_text is None:
        normalized_text = str(numeric_value)

    return numeric_value, normalized_text, notes


def _match_value_in_evidence(
    numeric_value: Optional[Decimal],
    text_value: Optional[str],
    evidence: Iterable[dict[str, Any]],
    sections: dict[str, Section],
    metric_payload: Optional[dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    tokens = _generate_value_tokens(numeric_value, text_value, metric_payload)
    if not tokens:
        return False, "no numeric tokens available for verification"

    for entry in evidence or []:
        span = entry.get("evidence_span")
        if not span:
            continue
        section = sections.get(str(span.get("section_id")))
        if section is None:
            continue
        start = max(int(span.get("start", 0)) - _SPAN_WINDOW, 0)
        end = min(int(span.get("end", start)) + _SPAN_WINDOW, len(section.content))
        snippet = section.content[start:end]
        for token in tokens:
            if token and token in snippet:
                return True, f"value '{token}' confirmed near evidence span"

    return False, "metric value not found near provided evidence span"


def _detect_outlier(
    numeric_value: Optional[Decimal],
    metric_payload: Optional[dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    if numeric_value is None:
        return False, None

    metric_name = (metric_payload or {}).get("name")
    if metric_name and "bleu" in metric_name.lower() and numeric_value > Decimal("100"):
        return True, f"BLEU value {numeric_value} exceeds maximum expected range"

    unit = (metric_payload or {}).get("unit")
    if unit and unit.strip().lower() in {"%", "percent", "percentage"}:
        if numeric_value < 0 or numeric_value > Decimal("100"):
            return True, f"percent metric out of bounds: {numeric_value}"

    return False, None


def _generate_value_tokens(
    numeric_value: Optional[Decimal],
    text_value: Optional[str],
    metric_payload: Optional[dict[str, Any]],
) -> set[str]:
    tokens: set[str] = set()
    if text_value:
        tokens.add(text_value)
        tokens.add(text_value.replace("%", "").strip())

    if numeric_value is not None:
        float_value = float(numeric_value)
        tokens.add(f"{float_value:.0f}")
        tokens.add(f"{float_value:.1f}")
        tokens.add(f"{float_value:.2f}")
        if text_value and "%" in text_value:
            tokens.add(f"{float_value:.0f}%")
            tokens.add(f"{float_value:.1f}%")
        unit = (metric_payload or {}).get("unit")
        if unit and unit.strip().lower() in {"%", "percent", "percentage"}:
            tokens.add(f"{float_value:.0f}%")
            tokens.add(f"{float_value:.1f}%")
            tokens.add(f"{float_value:.2f}%")
        if float_value >= 1:
            tokens.add(str(int(round(float_value))))
        else:
            tokens.add(f"{float_value:.3f}")

    tokens = {token for token in tokens if token}
    return tokens


def _extract_decimal(text_value: Optional[str]) -> Optional[Decimal]:
    if not text_value:
        return None
    cleaned = text_value.replace(",", "")
    match = _VALUE_PATTERN.search(cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group())
    except Exception:
        return None


def _is_percent_metric(
    metric_payload: Optional[dict[str, Any]],
    text_value: Optional[str],
) -> bool:
    if metric_payload and metric_payload.get("unit"):
        unit = metric_payload["unit"].strip().lower()
        if unit in {"%", "percent", "percentage"}:
            return True
    if text_value and "%" in text_value:
        return True
    return False


def _summary_result_to_model(paper_id: UUID, payload: dict[str, Any]) -> ResultCreate:
    method_id = _parse_uuid(((payload.get("method") or {}).get("id")))
    dataset_id = _parse_uuid(((payload.get("dataset") or {}).get("id")))
    metric_id = _parse_uuid(((payload.get("metric") or {}).get("id")))
    task_id = _parse_uuid(((payload.get("task") or {}).get("id")))

    value_numeric = payload.get("value_numeric")
    numeric_decimal = None
    if value_numeric is not None:
        numeric_decimal = Decimal(str(value_numeric))

    return ResultCreate(
        paper_id=paper_id,
        method_id=method_id,
        dataset_id=dataset_id,
        metric_id=metric_id,
        task_id=task_id,
        split=payload.get("split"),
        value_numeric=numeric_decimal,
        value_text=payload.get("value_text"),
        is_sota=bool(payload.get("is_sota")),
        confidence=payload.get("confidence"),
        evidence=list(payload.get("evidence") or []),
        verified=payload.get("verified"),
        verifier_notes=payload.get("verifier_notes"),
    )


def _serialize_result(
    result,
    *,
    method_payloads: dict[UUID, dict[str, Any]],
    dataset_payloads: dict[UUID, dict[str, Any]],
    metric_payloads: dict[UUID, dict[str, Any]],
    task_payloads: dict[UUID, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": str(result.id),
        "paper_id": str(result.paper_id),
        "method": _clone_entity(method_payloads.get(result.method_id)),
        "dataset": _clone_entity(dataset_payloads.get(result.dataset_id)),
        "metric": _clone_entity(metric_payloads.get(result.metric_id)),
        "task": _clone_entity(task_payloads.get(result.task_id)),
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


def _build_entity_lookup(items: Iterable[dict[str, Any]]) -> dict[UUID, dict[str, Any]]:
    lookup: dict[UUID, dict[str, Any]] = {}
    for item in items:
        identifier = _parse_uuid(item.get("id"))
        if identifier is None:
            continue
        lookup[identifier] = copy.deepcopy(item)
    return lookup


def _clone_entity(entity: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if entity is None:
        return None
    return copy.deepcopy(entity)


def _lookup_name(
    lookup: dict[UUID, dict[str, Any]],
    identifier: Optional[UUID],
) -> Optional[str]:
    if identifier is None:
        return None
    entity = lookup.get(identifier)
    if entity is None:
        return None
    return entity.get("name")


def _parse_uuid(value: Any) -> Optional[UUID]:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except Exception:
        return None


def _merge_tiers(existing: Union[Sequence[int], Sequence[str]], new: Sequence[int]) -> list[int]:
    tier_set: set[int] = set()
    for tier in existing:
        try:
            tier_set.add(int(tier))
        except (TypeError, ValueError):
            continue
    tier_set.update(new)
    return sorted(tier_set)


def _combine_notes(notes: Iterable[str]) -> Optional[str]:
    cleaned = [note for note in notes if note]
    if not cleaned:
        return None
    # Preserve order while removing duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for note in cleaned:
        if note in seen:
            continue
        seen.add(note)
        ordered.append(note)
    return "; ".join(ordered)


__all__ = ["run_tier3_verifier", "TIER_NAME"]