from __future__ import annotations

import copy
import hashlib
from decimal import Decimal
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from uuid import UUID

from app.models.ontology import ResultCreate
from app.services.ontology_store import append_results, ensure_dataset, ensure_method, ensure_metric, ensure_task
TIER_NAME = "tier3_verifier"
_BASE_CONFIDENCE_FALLBACK = 0.55
_JSON_VALID_BONUS = 0.05
_COREF_BONUS = 0.04
_TABLE_MATCH_BONUS = 0.1
_DUPLICATE_BONUS_STEP = 0.03
_DUPLICATE_BONUS_CAP = 0.09

_COREF_HEAD_PATTERN = re.compile(
    r"^(?:this|that|these|those|our|the|its|their|it|they)\b",
    re.IGNORECASE,
)
_COREF_NOUN_PATTERN = re.compile(
    r"(model|method|approach|system|framework|technique|architecture|algorithm)s?\b",
    re.IGNORECASE,
)
_NUMERIC_RESULT_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z0-9\-/+ ]{1,40})\s*(?:=|:)?\s*(?P<value>-?\d+(?:\.\d+)?)(?P<unit>%|\s*%|\s*pts|\s*pp)?",
    re.IGNORECASE,
)
_DATASET_RE = re.compile(r"on\s+(?P<dataset>[A-Za-z0-9\-_/\. ]{2,80})", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\b(test|dev|validation|val|train)\b", re.IGNORECASE)
_TASK_RE = re.compile(r"for\s+(?P<task>[A-Za-z][A-Za-z0-9\- ]{2,80})", re.IGNORECASE)
_CI_RE = re.compile(r"(?:\\u00B1\\s*\\d+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?\\s*-\\s*\\d+(?:\\.\\d+)?)")


@dataclass
class NumericMeasurement:
    metric: str
    value: float
    value_text: str
    unit: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    task: Optional[str] = None
    ci: Optional[str] = None
    normalization_hash: Optional[str] = None

    def compute_hash(self) -> str:
        parts = [
            _normalize_identifier(self.metric),
            f"{self.value:.6f}",
            (self.unit or "").strip().lower(),
            _normalize_identifier(self.dataset),
            _normalize_identifier(self.split),
            _normalize_identifier(self.task),
        ]
        joined = "|".join(parts)
        self.normalization_hash = hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()
        return self.normalization_hash

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric": self.metric,
            "value": self.value,
            "value_text": self.value_text,
        }
        if self.unit:
            payload["unit"] = self.unit
        if self.dataset:
            payload["dataset"] = self.dataset
        if self.split:
            payload["split"] = self.split
        if self.task:
            payload["task"] = self.task
        if self.ci:
            payload["confidence_interval"] = self.ci
        if self.normalization_hash:
            payload["normalization_hash"] = self.normalization_hash
        return payload


@dataclass
class CandidateWrapper:
    original_index: int
    order: tuple[int, int, int]
    data: dict[str, Any]
    section_id: Optional[str]
    sentence_index: Optional[int]
    measurement: Optional[NumericMeasurement] = None
    coref_resolved: bool = False
    table_match: bool = False


async def run_tier3_verifier(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if base_summary is None:
        raise ValueError("Tier-3 verifier requires Tier-1 and Tier-2 summaries")

    summary = copy.deepcopy(base_summary)
    candidates_raw = base_summary.get("triple_candidates") or []
    sections_raw = base_summary.get("sections") or []
    tables_raw = base_summary.get("tables") or []

    tiers = set(summary.get("tiers") or [])
    tiers.update({1, 2, 3})
    summary["tiers"] = sorted(tiers)

    if not candidates_raw:
        metadata = summary.setdefault("metadata", {})
        metadata["tier3"] = {
            "tier": TIER_NAME,
            "triple_count": 0,
            "normalized": 0,
            "table_matches": 0,
            "coref_resolved": 0,
        }
        return summary

    section_index = _build_section_index(sections_raw)
    table_index = _build_table_index(tables_raw)

    wrapped = _prepare_candidates(candidates_raw, section_index)
    _resolve_coreferences(wrapped)
    _attach_measurements(wrapped)
    _detect_table_matches(wrapped, table_index)
    _update_confidences(wrapped)

    summary["triple_candidates"] = [
        wrapper.data for wrapper in sorted(wrapped, key=lambda item: item.original_index)
    ]

    persisted_count = await _persist_structured_results(paper_id, wrapped)

    metadata = summary.setdefault("metadata", {})
    tier3_meta = {
        "tier": TIER_NAME,
        "triple_count": len(wrapped),
        "normalized": sum(1 for wrapper in wrapped if wrapper.measurement),
        "table_matches": sum(1 for wrapper in wrapped if wrapper.table_match),
        "coref_resolved": sum(1 for wrapper in wrapped if wrapper.coref_resolved),
    }
    if persisted_count:
        tier3_meta["persisted_results"] = persisted_count
    metadata["tier3"] = tier3_meta
    return summary


def _prepare_candidates(
    candidates: Sequence[dict[str, Any]],
    section_index: dict[str, dict[str, Any]],
) -> list[CandidateWrapper]:
    wrapped: list[CandidateWrapper] = []
    for idx, candidate in enumerate(candidates):
        section_id, sentence_index = _primary_span(candidate)
        section_meta = section_index.get(section_id or "", {})
        section_order = section_meta.get("order", 1_000_000)
        sentence_order = sentence_index if isinstance(sentence_index, int) else 1_000_000
        wrapper = CandidateWrapper(
            original_index=idx,
            order=(section_order, sentence_order, idx),
            data=copy.deepcopy(candidate),
            section_id=section_id,
            sentence_index=sentence_index if isinstance(sentence_index, int) else None,
        )
        wrapped.append(wrapper)
    wrapped.sort(key=lambda item: item.order)
    return wrapped


def _build_section_index(sections: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for order, section in enumerate(sections):
        section_id = str(section.get("section_id") or "").strip()
        if not section_id:
            continue
        index[section_id] = {
            "order": order,
            "sentences": section.get("sentence_spans") or [],
        }
    return index


def _build_table_index(tables: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        section_id = table.get("section_id")
        if not section_id:
            continue
        index.setdefault(str(section_id), []).append(table)
    return index


def _primary_span(candidate: dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    spans = candidate.get("evidence_spans") or []
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            section_id = span.get("section_id")
            if section_id:
                sentence_index = span.get("sentence_index")
                return str(section_id), sentence_index if isinstance(sentence_index, int) else None
    return None, None


def _resolve_coreferences(candidates: Sequence[CandidateWrapper]) -> None:
    history: dict[str, dict[str, list[str]]] = {}

    for wrapper in candidates:
        candidate = wrapper.data
        section_key = wrapper.section_id or ""
        section_history = history.setdefault(section_key, {})

        subject = str(candidate.get("subject") or "").strip()
        subject_type = str(candidate.get("subject_type_guess") or "unknown").lower()
        resolved_subject, subject_alias = _resolve_entity(subject, subject_type, section_history)
        candidate["subject"] = resolved_subject
        if subject_alias:
            candidate["subject_alias"] = subject_alias
            wrapper.coref_resolved = True
        if not _needs_coref_resolution(resolved_subject):
            section_history.setdefault(subject_type, []).append(resolved_subject)

        obj = str(candidate.get("object") or "").strip()
        object_type = str(candidate.get("object_type_guess") or "unknown").lower()
        resolved_object, object_alias = _resolve_entity(obj, object_type, section_history)
        candidate["object"] = resolved_object
        if object_alias:
            candidate["object_alias"] = object_alias
            wrapper.coref_resolved = True
        if not _needs_coref_resolution(resolved_object):
            section_history.setdefault(object_type, []).append(resolved_object)


def _resolve_entity(
    text: str,
    type_guess: str,
    section_history: dict[str, list[str]],
) -> tuple[str, Optional[str]]:
    if not _needs_coref_resolution(text):
        return text, None

    candidates = section_history.get(type_guess) or []
    if candidates:
        return candidates[-1], text

    # Fallback to any antecedent within the section
    for records in section_history.values():
        if records:
            return records[-1], text
    return text, None


def _needs_coref_resolution(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered in {"it", "they", "this", "that", "these", "those"}:
        return True
    if lowered.startswith("our ") or lowered.startswith("their "):
        return True
    if _COREF_HEAD_PATTERN.match(lowered) and _COREF_NOUN_PATTERN.search(lowered):
        return True
    return False


def _attach_measurements(candidates: Sequence[CandidateWrapper]) -> None:
    for wrapper in candidates:
        measurement = _extract_numeric_measure(wrapper.data)
        wrapper.measurement = measurement
        if measurement:
            measurement.compute_hash()
            wrapper.data["normalization"] = measurement.to_payload()
            verification = wrapper.data.setdefault("verification", {})
            verification["normalization_hash"] = measurement.normalization_hash


def _extract_numeric_measure(candidate: dict[str, Any]) -> Optional[NumericMeasurement]:
    texts: list[str] = []
    for key in ("object", "evidence"):
        value = candidate.get(key)
        if isinstance(value, str):
            texts.append(value)
    seen_metrics: set[str] = set()
    for text in texts:
        for match in _NUMERIC_RESULT_RE.finditer(text):
            metric = _clean_metric(match.group("metric"))
            if not metric or metric.lower() in seen_metrics:
                continue
            seen_metrics.add(metric.lower())
            value_text = match.group("value")
            try:
                value_numeric = float(value_text)
            except ValueError:
                continue
            unit = _clean_unit(match.group("unit"))
            dataset = _extract_dataset(text)
            split = _extract_split(text)
            task = _extract_task(text)
            ci = _extract_ci(text)
            measurement = NumericMeasurement(
                metric=metric,
                value=value_numeric,
                value_text=value_text,
                unit=unit,
                dataset=dataset,
                split=split,
                task=task,
                ci=ci,
            )
            return measurement
    return None


def _clean_metric(metric: str) -> str:
    cleaned = metric.strip(" :")
    cleaned = re.sub(r"\b(score|value)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    cleaned = unit.strip().replace(" ", "")
    if not cleaned:
        return None
    replacements = {"pts": "points", "%": "%", "pp": "pp"}
    return replacements.get(cleaned.lower(), cleaned)


def _extract_dataset(text: str) -> Optional[str]:
    match = _DATASET_RE.search(text)
    if not match:
        return None
    dataset = match.group("dataset").strip()
    dataset = re.split(r"\b(test|dev|validation|dataset|set|split)\b", dataset, maxsplit=1)[0].strip()
    dataset = dataset.strip(" ,.;:)")
    if len(dataset) < 2:
        return None
    return dataset


def _extract_split(text: str) -> Optional[str]:
    match = _SPLIT_RE.search(text)
    if not match:
        return None
    token = match.group(1).lower()
    mapping = {"val": "validation", "dev": "dev"}
    return mapping.get(token, token)


def _extract_task(text: str) -> Optional[str]:
    match = _TASK_RE.search(text)
    if not match:
        return None
    task = match.group("task").strip()
    task = re.split(r"\b(on|using|with)\b", task, maxsplit=1)[0].strip()
    task = task.strip(" ,.;:)")
    if len(task) < 3:
        return None
    return task


def _extract_ci(text: str) -> Optional[str]:
    match = _CI_RE.search(text)
    if not match:
        return None
    return match.group(0).strip()


def _detect_table_matches(
    candidates: Sequence[CandidateWrapper],
    table_index: dict[str, list[dict[str, Any]]],
) -> None:
    for wrapper in candidates:
        measurement = wrapper.measurement
        if not measurement or not wrapper.section_id:
            continue
        tables = table_index.get(wrapper.section_id)
        if not tables:
            continue
        aliases = _value_aliases(measurement)
        match_info: Optional[dict[str, Any]] = None
        for table in tables:
            for cell in table.get("cells", []):
                cell_text = str(cell.get("text") or "")
                if not cell_text:
                    continue
                lowered = cell_text.lower()
                if any(alias in cell_text for alias in aliases):
                    if measurement.metric and measurement.metric.lower() not in lowered:
                        # Allow matches even without metric mention but prefer those that do
                        pass
                    match_info = {
                        "table_id": table.get("table_id"),
                        "cell": cell,
                    }
                    break
            if match_info:
                break
        verification = wrapper.data.setdefault("verification", {})
        if match_info:
            wrapper.table_match = True
            verification["table_match"] = match_info
        else:
            verification.setdefault("table_match", False)


def _value_aliases(measurement: NumericMeasurement) -> list[str]:
    aliases = {measurement.value_text}
    try:
        numeric = float(measurement.value_text)
        formatted = f"{numeric:.4f}".rstrip("0").rstrip(".")
        aliases.add(formatted)
        aliases.add(str(int(numeric))) if numeric.is_integer() else None
    except ValueError:
        pass
    if measurement.unit and measurement.unit != "points":
        aliases.update({f"{alias}{measurement.unit}" for alias in list(aliases)})
    comma_aliases = {alias.replace(".", ",") for alias in aliases if "." in alias}
    aliases.update(comma_aliases)
    return [alias for alias in aliases if alias]


def _update_confidences(candidates: Sequence[CandidateWrapper]) -> None:
    normalization_counts = _count_normalizations(candidates)
    duplicate_counts = _count_duplicate_text(candidates)

    for wrapper in candidates:
        data = wrapper.data
        base_conf = data.get("triple_conf")
        try:
            score = float(base_conf)
        except (TypeError, ValueError):
            score = _BASE_CONFIDENCE_FALLBACK
        score = max(0.0, min(1.0, score))

        components: dict[str, float] = {"tier2_base": round(score, 4)}
        score = _apply_component(score, components, "json_valid", _JSON_VALID_BONUS)

        if wrapper.coref_resolved:
            score = _apply_component(score, components, "coref_resolution", _COREF_BONUS)
        if wrapper.table_match:
            score = _apply_component(score, components, "table_match", _TABLE_MATCH_BONUS)

        if wrapper.measurement and wrapper.measurement.normalization_hash:
            occurrences = normalization_counts.get(wrapper.measurement.normalization_hash, 0)
            if occurrences > 1:
                bonus = min(_DUPLICATE_BONUS_STEP * (occurrences - 1), _DUPLICATE_BONUS_CAP)
                score = _apply_component(score, components, "duplicate_support", bonus)
        else:
            signature = _triple_signature(data)
            occurrences = duplicate_counts.get(signature, 0)
            if occurrences > 1:
                bonus = min(0.02 * (occurrences - 1), 0.06)
                score = _apply_component(score, components, "duplicate_support", bonus)

        final_score = round(min(1.0, score), 4)
        components["final"] = final_score
        data["triple_conf"] = final_score
        data["confidence_components"] = components
        verification = data.setdefault("verification", {})
        verification.setdefault("coref_resolved", wrapper.coref_resolved)
        verification.setdefault("table_match", wrapper.table_match)



async def _persist_structured_results(
    paper_id: UUID,
    candidates: Sequence[CandidateWrapper],
) -> int:
    results: list[ResultCreate] = []
    local_keys: set[tuple[Any, ...]] = set()

    for wrapper in candidates:
        outcome = await _candidate_to_result(paper_id, wrapper)
        if outcome is None:
            continue
        result, key = outcome
        if key in local_keys:
            continue
        local_keys.add(key)
        results.append(result)

    if not results:
        return 0

    inserted = await append_results(paper_id, results)
    return len(inserted)


async def _candidate_to_result(
    paper_id: UUID,
    wrapper: CandidateWrapper,
) -> Optional[tuple[ResultCreate, tuple[Any, ...]]]:
    data = wrapper.data
    subject = (data.get("subject") or "").strip()
    obj = (data.get("object") or "").strip()
    subject_type = (data.get("subject_type_guess") or "unknown").lower()
    object_type = (data.get("object_type_guess") or "unknown").lower()
    relation_guess = (data.get("relation_type_guess") or "other").upper()
    measurement = wrapper.measurement

    method_name = _clean_entity_name(subject) if subject_type == "method" else None
    if not method_name and object_type == "method":
        method_name = _clean_entity_name(obj)
    if not method_name and relation_guess in {"MEASURES", "REPORTS", "ACHIEVES", "EVALUATED_ON", "USES", "OUTPERFORMS", "COMPARED_TO"}:
        method_name = _clean_entity_name(subject)

    dataset_name = None
    if measurement and measurement.dataset:
        dataset_name = _clean_entity_name(measurement.dataset)
    if not dataset_name and object_type == "dataset":
        dataset_name = _clean_entity_name(obj)
    if not dataset_name and relation_guess in {"EVALUATED_ON", "USES"}:
        dataset_name = _clean_entity_name(obj)

    metric_name = None
    metric_unit = None
    if measurement and measurement.metric:
        metric_name = _clean_entity_name(measurement.metric)
        metric_unit = measurement.unit
    if not metric_name and data.get("metric_inference"):
        metric_name = _clean_entity_name(data["metric_inference"].get("normalized_metric"))
    if not metric_name and object_type == "metric":
        metric_name = _clean_entity_name(obj)

    task_name = None
    if measurement and measurement.task:
        task_name = _clean_entity_name(measurement.task)
    if not task_name and object_type == "task":
        task_name = _clean_entity_name(obj)
    if not task_name and subject_type == "task":
        task_name = _clean_entity_name(subject)
    if not task_name and relation_guess == "PROPOSES":
        task_name = _clean_entity_name(obj)

    if subject_type == "task" and not method_name and object_type == "method":
        method_name = _clean_entity_name(obj)

    if not method_name:
        return None

    if not any([dataset_name, metric_name, task_name, measurement]):
        return None

    method_model = await ensure_method(method_name)
    dataset_model = await ensure_dataset(dataset_name) if dataset_name else None
    metric_model = await ensure_metric(metric_name, unit=metric_unit) if metric_name else None
    task_model = await ensure_task(task_name) if task_name else None

    value_numeric = None
    value_text: Optional[str] = None
    if measurement:
        value_text = measurement.value_text or obj or None
        if measurement.value is not None:
            try:
                value_numeric = Decimal(str(measurement.value))
            except (TypeError, ValueError):
                value_numeric = None
    if value_text is None:
        value_text = obj or None

    evidence: list[dict[str, Any]] = []
    snippet = data.get("evidence")
    if snippet:
        evidence_item = {
            "snippet": snippet,
            "section_id": wrapper.section_id,
            "candidate_id": data.get("candidate_id"),
            "relation": data.get("relation"),
            "tier": data.get("tier"),
        }
        evidence.append(evidence_item)

    confidence = data.get("triple_conf")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    result = ResultCreate(
        paper_id=paper_id,
        method_id=method_model.id,
        dataset_id=dataset_model.id if dataset_model else None,
        metric_id=metric_model.id if metric_model else None,
        task_id=task_model.id if task_model else None,
        split=measurement.split if measurement else None,
        value_numeric=value_numeric,
        value_text=value_text,
        is_sota=False,
        confidence=confidence_value,
        evidence=evidence,
    )

    key = (
        result.method_id,
        result.dataset_id,
        result.metric_id,
        result.task_id,
        result.split,
        (result.value_text or "").strip(),
        str(result.value_numeric) if result.value_numeric is not None else None,
    )
    return result, key


def _clean_entity_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _apply_component(score: float, components: dict[str, float], key: str, bonus: float) -> float:
    if bonus <= 0:
        return score
    components[key] = round(bonus, 4)
    return min(1.0, score + bonus)


def _count_normalizations(candidates: Sequence[CandidateWrapper]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for wrapper in candidates:
        if wrapper.measurement and wrapper.measurement.normalization_hash:
            key = wrapper.measurement.normalization_hash
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_duplicate_text(candidates: Sequence[CandidateWrapper]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for wrapper in candidates:
        signature = _triple_signature(wrapper.data)
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def _triple_signature(candidate: dict[str, Any]) -> str:
    parts = [
        _normalize_identifier(candidate.get("subject")),
        _normalize_identifier(candidate.get("relation")),
        _normalize_identifier(candidate.get("object")),
    ]
    return "|".join(parts)


def _normalize_identifier(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    return normalized
