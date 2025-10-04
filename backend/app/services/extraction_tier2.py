from __future__ import annotations

import asyncio
import copy
import difflib
import json
import logging
import re
import string
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from uuid import UUID

try:  # pragma: no cover - optional dependency guard
    import httpx
except ImportError:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore

try:  # pragma: no cover - optional dependency guard
    from jsonschema import Draft7Validator, ValidationError as JSONSchemaValidationError
except ImportError:  # pragma: no cover - optional dependency guard
    Draft7Validator = None  # type: ignore
    JSONSchemaValidationError = None  # type: ignore

from pydantic import ValidationError

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx  # type: ignore[import]
    except ImportError:  # pragma: no cover - optional dependency
        httpx = None  # type: ignore[assignment]


from app.core.config import settings
from app.schemas.tier2 import (
    RELATION_GUESS_VALUES,
    TYPE_GUESS_VALUES,
    TripleExtractionResponse,
    TriplePayload,
)
from app.models.triple_candidate import TripleCandidateRecord
from app.services.triple_candidates import replace_triple_candidates
from app.services.normalization import (
    METRIC_SYNONYM_MAP,
    extract_measurement,
    infer_metric_from_text,
    normalize_metric_name,
)



logger = logging.getLogger(__name__)

TIER_NAME = "tier2_llm_openie"
_DEFAULT_SYSTEM_PROMPT = settings.tier2_llm_system_prompt
DEFAULT_TRIPLE_CONFIDENCE = 0.55
DEFAULT_SCHEMA_SCORE = 1.0
MAX_LLM_ATTEMPTS = 2
METRIC_INFERENCE_CONF_PENALTY = 0.05
COMPARISON_GUIDANCE = (
    "Extract comparisons, ablations, and causal claims. Capture statements such as 'X improves Y', 'X vs. Y', 'with vs without', or reported gains/losses."
    " Infer implied metrics when terms like misclassification rate or error rate appear."
)


TRIPLE_JSON_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "subject",
                    "relation",
                    "object",
                    "evidence",
                    "subject_span",
                    "object_span",
                    "subject_type_guess",
                    "object_type_guess",
                    "relation_type_guess",
                ],
                "properties": {
                    "subject": {"type": "string", "minLength": 2, "maxLength": 120},
                    "relation": {"type": "string", "minLength": 2, "maxLength": 60},
                    "object": {"type": "string", "minLength": 1, "maxLength": 120},
                    "evidence": {"type": "string", "minLength": 10, "maxLength": 400},
                    "subject_span": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "object_span": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "subject_type_guess": {"type": "string", "enum": list(TYPE_GUESS_VALUES)},
                    "object_type_guess": {"type": "string", "enum": list(TYPE_GUESS_VALUES)},
                    "relation_type_guess": {
                        "type": "string",
                        "enum": list(RELATION_GUESS_VALUES),
                    },
                    "triple_conf": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "schema_match_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "section_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "default": [],
        },
        "warnings": {"type": "array", "items": {"type": "string"}, "default": []},
        "discarded": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["triples"],
    "additionalProperties": False,
}


def _system_prompt() -> str:
    """Return the configured Tier-2 system prompt augmented with comparison guidance."""

    prompt = getattr(settings, "tier2_llm_system_prompt", None)
    if isinstance(prompt, str):
        stripped = prompt.strip()
        base_prompt = stripped if stripped else _DEFAULT_SYSTEM_PROMPT
    else:
        base_prompt = _DEFAULT_SYSTEM_PROMPT

    guidance = "\n".join(COMPARISON_GUIDANCE).strip()
    if not guidance:
        return base_prompt

    if not base_prompt:
        return guidance

    return f"{base_prompt.rstrip()}\n\n{guidance}"


def _build_triple_json_schema(max_triples: int) -> dict[str, Any]:
    schema = copy.deepcopy(TRIPLE_JSON_SCHEMA_TEMPLATE)
    schema["properties"]["triples"]["maxItems"] = max_triples
    return schema


def get_triple_json_schema() -> dict[str, Any]:
    """Return the triple JSON schema using the current settings."""

    return _build_triple_json_schema(settings.tier2_llm_max_triples)


TRIPLE_JSON_SCHEMA: dict[str, Any] = get_triple_json_schema()


@dataclass
class SectionContext:
    section_id: str
    section_hash: str | None
    title: str | None
    page_number: int | None
    sentences: list[tuple[int, str]]
    captions: list[str]
    chunk_id: str | None = None

    def formatted_text(self) -> str:
        lines: list[str] = []
        for idx, text in self.sentences:
            lines.append(f"[{idx}] {text}")
        for caption in self.captions:
            lines.append(f"[caption] {caption}")
        return "\n".join(lines)


@dataclass
class GuardrailStats:
    pronoun_resolved_subjects: int = 0
    pronoun_resolved_objects: int = 0
    low_info_subject_dropped: int = 0
    low_info_object_dropped: int = 0
    deduplicated_triples: int = 0
    span_clamped: int = 0
    span_fallback: int = 0
    fuzzy_matches: int = 0
    unmatched_evidence: int = 0
    metrics_inferred: int = 0
    measurements_detected: int = 0

    def to_metadata(self) -> dict[str, int]:
        return asdict(self)

LOW_INFO_TERMS = frozenset(
    {
        "it",
        "its",
        "itself",
        "they",
        "them",
        "theirs",
        "themselves",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "this",
        "that",
        "these",
        "those",
        "this work",
        "that work",
        "our work",
        "the work",
        "the paper",
        "our paper",
        "the authors",
        "the method",
        "the model",
        "the approach",
        "the system",
        "the framework",
        "the technique",
        "the algorithm",
        "the pipeline",
        "the study",
        "our method",
        "our model",
        "our approach",
        "our system",
        "our framework",
        "our technique",
        "our algorithm",
        "this method",
        "this model",
        "this approach",
        "this system",
        "this framework",
        "this technique",
        "this algorithm",
        "that method",
        "that model",
        "that approach",
        "that system",
        "that framework",
        "their method",
        "their model",
        "their approach",
        "proposed method",
        "proposed model",
        "proposed approach",
        "the proposed method",
        "the proposed model",
        "the proposed approach",
    }
)
LOW_INFO_PREFIXES: tuple[str, ...] = (
    "this ",
    "that ",
    "these ",
    "those ",
    "our ",
    "their ",
    "its ",
    "the proposed ",
    "the presented ",
    "the aforementioned ",
    "such ",
)
LOW_INFO_BASE_TERMS = frozenset(
    {
        "method",
        "model",
        "approach",
        "system",
        "framework",
        "technique",
        "algorithm",
        "pipeline",
        "architecture",
        "study",
        "work",
    }
)
QUOTE_CHARS = "\"'\u2018\u2019\u201c\u201d"

MAX_ANTECEDENT_LOOKBACK = 5

ANTECEDENT_QUOTE_RE = re.compile(r'[\"\'\\u2018\\u2019\\u201c\\u201d]([^\"\'\\u2018\\u2019\\u201c\\u201d]{3,})[\"\'\\u2018\\u2019\\u201c\\u201d]')

ANTECEDENT_CAPITAL_RE = re.compile(r'([A-Z][A-Za-z0-9]*(?:[\s-][A-Z][A-Za-z0-9]*){0,4})')

CITATION_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s*\[[^\]]+\]\s*$"),
    re.compile(
        r"\s*\((?:[^)]*\b(?:doi|arxiv|vol\.|volume|no\.|issue|pp\.|pages|proc\.|conf\.|phys\.|rev\.|lett\.|\d{4})[^)]*)\)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b[A-Z][A-Za-z]{0,10}\.\s*){2,}(?:\b[A-Z][A-Za-z]{0,10}\.?)?(?:\s+\d{1,4}(?:[,;]\s*\d{1,6})*)?(?:\s*\(\d{4}\))?\s*$"
    ),
    re.compile(r"\b(?:vol|volume|no|issue)\.?\s*\d+(?:[,;]\s*\d+)?\s*$", re.IGNORECASE),
)

CITATION_ABBREVIATION_RE = re.compile(
    r"^(?:\b[A-Z][A-Za-z]{0,10}\.\s*){2,}(?:\b[A-Z][A-Za-z]{0,10}\.?)?(?:\s+\d{1,4}(?:[,;]\s*\d{1,6})*)?(?:\s*\(\d{4}\))?$"
)

CITATION_LOW_INFO_TOKENS = frozenset(
    {
        "acta",
        "adv",
        "conf",
        "ieee",
        "int",
        "j",
        "lett",
        "letters",
        "nat",
        "phys",
        "proc",
        "rev",
        "symp",
        "trans",
    }
)


def _normalize_whitespace(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_graph_text(value: Optional[str]) -> str:
    return _normalize_whitespace(value)


def _strip_citation_fragments(value: Optional[str]) -> str:
    if not value:
        return ""
    stripped = value.strip()
    previous = None
    while previous != stripped:
        previous = stripped
        for pattern in CITATION_SUFFIX_PATTERNS:
            stripped = pattern.sub("", stripped)
        stripped = stripped.strip()
    stripped = _normalize_whitespace(stripped)
    if CITATION_ABBREVIATION_RE.match(stripped):
        return ""
    return stripped


GRAPH_PUNCTUATION = frozenset(set(string.punctuation) - {"-", "_", "/"})


def _graph_value_passes_quality(value: Optional[str]) -> bool:
    if not value:
        return False
    if _is_low_info_text(value):
        return False
    if not re.search(r"[A-Za-z]", value):
        return False

    tokens = re.findall(r"[A-Za-z]+", value)
    if not tokens:
        return False

    lowered_tokens = [token.lower() for token in tokens]
    if lowered_tokens and all(token in CITATION_LOW_INFO_TOKENS for token in lowered_tokens):
        return False

    letters = sum(len(token) for token in tokens)
    alnum_chars = sum(1 for char in value if char.isalnum())
    if alnum_chars == 0:
        return False
    if letters < max(2, int(alnum_chars * 0.35)):
        return False

    punctuation_heavy = sum(1 for char in value if char in GRAPH_PUNCTUATION)
    if letters <= punctuation_heavy:
        return False

    return True


METRIC_KEYWORDS = frozenset(
    {
        "accuracy",
        "f1",
        "f-score",
        "f score",
        "precision",
        "recall",
        "bleu",
        "bleu score",
        "rouge",
        "wer",
        "word error rate",
        "error rate",
        "error",
        "loss",
        "perplexity",
        "auc",
        "auroc",
        "auprc",
        "psnr",
        "ssim",
        "mse",
        "rmse",
        "mae",
        "map",
        "mean average precision",
        "dice",
        "iou",
        "exact match",
        "top-1",
        "top 1",
        "top-5",
        "top 5",
        "cer",
        "mrr",
        "hit@1",
        "hit@5",
    }
    | {synonym.lower() for synonym in METRIC_SYNONYM_MAP}
    | {info["normalized_metric"].lower() for info in METRIC_SYNONYM_MAP.values()}
    | {info["variant"].lower() for info in METRIC_SYNONYM_MAP.values()}
)

METRIC_VALUE_UNIT_RE = re.compile(
    r"(?:%|percent|percentage|points|score|db|dB|ms|s|sec|second|seconds|minute|minutes|hour|hours|fps|flops|params|parameters|samples|epochs|iterations)",
    re.IGNORECASE,
)

DATASET_TASK_BANNED_PREFIXES = (
    "we ",
    "our ",
    "this ",
    "that ",
    "these ",
    "those ",
)

DATASET_TASK_VERB_RE = re.compile(
    r"\b(observe|observed|observes|observing|achieve|achieved|achieves|achieving|report|reported|reports|reporting|propose|proposed|proposes|proposing|introduce|introduced|introduces|introducing|improve|improved|improves|improving|show|showed|shows|showing|demonstrate|demonstrated|demonstrates|demonstrating|evaluate|evaluated|evaluates|evaluating)\b",
    re.IGNORECASE,
)


def _graph_value_valid_for_type(value: str, entity_type: str) -> bool:
    if not _graph_value_passes_quality(value):
        return False

    normalized_type = (entity_type or "").strip().lower()
    if not normalized_type:
        return True

    if normalized_type == "metric":
        lowered = value.lower()
        has_keyword = any(keyword in lowered for keyword in METRIC_KEYWORDS)
        has_number = bool(re.search(r"\d", value))
        has_unit = bool(METRIC_VALUE_UNIT_RE.search(value))
        if not (has_keyword or (has_number and has_unit)):
            return False
    elif normalized_type in {"dataset", "task"}:
        lowered = value.lower().strip()
        if any(lowered.startswith(prefix) for prefix in DATASET_TASK_BANNED_PREFIXES):
            return False
        if DATASET_TASK_VERB_RE.search(lowered):
            return False

    return True


def _normalized_graph_key(value: str) -> str:
    normalized = _normalize_graph_text(value)
    return normalized.lower()


def _collect_sentence_indices(matches: Sequence[Mapping[str, Any]]) -> list[int]:
    indices: set[int] = set()
    for match in matches:
        index = match.get("sentence_index")
        if isinstance(index, int):
            indices.add(index)
    return sorted(indices)


def _extract_graph_entities(
    subject_text: str,
    subject_type: str,
    object_text: str,
    object_type: str,
) -> dict[str, Optional[dict[str, str]]]:
    subject_entry = _build_entity_entry(subject_text, subject_type, role="subject")
    object_entry = _build_entity_entry(object_text, object_type, role="object")

    entities: dict[str, Optional[dict[str, str]]] = {
        "subject": subject_entry,
        "object": object_entry,
    }

    for entry in (subject_entry, object_entry):
        if not entry:
            continue
        entity_key = entry.get("type")
        if entity_key:
            entities.setdefault(entity_key, {"text": entry.get("text"), "normalized": entry.get("normalized"), "type": entity_key, "role": entry.get("role")})

    return entities


GRAPH_DATASET_RELATIONS = {"EVALUATED_ON", "USES"}
GRAPH_METRIC_RELATIONS = {"MEASURES", "REPORTS"}
GRAPH_TASK_RELATIONS = {"PROPOSES"}

_TYPE_GUESS_NORMALIZATION: dict[str, str] = {
    "method": "method",
    "model": "model",
    "dataset": "dataset",
    "metric": "metric",
    "task": "task",
    "concept": "concept",
    "material": "material",
    "organism": "organism",
    "finding": "finding",
    "process": "process",
    "result": "finding",
    "outcome": "finding",
    "unknown": "concept",
}

_RELATION_GUESS_TO_RELATION: dict[str, str] = {
    "EVALUATED_ON": "evaluates_on",
    "USES": "uses",
    "MEASURES": "reports",
    "REPORTS": "reports",
    "PROPOSES": "proposes",
    "COMPARED_TO": "compares",
    "OUTPERFORMS": "outperforms",
    "CAUSES": "causes",
    "PART_OF": "part_of",
    "IS_A": "is_a",
    "ASSUMES": "assumes",
}


def _normalize_entity_type_guess(type_guess: str) -> str:
    normalized = (type_guess or "").strip().lower()
    return _TYPE_GUESS_NORMALIZATION.get(normalized, "concept")


def _build_entity_entry(value: str, type_guess: str, *, role: str) -> Optional[dict[str, str]]:
    normalized_type = _normalize_entity_type_guess(type_guess)
    if not value:
        return None
    cleaned_value = _strip_citation_fragments(value)
    normalized_value = _normalize_graph_text(cleaned_value)
    if not normalized_value:
        return None
    entry = {
        "type": normalized_type,
        "text": cleaned_value,
        "normalized": normalized_value.lower(),
        "role": role,
    }
    return entry


def _build_graph_metadata(
    entities: Mapping[str, Any],
    *,
    relation_type_guess: Optional[str],
    matches: Sequence[Mapping[str, Any]],
    evidence_text: str,
    section_id: Optional[str],
    triple_conf: float,
    metric_inference: Optional[Mapping[str, Any]] = None,
    measurement: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"entities": {}, "pairs": []}

    subject_entry = entities.get("subject")
    object_entry = entities.get("object")
    if not isinstance(subject_entry, Mapping) or not isinstance(object_entry, Mapping):
        return {}

    def _store_entity(entry: Mapping[str, Any]) -> None:
        entry_type = entry.get("type")
        if not entry_type:
            return
        normalized_value = entry.get("normalized")
        if not _graph_value_valid_for_type(normalized_value, str(entry_type)):
            return
        metadata["entities"].setdefault(
            entry_type,
            {k: entry[k] for k in ("text", "normalized", "type") if k in entry},
        )

    _store_entity(subject_entry)
    _store_entity(object_entry)

    normalized_guess = (relation_type_guess or "").strip().upper()
    relation = _RELATION_GUESS_TO_RELATION.get(normalized_guess)

    if relation is None:
        subject_type = str(subject_entry.get("type") or "")
        object_type = str(object_entry.get("type") or "")
        if subject_type == "method" and object_type == "dataset":
            relation = "evaluates_on"
        elif subject_type == "method" and object_type == "metric":
            relation = "reports"
        elif subject_type == "method" and object_type == "task":
            relation = "proposes"

    if relation is None:
        return {}

    if not _graph_value_valid_for_type(
        subject_entry.get("normalized"), str(subject_entry.get("type"))
    ):
        return {}
    if not _graph_value_valid_for_type(
        object_entry.get("normalized"), str(object_entry.get("type"))
    ):
        return {}

    sentence_indices = _collect_sentence_indices(matches)
    confidence = max(0.0, min(1.0, float(triple_conf))) if triple_conf is not None else None

    def _build_endpoint(entry: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "type": entry.get("type"),
            "text": entry.get("text"),
            "normalized": entry.get("normalized"),
        }
        return {key: value for key, value in payload.items() if value}

    pair_payload: dict[str, Any] = {
        "source": _build_endpoint(subject_entry),
        "target": _build_endpoint(object_entry),
        "relation": relation,
        "confidence": confidence,
        "section_id": section_id,
        "sentence_indices": sentence_indices,
        "evidence": evidence_text,
        "provenance": "tier2_triple",
    }

    metadata["pairs"].append(pair_payload)

    metric_entity = metadata["entities"].get("metric")
    if metric_inference:
        metadata["metric_inference"] = dict(metric_inference)
        normalized_metric = metric_inference.get("normalized_metric")
        if normalized_metric and isinstance(metric_entity, dict):
            metric_entity.setdefault("canonical", normalized_metric)
    elif isinstance(metric_entity, dict):
        normalized_metric = normalize_metric_name(metric_entity.get("text"))
        if normalized_metric:
            metric_entity.setdefault("canonical", normalized_metric)

    if measurement:
        metadata["measurement"] = dict(measurement)

    return metadata


def extract_graph_entities(
    subject_text: str,
    subject_type: str,
    object_text: str,
    object_type: str,
) -> dict[str, Optional[dict[str, str]]]:
    """Public wrapper exposing graph entity extraction for other tiers."""

    return _extract_graph_entities(subject_text, subject_type, object_text, object_type)


def build_graph_metadata(
    entities: Mapping[str, Any],
    *,
    relation_type_guess: Optional[str],
    matches: Sequence[Mapping[str, Any]],
    evidence_text: str,
    section_id: Optional[str],
    triple_conf: float,
) -> dict[str, Any]:
    """Public wrapper exposing graph metadata construction for other tiers."""

    return _build_graph_metadata(
        entities,
        relation_type_guess=relation_type_guess,
        matches=matches,
        evidence_text=evidence_text,
        section_id=section_id,
        triple_conf=triple_conf,
    )


def _normalize_key_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return _normalize_whitespace(value).lower()


def _strip_outer_quotes(value: Optional[str]) -> str:
    if not value:
        return ""
    stripped = value.strip()
    while len(stripped) >= 2 and stripped[0] in QUOTE_CHARS and stripped[-1] in QUOTE_CHARS:
        stripped = stripped[1:-1].strip()
    return stripped


def _is_low_info_text(value: str) -> bool:
    normalized = _normalize_key_text(value)
    if not normalized:
        return True
    if normalized in LOW_INFO_TERMS:
        return True
    if normalized in LOW_INFO_BASE_TERMS:
        return True
    for prefix in LOW_INFO_PREFIXES:
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix):].strip()
            if not remainder or remainder in LOW_INFO_BASE_TERMS:
                return True
    return False


def _find_context_by_id(
    contexts: Sequence[SectionContext],
    section_id: Optional[str],
    *,
    chunk_id: Optional[str] = None,
) -> Optional[SectionContext]:
    if section_id:
        for context in contexts:
            if context.section_id == section_id and (
                chunk_id is None or context.chunk_id == chunk_id
            ):
                return context
    if chunk_id:
        for context in contexts:
            if context.chunk_id == chunk_id:
                return context
    return None


def _extract_antecedent_from_sentence(sentence: str) -> Optional[str]:
    if not sentence:
        return None
    for quoted in reversed(ANTECEDENT_QUOTE_RE.findall(sentence)):
        candidate = _normalize_whitespace(quoted)
        if candidate and not _is_low_info_text(candidate):
            return candidate
    for match in reversed(ANTECEDENT_CAPITAL_RE.findall(sentence)):
        candidate = _normalize_whitespace(match)
        if candidate and not _is_low_info_text(candidate):
            return candidate
    tokens = [token for token in re.findall(r"[A-Za-z0-9\-]+", sentence) if token]
    if tokens:
        tail = _normalize_whitespace(" ".join(tokens[-3:]))
        if tail and not _is_low_info_text(tail):
            return tail
    return None


def _find_antecedent(
    matches: Sequence[dict[str, Any]],
    contexts: Sequence[SectionContext],
) -> Optional[str]:
    for match in matches:
        section_id = match.get("section_id")
        context = _find_context_by_id(
            contexts,
            section_id,
            chunk_id=match.get("chunk_id"),
        )
        if context is None:
            continue
        sentence_index = match.get("sentence_index")
        if sentence_index is None:
            continue
        looked = 0
        for idx, text in reversed(context.sentences):
            if idx > sentence_index:
                continue
            if idx == sentence_index:
                continue
            candidate = _extract_antecedent_from_sentence(text)
            if candidate:
                return candidate
            looked += 1
            if looked >= MAX_ANTECEDENT_LOOKBACK:
                break
    return None


def _resolve_low_info_term(
    value: str,
    matches: Sequence[dict[str, Any]],
    contexts: Sequence[SectionContext],
) -> tuple[str, bool, bool]:
    cleaned = _normalize_whitespace(_strip_outer_quotes(value))
    if not cleaned:
        return "", False, False
    if not _is_low_info_text(cleaned):
        return cleaned, False, True
    antecedent = _find_antecedent(matches, contexts)
    if antecedent:
        return antecedent, True, True
    return cleaned, False, False


def _sanitize_span(
    span: Optional[Sequence[int]],
    evidence: str,
    fallback_text: str,
    *,
    match_length: Optional[int] = None,
    stats: Optional[GuardrailStats] = None,
) -> list[int]:
    evidence_text = evidence or ""
    evidence_length = len(evidence_text)
    adjusted = False
    fallback_used = False

    start = 0
    end = 0
    original_start: Optional[int] = None
    original_end: Optional[int] = None

    if span and len(span) >= 2:
        try:
            start = int(span[0])
            end = int(span[1])
            original_start = start
            original_end = end
        except (TypeError, ValueError):
            start = 0
            end = 0

    if end < start:
        start, end = end, start
        adjusted = True

    if start < 0:
        start = 0
        adjusted = True
    if end < 0:
        end = 0
        adjusted = True

    if start > evidence_length:
        start = evidence_length
        adjusted = True
    if end > evidence_length:
        end = evidence_length
        adjusted = True

    if end <= start:
        fallback = _normalize_whitespace(fallback_text)
        if fallback:
            lowered_evidence = evidence_text.lower()
            lowered_fallback = fallback.lower()
            position = lowered_evidence.find(lowered_fallback)
            if position != -1:
                start = position
                end = min(evidence_length, position + len(fallback))
            else:
                end = min(evidence_length, start + len(fallback))
        else:
            end = min(evidence_length, start + 1)
        fallback_used = True
        adjusted = True

    if match_length is not None and match_length >= 0:
        if start > match_length:
            start = max(0, match_length - 1)
            adjusted = True
        if end > match_length:
            end = match_length
            adjusted = True

    if stats:
        if adjusted and (original_start is not None or original_end is not None):
            stats.span_clamped += 1
        if fallback_used:
            stats.span_fallback += 1

    return [start, end]


def _make_dedupe_key(
    subject: str,
    relation: str,
    object_text: str,
    evidence: str,
    section_id: Optional[str],
    chunk_id: Optional[str],
) -> str:
    parts = (
        _normalize_key_text(subject),
        _normalize_key_text(relation),
        _normalize_key_text(object_text),
        _normalize_key_text(evidence),
        (section_id or "").lower(),
        (chunk_id or "").lower(),
    )
    return "|".join(parts)


def make_triple_dedupe_key(
    subject: str,
    relation: str,
    object_text: str,
    evidence: str,
    *,
    section_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> str:
    """Expose the triple dedupe key builder for downstream relation tiers."""

    return _make_dedupe_key(subject, relation, object_text, evidence, section_id, chunk_id)


class Tier2ValidationError(RuntimeError):
    """Raised when Tier-2 returns an invalid payload or cannot execute."""



    def formatted_text(self) -> str:
        lines: list[str] = []
        for idx, text in self.sentences:
            lines.append(f"[{idx}] {text}")
        for caption in self.captions:
            lines.append(f"[caption] {caption}")
        return "\n".join(lines)



async def run_tier2_structurer(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if base_summary is None:
        raise Tier2ValidationError("Tier-2 requires Tier-1 summary data")

    sections_raw = base_summary.get("sections")
    if not sections_raw:
        raise Tier2ValidationError("Tier-2 requires Tier-1 sections with sentence spans")

    contexts = _prepare_section_contexts(sections_raw)
    if not contexts:
        logger.info("[tier2] no suitable sections found for LLM extraction")
        return _augment_summary(
            base_summary,
            [],
            TripleExtractionResponse(),
            raw_responses=None,
            stats=GuardrailStats(),
        )

    guard_stats = GuardrailStats()
    seen_keys: set[str] = set()
    all_candidates: list[dict[str, Any]] = []
    all_persistence: list[TripleCandidateRecord] = []
    raw_responses: dict[str, Optional[str]] = {}
    payloads: list[TripleExtractionResponse] = []

    extraction_passes = [
        ("primary", _build_messages(contexts, mode="primary"), True, "primary extraction"),
        ("comparison", _build_messages(contexts, mode="comparison"), False, "comparison extraction"),
    ]

    for pass_label, messages, required, failure_reason in extraction_passes:
        try:
            validated, raw_json = await _request_llm_payload(
                paper_id,
                messages,
                required=required,
                failure_reason=failure_reason,
            )
        except Tier2ValidationError as exc:
            return _mark_needs_review(base_summary, contexts, exc)

        if not validated:
            continue

        payloads.append(validated)
        raw_responses[pass_label] = raw_json

        candidate_entries, persistence_payload, guard_stats, seen_keys = _build_candidates(
            validated,
            contexts,
            paper_id,
            stats=guard_stats,
            seen_keys=seen_keys,
            pass_label=pass_label,
        )
        all_candidates.extend(candidate_entries)
        all_persistence.extend(persistence_payload)

    merged_payload = (
        _merge_payloads(payloads) if payloads else TripleExtractionResponse()
    )

    summary = _augment_summary(
        base_summary,
        all_candidates,
        merged_payload,
        raw_responses=raw_responses if raw_responses else None,
        stats=guard_stats,
    )

    try:
        await replace_triple_candidates(paper_id, all_persistence)
    except Exception as exc:  # pragma: no cover - persistence failure should be surfaced but not fatal
        logger.error(
            "[tier2] failed to persist triple candidates for paper %s: %s",
            paper_id,
            exc,
        )
        metadata = summary.setdefault("metadata", {})
        tier2_meta = metadata.setdefault("tier2", {})
        tier2_meta["persistence_error"] = str(exc)

    return summary


def _mark_needs_review(
    base_summary: dict[str, Any],
    contexts: Sequence[SectionContext],
    error: Exception,
) -> dict[str, Any]:
    summary = copy.deepcopy(base_summary)
    metadata = summary.setdefault("metadata", {})
    metadata["tier2"] = {
        "tier": TIER_NAME,
        "status": "needs_review",
        "error": str(error),
    }
    entry = {
        "tier": TIER_NAME,
        "reason": str(error),
        "section_ids": [context.section_id for context in contexts],
    }
    summary.setdefault("needs_review", []).append(entry)
    return summary




def _merge_payloads(payloads: Sequence[TripleExtractionResponse]) -> TripleExtractionResponse:
    merged_triples: list[TriplePayload] = []
    warnings: list[str] = []
    discarded: list[str] = []
    seen_warnings: set[str] = set()
    seen_discarded: set[str] = set()

    for payload in payloads:
        merged_triples.extend(payload.triples)

        for warning in payload.warnings or []:
            normalized = warning.strip()
            key = normalized.lower() if normalized else warning
            if key in seen_warnings:
                continue
            seen_warnings.add(key)
            warnings.append(warning)

        for item in payload.discarded or []:
            normalized = item.strip()
            key = normalized.lower() if normalized else item
            if key in seen_discarded:
                continue
            seen_discarded.add(key)
            discarded.append(item)

    return TripleExtractionResponse(
        triples=list(merged_triples),
        warnings=warnings,
        discarded=discarded,
    )


async def _request_llm_payload(
    paper_id: UUID,
    messages: Sequence[dict[str, str]],
    *,
    required: bool,
    failure_reason: str,
) -> tuple[Optional[TripleExtractionResponse], Optional[str]]:
    attempts = 0
    last_error: Optional[Exception] = None
    pending_messages: Sequence[dict[str, str]] = list(messages)
    temperature_override: Optional[float] = None
    repair_context: Optional[RepairContext] = None

    while attempts < MAX_LLM_ATTEMPTS:
        attempts += 1
        parsed_payload: Optional[dict[str, Any]] = None
        try:
            raw_content = await _invoke_llm(
                pending_messages, temperature=temperature_override
            )
            candidate_payload = _parse_json(raw_content)
            if repair_context:
                candidate_payload = _apply_repair_patch(repair_context, candidate_payload)
                repair_context = None
                pending_messages = list(messages)
                temperature_override = None

            parsed_payload = candidate_payload
            validated = validate_triples(candidate_payload)
            final_json = json.dumps(candidate_payload, ensure_ascii=False)
            return validated, final_json
        except TripleSchemaError as exc:
            last_error = exc
            logger.warning(
                "[tier2] %s attempt %s schema repair needed for paper %s: %s",
                failure_reason,
                attempts,
                paper_id,
                exc,
            )
            if attempts >= MAX_LLM_ATTEMPTS:
                break
            if parsed_payload is None:
                raise
            try:
                pending_messages, repair_context, temperature_override = _build_repair_messages(
                    parsed_payload, exc.issues
                )
            except Tier2ValidationError as repair_exc:
                last_error = repair_exc
                break
            continue
        except Tier2ValidationError as exc:
            last_error = exc
            logger.warning(
                "[tier2] %s attempt %s failed for paper %s: %s",
                failure_reason,
                attempts,
                paper_id,
                exc,
            )
            if attempts >= MAX_LLM_ATTEMPTS:
                break
            pending_messages = list(messages)
            temperature_override = None
            repair_context = None

    if required:
        raise last_error or Tier2ValidationError(
            "Tier-2 validation failed without exception"
        )
    return None, None


def _prepare_section_contexts(sections: Sequence[dict[str, Any]]) -> list[SectionContext]:
    chunk_limit = max(1, settings.tier2_llm_max_sections or 1)
    raw_budget = (
        settings.tier2_llm_section_chunk_chars
        or settings.tier2_llm_max_section_chars
        or 2000
    )
    chunk_budget = max(120, raw_budget)
    chunk_overlap = max(0, settings.tier2_llm_section_chunk_overlap_sentences or 0)
    max_chunks_per_section = max(
        1, settings.tier2_llm_max_chunks_per_section or 1
    )

    def _section_sort_key(section: dict[str, Any]) -> tuple[int, int]:
        page = section.get("page_number")
        char_start = section.get("char_start")
        page_rank = int(page) if isinstance(page, int) else 1_000_000
        char_rank = int(char_start) if isinstance(char_start, int) else 1_000_000
        return page_rank, char_rank

    ordered = sorted(sections, key=_section_sort_key)
    contexts: list[SectionContext] = []

    for section in ordered:
        if len(contexts) >= chunk_limit:
            break
        section_id = str(section.get("section_id") or "").strip()
        if not section_id:
            continue
        section_hash = str(section.get("section_hash") or "")
        title = section.get("title")
        page_number = section.get("page_number") if isinstance(section.get("page_number"), int) else None
        sentences_payload = section.get("sentence_spans") or []
        if not isinstance(sentences_payload, list):
            continue
        sentences: list[tuple[int, str]] = []
        for sentence_index, sentence in enumerate(sentences_payload):
            if not isinstance(sentence, dict):
                continue
            raw_text = sentence.get("text")
            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text:
                continue
            sentences.append((sentence_index, text))
        if not sentences:
            continue
        captions_raw = section.get("captions") or []
        captions = [
            str(caption.get("text") or "").strip()
            for caption in captions_raw
            if isinstance(caption, dict) and caption.get("text")
        ]
        start_index = 0
        chunk_index = 0
        total_sentences = len(sentences)
        while (
            start_index < total_sentences
            and chunk_index < max_chunks_per_section
            and len(contexts) < chunk_limit
        ):
            idx = start_index
            chunk_sentences: list[tuple[int, str]] = []
            running_chars = 0
            while idx < total_sentences:
                sentence_entry = sentences[idx]
                formatted = f"[{sentence_entry[0]}] {sentence_entry[1]}"
                addition = len(formatted) + 1
                if chunk_sentences and running_chars + addition > chunk_budget:
                    break
                chunk_sentences.append(sentence_entry)
                running_chars += addition
                idx += 1
                if running_chars >= chunk_budget:
                    break

            if not chunk_sentences:
                break

            chunk_index += 1
            chunk_id = f"{section_id}#chunk-{chunk_index:02d}"
            contexts.append(
                SectionContext(
                    section_id=section_id,
                    section_hash=section_hash,
                    title=title,
                    page_number=page_number,
                    sentences=chunk_sentences,
                    captions=captions,
                    chunk_id=chunk_id,
                )
            )

            if len(contexts) >= chunk_limit or idx >= total_sentences:
                break

            next_start = max(idx - chunk_overlap, 0)
            if next_start <= start_index:
                next_start = idx
            start_index = next_start

    return contexts


def _build_messages(contexts: Sequence[SectionContext], *, mode: str = "primary") -> list[dict[str, str]]:
    type_vocab = ", ".join(TYPE_GUESS_VALUES)
    relation_vocab = ", ".join(RELATION_GUESS_VALUES)
    guidance = (
        "Analyse the provided paper snippets. Extract factual triples and only emit entries "
        "that are explicitly supported."
    )
    evidence_hint = (
        "For each triple, provide subject/object spans (character offsets) within the evidence sentence "
        "and include section_id if known."
    )
    output_hint = (
        "Type guesses must use the provided vocabularies. If none apply, use 'Unknown'."
    )

    content_lines = [
        guidance,
        evidence_hint,
        output_hint,
        f"Subject/Object type guesses: {type_vocab}.",
        f"Relation type guesses: {relation_vocab}.",
        "Return JSON only, matching the provided schema.",
        "",
    ]
    if mode == "comparison":
        content_lines.insert(0, COMPARISON_GUIDANCE)

    for idx, context in enumerate(contexts, start=1):
        header_parts = [f"Section {idx}", f"section_id={context.section_id}"]
        if context.title:
            header_parts.append(f"title={context.title}")
        if context.page_number is not None:
            header_parts.append(f"page={context.page_number}")
        if context.section_hash:
            header_parts.append(f"hash={context.section_hash}")
        if context.chunk_id:
            header_parts.append(f"chunk={context.chunk_id}")
        content_lines.append(" | ".join(header_parts))
        content_lines.append("---")
        content_lines.append(context.formatted_text())
        content_lines.append("")

    user_message = "\n".join(line for line in content_lines if line is not None)
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_message},
    ]




def _build_candidates(
    payload: TripleExtractionResponse,
    contexts: Sequence[SectionContext],
    paper_id: UUID,
    *,
    stats: Optional[GuardrailStats] = None,
    seen_keys: Optional[set[str]] = None,
    pass_label: str = "primary",
) -> tuple[list[dict[str, Any]], list[TripleCandidateRecord], GuardrailStats, set[str]]:
    stats = stats or GuardrailStats()
    candidates: list[dict[str, Any]] = []
    persistence_payload: list[TripleCandidateRecord] = []
    seen_keys = set(seen_keys or set())

    for index, triple in enumerate(payload.triples, start=1):
        evidence_text = (triple.evidence or "").strip()
        if not evidence_text:
            logger.debug(
                "[tier2] skipping triple %s for paper %s due to empty evidence",
                index,
                paper_id,
            )
            continue

        matches = _match_evidence(evidence_text, contexts, stats=stats)
        if not matches:
            stats.unmatched_evidence += 1
        candidate_section_id = triple.section_id or (matches[0]["section_id"] if matches else None)
        candidate_chunk_id = triple.chunk_id or (matches[0].get("chunk_id") if matches else None)
        primary_match_length = (
            matches[0]["end"] - matches[0]["start"] if matches else None
        )

        subject_text, subject_resolved, subject_valid = _resolve_low_info_term(
            triple.subject,
            matches,
            contexts,
        )
        if subject_resolved:
            stats.pronoun_resolved_subjects += 1
        elif not subject_valid:
            stats.low_info_subject_dropped += 1
            logger.debug(
                "[tier2] dropping triple %s for paper %s due to unresolved subject '%s'",
                index,
                paper_id,
                triple.subject,
            )
            continue

        object_text, object_resolved, object_valid = _resolve_low_info_term(
            triple.object,
            matches,
            contexts,
        )
        if object_resolved:
            stats.pronoun_resolved_objects += 1
        elif not object_valid:
            stats.low_info_object_dropped += 1
            logger.debug(
                "[tier2] dropping triple %s for paper %s due to unresolved object '%s'",
                index,
                paper_id,
                triple.object,
            )
            continue

        relation_text = _normalize_whitespace(triple.relation)
        if not relation_text:
            logger.debug(
                "[tier2] dropping triple %s for paper %s due to empty relation",
                index,
                paper_id,
            )
            continue

        subject_span = _sanitize_span(
            triple.subject_span,
            evidence_text,
            subject_text,
            match_length=primary_match_length,
            stats=stats,
        )
        object_span = _sanitize_span(
            triple.object_span,
            evidence_text,
            object_text,
            match_length=primary_match_length,
            stats=stats,
        )

        dedupe_key = _make_dedupe_key(
            subject_text,
            relation_text,
            object_text,
            evidence_text,
            candidate_section_id,
            candidate_chunk_id,
        )
        if dedupe_key in seen_keys:
            stats.deduplicated_triples += 1
            logger.debug(
                "[tier2] deduped triple %s for paper %s (subject=%s, relation=%s, object=%s)",
                index,
                paper_id,
                subject_text,
                relation_text,
                object_text,
            )
            continue
        seen_keys.add(dedupe_key)

        triple_conf = _coerce_float(triple.triple_conf, DEFAULT_TRIPLE_CONFIDENCE)
        schema_score = _coerce_float(triple.schema_match_score, DEFAULT_SCHEMA_SCORE)

        metric_inference = infer_metric_from_text(object_text, evidence_text)
        if metric_inference:
            stats.metrics_inferred += 1
            triple_conf = round(max(0.0, triple_conf - METRIC_INFERENCE_CONF_PENALTY), 4)

        measurement = extract_measurement(
            f"{object_text} {evidence_text}",
            metric_hint=metric_inference.get("normalized_metric") if metric_inference else None,
        )
        if measurement:
            stats.measurements_detected += 1

        candidate: dict[str, Any] = {
            "candidate_id": f"{TIER_NAME}_{index:03d}",
            "tier": TIER_NAME,
            "subject": subject_text,
            "relation": relation_text,
            "object": object_text,
            "subject_span": subject_span,
            "object_span": object_span,
            "subject_type_guess": triple.subject_type_guess,
            "relation_type_guess": triple.relation_type_guess,
            "object_type_guess": triple.object_type_guess,
            "triple_conf": triple_conf,
            "schema_match_score": schema_score,
            "evidence": evidence_text,
            "evidence_spans": matches,
        }
        if metric_inference:
            candidate["metric_inference"] = {**metric_inference, "confidence_penalty": METRIC_INFERENCE_CONF_PENALTY}

        if measurement:
            candidate["measurement"] = measurement

        if candidate_section_id:
            candidate["section_id"] = candidate_section_id
        if candidate_chunk_id:
            candidate["chunk_id"] = candidate_chunk_id
        if pass_label:
            candidate["pass"] = pass_label

        graph_entities = _extract_graph_entities(
            subject_text,
            triple.subject_type_guess,
            object_text,
            triple.object_type_guess,
        )
        graph_metadata = _build_graph_metadata(
            graph_entities,
            relation_type_guess=triple.relation_type_guess,
            matches=matches,
            evidence_text=evidence_text,
            section_id=candidate_section_id,
            triple_conf=triple_conf,
            metric_inference=metric_inference,
            measurement=measurement,
        )
        if not graph_metadata and (metric_inference or measurement):
            graph_metadata = {}
            if metric_inference:
                graph_metadata["metric_inference"] = dict(metric_inference)
            if measurement:
                graph_metadata["measurement"] = dict(measurement)
        if graph_metadata:
            candidate["graph_metadata"] = graph_metadata

        candidates.append(candidate)

        record_metadata: dict[str, Any] = {}
        if isinstance(graph_metadata, Mapping):
            record_metadata = dict(graph_metadata)

        persistence_payload.append(
            TripleCandidateRecord(
                paper_id=paper_id,
                section_id=candidate_section_id,
                subject=str(subject_text),
                relation=str(relation_text),
                object=str(object_text),
                subject_span=subject_span,
                object_span=object_span,
                subject_type_guess=triple.subject_type_guess,
                relation_type_guess=triple.relation_type_guess,
                object_type_guess=triple.object_type_guess,
                evidence=str(evidence_text),
                triple_conf=triple_conf,
                schema_match_score=schema_score,
                tier=TIER_NAME,
                graph_metadata=record_metadata,
            )
        )

    return candidates, persistence_payload, stats, seen_keys



def _match_evidence(
    evidence: str,
    contexts: Sequence[SectionContext],
    *,
    stats: Optional[GuardrailStats] = None,
) -> list[dict[str, Any]]:
    cleaned = _normalize_whitespace(evidence)
    if not cleaned:
        return []
    lowered_clean = cleaned.lower()
    matches: list[dict[str, Any]] = []

    for context in contexts:
        for sentence_index, sentence in context.sentences:
            lowered_sentence = sentence.lower()
            start = lowered_sentence.find(lowered_clean)
            end: Optional[int] = None
            fuzzy_used = False

            if start == -1:
                matcher = difflib.SequenceMatcher(None, lowered_sentence, lowered_clean)
                block = matcher.find_longest_match(0, len(lowered_sentence), 0, len(lowered_clean))
                ratio = matcher.ratio()
                if block.size >= max(5, int(len(lowered_clean) * 0.6)) and ratio >= 0.75:
                    start = block.a
                    end = start + block.size
                    fuzzy_used = True
                else:
                    continue

            if end is None:
                end = start + len(cleaned)

            matches.append(
                {
                    "section_id": context.section_id,
                    "chunk_id": context.chunk_id,
                    "sentence_index": sentence_index,
                    "start": start,
                    "end": end,
                }
            )
            if fuzzy_used and stats:
                stats.fuzzy_matches += 1
    return matches



def _augment_summary(
    base_summary: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    payload: TripleExtractionResponse,
    *,
    raw_responses: Optional[dict[str, Optional[str]]] = None,
    stats: Optional[GuardrailStats] = None,
) -> dict[str, Any]:
    summary = copy.deepcopy(base_summary)
    tiers = set(summary.get("tiers") or [])
    tiers.add(1)
    tiers.add(2)
    summary["tiers"] = sorted(tiers)

    summary["triple_candidates"] = list(candidates)

    metadata = summary.setdefault("metadata", {})
    tier2_meta = {
        "tier": TIER_NAME,
        "triple_count": len(candidates),
    }
    if payload.warnings:
        tier2_meta["warnings"] = list(payload.warnings)
    if payload.discarded:
        tier2_meta["discarded"] = list(payload.discarded)
    if raw_responses:
        tier2_meta["raw_responses"] = {key: value for key, value in raw_responses.items() if value is not None}
    if stats is not None:
        tier2_meta["guardrails"] = stats.to_metadata()
    metadata["tier2"] = tier2_meta

    return summary


def _coerce_float(candidate: Optional[float], default: float) -> float:
    try:
        if candidate is None:
            raise ValueError
        value = float(candidate)
    except (TypeError, ValueError):
        value = default
    return round(max(0.0, min(1.0, value)), 4)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Tier2ValidationError(f"Tier-2 LLM returned invalid JSON: {exc}") from exc


def _normalize_span_for_validation(span: Any) -> list[int]:
    if not isinstance(span, Sequence):
        return []

    normalized: list[int] = []
    for value in span:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
        if len(normalized) >= 2:
            break

    if len(normalized) == 1:
        normalized.append(normalized[0])

    if len(normalized) >= 2:
        return normalized[:2]

    return []


def _prepare_payload_for_validation(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(payload)
    triples = prepared.get("triples")
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            for key in ("subject_span", "object_span"):
                span = triple.get(key)
                normalized = _normalize_span_for_validation(span)
                triple[key] = normalized if normalized else [0, 0]
    return prepared


@dataclass
class TripleSchemaIssue:
    index: Optional[int]
    field: Optional[str]
    message: str
    validator: Optional[str] = None
    constraint: Optional[Any] = None


class TripleSchemaError(Tier2ValidationError):
    """Raised when triple JSON schema validation fails for specific entries."""

    def __init__(self, issues: Sequence[TripleSchemaIssue]):
        self.issues = list(issues)
        summary = "; ".join(
            f"index={issue.index}, field={issue.field}, error={issue.message}"
            for issue in self.issues
        )
        super().__init__(
            "Tier-2 payload failed triple JSON schema validation: " + summary
        )


@dataclass
class RepairContext:
    base_payload: dict[str, Any]
    target_indexes: list[int]
    issues: list[TripleSchemaIssue]


def _jsonschema_validator() -> Draft7Validator:
    if Draft7Validator is None:  # pragma: no cover - optional dependency guard
        raise Tier2ValidationError(
            "jsonschema is required for Tier-2 validation but is not installed"
        )
    schema = get_triple_json_schema()
    return Draft7Validator(schema)


def _coerce_triple_text(value: Any) -> str:
    return _normalize_whitespace(str(value or ""))


def _summarize_jsonschema_errors(
    errors: Sequence[JSONSchemaValidationError],
) -> list[TripleSchemaIssue]:
    issues: list[TripleSchemaIssue] = []
    for error in errors:
        index: Optional[int] = None
        field: Optional[str] = None
        path = list(error.absolute_path)
        if path and path[0] == "triples":
            if len(path) >= 2 and isinstance(path[1], int):
                index = int(path[1])
            if len(path) >= 3:
                field = str(path[2])
        issues.append(
            TripleSchemaIssue(
                index=index,
                field=field,
                message=error.message,
                validator=getattr(error, "validator", None),
                constraint=getattr(error, "validator_value", None),
            )
        )
    return issues


def validate_triples(payload: dict[str, Any]) -> TripleExtractionResponse:
    """Validate a raw LLM payload against the triple schema and sanitize text fields."""

    validator = _jsonschema_validator()
    schema_errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if schema_errors:
        issues = _summarize_jsonschema_errors(schema_errors)
        if any(issue.index is None for issue in issues):
            raise Tier2ValidationError(
                "Tier-2 payload failed top-level schema validation: "
                + "; ".join(issue.message for issue in issues)
            )
        raise TripleSchemaError(issues)

    try:
        prepared = _prepare_payload_for_validation(payload)
        response = TripleExtractionResponse.model_validate(prepared)
    except ValidationError as exc:
        raise Tier2ValidationError(f"Tier-2 payload failed schema validation: {exc}") from exc

    sanitized_triples: list[TriplePayload] = []
    for triple in response.triples:
        subject = _strip_citation_fragments(_coerce_triple_text(triple.subject))
        relation = _strip_citation_fragments(_coerce_triple_text(triple.relation))
        obj = _strip_citation_fragments(_coerce_triple_text(triple.object))
        evidence = _strip_citation_fragments(_coerce_triple_text(triple.evidence))
        sanitized_triples.append(
            TriplePayload.model_validate(
                {
                    **triple.model_dump(),
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "evidence": evidence,
                }
            )
        )

    return TripleExtractionResponse(
        triples=sanitized_triples,
        warnings=list(response.warnings or []),
        discarded=list(response.discarded or []),
    )


def _describe_field_constraints(field: str) -> str:
    schema = get_triple_json_schema()
    triple_schema = schema.get("properties", {}).get("triples", {}).get("items", {})
    properties = triple_schema.get("properties", {})
    field_schema = properties.get(field, {})
    parts: list[str] = []
    field_type = field_schema.get("type")
    if field_type:
        parts.append(f"type={field_type}")
    if "minLength" in field_schema:
        parts.append(f"minLength={field_schema['minLength']}")
    if "maxLength" in field_schema:
        parts.append(f"maxLength={field_schema['maxLength']}")
    if "enum" in field_schema:
        parts.append("enum=" + ",".join(map(str, field_schema["enum"])))
    if not parts:
        return "must satisfy schema requirements"
    return ", ".join(parts)


def _build_repair_messages(
    payload: dict[str, Any],
    issues: Sequence[TripleSchemaIssue],
) -> tuple[list[dict[str, str]], RepairContext, float]:
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise Tier2ValidationError("Cannot repair payload without a triples array")

    target_indexes = sorted(
        {issue.index for issue in issues if issue.index is not None}
    )
    if not target_indexes:
        raise Tier2ValidationError("No targeted triples available for repair")

    lines: list[str] = []
    lines.append(
        "The prior triple extraction JSON had schema violations. Fix ONLY the listed "
        "triples so they satisfy the schema."
    )
    lines.append(
        "Return strict JSON with a 'triples' array containing corrected entries in the "
        "same order as provided here."
    )
    lines.append(
        "Do not include any other keys. Each corrected triple must satisfy the schema "
        "constraints noted for its invalid fields."
    )
    lines.append("")

    for index in target_indexes:
        original = triples[index] if index < len(triples) else None
        lines.append(f"Triple index {index}:")
        if isinstance(original, Mapping):
            lines.append(json.dumps(original, indent=2, ensure_ascii=False))
        else:
            lines.append(f"(missing or invalid entry: {original!r})")

        for issue in issues:
            if issue.index != index:
                continue
            field_desc = f"Field '{issue.field}'" if issue.field else "Entry"
            constraint_desc = (
                _describe_field_constraints(issue.field)
                if issue.field
                else "must be a valid triple object"
            )
            lines.append(
                f"- {field_desc} violates: {issue.message}. Expected: {constraint_desc}."
            )
        lines.append("")

    user_message = "\n".join(lines).strip()
    repair_messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_message},
    ]

    repair_temperature = max(0.0, settings.tier2_llm_temperature / 2.0)
    context = RepairContext(
        base_payload=copy.deepcopy(payload),
        target_indexes=target_indexes,
        issues=list(issues),
    )

    return repair_messages, context, repair_temperature


def _apply_repair_patch(context: RepairContext, patch: dict[str, Any]) -> dict[str, Any]:
    triples_patch = patch.get("triples")
    if not isinstance(triples_patch, list):
        raise Tier2ValidationError(
            "Repair payload must include a 'triples' array with corrected entries"
        )

    if len(triples_patch) != len(context.target_indexes):
        raise Tier2ValidationError(
            "Repair payload triple count does not match requested fixes"
        )

    updated = copy.deepcopy(context.base_payload)
    triples = updated.get("triples")
    if not isinstance(triples, list):
        raise Tier2ValidationError("Base payload missing triples array during repair")

    for target_index, replacement in zip(context.target_indexes, triples_patch):
        if target_index >= len(triples):
            raise Tier2ValidationError(
                f"Repair target index {target_index} is out of range"
            )
        triples[target_index] = replacement

    return updated

    return TripleExtractionResponse(
        triples=sanitized_triples,
        warnings=list(response.warnings or []),
        discarded=list(response.discarded or []),
    )


def _describe_field_constraints(field: str) -> str:
    schema = get_triple_json_schema()
    triple_schema = schema.get("properties", {}).get("triples", {}).get("items", {})
    properties = triple_schema.get("properties", {})
    field_schema = properties.get(field, {})
    parts: list[str] = []
    field_type = field_schema.get("type")
    if field_type:
        parts.append(f"type={field_type}")
    if "minLength" in field_schema:
        parts.append(f"minLength={field_schema['minLength']}")
    if "maxLength" in field_schema:
        parts.append(f"maxLength={field_schema['maxLength']}")
    if "enum" in field_schema:
        parts.append("enum=" + ",".join(map(str, field_schema["enum"])))
    if not parts:
        return "must satisfy schema requirements"
    return ", ".join(parts)


def _build_repair_messages(
    payload: dict[str, Any],
    issues: Sequence[TripleSchemaIssue],
) -> tuple[list[dict[str, str]], RepairContext, float]:
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise Tier2ValidationError("Cannot repair payload without a triples array")

    target_indexes = sorted(
        {issue.index for issue in issues if issue.index is not None}
    )
    if not target_indexes:
        raise Tier2ValidationError("No targeted triples available for repair")

    lines: list[str] = []
    lines.append(
        "The prior triple extraction JSON had schema violations. Fix ONLY the listed "
        "triples so they satisfy the schema."
    )
    lines.append(
        "Return strict JSON with a 'triples' array containing corrected entries in the "
        "same order as provided here."
    )
    lines.append(
        "Do not include any other keys. Each corrected triple must satisfy the schema "
        "constraints noted for its invalid fields."
    )
    lines.append("")

    for index in target_indexes:
        original = triples[index] if index < len(triples) else None
        lines.append(f"Triple index {index}:")
        if isinstance(original, Mapping):
            lines.append(json.dumps(original, indent=2, ensure_ascii=False))
        else:
            lines.append(f"(missing or invalid entry: {original!r})")

        for issue in issues:
            if issue.index != index:
                continue
            field_desc = f"Field '{issue.field}'" if issue.field else "Entry"
            constraint_desc = (
                _describe_field_constraints(issue.field)
                if issue.field
                else "must be a valid triple object"
            )
            lines.append(
                f"- {field_desc} violates: {issue.message}. Expected: {constraint_desc}."
            )
        lines.append("")

    user_message = "\n".join(lines).strip()
    repair_messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_message},
    ]

    repair_temperature = max(0.0, settings.tier2_llm_temperature / 2.0)
    context = RepairContext(
        base_payload=copy.deepcopy(payload),
        target_indexes=target_indexes,
        issues=list(issues),
    )

    return repair_messages, context, repair_temperature


def _apply_repair_patch(context: RepairContext, patch: dict[str, Any]) -> dict[str, Any]:
    triples_patch = patch.get("triples")
    if not isinstance(triples_patch, list):
        raise Tier2ValidationError(
            "Repair payload must include a 'triples' array with corrected entries"
        )

    if len(triples_patch) != len(context.target_indexes):
        raise Tier2ValidationError(
            "Repair payload triple count does not match requested fixes"
        )

    updated = copy.deepcopy(context.base_payload)
    triples = updated.get("triples")
    if not isinstance(triples, list):
        raise Tier2ValidationError("Base payload missing triples array during repair")

    for target_index, replacement in zip(context.target_indexes, triples_patch):
        if target_index >= len(triples):
            raise Tier2ValidationError(
                f"Repair target index {target_index} is out of range"
            )
        triples[target_index] = replacement

    return updated


async def _invoke_llm(
    messages: Sequence[dict[str, str]], *, temperature: Optional[float] = None
) -> str:
    model = settings.tier2_llm_model
    if not model:
        raise Tier2ValidationError("Tier-2 LLM model is not configured")
    if httpx is None:  # pragma: no cover - optional dependency guard
        raise Tier2ValidationError("httpx is required for Tier-2 LLM requests")

    base_url = (settings.tier2_llm_base_url or "https://api.openai.com/v1").rstrip("/")
    path = (settings.tier2_llm_completion_path or "/chat/completions").lstrip("/")
    url = f"{base_url}/{path}" if path else base_url

    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": (
            temperature if temperature is not None else settings.tier2_llm_temperature
        ),
        "top_p": settings.tier2_llm_top_p,
        "max_tokens": settings.tier2_llm_max_output_tokens,
    }
    if settings.tier2_llm_force_json:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "triple_extraction",
                "schema": get_triple_json_schema(),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    if settings.openai_organization:
        headers["OpenAI-Organization"] = settings.openai_organization

    timeout_value = float(settings.tier2_llm_timeout_seconds or 120.0)
    timeout = httpx.Timeout(timeout_value)
    max_retries = max(1, int(getattr(settings, "tier2_llm_retry_attempts", 1) or 1))

    async with httpx.AsyncClient(timeout=timeout) as client:
        attempt = 0
        response: Optional[httpx.Response] = None
        while attempt < max_retries:
            attempt += 1
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                break
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                logger.warning(
                    "[tier2] LLM request attempt %s failed: %s",
                    attempt,
                    exc,
                )
                if attempt >= max_retries:
                    raise Tier2ValidationError(
                        f"Tier-2 LLM request failed after {max_retries} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(min(2.0, 0.5 * attempt))

        if response is None:
            raise Tier2ValidationError("Tier-2 LLM request failed without response")

    data = response.json()
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Tier2ValidationError(f"Unexpected LLM response format: {exc}") from exc

    if not isinstance(content, str) or not content.strip():
        raise Tier2ValidationError("Tier-2 LLM returned empty content")
    return content
