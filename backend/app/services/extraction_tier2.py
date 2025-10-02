from __future__ import annotations

import copy
import difflib
import json
import logging
import re
import string
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.tier2 import (
    RELATION_GUESS_VALUES,
    TYPE_GUESS_VALUES,
    TripleExtractionResponse,
    TriplePayload,
)
from app.models.triple_candidate import TripleCandidateRecord
from app.services.triple_candidates import replace_triple_candidates


logger = logging.getLogger(__name__)

TIER_NAME = "tier2_llm_openie"
SYSTEM_PROMPT = settings.tier2_llm_system_prompt
DEFAULT_TRIPLE_CONFIDENCE = 0.55
DEFAULT_SCHEMA_SCORE = 1.0
MAX_LLM_ATTEMPTS = 2
METRIC_INFERENCE_CONF_PENALTY = 0.05
COMPARISON_GUIDANCE = (
    "Extract comparisons, ablations, and causal claims. Capture statements such as 'X improves Y', 'X vs. Y', 'with vs without', or reported gains/losses."
    " Infer implied metrics when terms like misclassification rate or error rate appear."
)

METRIC_SYNONYM_MAP: dict[str, dict[str, str]] = {
    "misclassification rate": {
        "normalized_metric": "Accuracy",
        "variant": "1 - error",
        "reason": "Misclassification rate implies complement of accuracy",
    },
    "error rate": {
        "normalized_metric": "Accuracy",
        "variant": "1 - error",
        "reason": "Error rate implies complement of accuracy",
    },
    "word error rate": {
        "normalized_metric": "WER",
        "variant": "WER",
        "reason": "Word error rate corresponds to WER metric",
    },
    "false positive rate": {
        "normalized_metric": "Specificity",
        "variant": "1 - FPR",
        "reason": "False positive rate implies specificity complement",
    },
}


TRIPLE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "maxItems": 15,
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


@dataclass
class SectionContext:
    section_id: str
    section_hash: str | None
    title: str | None
    page_number: int | None
    sentences: list[tuple[int, str]]
    captions: list[str]

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
QUOTE_CHARS = '"\'“”‘’'

MAX_ANTECEDENT_LOOKBACK = 5

ANTECEDENT_QUOTE_RE = re.compile('["\'“”‘’]([^"\'“”‘’]{3,})["\'“”‘’]')

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


def _normalize_whitespace(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_graph_text(value: str) -> str:
    return _normalize_whitespace(value)


def _strip_citation_fragments(value: str) -> str:
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


def _graph_value_passes_quality(value: str) -> bool:
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
) -> dict[str, Optional[str]]:
    subject_guess = (subject_type or "").strip().lower()
    object_guess = (object_type or "").strip().lower()

    method_text: Optional[str] = None
    dataset_text: Optional[str] = None
    metric_text: Optional[str] = None
    task_text: Optional[str] = None

    if subject_guess == "method":
        method_text = subject_text
    elif object_guess == "method":
        method_text = object_text

    if object_guess == "dataset":
        dataset_text = object_text
    elif subject_guess == "dataset":
        dataset_text = subject_text

    if object_guess == "metric":
        metric_text = object_text
    elif subject_guess == "metric":
        metric_text = subject_text

    if object_guess == "task":
        task_text = object_text
    elif subject_guess == "task":
        task_text = subject_text

    return {
        "method": method_text,
        "dataset": dataset_text,
        "metric": metric_text,
        "task": task_text,
    }


GRAPH_DATASET_RELATIONS = {"EVALUATED_ON", "USES"}
GRAPH_METRIC_RELATIONS = {"MEASURES", "REPORTS"}
GRAPH_TASK_RELATIONS = {"PROPOSES"}


def _build_graph_metadata(
    entities: Mapping[str, Optional[str]],
    *,
    relation_type_guess: Optional[str],
    matches: Sequence[Mapping[str, Any]],
    evidence_text: str,
    section_id: Optional[str],
    triple_conf: float,
) -> dict[str, Any]:
    method_text = entities.get("method") or ""
    normalized_method = _normalize_graph_text(method_text)
    if not normalized_method:
        return {}

    metadata: dict[str, Any] = {
        "entities": {},
        "pairs": [],
    }

    def _register(entity_type: str, value: Optional[str]) -> Optional[dict[str, str]]:
        if not value:
            return None
        cleaned_value = _strip_citation_fragments(value)
        normalized_value = _normalize_graph_text(cleaned_value)
        if not normalized_value:
            return None
        if not _graph_value_valid_for_type(normalized_value, entity_type):
            return None
        entry = {
            "text": cleaned_value,
            "normalized": normalized_value.lower(),
        }
        metadata["entities"][entity_type] = entry
        return entry

    method_entry = _register("method", method_text)
    if method_entry is None:
        return {}

    sentence_indices = _collect_sentence_indices(matches)
    confidence = max(0.0, min(1.0, float(triple_conf))) if triple_conf is not None else None

    def _append_pair(target_type: str, raw_value: Optional[str], relation: str) -> None:
        target_entry = _register(target_type, raw_value)
        if not target_entry:
            return
        metadata["pairs"].append(
            {
                "source": {"type": "method", **method_entry},
                "target": {"type": target_type, **target_entry},
                "relation": relation,
                "confidence": confidence,
                "section_id": section_id,
                "sentence_indices": sentence_indices,
                "evidence": evidence_text,
                "source": "tier2_triple",
            }
        )

    normalized_guess = (relation_type_guess or "").strip().upper()

    if normalized_guess in GRAPH_DATASET_RELATIONS:
        _append_pair("dataset", entities.get("dataset"), "evaluates_on")
    if normalized_guess in GRAPH_METRIC_RELATIONS:
        _append_pair("metric", entities.get("metric"), "reports")
    if normalized_guess in GRAPH_TASK_RELATIONS:
        _append_pair("task", entities.get("task"), "proposes")

    if metadata["pairs"]:
        return metadata
    return {}


def _normalize_key_text(value: str) -> str:
    if not value:
        return ""
    return _normalize_whitespace(value).lower()


def _strip_outer_quotes(value: str) -> str:
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
) -> Optional[SectionContext]:
    if not section_id:
        return None
    for context in contexts:
        if context.section_id == section_id:
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
        context = _find_context_by_id(contexts, section_id)
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
            raw_response=None,
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
        raw_response=raw_responses if raw_responses else None,
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
    while attempts < MAX_LLM_ATTEMPTS:
        attempts += 1
        try:
            raw_json = await _invoke_llm(messages)
            payload_dict = _parse_json(raw_json)
            validated = _validate_payload(payload_dict)
            return validated, raw_json
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
                if required:
                    raise
                return None, None
    if required:
        raise last_error or Tier2ValidationError(
            "Tier-2 validation failed without exception"
        )
    return None, None


def _prepare_section_contexts(sections: Sequence[dict[str, Any]]) -> list[SectionContext]:
    limit = max(1, settings.tier2_llm_max_sections or 1)
    max_chars = max(400, settings.tier2_llm_max_section_chars or 2000)

    def _section_sort_key(section: dict[str, Any]) -> tuple[int, int]:
        page = section.get("page_number")
        char_start = section.get("char_start")
        page_rank = int(page) if isinstance(page, int) else 1_000_000
        char_rank = int(char_start) if isinstance(char_start, int) else 1_000_000
        return page_rank, char_rank

    ordered = sorted(sections, key=_section_sort_key)
    contexts: list[SectionContext] = []

    for section in ordered:
        if len(contexts) >= limit:
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
        running_chars = 0
        for sentence_index, sentence in enumerate(sentences_payload):
            if not isinstance(sentence, dict):
                continue
            raw_text = sentence.get("text")
            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text:
                continue
            formatted = f"[{sentence_index}] {text}"
            addition = len(formatted) + 1
            if running_chars + addition > max_chars and sentences:
                break
            sentences.append((sentence_index, text))
            running_chars += addition
        if not sentences:
            continue
        captions_raw = section.get("captions") or []
        captions = [
            str(caption.get("text") or "").strip()
            for caption in captions_raw
            if isinstance(caption, dict) and caption.get("text")
        ]
        contexts.append(
            SectionContext(
                section_id=section_id,
                section_hash=section_hash,
                title=title,
                page_number=page_number,
                sentences=sentences,
                captions=captions,
            )
        )

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
        content_lines.append(" | ".join(header_parts))
        content_lines.append("---")
        content_lines.append(context.formatted_text())
        content_lines.append("")

    user_message = "\n".join(line for line in content_lines if line is not None)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]



def _infer_metric_from_evidence(
    object_text: str,
    evidence_text: str,
) -> Optional[dict[str, str]]:
    combined = f"{object_text} {evidence_text}".lower()
    for synonym, info in METRIC_SYNONYM_MAP.items():
        if synonym in combined:
            return {
                "normalized_metric": info["normalized_metric"],
                "variant": info["variant"],
                "reason": info["reason"],
            }
    return None


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
            triple.chunk_id,
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

        metric_inference = _infer_metric_from_evidence(object_text, evidence_text)
        if metric_inference:
            stats.metrics_inferred += 1
            triple_conf = round(max(0.0, triple_conf - METRIC_INFERENCE_CONF_PENALTY), 4)

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

        if candidate_section_id:
            candidate["section_id"] = candidate_section_id
        if triple.chunk_id:
            candidate["chunk_id"] = triple.chunk_id
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
        )
        if graph_metadata:
            candidate["graph_metadata"] = graph_metadata

        candidates.append(candidate)

        persistence_payload.append(
            TripleCandidateRecord(
                paper_id=paper_id,
                section_id=candidate_section_id,
                subject=subject_text,
                relation=relation_text,
                object=object_text,
                subject_span=subject_span,
                object_span=object_span,
                subject_type_guess=triple.subject_type_guess,
                relation_type_guess=triple.relation_type_guess,
                object_type_guess=triple.object_type_guess,
                evidence=evidence_text,
                triple_conf=triple_conf,
                schema_match_score=schema_score,
                tier=TIER_NAME,
                graph_metadata=graph_metadata,
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
    raw_response: Optional[str],
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
    if raw_response is not None:
        tier2_meta["raw_response"] = raw_response
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


def _validate_payload(payload: dict[str, Any]) -> TripleExtractionResponse:
    try:
        prepared = _prepare_payload_for_validation(payload)
        return TripleExtractionResponse.model_validate(prepared)
    except ValidationError as exc:
        raise Tier2ValidationError(f"Tier-2 payload failed schema validation: {exc}") from exc


async def _invoke_llm(messages: Sequence[dict[str, str]]) -> str:
    model = settings.tier2_llm_model
    if not model:
        raise Tier2ValidationError("Tier-2 LLM model is not configured")

    base_url = (settings.tier2_llm_base_url or "https://api.openai.com/v1").rstrip("/")
    path = (settings.tier2_llm_completion_path or "/chat/completions").lstrip("/")
    url = f"{base_url}/{path}" if path else base_url

    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": settings.tier2_llm_temperature,
        "top_p": settings.tier2_llm_top_p,
        "max_tokens": settings.tier2_llm_max_output_tokens,
    }
    if settings.tier2_llm_force_json:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "triple_extraction",
                "schema": TRIPLE_JSON_SCHEMA,
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    if settings.openai_organization:
        headers["OpenAI-Organization"] = settings.openai_organization

    timeout = float(settings.tier2_llm_timeout_seconds or 120.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network failure path
        raise Tier2ValidationError(f"Tier-2 LLM request failed: {exc}") from exc

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



