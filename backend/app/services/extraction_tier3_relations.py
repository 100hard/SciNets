from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from app.models.triple_candidate import TripleCandidateRecord
from app.services.extraction_tier2 import (
    build_graph_metadata,
    extract_graph_entities,
    make_triple_dedupe_key,
)
from app.services.nlp_pipeline import process_text
from app.services.triple_candidates import replace_triple_candidates

try:  # pragma: no cover - optional dependency guard
    from spacy.matcher import DependencyMatcher  # type: ignore[import]
    from spacy.tokens import Doc, Token  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency guard
    DependencyMatcher = None  # type: ignore[assignment]
    Doc = None  # type: ignore[assignment]
    Token = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

TIER_NAME = "tier3_relations"
_RULE_CONFIDENCE = 0.72
_REPORT_CONFIDENCE = 0.68
_MIN_PHRASE_CHARS = 3

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[[^\]]+\]")
_NUMERIC_RE = re.compile(r"\d")

_METHOD_KEYWORDS = {"model", "method", "approach", "framework", "system", "architecture", "algorithm"}
_DATASET_KEYWORDS = {"dataset", "corpus", "bench", "benchmark", "set"}
_TASK_KEYWORDS = {
    "classification",
    "translation",
    "detection",
    "segmentation",
    "recognition",
    "prediction",
    "forecasting",
    "retrieval",
    "generation",
    "regression",
}
_METRIC_KEYWORDS = {
    "accuracy",
    "f1",
    "f1-score",
    "precision",
    "recall",
    "bleu",
    "rouge",
    "auc",
    "map",
    "mrr",
    "psnr",
    "dice",
    "iou",
    "wer",
    "cer",
}
_UNIT_TOKENS = {"%", "percent", "percentage", "points", "pts"}

_PRONOUNS = {"we", "our", "this", "that", "it", "they", "he", "she", "i"}

PATTERN_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "PROPOSES": {
        "rule": "PROPOSES",
        "nodes": ("VERB", "METHOD", "PREP", "TASK"),
        "pattern": [
            {"RIGHT_ID": "VERB", "RIGHT_ATTRS": {"LEMMA": {"IN": ["propose", "introduce", "present", "develop"]}}},
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "METHOD",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "attr"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "PREP",
                "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": {"IN": ["for", "to"]}},
            },
            {
                "LEFT_ID": "PREP",
                "REL_OP": ">",
                "RIGHT_ID": "TASK",
                "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
        ],
    },
    "EVALUATED_ON_SUBJ": {
        "rule": "EVALUATED_ON",
        "nodes": ("VERB", "METHOD", "PREP", "DATASET"),
        "pattern": [
            {"RIGHT_ID": "VERB", "RIGHT_ATTRS": {"LEMMA": {"IN": ["evaluate", "test", "assess", "train"]}}},
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "METHOD",
                "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
            {"LEFT_ID": "VERB", "REL_OP": ">", "RIGHT_ID": "PREP", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "on"}},
            {
                "LEFT_ID": "PREP",
                "REL_OP": ">",
                "RIGHT_ID": "DATASET",
                "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
        ],
    },
    "EVALUATED_ON_OBJ": {
        "rule": "EVALUATED_ON",
        "nodes": ("VERB", "METHOD", "PREP", "DATASET"),
        "pattern": [
            {"RIGHT_ID": "VERB", "RIGHT_ATTRS": {"LEMMA": {"IN": ["evaluate", "test", "assess", "train"]}}},
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "METHOD",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
            {"LEFT_ID": "VERB", "REL_OP": ">", "RIGHT_ID": "PREP", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "on"}},
            {
                "LEFT_ID": "PREP",
                "REL_OP": ">",
                "RIGHT_ID": "DATASET",
                "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
        ],
    },
    "REPORTS": {
        "rule": "REPORTS",
        "nodes": ("VERB", "METHOD", "VALUE"),
        "pattern": [
            {
                "RIGHT_ID": "VERB",
                "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["achieve", "obtain", "reach", "record", "report", "attain", "yield"]}
                },
            },
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "METHOD",
                "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}, "POS": {"IN": ["NOUN", "PROPN"]}},
            },
            {
                "LEFT_ID": "VERB",
                "REL_OP": ">",
                "RIGHT_ID": "VALUE",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr", "oprd"]}},
            },
        ],
    },
}

MATCHER_CACHE: Dict[str, DependencyMatcher] = {}


@dataclass
class SentenceContext:
    section_id: str
    sentence_index: int
    text: str
    start: int
    end: int


@dataclass
class CandidateData:
    subject: str
    subject_span: Tuple[int, int]
    subject_type: str
    relation_text: str
    relation_guess: str
    object: str
    object_span: Tuple[int, int]
    object_type: str
    confidence: float
    provenance: Dict[str, Any]


async def run_tier3_relations(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if base_summary is None:
        raise ValueError("Tier-3 relations require Tier-2 summary data")

    summary = copy.deepcopy(base_summary)
    tiers = set(summary.get("tiers") or [])
    tiers.update({1, 2, 3})
    summary["tiers"] = sorted(tiers)

    metadata = summary.setdefault("metadata", {})
    tier_meta: Dict[str, Any] = {"tier": TIER_NAME}
    metadata["tier3_relations"] = tier_meta

    if DependencyMatcher is None:
        tier_meta.update({"status": "skipped", "reason": "spaCy DependencyMatcher unavailable"})
        summary["triple_candidates"] = list(summary.get("triple_candidates") or [])
        return summary

    sections = summary.get("sections") or []
    contexts = _prepare_sentence_contexts(sections)
    if not contexts:
        tier_meta.update({"status": "skipped", "reason": "no section sentence spans"})
        summary["triple_candidates"] = list(summary.get("triple_candidates") or [])
        return summary

    existing_candidates = list(summary.get("triple_candidates") or [])
    combined_candidates = list(existing_candidates)
    existing_records = [_record_from_candidate_dict(paper_id, candidate) for candidate in existing_candidates]

    seen_keys: set[str] = {
        make_triple_dedupe_key(
            str(candidate.get("subject", "")),
            str(candidate.get("relation", "")),
            str(candidate.get("object", "")),
            str(candidate.get("evidence", "")),
            section_id=candidate.get("section_id"),
            chunk_id=candidate.get("chunk_id"),
        )
        for candidate in existing_candidates
    }

    stats = {"processed_sentences": len(contexts), "rule_hits": 0, "emitted": 0, "deduplicated": 0, "skipped": 0}
    rule_counts: Dict[str, int] = {}
    new_candidates: List[dict[str, Any]] = []
    new_records: List[TripleCandidateRecord] = []

    for context in contexts:
        docs = process_text(context.text)
        for cached in docs:
            matcher = _get_dependency_matcher(cached.pipeline.key, cached.pipeline.nlp)
            if matcher is None:
                continue
            matches = matcher(cached.doc)
            if not matches:
                continue
            for match_id, token_ids in matches:
                label = cached.pipeline.nlp.vocab.strings[match_id]
                pattern_def = PATTERN_DEFINITIONS.get(label)
                if not pattern_def:
                    continue
                stats["rule_hits"] += 1
                mapping = _map_pattern_tokens(pattern_def["nodes"], cached.doc, token_ids)
                if not mapping:
                    stats["skipped"] += 1
                    continue
                candidate_data = _build_candidate_from_match(pattern_def["rule"], cached.doc, mapping)
                if not candidate_data:
                    stats["skipped"] += 1
                    continue

                candidate = _candidate_to_dict(candidate_data, context)
                candidate["tier"] = TIER_NAME
                candidate["candidate_id"] = _build_candidate_id(
                    TIER_NAME, len(existing_candidates) + len(new_candidates) + 1
                )
                provenance = candidate.setdefault("provenance", {})
                provenance.setdefault("source", "tier3_dependency")
                provenance.setdefault("pattern", pattern_def["rule"].lower())
                provenance.setdefault("pipeline", cached.pipeline.key)
                provenance.setdefault("cached_doc", cached.from_cache)

                dedupe_key = make_triple_dedupe_key(
                    candidate["subject"],
                    candidate["relation"],
                    candidate["object"],
                    candidate["evidence"],
                    section_id=candidate.get("section_id"),
                    chunk_id=candidate.get("chunk_id"),
                )
                if dedupe_key in seen_keys:
                    stats["deduplicated"] += 1
                    continue
                seen_keys.add(dedupe_key)
                rule_counts[pattern_def["rule"]] = rule_counts.get(pattern_def["rule"], 0) + 1
                stats["emitted"] += 1

                graph_entities = extract_graph_entities(
                    candidate["subject"],
                    candidate["subject_type_guess"],
                    candidate["object"],
                    candidate["object_type_guess"],
                )
                graph_metadata = build_graph_metadata(
                    graph_entities,
                    relation_type_guess=candidate["relation_type_guess"],
                    matches=candidate.get("evidence_spans") or [],
                    evidence_text=candidate["evidence"],
                    section_id=candidate.get("section_id"),
                    triple_conf=candidate["triple_conf"],
                )
                if graph_metadata:
                    candidate["graph_metadata"] = graph_metadata

                new_candidates.append(candidate)
                new_records.append(_record_from_candidate_dict(paper_id, candidate))

    combined_candidates.extend(new_candidates)
    summary["triple_candidates"] = combined_candidates

    tier_meta.update(
        {
            "status": "completed" if new_candidates else "no_matches",
            "processed_sentences": stats["processed_sentences"],
            "rule_hits": stats["rule_hits"],
            "emitted": stats["emitted"],
            "deduplicated": stats["deduplicated"],
            "skipped": stats["skipped"],
            "patterns": rule_counts,
        }
    )

    persistence_payload = existing_records + new_records
    try:
        await replace_triple_candidates(paper_id, persistence_payload)
    except Exception as exc:  # pragma: no cover - persistence failure should not break pipeline
        logger.error("[tier3-rel] failed to persist triple candidates for paper %s: %s", paper_id, exc)
        tier_meta["persistence_error"] = str(exc)

    return summary


def _prepare_sentence_contexts(sections: Sequence[Mapping[str, Any]]) -> List[SentenceContext]:
    contexts: List[SentenceContext] = []
    for section in sections:
        section_id = str(section.get("section_id") or "").strip()
        if not section_id:
            continue
        sentences = section.get("sentence_spans") or []
        for entry in sentences:
            text = (entry.get("text") or "").strip()
            if not text or len(text) < _MIN_PHRASE_CHARS:
                continue
            try:
                sentence_index = int(entry.get("sentence_index"))
            except (TypeError, ValueError):
                sentence_index = 0
            start = int(entry.get("start") or 0)
            end = int(entry.get("end") or start + len(text))
            contexts.append(
                SentenceContext(
                    section_id=section_id,
                    sentence_index=sentence_index,
                    text=text,
                    start=start,
                    end=end,
                )
            )
    return contexts


def _get_dependency_matcher(key: str, nlp: Any) -> Optional[DependencyMatcher]:
    if DependencyMatcher is None:
        return None
    matcher = MATCHER_CACHE.get(key)
    if matcher is not None:
        return matcher
    matcher = DependencyMatcher(nlp.vocab)
    for label, definition in PATTERN_DEFINITIONS.items():
        try:
            matcher.add(label, [definition["pattern"]])
        except ValueError:
            matcher.remove(label)
            matcher.add(label, [definition["pattern"]])
    MATCHER_CACHE[key] = matcher
    return matcher


def _map_pattern_tokens(nodes: Sequence[str], doc: Doc, token_ids: Sequence[int]) -> Dict[str, Token]:
    if len(nodes) != len(token_ids):
        return {}
    mapping: Dict[str, Token] = {}
    for node, token_index in zip(nodes, token_ids):
        if token_index >= len(doc):
            return {}
        mapping[node] = doc[token_index]
    return mapping


def _build_candidate_from_match(
    rule: str,
    doc: Doc,
    mapping: Mapping[str, Token],
) -> Optional[CandidateData]:
    if rule == "PROPOSES":
        return _handle_proposes(doc, mapping)
    if rule == "EVALUATED_ON":
        return _handle_evaluated_on(doc, mapping)
    if rule == "REPORTS":
        return _handle_reports(doc, mapping)
    return None


def _handle_proposes(doc: Doc, mapping: Mapping[str, Token]) -> Optional[CandidateData]:
    method_token = mapping.get("METHOD")
    task_token = mapping.get("TASK")
    verb = mapping.get("VERB")
    if method_token is None or task_token is None:
        return None
    method_text, method_span = _phrase_for_token(method_token)
    task_text, task_span = _phrase_for_token(task_token)
    if not _looks_like_method(method_text) or not _looks_like_task(task_text):
        return None
    relation_text = verb.lemma_.lower() if verb is not None else "proposes"
    return CandidateData(
        subject=method_text,
        subject_span=method_span,
        subject_type="Method",
        relation_text=relation_text,
        relation_guess="PROPOSES",
        object=task_text,
        object_span=task_span,
        object_type="Task",
        confidence=_RULE_CONFIDENCE,
        provenance={"pattern": "proposes"},
    )


def _handle_evaluated_on(doc: Doc, mapping: Mapping[str, Token]) -> Optional[CandidateData]:
    method_token = mapping.get("METHOD")
    dataset_token = mapping.get("DATASET")
    verb = mapping.get("VERB")
    if method_token is None or dataset_token is None:
        return None
    method_text, method_span = _phrase_for_token(method_token)
    dataset_text, dataset_span = _phrase_for_token(dataset_token)
    if not _looks_like_method(method_text) or not _looks_like_dataset(dataset_text):
        return None
    relation_text = verb.lemma_.lower() if verb is not None else "evaluates"
    return CandidateData(
        subject=method_text,
        subject_span=method_span,
        subject_type="Method",
        relation_text=relation_text,
        relation_guess="EVALUATED_ON",
        object=dataset_text,
        object_span=dataset_span,
        object_type="Dataset",
        confidence=_RULE_CONFIDENCE,
        provenance={"pattern": "evaluated_on"},
    )


def _handle_reports(doc: Doc, mapping: Mapping[str, Token]) -> Optional[CandidateData]:
    method_token = mapping.get("METHOD")
    value_token = mapping.get("VALUE")
    verb = mapping.get("VERB")
    if method_token is None or value_token is None:
        return None
    method_text, method_span = _phrase_for_token(method_token)
    if not _looks_like_method(method_text):
        return None
    value_phrase, value_span = _expand_numeric_phrase(value_token)
    value_text = _normalize_phrase(value_phrase)
    if not _looks_like_value(value_text):
        return None
    metric_text, metric_span = _find_metric_phrase(doc, value_token)
    object_type = "Concept"
    combined_span = value_span
    object_text = value_text
    if metric_text:
        object_type = "Metric"
        object_text = f"{metric_text} {value_text}".strip()
        combined_span = (min(metric_span[0], value_span[0]), max(metric_span[1], value_span[1]))
    relation_text = verb.lemma_.lower() if verb is not None else "reports"
    confidence = _REPORT_CONFIDENCE if metric_text else max(_REPORT_CONFIDENCE - 0.04, 0.55)
    return CandidateData(
        subject=method_text,
        subject_span=method_span,
        subject_type="Method",
        relation_text=relation_text,
        relation_guess="REPORTS",
        object=object_text,
        object_span=combined_span,
        object_type=object_type,
        confidence=confidence,
        provenance={"pattern": "reports"},
    )


def _candidate_to_dict(data: CandidateData, context: SentenceContext) -> dict[str, Any]:
    evidence = context.text.strip()
    subject_span = _clamp_span(data.subject_span, len(evidence))
    object_span = _clamp_span(data.object_span, len(evidence))
    evidence_span = {
        "section_id": context.section_id,
        "sentence_index": context.sentence_index,
        "start": 0,
        "end": len(evidence),
    }
    candidate = {
        "subject": data.subject,
        "relation": data.relation_text,
        "object": data.object,
        "subject_span": subject_span,
        "object_span": object_span,
        "subject_type_guess": data.subject_type,
        "relation_type_guess": data.relation_guess,
        "object_type_guess": data.object_type,
        "triple_conf": round(float(data.confidence), 4),
        "schema_match_score": 1.0,
        "evidence": evidence,
        "evidence_spans": [evidence_span],
        "section_id": context.section_id,
        "provenance": dict(data.provenance),
    }
    return candidate


def _phrase_for_token(token: Token) -> Tuple[str, Tuple[int, int]]:
    tokens = list(token.subtree)
    tokens.sort(key=lambda item: item.idx)
    start = tokens[0].idx
    end = tokens[-1].idx + len(tokens[-1])
    raw_text = token.doc.text[start:end]
    cleaned = _normalize_phrase(raw_text)
    if not cleaned:
        cleaned = raw_text.strip()
    return cleaned, (start, end)


def _normalize_phrase(text: str) -> str:
    cleaned = _CITATION_RE.sub(" ", text)
    cleaned = _ARTICLE_RE.sub("", cleaned.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.;:")


def _looks_like_method(text: str) -> bool:
    lowered = text.lower()
    if not text or len(text) < _MIN_PHRASE_CHARS:
        return False
    if lowered in _PRONOUNS:
        return False
    if any(keyword in lowered for keyword in _METHOD_KEYWORDS):
        return True
    if any(ch.isupper() for ch in text if ch.isalpha()):
        return True
    return len(text.split()) >= 2


def _looks_like_dataset(text: str) -> bool:
    lowered = text.lower()
    if not text or len(text) < _MIN_PHRASE_CHARS:
        return False
    if any(keyword in lowered for keyword in _DATASET_KEYWORDS):
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if any(ch.isupper() for ch in text if ch.isalpha()):
        return True
    return len(text.split()) >= 2


def _looks_like_task(text: str) -> bool:
    lowered = text.lower()
    if not text or len(lowered) < _MIN_PHRASE_CHARS:
        return False
    if any(keyword in lowered for keyword in _TASK_KEYWORDS):
        return True
    if lowered.endswith("task") or lowered.endswith("problem"):
        return True
    return len(text.split()) >= 2


def _looks_like_value(text: str) -> bool:
    if not text:
        return False
    return bool(_NUMERIC_RE.search(text))


def _expand_numeric_phrase(token: Token) -> Tuple[str, Tuple[int, int]]:
    doc = token.doc
    start = token.idx
    end = token.idx + len(token)
    idx = token.i + 1
    while idx < len(doc):
        nxt = doc[idx]
        lower = nxt.text.lower()
        if nxt.is_space:
            idx += 1
            continue
        if nxt.like_num or lower in _UNIT_TOKENS:
            end = nxt.idx + len(nxt)
            idx += 1
            continue
        if lower == "to" and idx + 1 < len(doc) and doc[idx + 1].like_num:
            end = doc[idx + 1].idx + len(doc[idx + 1])
            idx += 2
            continue
        break
    phrase = doc.text[start:end]
    return phrase, (start, end)


def _find_metric_phrase(doc: Doc, value_token: Token) -> Tuple[Optional[str], Tuple[int, int]]:
    window_start = max(0, value_token.i - 6)
    window_end = min(len(doc), value_token.i + 7)
    candidate: Optional[Token] = None
    for idx in range(window_start, window_end):
        token = doc[idx]
        lemma = token.lemma_.lower()
        text = token.text.lower()
        if lemma in _METRIC_KEYWORDS or text in _METRIC_KEYWORDS or token.text.upper() in {"F1", "BLEU", "ROUGE"}:
            candidate = token
            break
    if candidate is None:
        return None, (value_token.idx, value_token.idx + len(value_token))
    start = candidate.i
    end = candidate.i
    while start > 0 and doc[start - 1].pos_ in {"ADJ", "NOUN", "PROPN"}:
        start -= 1
    while end + 1 < len(doc) and doc[end + 1].pos_ in {"NOUN", "PROPN"}:
        end += 1
    span_start = doc[start].idx
    span_end = doc[end].idx + len(doc[end])
    phrase = doc.text[span_start:span_end]
    return _normalize_phrase(phrase), (span_start, span_end)


def _clamp_span(span: Tuple[int, int], limit: int) -> List[int]:
    try:
        start = int(span[0])
        end = int(span[1])
    except (TypeError, ValueError):
        start, end = 0, 0
    start = max(0, min(start, limit))
    end = max(start, min(end, limit))
    return [start, end]


def _build_candidate_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:03d}"


def _record_from_candidate_dict(
    paper_id: UUID,
    candidate: Mapping[str, Any],
) -> TripleCandidateRecord:
    subject_span = _ensure_span(candidate.get("subject_span"))
    object_span = _ensure_span(candidate.get("object_span"))
    triple_conf = _coerce_float(candidate.get("triple_conf"), _RULE_CONFIDENCE)
    schema_score = _coerce_float(candidate.get("schema_match_score"), 1.0)

    provenance = candidate.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_payload = dict(provenance)
    elif provenance is None:
        provenance_payload = {}
    else:
        provenance_payload = {"source": str(provenance)}

    graph_metadata = candidate.get("graph_metadata")
    if isinstance(graph_metadata, Mapping):
        graph_payload = dict(graph_metadata)
    else:
        graph_payload = {}

    section_id = candidate.get("section_id")
    section_value = str(section_id) if section_id else None

    return TripleCandidateRecord(
        paper_id=paper_id,
        section_id=section_value,
        subject=str(candidate.get("subject", "")),
        relation=str(candidate.get("relation", "")),
        object=str(candidate.get("object", "")),
        subject_span=subject_span,
        object_span=object_span,
        subject_type_guess=str(candidate.get("subject_type_guess", "Unknown")),
        relation_type_guess=str(candidate.get("relation_type_guess", "OTHER")),
        object_type_guess=str(candidate.get("object_type_guess", "Unknown")),
        evidence=str(candidate.get("evidence", "")),
        triple_conf=triple_conf,
        schema_match_score=schema_score,
        tier=str(candidate.get("tier", TIER_NAME)),
        graph_metadata=graph_payload,
        provenance=provenance_payload,
    )


def _ensure_span(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = int(value[0])
            end = int(value[1])
        except (TypeError, ValueError):
            start, end = 0, 0
    else:
        start, end = 0, 0
    if end < start:
        end = start
    return [start, end]


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    return max(0.0, min(1.0, result))
