from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Optional, Sequence, Union
from uuid import UUID

from app.core.config import settings
from app.models.ontology import ClaimCategory, ClaimCreate, ResultCreate
from app.models.section import Section
from app.schemas.tier2 import (
    ClaimPayload,
    EvidenceSpan,
    MethodPayload,
    ResultPayload,
    Tier2LLMPayload,
)
from app.services.nlp_pipeline import CachedDoc, process_text
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

from typing import Optional
logger = logging.getLogger(__name__)

TIER_NAME = "spacy_structurer"
BASE_CONFIDENCE = 0.6
_NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_PERCENT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?\s*%")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MIN_SENTENCE_LENGTH = 35

_METRIC_KEYWORDS = {
    "accuracy",
    "f1",
    "f1-score",
    "precision",
    "recall",
    "bleu",
    "rouge",
    "meteor",
    "chrf",
    "psnr",
    "iou",
    "map",
    "mrr",
    "perplexity",
    "wer",
    "cer",
    "bleu-4",
}
_DATASET_HINTS = {
    "dataset",
    "corpus",
    "benchmark",
    "collection",
    "set",
}
_TASK_KEYWORDS = {
    "classification",
    "translation",
    "segmentation",
    "detection",
    "retrieval",
    "summarization",
    "generation",
    "analysis",
    "recognition",
    "answering",
}
_METHOD_NOVELTY_HINTS = {"novel", "new", "first", "state-of-the-art"}

_CLAIM_LIMITATION_HINTS = {"limitation", "limited", "failure", "weakness"}
_CLAIM_FUTURE_HINTS = {"future work", "future direction", "plan to"}
_CLAIM_ABLATION_HINTS = {"ablation"}


class Tier2ValidationError(RuntimeError):
    """Raised when Tier-2 is unable to generate a valid payload."""


@dataclass
class _Caches:
    methods: dict[str, Any]
    datasets: dict[str, Any]
    metrics: dict[str, Any]
    tasks: dict[str, Any]


@dataclass
class EntityOccurrence:
    section_id: Optional[str]
    start: int
    end: int
    pipeline: str
    source: str
    score: float
    sentence: str


@dataclass
class EntityRecord:
    name: str
    kind: str
    score: float
    aliases: set[str] = field(default_factory=set)
    occurrences: list[EntityOccurrence] = field(default_factory=list)
    is_new: bool = False

    def add_occurrence(
        self,
        occurrence: EntityOccurrence,
        *,
        alias: Optional[str] = None,
        score: float,
        is_new: bool = False,
    ) -> None:
        if score > self.score:
            self.score = score
        if alias and alias != self.name:
            self.aliases.add(alias)
        self.occurrences.append(occurrence)
        if is_new:
            self.is_new = True


class EntityAccumulator:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, EntityRecord]] = defaultdict(dict)

    def seed_from_summary(self, summary: Optional[dict[str, Any]]) -> None:
        if not summary:
            return
        for method in _summary_list(summary, "methods"):
            if isinstance(method, dict) and (name := method.get("name")):
                record = EntityRecord(name=name, kind="method", score=1.0)
                if isinstance(method.get("aliases"), list):
                    record.aliases.update(method["aliases"])
                if isinstance(method.get("is_new"), bool):
                    record.is_new = bool(method["is_new"])
                self._records["method"][_normalize_text(name)] = record
        for key in ("datasets", "metrics", "tasks"):
            for item in _summary_list(summary, key):
                if isinstance(item, dict) and (name := item.get("name")):
                    record = EntityRecord(name=name, kind=key[:-1], score=1.0)
                    self._records[key[:-1]][_normalize_text(name)] = record

    def add(
        self,
        *,
        name: str,
        kind: str,
        score: float,
        section: Optional[Section],
        start: int,
        end: int,
        pipeline: str,
        source: str,
        sentence: str,
        alias: Optional[str] = None,
        is_new: bool = False,
    ) -> None:
        key = _normalize_text(name)
        if not key or score < settings.nlp_min_span_score:
            return
        record = self._records[kind].get(key)
        if record is None:
            record = EntityRecord(name=name, kind=kind, score=score, is_new=is_new)
            self._records[kind][key] = record
        occurrence = EntityOccurrence(
            section_id=str(section.id) if section else None,
            start=start,
            end=end,
            pipeline=pipeline,
            source=source,
            score=score,
            sentence=sentence,
        )
        record.add_occurrence(occurrence, alias=alias, score=score, is_new=is_new)

    def records(self, kind: str) -> list[EntityRecord]:
        return list(self._records.get(kind, {}).values())

    def records_in_span(self, kind: str, section_id: Optional[str], start: int, end: int) -> list[EntityRecord]:
        matches: list[EntityRecord] = []
        for record in self._records.get(kind, {}).values():
            for occurrence in record.occurrences:
                if occurrence.section_id == section_id and occurrence.start < end and occurrence.end > start:
                    matches.append(record)
                    break
        return matches

    def best_record(self, kind: str) -> Optional[EntityRecord]:
        candidates = self.records(kind)
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.score)


class ResultBuilder:
    def __init__(self) -> None:
        self._results: list[ResultPayload] = []
        self._signatures: set[tuple[str, str, str, str, Optional[str]]] = set()

    def add(
        self,
        *,
        method: str,
        dataset: str,
        metric: str,
        value_text: str,
        sentence_start: int,
        sentence_end: int,
        section_id: Optional[str],
        task: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        signature = (
            _normalize_text(method),
            _normalize_text(dataset),
            _normalize_text(metric),
            value_text,
            section_id,
        )
        if signature in self._signatures:
            return
        self._signatures.add(signature)

        numeric_value: Union[float, int, str]
        cleaned = value_text.strip()
        try:
            numeric_value = float(cleaned)
        except ValueError:
            numeric_value = cleaned

        evidence_span = EvidenceSpan(
            section_id=section_id,
            start=sentence_start,
            end=sentence_end,
        )
        self._results.append(
            ResultPayload(
                method=method,
                dataset=dataset,
                metric=metric,
                value=numeric_value,
                split=split,
                task=task,
                evidence_span=evidence_span,
            )
        )

    @property
    def results(self) -> list[ResultPayload]:
        return self._results


class ClaimBuilder:
    def __init__(self) -> None:
        self._claims: list[ClaimPayload] = []
        self._signatures: set[tuple[Optional[str], str]] = set()

    def add(
        self,
        *,
        text: str,
        category: ClaimCategory,
        section_id: Optional[str],
        start: int,
        end: int,
    ) -> None:
        normalized = _normalize_text(text)
        signature = (section_id, normalized)
        if signature in self._signatures:
            return
        self._signatures.add(signature)
        self._claims.append(
            ClaimPayload(
                category=category.value,
                text=text.strip(),
                evidence_span=EvidenceSpan(section_id=section_id, start=start, end=end),
            )
        )

    @property
    def claims(self) -> list[ClaimPayload]:
        return self._claims

async def run_tier2_structurer(
    paper_id: UUID,
    *,
    base_summary: Optional[dict[str, Any]] = None,
    sections: Optional[Sequence[Section]] = None,
) -> dict[str, Any]:
    summary = base_summary or {}

    paper = await get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    if sections is None:
        sections = await list_sections(paper_id=paper_id, limit=500, offset=0)

    payload = _build_structured_payload(
        paper_title=paper.title or "",
        sections=sections,
        summary=summary,
    )

    caches = _Caches(methods={}, datasets={}, metrics={}, tasks={})
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


def _build_structured_payload(
    *,
    paper_title: str,
    sections: Sequence[Section],
    summary: Optional[dict[str, Any]],
) -> Tier2LLMPayload:
    accumulator = EntityAccumulator()
    accumulator.seed_from_summary(summary)
    result_builder = ResultBuilder()
    claim_builder = ClaimBuilder()

    for section in sections:
        text = (section.content or "").strip()
        if not text:
            continue
        cached_docs = process_text(text)
        for cached in cached_docs:
            _collect_entities_from_doc(accumulator, section, cached)
        primary = _select_primary_doc(cached_docs)
        if primary:
            _collect_sentence_relations(
                accumulator=accumulator,
                section=section,
                cached=primary,
                result_builder=result_builder,
                claim_builder=claim_builder,
            )

    methods_payload = [
        MethodPayload(
            name=record.name,
            aliases=sorted(record.aliases),
            is_new=record.is_new or None,
        )
        for record in sorted(
            accumulator.records("method"),
            key=lambda item: (-item.score, item.name.lower()),
        )
    ]

    datasets_payload = sorted({record.name for record in accumulator.records("dataset")})
    metrics_payload = sorted({record.name for record in accumulator.records("metric")})
    tasks_payload = sorted({record.name for record in accumulator.records("task")})

    if not methods_payload:
        best_method = accumulator.best_record("method")
        if best_method:
            methods_payload.append(
                MethodPayload(
                    name=best_method.name,
                    aliases=sorted(best_method.aliases),
                    is_new=best_method.is_new or None,
                )
            )

    return Tier2LLMPayload(
        paper_title=paper_title,
        methods=methods_payload,
        tasks=tasks_payload,
        datasets=datasets_payload,
        metrics=metrics_payload,
        results=result_builder.results,
        claims=claim_builder.claims,
    )

def _collect_entities_from_doc(accumulator: EntityAccumulator, section: Section, cached: CachedDoc) -> None:
    doc = cached.doc
    pipeline_key = cached.pipeline.key
    base_score = 0.62 if cached.pipeline.kind == "scispacy" else 0.58

    for ent in doc.ents:
        span_text = ent.text.strip()
        if len(span_text) < settings.nlp_min_span_char_length:
            continue
        kind = _infer_entity_kind(ent)
        if not kind:
            continue
        sentence = ent.sent.text.strip()
        is_new = kind == "method" and _sentence_has_novelty_hint(sentence)
        accumulator.add(
            name=span_text,
            alias=span_text,
            kind=kind,
            score=base_score,
            section=section,
            start=ent.start_char,
            end=ent.end_char,
            pipeline=pipeline_key,
            source=f"ner:{ent.label_}",
            sentence=sentence,
            is_new=is_new,
        )

    for chunk in doc.noun_chunks:
        span_text = chunk.text.strip()
        if len(span_text) < settings.nlp_min_span_char_length:
            continue
        kind = _infer_chunk_kind(chunk)
        if not kind:
            continue
        sentence = chunk.sent.text.strip()
        accumulator.add(
            name=span_text,
            alias=span_text,
            kind=kind,
            score=base_score - 0.03,
            section=section,
            start=chunk.start_char,
            end=chunk.end_char,
            pipeline=pipeline_key,
            source="chunk",
            sentence=sentence,
        )


def _collect_sentence_relations(
    *,
    accumulator: EntityAccumulator,
    section: Section,
    cached: CachedDoc,
    result_builder: ResultBuilder,
    claim_builder: ClaimBuilder,
) -> None:
    doc = cached.doc
    section_id = str(section.id) if section else None

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        start_char = sent.start_char
        end_char = sent.end_char

        lower_text = sent_text.lower()
        if len(sent_text) >= _MIN_SENTENCE_LENGTH and _contains_claim_signal(lower_text):
            category = _classify_claim(lower_text)
            claim_builder.add(
                text=sent_text,
                category=category,
                section_id=section_id,
                start=start_char,
                end=end_char,
            )

        method_records = accumulator.records_in_span("method", section_id, start_char, end_char)
        dataset_records = accumulator.records_in_span("dataset", section_id, start_char, end_char)
        metric_records = accumulator.records_in_span("metric", section_id, start_char, end_char)
        task_records = accumulator.records_in_span("task", section_id, start_char, end_char)

        if not metric_records:
            for keyword in _METRIC_KEYWORDS:
                if keyword in lower_text:
                    accumulator.add(
                        name=keyword.upper(),
                        kind="metric",
                        score=0.55,
                        section=section,
                        start=start_char,
                        end=end_char,
                        pipeline=cached.pipeline.key,
                        source="heuristic",
                        sentence=sent_text,
                    )
                    metric_records = accumulator.records_in_span("metric", section_id, start_char, end_char)
                    break

        if not (method_records and dataset_records and metric_records):
            continue

        number_match = _find_primary_value(sent_text)
        if not number_match:
            continue

        method_name = max(method_records, key=lambda record: record.score).name
        dataset_name = max(dataset_records, key=lambda record: record.score).name
        metric_name = max(metric_records, key=lambda record: record.score).name
        value_text = number_match.group().strip(" %")
        split = _detect_split(lower_text)
        task_name = (
            max(task_records, key=lambda record: record.score).name
            if task_records
            else None
        )

        result_builder.add(
            method=method_name,
            dataset=dataset_name,
            metric=metric_name,
            value_text=value_text,
            sentence_start=start_char,
            sentence_end=end_char,
            section_id=section_id,
            task=task_name,
            split=split,
        )


def _select_primary_doc(cached_docs: Sequence[CachedDoc]) -> Optional[CachedDoc]:
    for cached in cached_docs:
        if cached.pipeline.kind == "scispacy":
            return cached
    return cached_docs[0] if cached_docs else None


def _sentence_has_novelty_hint(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in _METHOD_NOVELTY_HINTS)


def _contains_claim_signal(lowered_sentence: str) -> bool:
    return (
        "we " in lowered_sentence
        or "our " in lowered_sentence
        or "this paper" in lowered_sentence
    )


def _classify_claim(lowered_sentence: str) -> ClaimCategory:
    if any(hint in lowered_sentence for hint in _CLAIM_LIMITATION_HINTS):
        return ClaimCategory.LIMITATION
    if any(hint in lowered_sentence for hint in _CLAIM_ABLATION_HINTS):
        return ClaimCategory.ABLATION
    if any(hint in lowered_sentence for hint in _CLAIM_FUTURE_HINTS):
        return ClaimCategory.FUTURE_WORK
    return ClaimCategory.CONTRIBUTION


def _find_primary_value(sentence: str) -> Optional[re.Match[str]]:
    percent_match = _PERCENT_PATTERN.search(sentence)
    if percent_match:
        return percent_match
    return _NUMERIC_PATTERN.search(sentence)


def _detect_split(lowered_sentence: str) -> Optional[str]:
    if "test" in lowered_sentence:
        return "test"
    if "validation" in lowered_sentence or " val " in lowered_sentence:
        return "validation"
    if "dev" in lowered_sentence:
        return "dev"
    return None


def _infer_entity_kind(span) -> Optional[str]:
    text = span.text.strip()
    lower = text.lower()
    label = getattr(span, "label_", "")

    if _looks_like_metric(text, lower, label):
        return "metric"
    if _looks_like_dataset(span, lower):
        return "dataset"
    if _looks_like_task(text, lower):
        return "task"
    return "method"


def _infer_chunk_kind(chunk) -> Optional[str]:
    text = chunk.text.strip()
    lower = text.lower()
    if _looks_like_dataset(chunk, lower):
        return "dataset"
    if _looks_like_task(text, lower):
        return "task"
    return None


def _looks_like_metric(text: str, lower: str, label: str) -> bool:
    if lower in _METRIC_KEYWORDS:
        return True
    if text.upper() in {"BLEU", "ROUGE", "ROUGE-L", "PSNR", "WER", "CER"}:
        return True
    if label in {"PERCENT", "CARDINAL", "QUANTITY"} and any(
        keyword in lower for keyword in _METRIC_KEYWORDS
    ):
        return True
    if "score" in lower and _NUMERIC_PATTERN.search(text):
        return True
    return False


def _looks_like_dataset(span, lower: str) -> bool:
    if any(lower.endswith(hint) for hint in _DATASET_HINTS):
        return True
    if "dataset" in lower or "corpus" in lower:
        return True
    head = getattr(span, "root", None)
    if head is not None:
        if head.dep_ == "pobj" and head.head is not None and head.head.text.lower() in {"on", "over", "against", "using"}:
            return True
        head_text = head.text.lower()
        head_head = head.head.text.lower() if head.head is not None else ""
        if head_text in _DATASET_HINTS or head_head in _DATASET_HINTS:
            return True
    upper_ratio = sum(1 for ch in span.text if ch.isupper()) / max(1, len(span.text))
    if upper_ratio > 0.6 and len(span.text) <= 20:
        return True
    return False


def _looks_like_task(text: str, lower: str) -> bool:
    if "task" in lower or "problem" in lower:
        return True
    return any(keyword in lower for keyword in _TASK_KEYWORDS)

def _summary_list(summary: dict[str, Any], key: str) -> list[Any]:
    value = summary.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


async def _ensure_catalog_from_summary(summary: dict[str, Any], caches: _Caches) -> None:
    for item in _summary_list(summary, "methods"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_method(name, caches, aliases=item.get("aliases"))
    for item in _summary_list(summary, "datasets"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_dataset(name, caches, aliases=item.get("aliases"))
    for item in _summary_list(summary, "metrics"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_metric(name, caches, unit=item.get("unit"), aliases=item.get("aliases"))
    for item in _summary_list(summary, "tasks"):
        if isinstance(item, dict) and (name := item.get("name")):
            await _ensure_task(name, caches, aliases=item.get("aliases"))


async def _ensure_catalog_from_payload(payload: Tier2LLMPayload, caches: _Caches) -> None:
    for method in payload.methods:
        await _ensure_method(method.name, caches, aliases=method.aliases or None)
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
        if isinstance(result, dict):
            converted_model = await _summary_result_to_model(paper_id, result, caches)
            if converted_model:
                converted.append(converted_model)
    return converted


async def _summary_result_to_model(
    paper_id: UUID,
    payload: dict[str, Any],
    caches: _Caches,
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


async def _extract_summary_ontology(
    payload: Optional[dict[str, Any]],
    ensure_func,
    caches: _Caches,
):
    if payload and (name := payload.get("name")):
        return await ensure_func(name, caches, aliases=payload.get("aliases"))
    return None


async def _extract_summary_method(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_method, caches)


async def _extract_summary_dataset(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_dataset, caches)


async def _extract_summary_metric(payload: Optional[dict[str, Any]], caches: _Caches):
    if payload and (name := payload.get("name")):
        return await _ensure_metric(name, caches, unit=payload.get("unit"), aliases=payload.get("aliases"))
    return None


async def _extract_summary_task(payload: Optional[dict[str, Any]], caches: _Caches):
    return await _extract_summary_ontology(payload, _ensure_task, caches)


async def _convert_payload_results(
    paper_id: UUID,
    payload: Tier2LLMPayload,
    caches: _Caches,
) -> list[ResultCreate]:
    converted: list[ResultCreate] = []
    method_lookup = {_normalize_text(method.name): method for method in payload.methods}

    for result in payload.results:
        method_meta = method_lookup.get(_normalize_text(result.method))
        method_aliases = method_meta.aliases if method_meta else None

        method_model = await _ensure_method(result.method, caches, aliases=method_aliases)
        dataset_model = await _ensure_dataset(result.dataset, caches)
        metric_model = await _ensure_metric(result.metric, caches)

        task_name = result.task or (payload.tasks[0] if payload.tasks else None)
        task_model = await _ensure_task(task_name, caches) if task_name else None

        value_text = None
        numeric_decimal = None
        if result.value is not None:
            value_text = str(result.value)
            try:
                numeric_decimal = Decimal(value_text)
            except Exception:
                numeric_decimal = None

        evidence_payload: list[dict[str, Any]] = []
        if result.evidence_span:
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
        try:
            category = ClaimCategory(claim.category.strip().lower())
        except ValueError:
            category = ClaimCategory.OTHER

        evidence_payload: list[dict[str, Any]] = []
        if claim.evidence_span:
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


async def _convert_summary_claims(
    paper_id: UUID,
    summary: dict[str, Any],
) -> list[ClaimCreate]:
    converted: list[ClaimCreate] = []
    for claim in _summary_list(summary, "claims"):
        if isinstance(claim, dict) and (text := claim.get("text")):
            try:
                category = ClaimCategory(claim.get("category", "").strip().lower())
            except ValueError:
                category = ClaimCategory.OTHER

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
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
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
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
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
    unit: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
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
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
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


def _merge_tiers(existing: Optional[Sequence[Union[int, str]]], new: Sequence[int]) -> list[int]:
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
        "description": getattr(model, "description", None),
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


_serialize_method = _serialize_ontology
_serialize_dataset = _serialize_ontology
_serialize_task = _serialize_ontology


def _serialize_metric(model) -> dict[str, Any]:
    payload = _serialize_ontology(model)
    payload["unit"] = getattr(model, "unit", None)
    return payload


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