from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from uuid import UUID

from app.core.config import settings
from app.models.concept import Concept, ConceptCreate
from app.models.paper import Paper
from app.models.section import SectionBase
from app.services.concepts import replace_concepts
from app.services.papers import get_paper
from app.services.relations import replace_paper_concept_relations
from app.services.sections import list_sections

try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from spacy.language import Language  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    Language = None  # type: ignore[assignment]


MIN_CHAR_LENGTH = 3
SNIPPET_WINDOW = 120

COMMON_SECTION_TITLES = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "methods",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
}

STOPWORDS: Set[str] = set(settings.concept_extraction.tuning.stopwords)

_SPACY_MODEL_CACHE: Dict[str, Union[Language, bool]] = {}

FILLER_PREFIXES: Set[str] = set(
    settings.concept_extraction.tuning.filler_prefixes
)

FILLER_SUFFIXES: Set[str] = set(
    settings.concept_extraction.tuning.filler_suffixes
)


@dataclass
class ExtractedConcept:
    name: str
    type: Optional[str]
    description: Optional[str]
    score: float
    canonical_id: Optional[str] = None
    canonical_score: Optional[float] = None


@dataclass
class _Candidate:
    name: str
    normalized: str
    type: Optional[str]
    description: Optional[str]
    score: float
    occurrences: int = 1
    canonical_id: Optional[str] = None
    canonical_score: Optional[float] = None


@dataclass
class ConceptExtractionRuntimeConfig:
    max_concepts: int
    max_tokens: int
    stopwords: Set[str]
    filler_prefixes: Set[str]
    filler_suffixes: Set[str]
    provider_priority: Tuple[str, ...]
    scispacy_models: Tuple[str, ...]
    domain_model: Optional[str]
    llm_prompt: Optional[str]
    entity_hints: Dict[str, Set[str]] = field(default_factory=dict)
    domain_key: Optional[str] = None


def extract_concepts_from_sections(
    sections: Sequence[SectionBase],
    *,
    config: Optional[ConceptExtractionRuntimeConfig] = None,
    paper: Optional[Paper] = None,
) -> List[ExtractedConcept]:
    filtered = [section for section in sections if section.content.strip()]
    if not filtered:
        return []

    runtime = config or _build_runtime_config(paper)
    candidates: Dict[str, _Candidate] = {}

    provider_available, provider_candidates = _collect_model_candidates(filtered, runtime)
    for candidate in provider_candidates:
        _merge_candidate(candidates, candidate)

    if not provider_available:
        for candidate in _extract_with_heuristics(filtered, runtime):
            _merge_candidate(candidates, candidate)

    if not candidates:
        return []

    _apply_method_post_filters(candidates, runtime)

    ranked = sorted(
        candidates.values(), key=lambda item: (-_final_score(item), item.name.lower())
    )
    top_ranked = ranked[: runtime.max_concepts]
    return [
        ExtractedConcept(
            name=candidate.name,
            type=candidate.type,
            description=candidate.description,
            score=round(_final_score(candidate), 4),
            canonical_id=candidate.canonical_id,
            canonical_score=
            round(candidate.canonical_score, 4)
            if candidate.canonical_score is not None
            else None,
        )
        for candidate in top_ranked
    ]


def _collect_model_candidates(
    sections: Sequence[SectionBase],
    config: ConceptExtractionRuntimeConfig,
) -> Tuple[bool, List[_Candidate]]:
    collected: List[_Candidate] = []
    provider_available = False
    for provider in config.provider_priority:
        normalized_provider = provider.lower()
        if normalized_provider == "scispacy":
            model = _load_scispacy_model(config)
            if model is None:
                continue
            provider_available = True
            model_meta = getattr(model, "meta", {}) or {}
            provider_meta = {
                "model": model_meta.get("name"),
                "lang": getattr(model, "lang", None),
            }
            collected.extend(
                _extract_with_spacy_model(
                    model,
                    sections,
                    config,
                    provider="scispacy",
                    provider_metadata=provider_meta,
                )
            )
            break
        if normalized_provider == "domain_ner":
            if not config.domain_model:
                continue
            model = _load_spacy_model(config.domain_model)
            if model is None:
                continue
            provider_available = True
            model_meta = getattr(model, "meta", {}) or {}
            provider_meta = {
                "model": model_meta.get("name"),
                "lang": getattr(model, "lang", None),
            }
            collected.extend(
                _extract_with_spacy_model(
                    model,
                    sections,
                    config,
                    provider="domain_ner",
                    provider_metadata=provider_meta,
                )
            )
            break
        if normalized_provider == "llm":
            if not _llm_prompt_available(config):
                continue
            provider_available = True
            collected.extend(_extract_with_llm(sections, config))
            break
    return provider_available, collected


async def extract_and_store_concepts(
    paper_id: UUID,
    sections: Optional[Sequence[SectionBase]] = None,
) -> List[Concept]:
    if sections is None:
        sections = await _load_sections_for_paper(paper_id)

    paper = await get_paper(paper_id)
    config = _build_runtime_config(paper)
    concepts = extract_concepts_from_sections(sections, config=config, paper=paper)
    concept_models = [
        ConceptCreate(
            paper_id=paper_id,
            name=concept.name,
            type=concept.type,
            description=concept.description,
            metadata=_build_concept_metadata(concept),
        )
        for concept in concepts
    ]
    stored = await replace_concepts(paper_id, concept_models)
    try:
        await replace_paper_concept_relations(paper_id, stored, relation_type="mentions")
    except Exception as exc:  # pragma: no cover - background task logging
        print(
            f"[concept_extraction] Failed to sync paper {paper_id} concept relations: {exc}"
        )
    return stored


def _build_runtime_config(paper: Optional[Paper]) -> ConceptExtractionRuntimeConfig:
    concept_settings = settings.concept_extraction
    domain_key = _infer_domain_key(paper)
    overrides = (
        concept_settings.domain_overrides.get(domain_key)
        if domain_key and concept_settings.domain_overrides
        else None
    )

    max_tokens = concept_settings.tuning.max_tokens
    if overrides and overrides.max_tokens is not None:
        max_tokens = overrides.max_tokens

    provider_priority: Tuple[str, ...]
    if overrides and overrides.provider_priority:
        provider_priority = tuple(overrides.provider_priority)
    else:
        base_providers = concept_settings.providers or ["scispacy"]
        provider_priority = tuple(base_providers)

    base_stopwords = set(concept_settings.tuning.stopwords)
    base_filler_prefixes = set(concept_settings.tuning.filler_prefixes)
    base_filler_suffixes = set(concept_settings.tuning.filler_suffixes)

    if overrides and overrides.stopwords:
        base_stopwords.update(word.lower() for word in overrides.stopwords)
    if overrides and overrides.filler_prefixes:
        base_filler_prefixes.update(word.lower() for word in overrides.filler_prefixes)
    if overrides and overrides.filler_suffixes:
        base_filler_suffixes.update(word.lower() for word in overrides.filler_suffixes)

    llm_prompt = overrides.llm_prompt if overrides and overrides.llm_prompt else concept_settings.llm_prompt

    entity_hints: Dict[str, Set[str]] = {}
    if overrides and overrides.entity_hints:
        for entity_type, hints in overrides.entity_hints.items():
            entity_hints[entity_type] = {hint.lower() for hint in hints}

    return ConceptExtractionRuntimeConfig(
        max_concepts=concept_settings.max_concepts,
        max_tokens=max(1, max_tokens),
        stopwords=base_stopwords,
        filler_prefixes=base_filler_prefixes,
        filler_suffixes=base_filler_suffixes,
        provider_priority=provider_priority,
        scispacy_models=tuple(concept_settings.scispacy_models),
        domain_model=overrides.ner_model if overrides else None,
        llm_prompt=llm_prompt,
        entity_hints=entity_hints,
        domain_key=domain_key,
    )


def _infer_domain_key(paper: Optional[Paper]) -> Optional[str]:
    if paper is None:
        return None
    fields = [paper.venue, paper.file_content_type, paper.title]
    combined = " ".join(filter(None, (value or "" for value in fields))).strip()
    if not combined:
        return None
    lowered = combined.lower()
    biology_cues = (
        "biology",
        "biochem",
        "genom",
        "microbio",
        "immun",
        "cell",
        "organism",
        "protein",
        "enzyme",
        "med",
    )
    if any(cue in lowered for cue in biology_cues):
        return "biology"

    materials_cues = (
        "material",
        "materials",
        "alloy",
        "polymer",
        "ceramic",
        "perovskite",
        "graphene",
        "nanotube",
        "battery",
        "cathode",
        "anode",
        "electrode",
        "composite",
        "crystal",
        "oxide",
    )
    if any(cue in lowered for cue in materials_cues):
        return "materials"

    return None


def _llm_prompt_available(config: ConceptExtractionRuntimeConfig) -> bool:
    if not config.llm_prompt:
        return False
    if settings.openai_api_key or settings.tier2_llm_model:
        return True
    return False


def _extract_with_llm(
    sections: Sequence[SectionBase],
    config: ConceptExtractionRuntimeConfig,
) -> List[_Candidate]:
    # Placeholder hook for a future LLM-backed extractor. We currently
    # require credentials to be configured before attempting to call an LLM.
    # Without them, we fall back to heuristic extraction.
    return []


async def _load_sections_for_paper(paper_id: UUID) -> List[SectionBase]:
    sections: List[SectionBase] = []
    offset = 0
    limit = 200
    while True:
        batch = await list_sections(paper_id=paper_id, limit=limit, offset=offset)
        if not batch:
            break
        sections.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return sections


def _merge_candidate(registry: Dict[str, _Candidate], candidate: _Candidate) -> None:
    if candidate.normalized in COMMON_SECTION_TITLES:
        return
    existing = registry.get(candidate.normalized)
    if existing is None:
        registry[candidate.normalized] = candidate
        return
    existing.score += candidate.score
    existing.occurrences += candidate.occurrences
    existing_canonical_score = (
        existing.canonical_score if existing.canonical_score is not None else float("-inf")
    )
    candidate_canonical_score = (
        candidate.canonical_score if candidate.canonical_score is not None else float("-inf")
    )
    if candidate.canonical_id and (
        not existing.canonical_id or candidate_canonical_score > existing_canonical_score
    ):
        existing.canonical_id = candidate.canonical_id
        existing.canonical_score = candidate.canonical_score
    if candidate.type and (not existing.type or existing.type == "keyword"):
        existing.type = candidate.type
    if candidate.description and (
        not existing.description or len(candidate.description) < len(existing.description)
    ):
        existing.description = candidate.description
    existing_words = len(existing.name.split())
    candidate_words = len(candidate.name.split())
    if candidate_words < existing_words or (
        candidate_words == existing_words and len(candidate.name) < len(existing.name)
    ):
        existing.name = candidate.name


def _resolve_absolute_span(
    section: SectionBase, local_start: Optional[int], local_end: Optional[int]
) -> Tuple[Optional[int], Optional[int]]:
    if local_start is None and local_end is None:
        return None, None
    base_offset = section.char_start or 0
    if section.char_start is None:
        return local_start, local_end
    absolute_start = local_start + base_offset if local_start is not None else None
    absolute_end = local_end + base_offset if local_end is not None else None
    return absolute_start, absolute_end


def _apply_method_post_filters(
    registry: Dict[str, _Candidate], config: ConceptExtractionRuntimeConfig
) -> None:
    for candidate in registry.values():
        if candidate.type != "method":
            continue
        if candidate.occurrences > 1:
            continue
        if _is_noisy_method_phrase(candidate.name, config):
            candidate.type = "keyword"


def _is_noisy_method_phrase(
    name: str, config: ConceptExtractionRuntimeConfig
) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return False
    tokens = lowered.split()
    if not tokens:
        return False
    if len(tokens) > 8 or len(lowered) > 80:
        return True
    first_token = tokens[0]
    if first_token in config.stopwords or first_token in config.filler_prefixes:
        return True
    return False


def _extract_with_spacy_model(
    model: Language,
    sections: Sequence[SectionBase],
    config: ConceptExtractionRuntimeConfig,
    *,
    provider: str,
    provider_metadata: Optional[Dict[str, Any]] = None,
) -> Iterable[_Candidate]:
    for section in sections:
        doc = model(section.content)
        for entity in getattr(doc, "ents", []):
            raw_name = entity.text
            long_form = getattr(getattr(entity, "_", None), "long_form", None)
            if long_form:
                long_text = getattr(long_form, "text", None) or str(long_form)
                if long_text:
                    raw_name = long_text
            name = _format_phrase(raw_name)
            normalized = _normalize_concept_name(name, config)
            if not normalized:
                continue
            snippet = _build_snippet(section.content, entity.start_char, entity.end_char)
            description = _compose_description(section, snippet)
            label = (entity.label_ or "entity").lower()
            score = 4.0 + min(len(normalized.split()), 4)
            kb_ents = getattr(getattr(entity, "_", None), "kb_ents", None)
            canonical_id: Optional[str] = None
            canonical_score: Optional[float] = None
            if kb_ents:
                kb_list = list(kb_ents)
                if kb_list:
                    best = max(kb_list, key=lambda item: item[1] if len(item) > 1 else 0.0)
                    canonical_id = str(best[0])
                    canonical_score = float(best[1]) if len(best) > 1 else None
            yield _Candidate(
                name=name,
                normalized=normalized,
                type=label,
                description=description,
                score=score,
                canonical_id=canonical_id,
                canonical_score=canonical_score,
            )


def _extract_with_heuristics(
    sections: Sequence[SectionBase],
    config: ConceptExtractionRuntimeConfig,
) -> Iterable[_Candidate]:
    for section in sections:
        for phrase, start, end in _iter_phrases(section.content, config):
            normalized = _normalize_concept_name(phrase, config)
            if not normalized:
                continue
            snippet = _build_snippet(section.content, start, end)
            description = _compose_description(section, snippet)
            score = 1.0 + 0.3 * min(len(normalized.split()), 4)
            score += 0.2 if section.title else 0.0
            score += 0.4 if phrase.isupper() else 0.0
            absolute_start, absolute_end = _resolve_absolute_span(section, start, end)
            metadata = {
                "strategy": "token_phrase",
                "relative_span": [start, end],
            }
            provenance = ConceptProvenance(
                section_id=getattr(section, "id", None),
                char_start=absolute_start,
                char_end=absolute_end,
                snippet=snippet,
                provider="heuristic",
                provider_metadata=metadata,
            )
            yield _Candidate(
                name=_format_phrase(phrase),
                normalized=normalized,
                type=_infer_concept_type(phrase, entity_hints=config.entity_hints),
                description=description,
                score=score,
                provenance=[provenance],
            )


def _iter_phrases(
    text: str,
    config: ConceptExtractionRuntimeConfig,
) -> Iterable[Tuple[str, int, int]]:
    tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]*", text))
    current: List[Tuple[str, int, int]] = []
    for match in tokens:
        token = match.group(0)
        normalized = token.lower()
        if normalized in config.stopwords:
            yield from _flush_phrase(current, config)
            current = []
            continue
        current.append((token, match.start(), match.end()))
        if len(current) >= config.max_tokens:
            yield from _flush_phrase(current, config)
            current = []
    yield from _flush_phrase(current, config)


def _flush_phrase(
    current: List[Tuple[str, int, int]],
    config: ConceptExtractionRuntimeConfig,
) -> Iterable[Tuple[str, int, int]]:
    if not current:
        return []
    words = [item[0] for item in current]
    start = current[0][1]
    end = current[-1][2]
    phrase = " ".join(words)
    tokens = len(words)
    if tokens == 1:
        word = words[0]
        if len(word) < 4 and not any(char.isdigit() for char in word):
            return []
        if word.islower():
            return []
    if tokens == 0:
        return []
    cleaned = _normalize_concept_name(phrase, config)
    if not cleaned or len(cleaned) < MIN_CHAR_LENGTH:
        return []
    if not any(char.isalpha() for char in cleaned):
        return []
    return [(phrase, start, end)]


def _normalize_concept_name(
    name: str, config: Optional[ConceptExtractionRuntimeConfig] = None
) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip()
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\(\)\[\]\{\}]+", " ", normalized)
    normalized = re.sub(r"[^A-Za-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    if not normalized:
        return ""
    parts = normalized.split()
    trimmed = _trim_tokens(parts, config)
    if trimmed:
        parts = trimmed
    stemmed = []
    for part in parts:
        if len(part) > 4 and part.endswith("s"):
            stemmed.append(part[:-1])
        else:
            stemmed.append(part)
    return " ".join(stemmed)


def _trim_tokens(
    tokens: List[str], config: Optional[ConceptExtractionRuntimeConfig]
) -> List[str]:
    start = 0
    end = len(tokens)
    filler_prefixes = config.filler_prefixes if config else FILLER_PREFIXES
    filler_suffixes = config.filler_suffixes if config else FILLER_SUFFIXES
    while start < end and tokens[start] in filler_prefixes:
        start += 1
    while end > start and tokens[end - 1] in filler_suffixes:
        end -= 1
    trimmed = list(tokens[start:end])
    if not trimmed:
        trimmed = list(tokens)
    while len(trimmed) > 1 and _looks_like_acronym(trimmed[-1], trimmed[:-1]):
        trimmed.pop()
    return trimmed


def _looks_like_acronym(token: str, prior_tokens: Sequence[str]) -> bool:
    if len(prior_tokens) < 2:
        return False
    stripped = token.rstrip("s")
    if len(stripped) < 2 or len(stripped) > len(prior_tokens):
        return False
    if not stripped.isalpha():
        return False
    initials = "".join(part[0] for part in prior_tokens[: len(stripped)] if part)
    return stripped == initials.lower()


def _format_phrase(phrase: str) -> str:
    cleaned = unicodedata.normalize("NFKC", phrase).strip()
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.islower():
        return cleaned.title()
    return cleaned


def _build_snippet(text: str, start: int, end: int) -> str:
    snippet_start = max(0, start - SNIPPET_WINDOW)
    snippet_end = min(len(text), end + SNIPPET_WINDOW)
    snippet = text[snippet_start:snippet_end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > SNIPPET_WINDOW * 2:
        snippet = snippet[: SNIPPET_WINDOW * 2].rstrip()
        snippet += "…"
    return snippet


def _compose_description(section: SectionBase, snippet: str) -> Optional[str]:
    title = (section.title or "").strip()
    normalized_title = _normalize_concept_name(title)
    parts: List[str] = []
    if title and normalized_title not in COMMON_SECTION_TITLES:
        parts.append(title)
    if snippet:
        parts.append(snippet)
    if not parts:
        return None
    return " · ".join(parts)


DATASET_HINTS = {
    "dataset",
    "data set",
    "corpus",
    "benchmark",
    "collection",
    "library",
    "suite",
    "set",
    "track",
    "challenge",
}

METRIC_HINTS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f-score",
    "bleu",
    "rouge",
    "meteor",
    "perplexity",
    "auc",
    "mae",
    "rmse",
    "error",
    "loss",
    "score",
    "psnr",
    "ssim",
}

TASK_HINTS = {
    "classification",
    "segmentation",
    "translation",
    "labeling",
    "detection",
    "regression",
    "prediction",
    "generation",
    "retrieval",
    "forecasting",
    "recognition",
    "synthesis",
    "analysis",
    "clustering",
    "matching",
}

METHOD_HINTS = {
    "network",
    "transformer",
    "model",
    "encoder",
    "decoder",
    "framework",
    "approach",
    "algorithm",
    "architecture",
    "pipeline",
    "system",
    "gan",
    "cnn",
    "rnn",
    "bert",
    "gpt",
    "resnet",
    "lstm",
    "cas9",
}

_METHOD_SUFFIX_PATTERN = re.compile(
    r"(?:(?:auto)?encoder|decoder|network|net|transformer|former|gan|cnn|rnn|lstm|bert|gpt|resnet)$",
    re.IGNORECASE,
)

KNOWN_DATASETS = {
    "imagenet",
    "coco",
    "mnist",
    "cifar10",
    "cifar-10",
    "cifar100",
    "cifar-100",
    "wmt",
    "wmt14",
    "squad",
    "wikidata",
    "openwebtext",
    "msmarco",
    "libri",
    "librispeech",
    "glue",
    "superglue",
    "kitti",
    "nyuv2",
}


def _infer_concept_type(
    phrase: str, *, entity_hints: Optional[Dict[str, Set[str]]] = None
) -> str:
    normalized = phrase.lower()
    tokens = normalized.replace("-", " ").split()
    token_set = set(tokens)

    def _has_method_cue() -> bool:
        if not tokens:
            return False
        if token_set & METHOD_HINTS:
            return True
        last_token = tokens[-1]
        if _METHOD_SUFFIX_PATTERN.search(last_token):
            return True
        if len(tokens) == 1 and _METHOD_SUFFIX_PATTERN.search(tokens[0]):
            return True
        return False

    if any(char.isdigit() for char in phrase) and len(tokens) <= 3:
        if any(hint in normalized for hint in DATASET_HINTS) or normalized.replace("-", "") in KNOWN_DATASETS:
            return "dataset"
        if _has_method_cue():
            return "method"
        return "identifier"

    if token_set & METRIC_HINTS:
        return "metric"

    if token_set & DATASET_HINTS or any(name in normalized for name in KNOWN_DATASETS):
        return "dataset"

    if token_set & TASK_HINTS or any(normalized.endswith(suffix) for suffix in TASK_HINTS):
        return "task"

    if _has_method_cue():
        return "method"

    if phrase.isupper() and len(phrase) <= 12:
        return "acronym"

    if entity_hints:
        for entity_type, hints in entity_hints.items():
            for hint in hints:
                if hint and hint in normalized:
                    return entity_type

    return "keyword"


def _final_score(candidate: _Candidate) -> float:
    return candidate.score + 0.2 * candidate.occurrences


def _build_concept_metadata(concept: ExtractedConcept) -> Dict[str, Any]:
    provenance_payload = [prov.to_payload() for prov in concept.provenance]
    metadata: Dict[str, Any] = {
        "occurrences": concept.occurrences,
        "score": concept.score,
        "provenance": provenance_payload,
    }
    return metadata


def _load_scispacy_model(
    config: ConceptExtractionRuntimeConfig,
) -> Optional[Language]:
    if spacy is None or Language is None:
        return None
    for model_name in config.scispacy_models:
        model = _load_spacy_model(model_name)
        if model is not None:
            return model
    return None


def _load_spacy_model(model_name: str) -> Optional[Language]:
    if not model_name:
        return None
    cached = _SPACY_MODEL_CACHE.get(model_name)
    if cached is False:
        return None
    if cached not in (None, False):
        return cached  # type: ignore[return-value]
    if spacy is None or Language is None:
        _SPACY_MODEL_CACHE[model_name] = False
        return None
    try:
        model = spacy.load(model_name)  # type: ignore[call-arg]
    except (OSError, IOError, ImportError):  # pragma: no cover - missing model
        _SPACY_MODEL_CACHE[model_name] = False
        return None
    _configure_scispacy_components(model)
    _SPACY_MODEL_CACHE[model_name] = model
    return model


def _configure_scispacy_components(model: Language) -> None:
    try:
        _ensure_abbreviation_detector(model)
    except Exception:  # pragma: no cover - defensive guard
        pass
    try:
        _ensure_entity_linker(model)
    except Exception:  # pragma: no cover - defensive guard
        pass


def _ensure_abbreviation_detector(model: Language) -> None:
    if "abbreviation_detector" in model.pipe_names:
        return
    try:
        model.add_pipe("abbreviation_detector")
        return
    except Exception:
        pass
    try:
        from scispacy.abbreviation import AbbreviationDetector  # type: ignore[import]
    except ImportError:  # pragma: no cover - optional dependency
        return
    try:
        abbreviation_pipe = AbbreviationDetector(model)
        model.add_pipe(abbreviation_pipe, name="abbreviation_detector")
    except Exception:  # pragma: no cover - best effort
        return


def _ensure_entity_linker(model: Language) -> None:
    existing = set(model.pipe_names)
    if {"scispacy_linker", "entity_linker"} & existing:
        return
    linker_added = False
    try:
        for linker_name in ("umls", "wikidata"):
            try:
                model.add_pipe(
                    "scispacy_linker",
                    last=True,
                    config={"resolve_abbreviations": True, "linker_name": linker_name},
                )
                linker_added = True
                break
            except Exception:
                continue
    except Exception:
        pass
    if linker_added:
        return
    try:
        from scispacy.linking import EntityLinker  # type: ignore[import]
    except ImportError:  # pragma: no cover - optional dependency
        return
    for linker_name in ("umls", "wikidata"):
        try:
            linker = EntityLinker(resolve_abbreviations=True, name=linker_name)
        except Exception:  # pragma: no cover - missing resources
            continue
        for pipe_name in ("scispacy_linker", "entity_linker"):
            try:
                model.add_pipe(linker, name=pipe_name, last=True)
                return
            except Exception:
                continue
