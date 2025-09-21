from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from app.models.concept import Concept, ConceptCreate
from app.models.section import SectionBase
from app.services.concepts import replace_concepts
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


MAX_CONCEPTS = 50
MAX_TOKENS = 6
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

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "if",
    "but",
    "on",
    "in",
    "into",
    "by",
    "for",
    "of",
    "with",
    "we",
    "our",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "these",
    "those",
    "using",
    "used",
    "use",
    "can",
    "may",
    "might",
    "should",
    "could",
    "to",
    "from",
    "as",
    "at",
    "be",
    "been",
    "it",
    "its",
    "their",
    "they",
    "them",
    "than",
    "then",
    "also",
    "such",
    "however",
    "between",
    "within",
    "through",
    "across",
    "each",
    "both",
    "either",
    "neither",
    "because",
    "due",
    "after",
    "before",
    "over",
    "under",
    "more",
    "most",
    "less",
    "least",
    "many",
    "much",
    "several",
    "various",
    "against",
    "include",
    "includes",
    "including",
    "based",
    "extend",
    "extends",
    "extending",
    "extended",
    "leveraging",
    "leverage",
    "leverages",
    "utilizing",
    "utilize",
    "utilizes",
    "via",
    "around",
    "among",
    "amongst",
    "towards",
    "toward",
    "accompanying",
    "accompanied",
    "accompanies",
    "compared",
    "comparing",
    "compare",
    "compares",
}

SCISPACY_MODEL_NAMES = (
    "en_core_sci_sm",
    "en_core_sci_md",
    "en_core_web_sm",
)

_SCISPACY_MODEL: Optional[Language] | bool = None

FILLER_PREFIXES = {
    "baseline",
    "baselines",
    "compare",
    "compares",
    "compared",
    "comparing",
    "proposed",
    "propose",
    "proposes",
    "introduce",
    "introduces",
    "introducing",
    "novel",
    "new",
    "simple",
    "improved",
    "fast",
    "robust",
    "efficient",
    "effective",
    "powerful",
    "general",
}

FILLER_SUFFIXES = {
    "approach",
    "approaches",
    "method",
    "methods",
    "technique",
    "techniques",
    "architecture",
    "architectures",
    "pipeline",
    "pipelines",
    "framework",
    "frameworks",
    "model",
    "models",
    "system",
    "systems",
    "strategy",
    "strategies",
    "procedure",
    "procedures",
    "scheme",
    "schemes",
}


@dataclass
class ExtractedConcept:
    name: str
    type: Optional[str]
    description: Optional[str]
    score: float


@dataclass
class _Candidate:
    name: str
    normalized: str
    type: Optional[str]
    description: Optional[str]
    score: float
    occurrences: int = 1


def extract_concepts_from_sections(
    sections: Sequence[SectionBase],
) -> List[ExtractedConcept]:
    filtered = [section for section in sections if section.content.strip()]
    if not filtered:
        return []

    candidates: Dict[str, _Candidate] = {}

    model = _load_scispacy_model()
    if model is not None:
        for candidate in _extract_with_scispacy(model, filtered):
            _merge_candidate(candidates, candidate)

    for candidate in _extract_with_heuristics(filtered):
        _merge_candidate(candidates, candidate)

    if not candidates:
        return []

    ranked = sorted(
        candidates.values(), key=lambda item: (-_final_score(item), item.name.lower())
    )
    top_ranked = ranked[:MAX_CONCEPTS]
    return [
        ExtractedConcept(
            name=candidate.name,
            type=candidate.type,
            description=candidate.description,
            score=round(_final_score(candidate), 4),
        )
        for candidate in top_ranked
    ]


async def extract_and_store_concepts(
    paper_id: UUID,
    sections: Optional[Sequence[SectionBase]] = None,
) -> List[Concept]:
    if sections is None:
        sections = await _load_sections_for_paper(paper_id)

    concepts = extract_concepts_from_sections(sections)
    concept_models = [
        ConceptCreate(
            paper_id=paper_id,
            name=concept.name,
            type=concept.type,
            description=concept.description,
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


def _extract_with_scispacy(
    model: Language, sections: Sequence[SectionBase]
) -> Iterable[_Candidate]:
    for section in sections:
        doc = model(section.content)
        for entity in getattr(doc, "ents", []):
            name = _format_phrase(entity.text)
            normalized = _normalize_concept_name(name)
            if not normalized:
                continue
            snippet = _build_snippet(section.content, entity.start_char, entity.end_char)
            description = _compose_description(section, snippet)
            label = (entity.label_ or "entity").lower()
            score = 4.0 + min(len(normalized.split()), 4)
            yield _Candidate(
                name=name,
                normalized=normalized,
                type=label,
                description=description,
                score=score,
            )


def _extract_with_heuristics(sections: Sequence[SectionBase]) -> Iterable[_Candidate]:
    for section in sections:
        for phrase, start, end in _iter_phrases(section.content):
            normalized = _normalize_concept_name(phrase)
            if not normalized:
                continue
            snippet = _build_snippet(section.content, start, end)
            description = _compose_description(section, snippet)
            score = 1.0 + 0.3 * min(len(normalized.split()), 4)
            score += 0.2 if section.title else 0.0
            score += 0.4 if phrase.isupper() else 0.0
            yield _Candidate(
                name=_format_phrase(phrase),
                normalized=normalized,
                type=_infer_concept_type(phrase),
                description=description,
                score=score,
            )


def _iter_phrases(text: str) -> Iterable[Tuple[str, int, int]]:
    tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]*", text))
    current: List[Tuple[str, int, int]] = []
    for match in tokens:
        token = match.group(0)
        normalized = token.lower()
        if normalized in STOPWORDS:
            yield from _flush_phrase(current)
            current = []
            continue
        current.append((token, match.start(), match.end()))
        if len(current) >= MAX_TOKENS:
            yield from _flush_phrase(current)
            current = []
    yield from _flush_phrase(current)


def _flush_phrase(
    current: List[Tuple[str, int, int]]
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
    cleaned = _normalize_concept_name(phrase)
    if not cleaned or len(cleaned) < MIN_CHAR_LENGTH:
        return []
    if not any(char.isalpha() for char in cleaned):
        return []
    return [(phrase, start, end)]


def _normalize_concept_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip()
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\(\)\[\]\{\}]+", " ", normalized)
    normalized = re.sub(r"[^A-Za-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    if not normalized:
        return ""
    parts = normalized.split()
    trimmed = _trim_tokens(parts)
    if trimmed:
        parts = trimmed
    stemmed = []
    for part in parts:
        if len(part) > 4 and part.endswith("s"):
            stemmed.append(part[:-1])
        else:
            stemmed.append(part)
    return " ".join(stemmed)


def _trim_tokens(tokens: List[str]) -> List[str]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in FILLER_PREFIXES:
        start += 1
    while end > start and tokens[end - 1] in FILLER_SUFFIXES:
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


def _infer_concept_type(phrase: str) -> str:
    if any(char.isdigit() for char in phrase):
        return "identifier"
    if phrase.isupper():
        return "acronym"
    tokens = phrase.replace("-", " ").split()
    if len(tokens) >= 3:
        return "method"
    return "keyword"


def _final_score(candidate: _Candidate) -> float:
    return candidate.score + 0.2 * candidate.occurrences


def _load_scispacy_model() -> Optional[Language]:
    global _SCISPACY_MODEL
    if _SCISPACY_MODEL is False:
        return None
    if _SCISPACY_MODEL not in (None, False):
        return _SCISPACY_MODEL  # type: ignore[return-value]
    if spacy is None or Language is None:
        _SCISPACY_MODEL = False
        return None
    for model_name in SCISPACY_MODEL_NAMES:
        try:
            model = spacy.load(model_name)  # type: ignore[call-arg]
        except (OSError, IOError, ImportError):  # pragma: no cover - missing model
            continue
        _SCISPACY_MODEL = model
        return model
    _SCISPACY_MODEL = False
    return None
