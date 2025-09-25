from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from ..core.config import settings

from typing import Optional
logger = logging.getLogger(__name__)

_PIPELINE_SPEC_DELIMITER = ","
_PIPELINE_SPEC_SEPARATOR = ":"
_CACHE_VERSION = "v1"


@dataclass(frozen=True)
class PipelineHandle:
    """Holds metadata and the loaded spaCy pipeline."""

    key: str
    kind: str
    model: str
    nlp: Language


@dataclass
class CachedDoc:
    pipeline: PipelineHandle
    doc: Doc
    from_cache: bool


class DocCache:
    """Manages persistent Doc storage so NLP runs can be reused."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = (base_dir / _CACHE_VERSION).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[tuple[str, str], Doc] = {}

    def load(self, pipeline_key: str, text_hash: str, nlp: Language) ->Optional[Doc]:
        cache_key = (pipeline_key, text_hash)
        if cache_key in self._memory:
            return self._memory[cache_key]
        path = self._path_for(pipeline_key, text_hash)
        if not path.exists():
            return None
        try:
            doc_bin = DocBin().from_bytes(path.read_bytes())
            docs = doc_bin.get_docs(nlp.vocab)
            doc = next(docs, None)
        except Exception as exc:  # pragma: no cover - best effort cache
            logger.warning(
                "[nlp-cache] failed to load cached doc pipeline=%s hash=%s error=%s",
                pipeline_key,
                text_hash,
                exc,
            )
            return None
        if doc is None:
            return None
        self._memory[cache_key] = doc
        return doc

    def store(self, pipeline_key: str, text_hash: str, doc: Doc) -> None:
        cache_key = (pipeline_key, text_hash)
        self._memory[cache_key] = doc
        path = self._path_for(pipeline_key, text_hash)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc_bin = DocBin(store_user_data=True)
            doc_bin.add(doc)
            path.write_bytes(doc_bin.to_bytes())
        except Exception as exc:  # pragma: no cover - best effort cache
            logger.warning(
                "[nlp-cache] failed to persist doc pipeline=%s hash=%s error=%s",
                pipeline_key,
                text_hash,
                exc,
            )

    def _path_for(self, pipeline_key: str, text_hash: str) -> Path:
        safe_pipeline = pipeline_key.replace("/", "_")
        return self._base_dir / safe_pipeline / f"{text_hash}.spacy"


def get_doc_cache() -> DocCache:
    return _get_doc_cache(settings.nlp_cache_dir)


@lru_cache(maxsize=4)
def _get_doc_cache(cache_dir: str) -> DocCache:
    return DocCache(Path(cache_dir))


def hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def process_text(text: str) -> List[CachedDoc]:
    text_hash = hash_text(text)
    cache = get_doc_cache()
    docs: List[CachedDoc] = []
    for pipeline in get_pipelines():
        doc = cache.load(pipeline.key, text_hash, pipeline.nlp)
        from_cache = doc is not None
        if doc is None:
            doc = pipeline.nlp(text)
            cache.store(pipeline.key, text_hash, doc)
        docs.append(CachedDoc(pipeline=pipeline, doc=doc, from_cache=from_cache))
    return docs


@lru_cache(maxsize=1)
def get_pipelines() -> Tuple[PipelineHandle, ...]:
    handles: list[PipelineHandle] = []
    for kind, model in _parse_pipeline_spec(settings.nlp_pipeline_spec):
        nlp = _load_pipeline(kind, model)
        key = _normalise_key(kind, model)
        handles.append(PipelineHandle(key=key, kind=kind, model=model, nlp=nlp))
    if not handles:
        raise RuntimeError("No NLP pipelines configured; check NLP_PIPELINE_SPEC")
    return tuple(handles)


def _parse_pipeline_spec(spec: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for raw in spec.split(_PIPELINE_SPEC_DELIMITER):
        token = raw.strip()
        if not token:
            continue
        if _PIPELINE_SPEC_SEPARATOR in token:
            kind, model = token.split(_PIPELINE_SPEC_SEPARATOR, maxsplit=1)
        else:
            kind, model = "spacy", token
        items.append((kind.strip(), model.strip()))
    return items


def _normalise_key(kind: str, model: str) -> str:
    safe = model.replace(":", "_").replace("/", "_")
    return f"{kind}_{safe}"


def _load_pipeline(kind: str, model: str) -> Language:
    logger.info("[nlp] loading pipeline kind=%s model=%s", kind, model)
    nlp = spacy.load(model)
    if kind.lower() == "scispacy" and "abbreviation_detector" not in nlp.pipe_names:
        try:
            nlp.add_pipe("abbreviation_detector")
        except ValueError:
            logger.warning("[nlp] abbreviation_detector unavailable for model=%s", model)
    if "scinets_normalizer" not in nlp.pipe_names:
        nlp.add_pipe("scinets_normalizer", name="scinets_normalizer", first=True)
    return nlp


@Language.component("scinets_normalizer")
def scinets_normalizer(doc: Doc) -> Doc:
    """Derive a lightly-normalized variant of the text for downstream rules."""

    if not Doc.has_extension("scinets_normalized_text"):
        Doc.set_extension("scinets_normalized_text", default=None, force=True)
    if doc._.scinets_normalized_text is None:
        normalized = _normalize_text(doc.text)
        doc._.scinets_normalized_text = normalized
    return doc


def _normalize_text(text: str) -> str:
    # Basic cleanup: collapse whitespace and drop numerical citations like [12].
    import re

    without_citations = re.sub(r"\[(?:\d+[\-,\s]*)+\]", " ", text)
    collapsed = re.sub(r"\s+", " ", without_citations)
    return collapsed.strip()