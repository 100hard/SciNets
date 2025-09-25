from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import List, Sequence

from app.core.config import settings
from app.models.search import SearchResult
from app.services.embeddings import (
    EmbeddingBackend,
    VectorSearchResult,
    VectorStore,
    VectorStoreNotAvailableError,
    build_embedding_backend,
)

from typing import Optional
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_MAX_LIMIT = 20


@dataclass
class _RankedMatch:
    vector_id: str
    payload: dict[str, object]
    base_score: float
    lexical_score: float
    final_score: float


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token}


def _lexical_similarity(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    if overlap == 0:
        return 0.0
    return overlap / float(len(query_tokens))


class SemanticSearchService:
    def __init__(
        self,
        *,
        backend: Optional[EmbeddingBackend] = None,
        vector_store: Optional[VectorStore] = None,
        rerank_top_k: int = 3,
        lexical_weight: float = 0.3,
        max_limit: int = _DEFAULT_MAX_LIMIT,
    ) -> None:
        self._model_name = settings.embedding_model_name
        self._dimension = max(1, settings.embedding_dimension)
        self._batch_size = max(1, settings.embedding_batch_size)
        self._backend = backend or build_embedding_backend(
            self._model_name, self._dimension
        )
        self._vector_store = vector_store or VectorStore(
            settings.qdrant_collection_name, self._dimension
        )
        self._rerank_top_k = max(0, rerank_top_k)
        self._lexical_weight = float(min(max(lexical_weight, 0.0), 1.0))
        self._max_limit = max(1, max_limit)

    async def search(self, *, query: str, limit: int = 5) -> List[SearchResult]:
        normalized = query.strip()
        if not normalized:
            return []

        limit = max(1, min(limit, self._max_limit))

        embeddings = await self._backend.embed([normalized], self._batch_size)
        if not embeddings:
            return []
        vector = list(map(float, embeddings[0]))
        if not vector:
            return []

        search_limit = max(limit, self._rerank_top_k)
        raw_results = await self._vector_store.search(vector, search_limit)
        if not raw_results:
            return []

        ranked = self._rank_results(normalized, raw_results)
        if not ranked:
            return []

        top_results = ranked[:limit]
        return [self._build_result(match) for match in top_results if match is not None]

    def _rank_results(
        self, query: str, results: Sequence[VectorSearchResult]
    ) -> List[_RankedMatch]:
        ranked: List[_RankedMatch] = []
        for result in results:
            payload = dict(result.payload)
            snippet = self._extract_snippet(payload)
            lexical_score = _lexical_similarity(query, snippet)
            final_score = self._combine_scores(result.score, lexical_score)
            ranked.append(
                _RankedMatch(
                    vector_id=result.vector_id,
                    payload=payload,
                    base_score=float(result.score),
                    lexical_score=lexical_score,
                    final_score=final_score,
                )
            )

        if not ranked:
            return []

        rerank_count = min(self._rerank_top_k, len(ranked))
        if rerank_count:
            head = sorted(
                ranked[:rerank_count],
                key=lambda item: item.final_score,
                reverse=True,
            )
            tail = ranked[rerank_count:]
            return head + tail
        return ranked

    def _combine_scores(self, base: float, lexical: float) -> float:
        base_clamped = max(-1.0, min(1.0, float(base)))
        lexical_clamped = max(0.0, min(1.0, float(lexical)))
        weight = self._lexical_weight
        return (1.0 - weight) * base_clamped + weight * lexical_clamped

    def _extract_snippet(self, payload: dict[str, object]) -> str:
        snippet = payload.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            return snippet
        content = payload.get("content")
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned
        return ""

    def _build_result(self, match: _RankedMatch) ->Optional[SearchResult]:
        paper_id = match.payload.get("paper_id")
        snippet = self._extract_snippet(match.payload)
        if not snippet or not paper_id:
            return None
        section_title = match.payload.get("section_title")
        if section_title is not None and not isinstance(section_title, str):
            section_title = str(section_title)
        return SearchResult(
            paper_id=paper_id,
            section_id=match.payload.get("section_id"),
            section_title=section_title,
            snippet=snippet,
            score=float(match.final_score),
            page_number=match.payload.get("page_number"),
            char_start=match.payload.get("char_start"),
            char_end=match.payload.get("char_end"),
        )


_service: Optional[SemanticSearchService] = None
_service_lock = threading.Lock()


def get_search_service() -> SemanticSearchService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = SemanticSearchService()
    return _service


__all__ = [
    "SemanticSearchService",
    "get_search_service",
    "VectorStoreNotAvailableError",
]