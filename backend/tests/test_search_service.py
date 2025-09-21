from __future__ import annotations

import asyncio
from typing import List
from uuid import uuid4

import pytest

from app.core.config import settings
from app.models.search import SearchResult
from app.services.embeddings import EmbeddingBackend, VectorSearchResult
from app.services.search import SemanticSearchService


class FakeBackend(EmbeddingBackend):
    def __init__(self, vector: List[float]) -> None:
        self._vector = list(vector)
        self.calls: List[tuple[List[str], int]] = []

    async def embed(self, texts: List[str], batch_size: int) -> List[List[float]]:  # type: ignore[override]
        self.calls.append((list(texts), batch_size))
        return [list(self._vector) for _ in texts]


class FakeVectorStore:
    def __init__(self, results: List[VectorSearchResult]) -> None:
        self._results = list(results)
        self.calls: List[tuple[List[float], int]] = []

    async def search(self, vector: List[float], limit: int) -> List[VectorSearchResult]:
        self.calls.append((list(vector), limit))
        return list(self._results)


@pytest.fixture
def configure_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension", 4, raising=False)
    monkeypatch.setattr(settings, "embedding_batch_size", 8, raising=False)
    monkeypatch.setattr(settings, "qdrant_collection_name", "test-search", raising=False)


def test_similarity_search_reranks_results(configure_settings: None) -> None:
    asyncio.run(_run_similarity_search_reranks_results())


async def _run_similarity_search_reranks_results() -> None:
    query = "quantum entanglement"
    backend = FakeBackend([0.1, 0.2, 0.3, 0.4])

    paper_ids = [uuid4() for _ in range(3)]
    section_ids = [uuid4() for _ in range(3)]

    vector_results = [
        VectorSearchResult(
            vector_id="vec-1",
            score=0.82,
            payload={
                "paper_id": str(paper_ids[0]),
                "section_id": str(section_ids[0]),
                "snippet": "Background on general relativity and spacetime curvature.",
                "page_number": 3,
            },
        ),
        VectorSearchResult(
            vector_id="vec-2",
            score=0.78,
            payload={
                "paper_id": str(paper_ids[1]),
                "section_id": str(section_ids[1]),
                "snippet": "We explore quantum entanglement experiments across photonic lattices.",
                "page_number": 7,
                "char_start": 12,
                "char_end": 86,
            },
        ),
        VectorSearchResult(
            vector_id="vec-3",
            score=0.74,
            payload={
                "paper_id": str(paper_ids[2]),
                "section_id": str(section_ids[2]),
                "content": "Entangled particles show non-classical correlations in solid-state systems.",
                "page_number": 10,
            },
        ),
    ]

    vector_store = FakeVectorStore(vector_results)
    service = SemanticSearchService(
        backend=backend,
        vector_store=vector_store,  # type: ignore[arg-type]
        rerank_top_k=3,
        lexical_weight=0.3,
    )

    results = await service.search(query=query, limit=2)

    assert backend.calls == [([query], settings.embedding_batch_size)]
    assert vector_store.calls and vector_store.calls[0][1] >= 2

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].paper_id == paper_ids[1]
    assert results[0].section_id == section_ids[1]
    assert "quantum entanglement" in results[0].snippet.lower()
    assert results[0].score > results[1].score


def test_similarity_search_handles_blank_queries(configure_settings: None) -> None:
    asyncio.run(_run_similarity_search_handles_blank_queries())


async def _run_similarity_search_handles_blank_queries() -> None:
    backend = FakeBackend([0.5, 0.5, 0.5, 0.5])
    vector_store = FakeVectorStore([])
    service = SemanticSearchService(
        backend=backend,
        vector_store=vector_store,  # type: ignore[arg-type]
    )

    results = await service.search(query="   \n\t", limit=5)

    assert results == []
    assert backend.calls == []
    assert vector_store.calls == []
