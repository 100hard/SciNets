from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.search import api_similarity_search
from app.models.search import SearchResult
from app.services.search import VectorStoreNotAvailableError


class FakeSearchService:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def search(self, *, query: str, limit: int) -> list[SearchResult]:
        self.calls.append((query, limit))
        return list(self._results)


class ErrorSearchService:
    async def search(self, *, query: str, limit: int) -> list[SearchResult]:
        raise VectorStoreNotAvailableError("offline")


def test_api_similarity_search_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_similarity_search_returns_results(monkeypatch))


async def _run_api_similarity_search_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    section_id = uuid4()
    expected = SearchResult(
        paper_id=paper_id,
        section_id=section_id,
        snippet="Coherent search results snippet.",
        score=0.88,
        page_number=4,
        char_start=5,
        char_end=64,
    )

    service = FakeSearchService([expected])
    monkeypatch.setattr("app.api.search.get_search_service", lambda: service)

    results = await api_similarity_search(q="coherent", limit=5)

    assert results == [expected]
    assert service.calls == [("coherent", 5)]


def test_api_similarity_search_maps_vector_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_similarity_search_maps_vector_errors(monkeypatch))


async def _run_api_similarity_search_maps_vector_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.search.get_search_service", lambda: ErrorSearchService())

    with pytest.raises(HTTPException) as exc:
        await api_similarity_search(q="unstable", limit=3)

    assert exc.value.status_code == 503
    assert "Vector store unavailable" in str(exc.value.detail)
