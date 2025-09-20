from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID, uuid4

import pytest

from app.core.config import settings
from app.models.evidence import EvidenceCreate
from app.models.section import Section
from app.services.embeddings import (
    EmbeddingBackend,
    EmbeddingCacheRepository,
    EmbeddingService,
)


class FakeBackend(EmbeddingBackend):
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls: List[List[str]] = []

    async def embed(self, texts: List[str], batch_size: int) -> List[List[float]]:  # type: ignore[override]
        self.calls.append(list(texts))
        return [
            [float(len(text) + index) for index in range(self.dimension)]
            for text in texts
        ]


class FakeCache(EmbeddingCacheRepository):
    def __init__(self, initial: Dict[str, List[float]] | None = None) -> None:
        self.store: Dict[str, List[float]] = dict(initial or {})
        self.upserts: List[Dict[str, List[float]]] = []

    async def get_many(self, model: str, hashes: List[str]) -> Dict[str, List[float]]:  # type: ignore[override]
        return {text_hash: self.store[text_hash] for text_hash in hashes if text_hash in self.store}

    async def upsert_many(self, model: str, values: Dict[str, List[float]]) -> None:  # type: ignore[override]
        self.upserts.append(dict(values))
        self.store.update({key: list(map(float, value)) for key, value in values.items()})


class FakeVectorStore:
    def __init__(self) -> None:
        self.replace_calls: List[tuple[UUID, List[Any], int]] = []
        self.delete_calls: List[UUID] = []
        self._lock = threading.Lock()

    async def replace_for_paper(
        self, paper_id: UUID, records: List[Any], batch_size: int
    ) -> None:
        with self._lock:
            self.replace_calls.append((paper_id, list(records), batch_size))

    async def delete_for_paper(self, paper_id: UUID) -> None:
        with self._lock:
            self.delete_calls.append(paper_id)


class FakeEvidenceStore:
    def __init__(self) -> None:
        self.deleted: List[UUID] = []
        self.created: List[EvidenceCreate] = []

    async def delete(self, paper_id: UUID) -> None:
        self.deleted.append(paper_id)

    async def create(self, payload: EvidenceCreate) -> EvidenceCreate:
        self.created.append(payload)
        return payload


@pytest.fixture
def configure_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model_name", "test-model", raising=False)
    monkeypatch.setattr(settings, "embedding_dimension", 3, raising=False)
    monkeypatch.setattr(settings, "embedding_batch_size", 2, raising=False)
    monkeypatch.setattr(settings, "qdrant_collection_name", "test-collection", raising=False)
    monkeypatch.setattr(settings, "qdrant_upsert_batch_size", 2, raising=False)


def _make_section(paper_id: UUID, content: str, *, snippet: str | None = None) -> Section:
    now = datetime.now(timezone.utc)
    return Section(
        id=uuid4(),
        paper_id=paper_id,
        title="Section",
        content=content,
        char_start=0,
        char_end=len(content),
        page_number=1,
        snippet=snippet,
        created_at=now,
        updated_at=now,
    )


def test_embed_service_uses_cache_and_batches(configure_settings: None) -> None:
    asyncio.run(_run_embed_service_uses_cache_and_batches())


async def _run_embed_service_uses_cache_and_batches() -> None:
    paper_id = uuid4()
    cached_text = "Cached content"
    new_text = "Fresh content"

    cached_hash = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
    cache = FakeCache({cached_hash: [0.1, 0.2, 0.3]})
    backend = FakeBackend(dimension=3)
    vector_store = FakeVectorStore()
    evidence_store = FakeEvidenceStore()

    sections = [
        _make_section(paper_id, cached_text, snippet="cached snippet"),
        _make_section(paper_id, new_text, snippet=None),
        _make_section(paper_id, new_text, snippet="duplicate snippet"),
    ]

    async def list_sections(*, paper_id: UUID, limit: int, offset: int) -> List[Section]:
        assert limit >= len(sections)
        return sections if offset == 0 else []

    service = EmbeddingService(
        backend=backend,
        cache=cache,
        vector_store=vector_store,  # type: ignore[arg-type]
        sections_fetcher=list_sections,
        create_evidence_fn=evidence_store.create,
        delete_evidence_fn=evidence_store.delete,
    )

    await service.embed_paper_sections(paper_id)

    assert evidence_store.deleted == [paper_id]
    assert len(evidence_store.created) == 3

    # Only the uncached text should have triggered an embedding call.
    assert backend.calls == [[new_text]]

    new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    assert new_hash in cache.store
    assert cache.upserts and new_hash in cache.upserts[0]

    assert len(vector_store.replace_calls) == 1
    replace_paper_id, records, batch_size = vector_store.replace_calls[0]
    assert replace_paper_id == paper_id
    assert batch_size == settings.qdrant_upsert_batch_size
    assert len(records) == 3
    assert all(len(record.vector) == settings.embedding_dimension for record in records)


def test_embed_service_handles_empty_sections(configure_settings: None) -> None:
    asyncio.run(_run_embed_service_handles_empty_sections())


async def _run_embed_service_handles_empty_sections() -> None:
    paper_id = uuid4()
    cache = FakeCache()
    backend = FakeBackend(dimension=3)
    vector_store = FakeVectorStore()
    evidence_store = FakeEvidenceStore()

    async def list_sections(*, paper_id: UUID, limit: int, offset: int) -> List[Section]:
        return []

    service = EmbeddingService(
        backend=backend,
        cache=cache,
        vector_store=vector_store,  # type: ignore[arg-type]
        sections_fetcher=list_sections,
        create_evidence_fn=evidence_store.create,
        delete_evidence_fn=evidence_store.delete,
    )

    await service.embed_paper_sections(paper_id)

    assert backend.calls == []
    assert cache.upserts == []
    assert vector_store.replace_calls == []
    assert vector_store.delete_calls == [paper_id]
    assert evidence_store.deleted == [paper_id]
    assert evidence_store.created == []
