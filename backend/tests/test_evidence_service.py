from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple
from uuid import uuid4

import pytest

from app.models.evidence import EvidenceCreate
from app.services import evidence as evidence_service


class FakeConnection:
    def __init__(self) -> None:
        self.fetch_calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.fetchrow_calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.rows_to_return: Sequence[dict[str, Any]] = []
        self.row_to_return: Optional[dict[str, Any]] = None

    async def fetch(self, query: str, *params: Any) -> Sequence[dict[str, Any]]:
        self.fetch_calls.append((query, params))
        return list(self.rows_to_return)

    async def fetchrow(self, query: str, *params: Any) -> Optional[dict[str, Any]]:
        self.fetchrow_calls.append((query, params))
        return self.row_to_return

    async def execute(self, query: str, *params: Any) -> None:
        return None


class _AcquireContext:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._connection)


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakeConnection:
    connection = FakeConnection()
    pool = FakePool(connection)
    monkeypatch.setattr(evidence_service, "get_pool", lambda: pool)
    return connection


def _base_row(**overrides: Any) -> dict[str, Any]:
    paper_id = overrides.get("paper_id", uuid4())
    section_id = overrides.get("section_id")
    concept_id = overrides.get("concept_id")
    relation_id = overrides.get("relation_id")
    now = datetime.now(timezone.utc)
    row = {
        "id": overrides.get("id", uuid4()),
        "paper_id": paper_id,
        "section_id": section_id,
        "concept_id": concept_id,
        "relation_id": relation_id,
        "snippet": overrides.get("snippet", "Snippet"),
        "vector_id": overrides.get("vector_id", "vector"),
        "embedding_model": overrides.get("embedding_model", "model"),
        "score": overrides.get("score"),
        "metadata": overrides.get("metadata"),
        "provenance": overrides.get("provenance", {}),
        "created_at": overrides.get("created_at", now),
        "updated_at": overrides.get("updated_at", now),
    }
    return row


def test_create_evidence_serializes_metadata(fake_pool: FakeConnection) -> None:
    asyncio.run(_run_create_evidence_serializes_metadata(fake_pool))


async def _run_create_evidence_serializes_metadata(fake_pool: FakeConnection) -> None:
    paper_id = uuid4()
    metadata = {"text_hash": "abc"}
    provenance = {"source": "pipeline"}
    fake_pool.row_to_return = _base_row(
        paper_id=paper_id,
        metadata=metadata,
        provenance=provenance,
    )
    payload = EvidenceCreate(
        paper_id=paper_id,
        section_id=None,
        concept_id=None,
        relation_id=None,
        snippet="Snippet",
        vector_id="vector",
        embedding_model="model",
        score=0.5,
        metadata=metadata,
        provenance=provenance,
    )

    evidence = await evidence_service.create_evidence(payload)

    assert fake_pool.fetchrow_calls
    _, params = fake_pool.fetchrow_calls[-1]
    assert params[-2] == metadata
    assert params[-1] == provenance
    assert evidence.metadata == metadata
    assert evidence.provenance == provenance


def test_list_evidence_decodes_metadata(fake_pool: FakeConnection) -> None:
    asyncio.run(_run_list_evidence_decodes_metadata(fake_pool))


async def _run_list_evidence_decodes_metadata(fake_pool: FakeConnection) -> None:
    metadata = {"value": 42}
    fake_pool.rows_to_return = [
        _base_row(metadata=metadata, provenance={"stage": "test"}),
        _base_row(metadata=None, provenance=None),
    ]

    results = await evidence_service.list_evidence(uuid4())

    assert len(results) == 2
    assert results[0].metadata == metadata
    assert results[1].metadata is None
    assert results[0].provenance == {"stage": "test"}
    assert results[1].provenance == {}


def test_get_evidence_handles_invalid_metadata(fake_pool: FakeConnection) -> None:
    asyncio.run(_run_get_evidence_handles_invalid_metadata(fake_pool))


async def _run_get_evidence_handles_invalid_metadata(fake_pool: FakeConnection) -> None:
    fake_pool.row_to_return = _base_row(metadata="not-json", provenance=None)

    evidence = await evidence_service.get_evidence(uuid4())

    assert evidence is not None
    assert evidence.metadata is None
    assert evidence.provenance == {}
