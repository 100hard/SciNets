import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

import app.services.ontology_store as ontology_store
from app.services.ontology_store import (
    _clean_aliases,
    _clean_evidence,
    _method_from_row,
)
from app.models.ontology import ResultCreate


def test_clean_aliases_from_json_string() -> None:
    raw = '["Transformer model", "Transformer architecture"]'
    assert _clean_aliases(raw) == [
        "Transformer model",
        "Transformer architecture",
    ]


def test_clean_aliases_trims_and_deduplicates() -> None:
    raw = [" Transformer model ", "Transformer model", ""]
    assert _clean_aliases(raw) == ["Transformer model"]


def test_method_from_row_coerces_alias_payload() -> None:
    now = datetime.now(timezone.utc)
    method = _method_from_row(
        {
            "id": uuid4(),
            "name": "Transformer",
            "aliases": '["Transformer model", "Transformer architecture"]',
            "description": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    assert method.aliases == [
        "Transformer model",
        "Transformer architecture",
    ]


def test_clean_evidence_filters_non_dict_entries() -> None:
    raw = '[{"text": "foo"}, {"span": 1}, "ignored"]'
    assert _clean_evidence(raw) == [{"text": "foo"}, {"span": 1}]


class _DummyConn:
    def __init__(self, rows: list[dict[str, str]]):
        self._rows = rows
        self.calls = 0

    async def fetch(self, *_: object) -> list[dict[str, str]]:
        self.calls += 1
        return self._rows


def test_results_supports_verification_detects_columns() -> None:
    ontology_store._RESULTS_VERIFICATION_SUPPORTED = None
    conn = _DummyConn([
        {"column_name": "verified"},
        {"column_name": "verifier_notes"},
    ])
    try:
        assert asyncio.run(ontology_store._results_supports_verification(conn)) is True
        assert conn.calls == 1
        # Cached value should short-circuit without re-querying
        assert asyncio.run(ontology_store._results_supports_verification(conn)) is True
        assert conn.calls == 1
    finally:
        ontology_store._RESULTS_VERIFICATION_SUPPORTED = None


def test_results_supports_verification_handles_missing_columns() -> None:
    ontology_store._RESULTS_VERIFICATION_SUPPORTED = None
    conn = _DummyConn([{ "column_name": "verified" }])
    try:
        assert asyncio.run(ontology_store._results_supports_verification(conn)) is False
    finally:
        ontology_store._RESULTS_VERIFICATION_SUPPORTED = None


def test_replace_results_downgrades_when_verification_columns_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    ontology_store._RESULTS_VERIFICATION_SUPPORTED = None
    paper_id = uuid4()

    class _DummyTransaction:
        async def __aenter__(self) -> "_DummyTransaction":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _DummyConn:
        def __init__(self) -> None:
            self.raise_once = True
            self.insert_sqls: list[str] = []

        def transaction(self) -> _DummyTransaction:
            return _DummyTransaction()

        async def execute(self, query: str, *_: object) -> None:
            return None

        async def fetch(self, *_: object) -> list[dict[str, str]]:
            return [
                {"column_name": "verified"},
                {"column_name": "verifier_notes"},
            ]

        async def fetchrow(self, sql: str, *params: object) -> dict[str, object]:
            self.insert_sqls.append(sql)
            if "verified" in sql and self.raise_once:
                self.raise_once = False
                raise asyncpg.UndefinedColumnError('column "verified" does not exist')
            now = datetime.now(timezone.utc)
            return {
                "id": uuid4(),
                "paper_id": paper_id,
                "method_id": None,
                "dataset_id": None,
                "metric_id": None,
                "task_id": None,
                "split": None,
                "value_numeric": None,
                "value_text": None,
                "is_sota": False,
                "confidence": None,
                "evidence": "[]",
                "created_at": now,
                "updated_at": now,
            }

    class _Acquire:
        def __init__(self, conn: _DummyConn) -> None:
            self._conn = conn

        async def __aenter__(self) -> _DummyConn:
            return self._conn

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _DummyPool:
        def __init__(self, conn: _DummyConn) -> None:
            self._conn = conn

        def acquire(self) -> _Acquire:
            return _Acquire(self._conn)

    conn = _DummyConn()
    pool = _DummyPool(conn)
    monkeypatch.setattr(ontology_store, "get_pool", lambda: pool)

    try:
        result = ResultCreate(paper_id=paper_id)
        inserted = asyncio.run(ontology_store.replace_results(paper_id, [result]))
        verification_flag = ontology_store._RESULTS_VERIFICATION_SUPPORTED
    finally:
        ontology_store._RESULTS_VERIFICATION_SUPPORTED = None

    assert len(conn.insert_sqls) == 2
    assert "verified" in conn.insert_sqls[0]
    assert "verified" not in conn.insert_sqls[1]
    assert len(inserted) == 1
    assert inserted[0].verified is None

