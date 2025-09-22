from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import app.services.ontology_store as ontology_store
from app.services.ontology_store import (
    _clean_aliases,
    _clean_evidence,
    _method_from_row,
)


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
