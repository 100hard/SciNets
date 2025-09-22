from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

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
