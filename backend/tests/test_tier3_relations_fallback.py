import json
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services.extraction_tier3_relations import (
    FALLBACK_SOURCE,
    maybe_apply_relation_llm_fallback,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(section_id: str, text: str) -> dict[str, object]:
    return {
        "section_id": section_id,
        "section_hash": f"hash-{section_id}",
        "page_number": 1,
        "char_start": 0,
        "char_end": len(text),
        "sentence_spans": [
            {
                "start": 0,
                "end": len(text),
                "text": text,
            }
        ],
    }


@pytest.mark.anyio
async def test_tier3_fallback_generates_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tier3_relation_fallback_enabled", True)
    monkeypatch.setattr(settings, "tier3_relation_fallback_min_rule_candidates", 2)
    monkeypatch.setattr(settings, "tier3_relation_fallback_max_attempts", 2)

    paper_id = uuid4()
    summary = {
        "sections": [
            _build_section(
                "sec-1",
                "AlphaNet evaluates on CIFAR-10 and achieves 95.2% accuracy on the test set.",
            )
        ],
        "triple_candidates": [],
    }

    payload = {
        "triples": [
            {
                "subject": "AlphaNet",
                "relation": "evaluated on",
                "object": "CIFAR-10",
                "evidence": "AlphaNet evaluates on CIFAR-10 and achieves 95.2% accuracy on the test set.",
                "subject_span": [0, 8],
                "object_span": [19, 27],
                "relation_type_guess": "EVALUATED_ON",
                "subject_type_guess": "Method",
                "object_type_guess": "Dataset",
                "section_id": "sec-1",
                "chunk_id": "sec-1#chunk-01",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    async def fake_invoke(messages, *, temperature=None):  # type: ignore[override]
        return json.dumps(payload)

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.tier2._invoke_llm",
        fake_invoke,
    )

    candidates, meta = await maybe_apply_relation_llm_fallback(
        paper_id,
        summary,
    )

    assert meta["triggered"] is True
    assert meta["status"] == "succeeded"
    assert meta["accepted"] == 1
    assert meta.get("errors") == []

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["source"] == FALLBACK_SOURCE
    assert candidate["tier"] == FALLBACK_SOURCE
    assert candidate["retry_count"] == 0
    assert candidate["provenance"] == FALLBACK_SOURCE


@pytest.mark.anyio
async def test_tier3_fallback_retries_after_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tier3_relation_fallback_enabled", True)
    monkeypatch.setattr(settings, "tier3_relation_fallback_min_rule_candidates", 1)
    monkeypatch.setattr(settings, "tier3_relation_fallback_max_attempts", 3)

    paper_id = uuid4()
    summary = {
        "sections": [
            _build_section(
                "sec-2",
                "BetaNet reports 87.4 F1 on the dev split of the CustomEval benchmark.",
            )
        ],
        "triple_candidates": [],
    }

    valid_payload = {
        "triples": [
            {
                "subject": "BetaNet",
                "relation": "reports",
                "object": "87.4 F1 on CustomEval dev",
                "evidence": "BetaNet reports 87.4 F1 on the dev split of the CustomEval benchmark.",
                "subject_span": [0, 7],
                "object_span": [15, 40],
                "relation_type_guess": "REPORTS",
                "subject_type_guess": "Method",
                "object_type_guess": "Metric",
                "section_id": "sec-2",
                "chunk_id": "sec-2#chunk-01",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    responses = iter([
        "{",  # invalid JSON to trigger retry
        json.dumps(valid_payload),
    ])

    async def fake_invoke(messages, *, temperature=None):  # type: ignore[override]
        return next(responses)

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.tier2._invoke_llm",
        fake_invoke,
    )

    candidates, meta = await maybe_apply_relation_llm_fallback(
        paper_id,
        summary,
    )

    assert meta["triggered"] is True
    assert meta["status"] == "succeeded"
    assert meta["attempts"] == 2
    assert meta["accepted"] == 1
    assert meta["errors"]

    candidate = candidates[0]
    assert candidate["retry_count"] == 1
    assert candidate["fallback_attempt"] == 2
    assert candidate["source"] == FALLBACK_SOURCE
