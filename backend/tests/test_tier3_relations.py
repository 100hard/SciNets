import copy
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from app.services.extraction_tier3_relations import (
    FALLBACK_SOURCE,
    run_tier3_relations,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_run_tier3_relations_merges_fallback_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "sections": [],
        "triple_candidates": [
            {
                "subject": "AlphaNet",
                "relation": "evaluated on",
                "object": "CIFAR-10",
                "evidence": "AlphaNet evaluates on CIFAR-10.",
                "subject_type_guess": "Method",
                "relation_type_guess": "EVALUATED_ON",
                "object_type_guess": "Dataset",
                "subject_span": [0, 8],
                "object_span": [20, 28],
            }
        ],
        "metadata": {},
    }

    summary_after_core = copy.deepcopy(base_summary)
    summary_after_core.setdefault("metadata", {})["tier3_relations"] = {
        "tier": "tier3_relations",
        "status": "completed",
    }

    async def fake_core(
        paper_id: UUID,
        *,
        base_summary: dict[str, Any],
        persist: bool,
        enable_llm_fallback: bool,
    ) -> dict[str, Any]:
        assert persist is False
        assert enable_llm_fallback is False
        return copy.deepcopy(summary_after_core)

    fallback_candidate = {
        "candidate_id": "tier3_relations_002",
        "subject": "AlphaNet++",
        "relation": "reports",
        "object": "95 accuracy",
        "evidence": "AlphaNet++ reports 95 accuracy.",
        "subject_type_guess": "Method",
        "relation_type_guess": "REPORTS",
        "object_type_guess": "Metric",
        "subject_span": [0, 10],
        "object_span": [18, 29],
        "source": FALLBACK_SOURCE,
        "tier": FALLBACK_SOURCE,
    }
    fallback_meta = {
        "triggered": True,
        "status": "succeeded",
        "attempts": 1,
        "accepted": 1,
        "errors": [],
    }

    async def fake_fallback(
        paper_id: UUID,
        summary: dict[str, Any],
        *,
        enabled: Optional[bool] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert enabled is None
        return [fallback_candidate], fallback_meta

    persisted_records: list = []

    async def fake_replace(_: UUID, payload: list) -> None:
        persisted_records.extend(payload)

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations._run_tier3_relations_core",
        fake_core,
    )
    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.maybe_apply_relation_llm_fallback",
        fake_fallback,
    )
    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.replace_triple_candidates",
        fake_replace,
    )

    result = await run_tier3_relations(paper_id, base_summary=base_summary)

    assert len(result["triple_candidates"]) == 2
    assert result["triple_candidates"][-1]["subject"] == "AlphaNet++"
    tier_meta = result["metadata"]["tier3_relations"]
    assert tier_meta["fallback"] == fallback_meta
    assert tier_meta["accepted_fallback"] == 1
    assert len(persisted_records) == 2
    assert persisted_records[-1].subject == "AlphaNet++"


@pytest.mark.anyio
async def test_run_tier3_relations_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    base_summary = {
        "sections": [],
        "triple_candidates": [],
        "metadata": {},
    }

    summary_after_core = copy.deepcopy(base_summary)
    summary_after_core.setdefault("metadata", {})["tier3_relations"] = {
        "tier": "tier3_relations",
        "status": "completed",
    }

    async def fake_core(
        paper_id: UUID,
        *,
        base_summary: dict[str, Any],
        persist: bool,
        enable_llm_fallback: bool,
    ) -> dict[str, Any]:
        return copy.deepcopy(summary_after_core)

    async def fake_fallback(
        paper_id: UUID,
        summary: dict[str, Any],
        *,
        enabled: Optional[bool] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return [], {
            "triggered": False,
            "status": "disabled",
            "attempts": 0,
            "accepted": 0,
            "errors": [],
        }

    persisted_records: list = []

    async def fake_replace(_: UUID, payload: list) -> None:
        persisted_records.extend(payload)

    monkeypatch.setattr(
        "app.services.extraction_tier3_relations._run_tier3_relations_core",
        fake_core,
    )
    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.maybe_apply_relation_llm_fallback",
        fake_fallback,
    )
    monkeypatch.setattr(
        "app.services.extraction_tier3_relations.replace_triple_candidates",
        fake_replace,
    )

    result = await run_tier3_relations(paper_id, base_summary=base_summary)

    assert result["triple_candidates"] == []
    tier_meta = result["metadata"]["tier3_relations"]
    assert tier_meta["fallback"]["status"] == "disabled"
    assert len(persisted_records) == 0


@pytest.mark.anyio
async def test_run_tier3_relations_invalid_summary_raises() -> None:
    with pytest.raises(ValueError):
        await run_tier3_relations(uuid4(), base_summary=None)  # type: ignore[arg-type]
