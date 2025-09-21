from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models.ontology import Result
from app.models.paper import Paper
from app.models.section import Section
from app.services.extraction_tier3 import run_tier3_verifier


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(section_id: UUID, content: str) -> Section:
    now = datetime.now(timezone.utc)
    return Section.model_construct(
        id=section_id,
        paper_id=uuid4(),
        title="Results",
        content=content,
        char_start=0,
        char_end=len(content),
        page_number=1,
        snippet=content[:120],
        created_at=now,
        updated_at=now,
    )


def _build_paper(paper_id: UUID) -> Paper:
    now = datetime.now(timezone.utc)
    return Paper.model_construct(
        id=paper_id,
        title="Verified Evaluations",
        authors="Doe et al.",
        venue="TestConf",
        year=2024,
        status="parsed",
        file_path=None,
        file_name="paper.pdf",
        file_size=1024,
        file_content_type="application/pdf",
        created_at=now,
        updated_at=now,
    )


def _build_summary_payload(
    *,
    paper_id: UUID,
    method_id: UUID,
    dataset_id: UUID,
    metric_id: UUID,
    section_id: UUID,
    value_numeric: float | None,
    value_text: str | None,
    confidence: float,
    metric_name: str,
    metric_unit: str | None = None,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "paper_id": str(paper_id),
        "tiers": [1, 2],
        "methods": [
            {
                "id": str(method_id),
                "name": "Test Method",
                "aliases": [],
                "description": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        ],
        "datasets": [
            {
                "id": str(dataset_id),
                "name": "Test Dataset",
                "aliases": [],
                "description": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        ],
        "metrics": [
            {
                "id": str(metric_id),
                "name": metric_name,
                "unit": metric_unit,
                "aliases": [],
                "description": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        ],
        "tasks": [],
        "results": [
            {
                "id": str(uuid4()),
                "paper_id": str(paper_id),
                "method": {"id": str(method_id)},
                "dataset": {"id": str(dataset_id)},
                "metric": {"id": str(metric_id)},
                "task": None,
                "split": "test",
                "value_numeric": value_numeric,
                "value_text": value_text,
                "is_sota": False,
                "confidence": confidence,
                "evidence": [
                    {
                        "tier": "llm_structurer",
                        "evidence_span": {
                            "section_id": str(section_id),
                            "start": 10,
                            "end": 60,
                        },
                    }
                ],
            }
        ],
        "claims": [],
    }


@pytest.mark.anyio("asyncio")
async def test_tier3_verifier_confirms_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    method_id = uuid4()
    dataset_id = uuid4()
    metric_id = uuid4()
    section_id = uuid4()

    paper = _build_paper(paper_id)
    section = _build_section(section_id, "The model reaches 87% accuracy on the benchmark.")
    summary = _build_summary_payload(
        paper_id=paper_id,
        method_id=method_id,
        dataset_id=dataset_id,
        metric_id=metric_id,
        section_id=section_id,
        value_numeric=0.87,
        value_text="0.87",
        confidence=0.7,
        metric_name="Accuracy",
        metric_unit="%",
    )

    stored_results: list[Result] = []

    async def fake_get_paper(_: UUID) -> Paper:
        return paper

    async def fake_list_sections(*_: object, **__: object) -> list[Section]:
        return [section]

    async def fake_replace_results(_: UUID, models) -> list[Result]:
        stored_results.clear()
        now = datetime.now(timezone.utc)
        for item in models:
            result = Result.model_construct(
                id=uuid4(),
                paper_id=paper_id,
                method_id=item.method_id,
                dataset_id=item.dataset_id,
                metric_id=item.metric_id,
                task_id=item.task_id,
                split=item.split,
                value_numeric=item.value_numeric,
                value_text=item.value_text,
                is_sota=item.is_sota,
                confidence=item.confidence,
                evidence=item.evidence,
                verified=item.verified,
                verifier_notes=item.verifier_notes,
                created_at=now,
                updated_at=now,
            )
            stored_results.append(result)
        return stored_results

    monkeypatch.setattr("app.services.extraction_tier3.get_paper", fake_get_paper)
    monkeypatch.setattr("app.services.extraction_tier3.list_sections", fake_list_sections)
    monkeypatch.setattr("app.services.extraction_tier3.replace_results", fake_replace_results)

    verified_summary = await run_tier3_verifier(paper_id, base_summary=summary)

    assert stored_results
    result = verified_summary["results"][0]
    assert result["verified"] is True
    assert pytest.approx(result["value_numeric"], rel=1e-5) == 87.0
    assert result["confidence"] == pytest.approx(0.8, rel=1e-5)
    assert "scaled fractional percent" in (result["verifier_notes"] or "")
    assert verified_summary["audit_log"][0]["verified"] is True


@pytest.mark.anyio("asyncio")
async def test_tier3_verifier_flags_outlier(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    method_id = uuid4()
    dataset_id = uuid4()
    metric_id = uuid4()
    section_id = uuid4()

    paper = _build_paper(paper_id)
    section = _build_section(section_id, "Reported BLEU score of 125 on validation.")
    summary = _build_summary_payload(
        paper_id=paper_id,
        method_id=method_id,
        dataset_id=dataset_id,
        metric_id=metric_id,
        section_id=section_id,
        value_numeric=125,
        value_text="125",
        confidence=0.6,
        metric_name="BLEU",
    )

    async def fake_get_paper(_: UUID) -> Paper:
        return paper

    async def fake_list_sections(*_: object, **__: object) -> list[Section]:
        return [section]

    async def fake_replace_results(_: UUID, models) -> list[Result]:
        now = datetime.now(timezone.utc)
        return [
            Result.model_construct(
                id=uuid4(),
                paper_id=paper_id,
                method_id=item.method_id,
                dataset_id=item.dataset_id,
                metric_id=item.metric_id,
                task_id=item.task_id,
                split=item.split,
                value_numeric=item.value_numeric,
                value_text=item.value_text,
                is_sota=item.is_sota,
                confidence=item.confidence,
                evidence=item.evidence,
                verified=item.verified,
                verifier_notes=item.verifier_notes,
                created_at=now,
                updated_at=now,
            )
            for item in models
        ]

    monkeypatch.setattr("app.services.extraction_tier3.get_paper", fake_get_paper)
    monkeypatch.setattr("app.services.extraction_tier3.list_sections", fake_list_sections)
    monkeypatch.setattr("app.services.extraction_tier3.replace_results", fake_replace_results)

    verified_summary = await run_tier3_verifier(paper_id, base_summary=summary)

    result = verified_summary["results"][0]
    assert result["verified"] is False
    assert "BLEU value" in (result["verifier_notes"] or "")
    assert verified_summary["audit_log"][0]["verified"] is False
