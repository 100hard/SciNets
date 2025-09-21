from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models.ontology import Claim, Dataset, Method, Metric, Result, Task
from app.models.paper import Paper
from app.models.section import Section
from app.services.extraction_tier1 import extract_signals, load_mt_lexicon, run_tier1_extraction


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(content: str, title: str | None = None) -> Section:
    now = datetime.now(timezone.utc)
    return Section.model_construct(
        id=uuid4(),
        paper_id=uuid4(),
        title=title,
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
        title="Neural Machine Translation",
        authors="Doe et al.",
        venue="MT Summit",
        year=2024,
        status="parsed",
        file_path=None,
        file_name="paper.pdf",
        file_size=1024,
        file_content_type="application/pdf",
        created_at=now,
        updated_at=now,
    )


def test_extract_signals_identifies_entities() -> None:
    lexicon = load_mt_lexicon()
    section = _build_section(
        "We propose the NMT-Transformer architecture for machine translation. "
        "Our approach is evaluated on the WMT14 En-De benchmark and achieves BLEU = 29.3 on the test set."
    )

    artifacts = extract_signals([section], lexicon=lexicon, table_texts=["Dataset WMT14 En-De"])

    method_names = {entity.name for entity in artifacts.methods.values()}
    dataset_names = {entity.name for entity in artifacts.datasets.values()}
    metric_names = {entity.name for entity in artifacts.metrics.values()}
    task_names = {entity.name for entity in artifacts.tasks.values()}

    assert "Neural Machine Translation Transformer" in method_names
    assert "WMT14 English-German" in dataset_names
    assert "BLEU" in metric_names
    assert "Machine Translation" in task_names

    assert artifacts.results
    result = artifacts.results[0]
    assert result.metric_name == "BLEU"
    assert result.dataset_name == "WMT14 English-German"
    assert result.method_name in method_names
    assert pytest.approx(result.value_numeric or 0.0, rel=1e-5) == 29.3
    assert result.evidence and result.evidence[0]["source"] == "section"


def test_extract_signals_detects_dataset_with_intervening_words() -> None:
    lexicon = load_mt_lexicon()
    section = _build_section(
        "We evaluate our new FooNet model on the CIFAR-10 dataset and report Accuracy = 92.3%."
    )

    artifacts = extract_signals([section], lexicon=lexicon)

    dataset_names = {entity.name for entity in artifacts.datasets.values()}
    assert any("CIFAR-10" in name for name in dataset_names)
    assert any(
        result.dataset_name and "CIFAR-10" in result.dataset_name for result in artifacts.results
    )


@pytest.mark.anyio("asyncio")
async def test_run_tier1_extraction_persists_results(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    paper = _build_paper(paper_id)
    section = _build_section(
        "This work introduces the NMT-Transformer model. "
        "We achieve BLEU: 30.5 on the test set for neural machine translation.",
        title="Results",
    )
    sections = [section]
    lexicon = load_mt_lexicon()

    stored_methods: dict[str, Method] = {}
    stored_datasets: dict[str, Dataset] = {}
    stored_metrics: dict[str, Metric] = {}
    stored_tasks: dict[str, Task] = {}
    stored_results: list[Result] = []
    claims_calls: list[list[Claim]] = []

    async def fake_get_paper(_: UUID) -> Paper:
        return paper

    async def fake_list_sections(*_: object, **__: object) -> list[Section]:
        return sections

    async def fake_download_pdf(*_: object, **__: object) -> bytes:
        return b""

    async def fake_ensure_method(
        name: str,
        *,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> Method:
        key = name.lower()
        model = stored_methods.get(key)
        if model is None:
            now = datetime.now(timezone.utc)
            model = Method.model_construct(
                id=uuid4(),
                name=name,
                aliases=list(aliases or []),
                description=description,
                created_at=now,
                updated_at=now,
            )
            stored_methods[key] = model
        return model

    async def fake_ensure_dataset(
        name: str,
        *,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> Dataset:
        key = name.lower()
        model = stored_datasets.get(key)
        if model is None:
            now = datetime.now(timezone.utc)
            model = Dataset.model_construct(
                id=uuid4(),
                name=name,
                aliases=list(aliases or []),
                description=description,
                created_at=now,
                updated_at=now,
            )
            stored_datasets[key] = model
        return model

    async def fake_ensure_metric(
        name: str,
        *,
        unit: str | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> Metric:
        key = name.lower()
        model = stored_metrics.get(key)
        if model is None:
            now = datetime.now(timezone.utc)
            model = Metric.model_construct(
                id=uuid4(),
                name=name,
                unit=unit,
                aliases=list(aliases or []),
                description=description,
                created_at=now,
                updated_at=now,
            )
            stored_metrics[key] = model
        return model

    async def fake_ensure_task(
        name: str,
        *,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> Task:
        key = name.lower()
        model = stored_tasks.get(key)
        if model is None:
            now = datetime.now(timezone.utc)
            model = Task.model_construct(
                id=uuid4(),
                name=name,
                aliases=list(aliases or []),
                description=description,
                created_at=now,
                updated_at=now,
            )
            stored_tasks[key] = model
        return model

    async def fake_replace_results(_: UUID, models: list) -> list[Result]:
        stored_results.clear()
        now = datetime.now(timezone.utc)
        for item in models:
            stored_results.append(
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
            )
        return list(stored_results)

    async def fake_replace_claims(_: UUID, models: list) -> list[Claim]:
        claims_calls.append(models)
        return []

    monkeypatch.setattr("app.services.extraction_tier1.get_paper", fake_get_paper)
    monkeypatch.setattr("app.services.extraction_tier1.list_sections", fake_list_sections)
    monkeypatch.setattr("app.services.extraction_tier1.download_pdf_from_storage", fake_download_pdf)
    monkeypatch.setattr("app.services.extraction_tier1.ensure_method", fake_ensure_method)
    monkeypatch.setattr("app.services.extraction_tier1.ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr("app.services.extraction_tier1.ensure_metric", fake_ensure_metric)
    monkeypatch.setattr("app.services.extraction_tier1.ensure_task", fake_ensure_task)
    monkeypatch.setattr("app.services.extraction_tier1.replace_results", fake_replace_results)
    monkeypatch.setattr("app.services.extraction_tier1.replace_claims", fake_replace_claims)

    summary = await run_tier1_extraction(
        paper_id,
        lexicon=lexicon,
        table_texts=["Metric BLEU 30.5", "Dataset WMT14 En-De"],
    )

    assert stored_methods
    assert stored_datasets
    assert stored_metrics
    assert stored_tasks
    assert stored_results

    assert any(result.dataset_id is not None for result in stored_results)

    summary_result = next(
        res for res in summary["results"] if res.get("dataset") is not None
    )
    assert pytest.approx(summary_result["value_numeric"], rel=1e-5) == 30.5
    assert summary_result["metric"]["name"] == "BLEU"
    assert summary_result["dataset"]["name"] == "WMT14 English-German"
    assert any(res.get("method") for res in summary["results"])
    assert summary_result["task"]["name"] == "Machine Translation"
    assert summary_result["evidence"]

    assert summary["claims"] == []
    assert claims_calls == [[]]

