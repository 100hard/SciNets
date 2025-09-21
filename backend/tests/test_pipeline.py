from __future__ import annotations

import asyncio
from io import BytesIO
from typing import TYPE_CHECKING

import fitz
from fastapi import BackgroundTasks
from starlette.datastructures import UploadFile

from app.api.papers import api_list_papers, api_upload_paper
from app.api.sections import api_list_sections
from app.services.tasks import parse_pdf_task

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from .conftest import InMemoryDataStore


def _create_sample_pdf() -> bytes:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Introduction", fontsize=18)
        page.insert_text(
            (72, 108),
            "This section describes the goals of the integration test suite.",
            fontsize=12,
        )
        page.insert_text((72, 160), "Methods", fontsize=18)
        page.insert_text(
            (72, 196),
            "We parse uploaded PDFs and verify that sections are stored correctly.",
            fontsize=12,
        )
        page.insert_text((72, 248), "Results", fontsize=18)
        page.insert_text(
            (72, 284),
            "Successful parsing should produce multiple structured sections.",
            fontsize=12,
        )
        return document.tobytes()
    finally:
        document.close()


def test_upload_parse_display(datastore: "InMemoryDataStore") -> None:
    asyncio.run(_run_upload_parse_display(datastore))


def test_corrupted_pdf_marks_paper_failed(datastore: "InMemoryDataStore") -> None:
    asyncio.run(_run_corrupted_pdf(datastore))


async def _run_upload_parse_display(datastore: "InMemoryDataStore") -> None:
    pdf_bytes = _create_sample_pdf()
    upload = UploadFile(
        filename="integration.pdf",
        file=BytesIO(pdf_bytes),
        headers={"content-type": "application/pdf"},
    )

    background_tasks = BackgroundTasks()
    paper = await api_upload_paper(
        background_tasks=background_tasks,
        file=upload,
        title=None,
        authors=None,
        venue=None,
        year=None,
    )

    await parse_pdf_task(paper.id)
    parsed_paper = await datastore.wait_for_status(paper.id, "parsed")
    assert parsed_paper.status == "parsed"

    papers = await api_list_papers(limit=50, offset=0, q=None)
    assert any(item.id == paper.id and item.status == "parsed" for item in papers)

    sections = await api_list_sections(paper_id=paper.id, limit=100, offset=0)
    assert sections
    assert all(section.content.strip() for section in sections)
    assert any(section.title for section in sections)
    assert datastore.extraction_log == [
        ("tier1", paper.id),
        ("tier2", paper.id),
        ("tier3", paper.id),
    ]


async def _run_corrupted_pdf(datastore: "InMemoryDataStore") -> None:
    corrupt_bytes = b"%PDF-1.4\n% Corrupted payload that cannot be parsed"
    upload = UploadFile(
        filename="corrupted.pdf",
        file=BytesIO(corrupt_bytes),
        headers={"content-type": "application/pdf"},
    )

    paper = await api_upload_paper(
        background_tasks=BackgroundTasks(),
        file=upload,
        title=None,
        authors=None,
        venue=None,
        year=None,
    )

    await parse_pdf_task(paper.id)
    failed_paper = await datastore.wait_for_status(paper.id, "failed")
    assert failed_paper.status == "failed"

    sections = await api_list_sections(paper_id=paper.id, limit=100, offset=0)
    assert sections == []


def test_pipeline_marks_failed_when_extraction_fails(
    datastore: "InMemoryDataStore", monkeypatch: "MonkeyPatch"
) -> None:
    asyncio.run(_run_extraction_failure(datastore, monkeypatch))


async def _run_extraction_failure(
    datastore: "InMemoryDataStore", monkeypatch: "MonkeyPatch"
) -> None:
    pdf_bytes = _create_sample_pdf()
    upload = UploadFile(
        filename="integration.pdf",
        file=BytesIO(pdf_bytes),
        headers={"content-type": "application/pdf"},
    )

    background_tasks = BackgroundTasks()
    paper = await api_upload_paper(
        background_tasks=background_tasks,
        file=upload,
        title=None,
        authors=None,
        venue=None,
        year=None,
    )

    async def failing_tier1_extraction(paper_id, **_):
        datastore.extraction_log.append(("tier1", paper_id))
        raise RuntimeError("tier1 failure")

    monkeypatch.setattr(
        "app.services.tasks.run_tier1_extraction", failing_tier1_extraction
    )

    await parse_pdf_task(paper.id)
    failed_paper = await datastore.wait_for_status(paper.id, "failed")
    assert failed_paper.status == "failed"
    assert datastore.extraction_log == [("tier1", paper.id)]
