from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models.paper import Paper
from app.models.section import Section
from app.services.extraction_tier1 import run_tier1_extraction


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(content: str, *, title: str | None = None, page: int = 1) -> Section:
    now = datetime.now(timezone.utc)
    return Section.model_construct(
        id=uuid4(),
        paper_id=uuid4(),
        title=title,
        content=content,
        char_start=0,
        char_end=len(content),
        page_number=page,
        snippet=content[:120],
        created_at=now,
        updated_at=now,
    )


def _build_paper(paper_id: UUID) -> Paper:
    now = datetime.now(timezone.utc)
    return Paper.model_construct(
        id=paper_id,
        title="Example Paper",
        authors="Doe et al.",
        venue="ICML",
        year=2024,
        status="parsed",
        file_path=None,
        file_name="paper.pdf",
        file_size=1024,
        file_content_type="application/pdf",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.anyio
async def test_run_tier1_extraction_emits_structural_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    paper = _build_paper(paper_id)
    section = _build_section(
        """
        Introduction. Table 1: Overview of results.
        The Transformer is defined as a neural architecture [1].
        We evaluate it on WMT14 En-Fr and achieve strong results.
        """.strip(),
        title="Introduction",
    )

    async def fake_get_paper(_: UUID) -> Paper:
        return paper

    async def fake_list_sections(*_: object, **__: object) -> list[Section]:
        return [section]

    async def fake_download_pdf(*_: object, **__: object) -> bytes:
        return b""

    def fake_tables(_: bytes) -> list[dict[str, object]]:
        return [
            {
                "table_id": "table_1_1",
                "page_number": 1,
                "section_id": str(section.id),
                "cells": [
                    {
                        "row": 0,
                        "column": 0,
                        "text": "Metric",
                    },
                    {
                        "row": 0,
                        "column": 1,
                        "text": "BLEU 41.8",
                    },
                ],
            }
        ]

    monkeypatch.setattr("app.services.extraction_tier1.get_paper", fake_get_paper)
    monkeypatch.setattr("app.services.extraction_tier1.list_sections", fake_list_sections)
    monkeypatch.setattr("app.services.extraction_tier1.download_pdf_from_storage", fake_download_pdf)
    monkeypatch.setattr(
        "app.services.extraction_tier1._extract_tables_with_coordinates",
        fake_tables,
    )

    summary = await run_tier1_extraction(paper_id)

    assert summary["paper_id"] == str(paper_id)
    assert summary["metadata"]["section_count"] == 1
    assert summary["sections"], "expected sections payload"

    section_payload = summary["sections"][0]
    assert section_payload["section_hash"], "section hash should be present"
    sentences = section_payload["sentence_spans"]
    assert sentences, "expected extracted sentences"
    assert any("defined as" in sentence["text"].lower() for sentence in sentences)

    citations = summary["citations"]
    assert citations and citations[0]["text"] == "[1]"

    definition_sentences = summary["definition_sentences"]
    assert definition_sentences and definition_sentences[0]["section_id"] == section_payload["section_id"]

    assert section_payload.get("table_refs") == ["table_1_1"]
    assert summary["tables"][0]["cells"][1]["text"] == "BLEU 41.8"