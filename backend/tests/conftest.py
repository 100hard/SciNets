from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
from uuid import UUID, uuid4

import pytest
from fastapi import UploadFile
from app.models.paper import Paper, PaperCreate
from app.models.section import Section, SectionCreate
from app.services.storage import StorageUploadResult


class InMemoryDataStore:
    def __init__(self) -> None:
        self._papers: Dict[UUID, Dict[str, Any]] = {}
        self._sections: Dict[UUID, List[Dict[str, Any]]] = {}
        self._objects: Dict[str, bytes] = {}
        self.extraction_log: List[tuple[str, UUID]] = []

    async def upload_pdf_to_storage(self, file: UploadFile) -> StorageUploadResult:
        data = await file.read()
        if not data:
            raise ValueError("Uploaded file is empty")
        object_name = f"{uuid4().hex}/{Path(file.filename or 'uploaded.pdf').name}"
        self._objects[object_name] = data
        result = StorageUploadResult(
            bucket="test-bucket",
            object_name=object_name,
            file_name=Path(file.filename or "uploaded.pdf").name,
            size=len(data),
            content_type=(file.content_type or "application/pdf").lower(),
        )
        await file.close()
        return result

    async def download_pdf_from_storage(self, object_name: str) -> bytes:
        return self._objects[object_name]

    async def create_paper(self, data: PaperCreate) -> Paper:
        paper_id = uuid4()
        now = datetime.now(timezone.utc)
        record = {
            "id": paper_id,
            "title": data.title,
            "authors": data.authors,
            "venue": data.venue,
            "year": data.year,
            "status": data.status or "uploaded",
            "file_path": data.file_path,
            "file_name": data.file_name,
            "file_size": data.file_size,
            "file_content_type": data.file_content_type,
            "created_at": now,
            "updated_at": now,
        }
        self._papers[paper_id] = record
        return Paper(**record)

    async def get_paper(self, paper_id: UUID) -> Paper | None:
        record = self._papers.get(paper_id)
        return Paper(**record) if record else None

    async def update_paper_status(self, paper_id: UUID, status: str) -> Paper | None:
        record = self._papers.get(paper_id)
        if record is None:
            return None
        updated = dict(record)
        updated["status"] = status
        updated["updated_at"] = datetime.now(timezone.utc)
        self._papers[paper_id] = updated
        return Paper(**updated)

    async def list_papers(
        self, limit: int = 50, offset: int = 0, q: str | None = None
    ) -> List[Paper]:
        records = list(self._papers.values())
        if q:
            normalized = q.lower()
            records = [
                record
                for record in records
                if normalized in (record["title"] or "").lower()
            ]
        records.sort(key=lambda rec: rec["created_at"], reverse=True)
        sliced = records[offset : offset + limit]
        return [Paper(**record) for record in sliced]

    async def replace_sections(
        self, paper_id: UUID, sections: Sequence[SectionCreate]
    ) -> None:
        now = datetime.now(timezone.utc)
        section_records: List[Dict[str, Any]] = []
        for section in sections:
            section_records.append(
                {
                    "id": uuid4(),
                    "paper_id": paper_id,
                    "title": section.title,
                    "content": section.content,
                    "char_start": section.char_start,
                    "char_end": section.char_end,
                    "page_number": section.page_number,
                    "snippet": section.snippet,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        self._sections[paper_id] = section_records

    async def list_sections(
        self, paper_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[Section]:
        records = self._sections.get(paper_id, [])
        sliced = records[offset : offset + limit]
        return [Section(**record) for record in sliced]

    async def wait_for_status(
        self, paper_id: UUID, expected: str, timeout: float = 5.0
    ) -> Paper:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            record = self._papers.get(paper_id)
            if record and record["status"] == expected:
                return Paper(**record)
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"Status for paper {paper_id} did not reach '{expected}' within {timeout} seconds"
        )


@pytest.fixture
def datastore(monkeypatch: pytest.MonkeyPatch) -> InMemoryDataStore:
    store = InMemoryDataStore()

    async def async_noop(*_: Any, **__: Any) -> None:
        return None

    async def run_tier1_extraction(paper_id: UUID, **_: Any) -> dict[str, Any]:
        store.extraction_log.append(("tier1", paper_id))
        return {"paper_id": str(paper_id), "tiers": [1]}

    async def run_tier2_structurer(
        paper_id: UUID,
        *,
        base_summary: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        store.extraction_log.append(("tier2", paper_id))
        summary = dict(base_summary or {"paper_id": str(paper_id), "tiers": []})
        tiers = list(summary.get("tiers", []))
        if 2 not in tiers:
            tiers.append(2)
        summary["tiers"] = tiers
        return summary

    async def run_tier3_verifier(
        paper_id: UUID,
        *,
        base_summary: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        store.extraction_log.append(("tier3", paper_id))
        summary = dict(base_summary or {"paper_id": str(paper_id), "tiers": []})
        tiers = list(summary.get("tiers", []))
        if 3 not in tiers:
            tiers.append(3)
        summary["tiers"] = tiers
        return summary

    monkeypatch.setattr("app.db.database.test_postgres_connection", async_noop)
    monkeypatch.setattr("app.services.storage.ensure_bucket_exists", async_noop)
    monkeypatch.setattr("app.db.migrate.apply_migrations", async_noop)
    monkeypatch.setattr("app.db.pool.init_pool", async_noop)
    monkeypatch.setattr("app.db.pool.close_pool", async_noop)

    monkeypatch.setattr(
        "app.services.storage.upload_pdf_to_storage", store.upload_pdf_to_storage
    )
    monkeypatch.setattr(
        "app.services.storage.download_pdf_from_storage",
        store.download_pdf_from_storage,
    )
    monkeypatch.setattr("app.services.papers.create_paper", store.create_paper)
    monkeypatch.setattr("app.services.papers.get_paper", store.get_paper)
    monkeypatch.setattr(
        "app.services.papers.update_paper_status", store.update_paper_status
    )
    monkeypatch.setattr("app.services.papers.list_papers", store.list_papers)
    monkeypatch.setattr("app.services.sections.replace_sections", store.replace_sections)
    monkeypatch.setattr("app.services.sections.list_sections", store.list_sections)

    # Patch API module aliases to ensure they use the in-memory implementations
    monkeypatch.setattr("app.api.papers.upload_pdf_to_storage", store.upload_pdf_to_storage)
    monkeypatch.setattr("app.api.papers.create_paper", store.create_paper)
    monkeypatch.setattr("app.api.papers.get_paper", store.get_paper)
    monkeypatch.setattr("app.api.papers.list_papers", store.list_papers)
    monkeypatch.setattr("app.api.sections.list_sections", store.list_sections)

    # Patch task module references used inside parse_pdf_task
    monkeypatch.setattr("app.services.tasks.get_paper", store.get_paper)
    monkeypatch.setattr("app.services.tasks.update_paper_status", store.update_paper_status)
    monkeypatch.setattr("app.services.tasks.replace_sections", store.replace_sections)
    monkeypatch.setattr("app.services.tasks.download_pdf_from_storage", store.download_pdf_from_storage)
    monkeypatch.setattr("app.services.tasks.embed_paper_sections", async_noop)
    monkeypatch.setattr("app.services.tasks.extract_and_store_concepts", async_noop)
    monkeypatch.setattr(
        "app.services.tasks.run_tier1_extraction",
        run_tier1_extraction,
    )
    monkeypatch.setattr(
        "app.services.tasks.run_tier2_structurer",
        run_tier2_structurer,
    )
    monkeypatch.setattr(
        "app.services.tasks.run_tier3_verifier",
        run_tier3_verifier,
    )

    return store
