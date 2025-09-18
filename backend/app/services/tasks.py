from __future__ import annotations

from uuid import UUID


def parse_pdf_task(paper_id: UUID) -> None:
    """Placeholder task hook for triggering the PDF parsing pipeline.

    For the MVP we simply log the invocation. In future iterations this will
    enqueue a background job (Celery/Arq/etc.) that performs extraction and
    updates the paper status.
    """

    print(f"[parse_pdf_task] Triggered for paper {paper_id}")
