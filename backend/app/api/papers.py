from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile

from app.models.paper import Paper, PaperCreate
from app.services.papers import create_paper, get_paper, list_papers
from app.services.storage import create_presigned_download_url, upload_pdf_to_storage
from app.services.tasks import parse_pdf_task


from typing import Optional
router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=List[Paper])
@router.get("/", response_model=List[Paper])
async def api_list_papers(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: Optional[str] = Query(default=None, description="Search in title"),
):
    return await list_papers(limit=limit, offset=offset, q=q)


@router.post("", response_model=Paper, status_code=201)
@router.post("/", response_model=Paper, status_code=201)
async def api_create_paper(data: PaperCreate):
    return await create_paper(data)


@router.get("/{paper_id}", response_model=Paper)
async def api_get_paper(paper_id: UUID):
    paper = await get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/upload", response_model=Paper, status_code=201)
async def api_upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: Optional[str] = Form(default=None),
    authors: Optional[str] = Form(default=None),
    venue: Optional[str] = Form(default=None),
    year: Optional[int] = Form(default=None),
):
    try:
        stored = await upload_pdf_to_storage(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    derived_title = title or Path(stored.file_name).stem or "Untitled"

    paper = await create_paper(
        PaperCreate(
            title=derived_title,
            authors=authors,
            venue=venue,
            year=year,
            status="uploaded",
            file_path=stored.object_name,
            file_name=stored.file_name,
            file_size=stored.size,
            file_content_type=stored.content_type,
        )
    )

    background_tasks.add_task(parse_pdf_task, paper.id)

    return paper


@router.get("/{paper_id}/download")
async def api_download_paper(
    paper_id: UUID,
    expires_in: int = Query(
        default=3600,
        ge=60,
        le=24 * 3600,
        description="Signed URL validity in seconds (min 60, max 86400).",
    ),
):
    paper = await get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if not paper.file_path:
        raise HTTPException(status_code=404, detail="Paper does not have an uploaded file")

    try:
        download_url = create_presigned_download_url(paper.file_path, expires_in=expires_in)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"download_url": download_url, "expires_in": expires_in}