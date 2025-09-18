from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.paper import Paper, PaperCreate
from app.services.papers import create_paper, get_paper, list_papers


router = APIRouter(prefix="/api/papers", tags=["papers"])


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

