from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.section import Section, SectionCreate
from app.services.sections import create_section, get_section, list_sections


router = APIRouter(prefix="/sections", tags=["sections"])


@router.get("", response_model=List[Section])
@router.get("/", response_model=List[Section])
async def api_list_sections(
    paper_id: UUID = Query(..., description="Filter sections for a paper"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return await list_sections(paper_id=paper_id, limit=limit, offset=offset)


@router.post("", response_model=Section, status_code=201)
@router.post("/", response_model=Section, status_code=201)
async def api_create_section(data: SectionCreate) -> Section:
    return await create_section(data)


@router.get("/{section_id}", response_model=Section)
async def api_get_section(section_id: UUID) -> Section:
    section = await get_section(section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    return section

