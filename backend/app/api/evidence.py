from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.evidence import Evidence, EvidenceCreate
from app.services.evidence import create_evidence, get_evidence, list_evidence


router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("", response_model=List[Evidence])
@router.get("/", response_model=List[Evidence])
async def api_list_evidence(
    paper_id: UUID = Query(..., description="Filter evidence for a paper"),
    section_id: Optional[UUID] = Query(default=None, description="Filter by originating section"),
    concept_id: Optional[UUID] = Query(default=None, description="Filter by linked concept"),
    relation_id: Optional[UUID] = Query(default=None, description="Filter by relation context"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return await list_evidence(
        paper_id=paper_id,
        section_id=section_id,
        concept_id=concept_id,
        relation_id=relation_id,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=Evidence, status_code=201)
@router.post("/", response_model=Evidence, status_code=201)
async def api_create_evidence(data: EvidenceCreate) -> Evidence:
    return await create_evidence(data)


@router.get("/{evidence_id}", response_model=Evidence)
async def api_get_evidence(evidence_id: UUID) -> Evidence:
    evidence = await get_evidence(evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return evidence

