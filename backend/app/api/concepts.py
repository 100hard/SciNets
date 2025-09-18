from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.concept import Concept, ConceptCreate
from app.services.concepts import create_concept, get_concept, list_concepts


router = APIRouter(prefix="/concepts", tags=["concepts"])


@router.get("", response_model=List[Concept])
@router.get("/", response_model=List[Concept])
async def api_list_concepts(
    paper_id: UUID = Query(..., description="Filter concepts extracted from a paper"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return await list_concepts(paper_id=paper_id, limit=limit, offset=offset)


@router.post("", response_model=Concept, status_code=201)
@router.post("/", response_model=Concept, status_code=201)
async def api_create_concept(data: ConceptCreate) -> Concept:
    return await create_concept(data)


@router.get("/{concept_id}", response_model=Concept)
async def api_get_concept(concept_id: UUID) -> Concept:
    concept = await get_concept(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    return concept

