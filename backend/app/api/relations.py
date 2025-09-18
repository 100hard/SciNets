from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.relation import Relation, RelationCreate
from app.services.relations import create_relation, get_relation, list_relations


router = APIRouter(prefix="/relations", tags=["relations"])


@router.get("", response_model=List[Relation])
@router.get("/", response_model=List[Relation])
async def api_list_relations(
    paper_id: UUID = Query(..., description="Filter relations for a paper"),
    concept_id: Optional[UUID] = Query(default=None, description="Filter by primary concept"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return await list_relations(
        paper_id=paper_id,
        concept_id=concept_id,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=Relation, status_code=201)
@router.post("/", response_model=Relation, status_code=201)
async def api_create_relation(data: RelationCreate) -> Relation:
    return await create_relation(data)


@router.get("/{relation_id}", response_model=Relation)
async def api_get_relation(relation_id: UUID) -> Relation:
    relation = await get_relation(relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="Relation not found")
    return relation

