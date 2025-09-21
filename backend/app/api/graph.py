from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.graph import GraphResponse
from app.services.graph import (
    GraphEntityNotFoundError,
    get_graph_neighborhood,
    get_graph_overview,
)


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/overview", response_model=GraphResponse)
async def api_graph_overview(
    limit: int = Query(default=100, ge=1, le=500),
    paper_id: Optional[UUID] = Query(default=None),
    concept_type: Optional[str] = Query(default=None, min_length=1),
) -> GraphResponse:
    return await get_graph_overview(limit=limit, paper_id=paper_id, concept_type=concept_type)


@router.get("/neighborhood/{node_id}", response_model=GraphResponse)
async def api_graph_neighborhood(
    node_id: UUID,
    limit: int = Query(default=50, ge=1, le=500),
) -> GraphResponse:
    try:
        return await get_graph_neighborhood(node_id, limit=limit)
    except GraphEntityNotFoundError as exc:  # pragma: no cover - handled for completeness
        raise HTTPException(status_code=404, detail=str(exc)) from exc
