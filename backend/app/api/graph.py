from __future__ import annotations

from typing import Optional, Sequence
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
    min_conf: float = Query(default=0.6, ge=0.0, le=1.0),
    types: Optional[Sequence[str]] = Query(default=None),
    relations: Optional[Sequence[str]] = Query(default=None),
) -> GraphResponse:
    try:
        return await get_graph_overview(
            limit=limit,
            types=types,
            relations=relations,
            min_conf=min_conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/neighborhood/{node_id}", response_model=GraphResponse)
async def api_graph_neighborhood(
    node_id: UUID,
    limit: int = Query(default=50, ge=1, le=500),
    min_conf: float = Query(default=0.6, ge=0.0, le=1.0),
    types: Optional[Sequence[str]] = Query(default=None),
    relations: Optional[Sequence[str]] = Query(default=None),
) -> GraphResponse:
    try:
        return await get_graph_neighborhood(
            node_id,
            limit=limit,
            types=types,
            relations=relations,
            min_conf=min_conf,
        )
    except GraphEntityNotFoundError as exc:  # pragma: no cover - handled for completeness
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
