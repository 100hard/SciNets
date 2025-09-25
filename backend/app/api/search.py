from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.models.search import SearchResult
from app.services.search import (
    VectorStoreNotAvailableError,
    get_search_service,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/similarity", response_model=List[SearchResult])
async def api_similarity_search(
    q: str = Query(..., description="Free-text search query"),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum number of results"),
) -> List[SearchResult]:
    service = get_search_service()
    try:
        return await service.search(query=q, limit=limit)
    except VectorStoreNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc