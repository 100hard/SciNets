from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.ontology import CanonicalizationReport, ConceptResolutionType
from app.services.canonicalization import canonicalize


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/canonicalize", response_model=CanonicalizationReport)
async def api_canonicalize(
    types: list[ConceptResolutionType] | None = Query(default=None),
) -> CanonicalizationReport:
    selected = types or list(ConceptResolutionType)
    return await canonicalize(selected)
