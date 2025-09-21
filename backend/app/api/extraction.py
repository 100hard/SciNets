from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.services.extraction_tier1 import run_tier1_extraction


router = APIRouter(prefix="/extract", tags=["extraction"])


@router.post("/{paper_id}")
async def api_run_extraction(
    paper_id: UUID,
    tiers: str = Query(
        default="1",
        description="Comma separated list of extraction tiers to execute (currently only tier 1 is supported).",
    ),
):
    requested = {tier.strip() for tier in tiers.split(",") if tier.strip()}
    if not requested:
        requested = {"1"}

    unsupported = sorted(tier for tier in requested if tier != "1")
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extraction tiers requested: {', '.join(unsupported)}. Only tier 1 is available.",
        )

    try:
        summary = await run_tier1_extraction(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return summary

