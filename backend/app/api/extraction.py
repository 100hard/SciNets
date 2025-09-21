from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.services.extraction_tier1 import run_tier1_extraction
from app.services.extraction_tier2 import Tier2ValidationError, run_tier2_structurer


router = APIRouter(prefix="/extract", tags=["extraction"])


@router.post("/{paper_id}")
async def api_run_extraction(
    paper_id: UUID,
    tiers: str = Query(
        default="1",
        description=(
            "Comma separated list of extraction tiers to execute "
            "(supported tiers: 1, 2)."
        ),
    ),
):
    requested = {tier.strip() for tier in tiers.split(",") if tier.strip()}
    if not requested:
        requested = {"1"}

    unsupported = sorted(tier for tier in requested if tier not in {"1", "2"})
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported extraction tiers requested: "
                f"{', '.join(unsupported)}. Supported tiers are 1 and 2."
            ),
        )

    summary: dict | None = None
    try:
        if "1" in requested:
            summary = await run_tier1_extraction(paper_id)
        if "2" in requested:
            summary = await run_tier2_structurer(paper_id, base_summary=summary)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Tier2ValidationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return summary

