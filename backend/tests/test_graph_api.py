from __future__ import annotations

import asyncio
from typing import Sequence
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.api.graph import api_graph_clear, api_graph_neighborhood, api_graph_overview
from app.models.graph import (
    GraphEdge,
    GraphEdgeData,
    GraphMeta,
    GraphNode,
    GraphNodeData,
    GraphNodeLink,
    GraphResponse,
)
from app.services.graph import GraphEntityNotFoundError


PAPER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CONCEPT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _sample_response() -> GraphResponse:
    return GraphResponse(
        nodes=[
            GraphNode(
                data=GraphNodeData(
                    id=f"method:{CONCEPT_ID}",
                    type="method",
                    label="Sample Method",
                    entity_id=CONCEPT_ID,
                    paper_count=3,
                    aliases=["Baseline"],
                    top_links=[
                        GraphNodeLink(
                            id=f"dataset:{PAPER_ID}",
                            label="Dataset",
                            type="dataset",
                            relation="evaluates_on",
                            weight=1.8,
                        )
                    ],
                    evidence=[],
                    metadata=None,
                )
            ),
            GraphNode(
                data=GraphNodeData(
                    id=f"dataset:{PAPER_ID}",
                    type="dataset",
                    label="Dataset",
                    entity_id=PAPER_ID,
                    paper_count=2,
                    aliases=[],
                    top_links=[],
                    evidence=[],
                    metadata=None,
                )
            ),
        ],
        edges=[
            GraphEdge(
                data=GraphEdgeData(
                    id=f"edge:evaluates_on:method:{CONCEPT_ID}->dataset:{PAPER_ID}",
                    source=f"method:{CONCEPT_ID}",
                    target=f"dataset:{PAPER_ID}",
                    type="evaluates_on",
                    weight=1.8,
                    paper_count=2,
                    average_confidence=0.9,
                    metadata=None,
                )
            )
        ],
        meta=GraphMeta(limit=10, node_count=2, edge_count=1),
    )


def test_api_graph_overview_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_overview(monkeypatch))


async def _run_api_graph_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _sample_response()

    async def fake_get_graph_overview(
        *,
        limit: int,
        types: Sequence[str] | None,
        relations: Sequence[str] | None,
        min_conf: float,
    ) -> GraphResponse:
        assert limit == 25
        assert types is None
        assert relations is None
        assert min_conf == 0.75
        return expected

    monkeypatch.setattr("app.api.graph.get_graph_overview", fake_get_graph_overview)

    response = await api_graph_overview(limit=25, min_conf=0.75, types=None, relations=None)
    assert response == expected


def test_api_graph_neighborhood_success(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_neighborhood_success(monkeypatch))


async def _run_api_graph_neighborhood_success(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _sample_response()

    async def fake_get_graph_neighborhood(
        node_id: UUID,
        *,
        limit: int,
        types: Sequence[str] | None,
        relations: Sequence[str] | None,
        min_conf: float,
    ) -> GraphResponse:
        assert node_id == CONCEPT_ID
        assert limit == 5
        assert types == ["method"]
        assert relations == ["reports"]
        assert min_conf == 0.65
        return expected

    monkeypatch.setattr("app.api.graph.get_graph_neighborhood", fake_get_graph_neighborhood)

    response = await api_graph_neighborhood(
        CONCEPT_ID,
        limit=5,
        types=["method"],
        relations=["reports"],
        min_conf=0.65,
    )
    assert response == expected



def test_api_graph_clear_triggers_service(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_clear(monkeypatch))


async def _run_api_graph_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_clear_graph_data() -> None:
        calls["count"] += 1

    monkeypatch.setattr("app.api.graph.clear_graph_data", fake_clear_graph_data)

    response = await api_graph_clear()
    assert calls["count"] == 1
    assert response.status_code == 204



def test_api_graph_neighborhood_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_neighborhood_not_found(monkeypatch))


async def _run_api_graph_neighborhood_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_graph_neighborhood(
        node_id: UUID,
        *,
        limit: int,
        types: Sequence[str] | None,
        relations: Sequence[str] | None,
        min_conf: float,
    ) -> GraphResponse:
        raise GraphEntityNotFoundError("missing")

    monkeypatch.setattr("app.api.graph.get_graph_neighborhood", fake_get_graph_neighborhood)

    with pytest.raises(HTTPException) as exc:
        await api_graph_neighborhood(CONCEPT_ID, limit=5)

    assert exc.value.status_code == 404
    assert "missing" in str(exc.value.detail)
