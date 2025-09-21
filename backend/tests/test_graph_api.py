from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.api.graph import api_graph_neighborhood, api_graph_overview
from app.models.graph import GraphEdge, GraphEdgeData, GraphMeta, GraphNode, GraphNodeData, GraphResponse
from app.services.graph import GraphEntityNotFoundError


PAPER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CONCEPT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _sample_response() -> GraphResponse:
    return GraphResponse(
        nodes=[
            GraphNode(
                data=GraphNodeData(
                    id=f"paper:{PAPER_ID}",
                    type="paper",
                    label="Sample Paper",
                    paper_id=PAPER_ID,
                    metadata=None,
                )
            ),
            GraphNode(
                data=GraphNodeData(
                    id=f"concept:{CONCEPT_ID}",
                    type="concept",
                    label="Concept",
                    paper_id=PAPER_ID,
                    concept_id=CONCEPT_ID,
                    metadata=None,
                )
            ),
        ],
        edges=[
            GraphEdge(
                data=GraphEdgeData(
                    id=f"edge:paper:{PAPER_ID}->concept:{CONCEPT_ID}",
                    source=f"paper:{PAPER_ID}",
                    target=f"concept:{CONCEPT_ID}",
                    type="mentions",
                    paper_id=PAPER_ID,
                    concept_id=CONCEPT_ID,
                )
            )
        ],
        meta=GraphMeta(limit=10, node_count=2, edge_count=1),
    )


def test_api_graph_overview_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_overview(monkeypatch))


async def _run_api_graph_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _sample_response()

    async def fake_get_graph_overview(*, limit: int, paper_id: UUID | None, concept_type: str | None) -> GraphResponse:
        assert limit == 25
        assert paper_id is None
        assert concept_type is None
        return expected

    monkeypatch.setattr("app.api.graph.get_graph_overview", fake_get_graph_overview)

    response = await api_graph_overview(limit=25, paper_id=None, concept_type=None)
    assert response == expected


def test_api_graph_neighborhood_success(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_neighborhood_success(monkeypatch))


async def _run_api_graph_neighborhood_success(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _sample_response()

    async def fake_get_graph_neighborhood(node_id: UUID, *, limit: int) -> GraphResponse:
        assert node_id == CONCEPT_ID
        assert limit == 5
        return expected

    monkeypatch.setattr("app.api.graph.get_graph_neighborhood", fake_get_graph_neighborhood)

    response = await api_graph_neighborhood(CONCEPT_ID, limit=5)
    assert response == expected


def test_api_graph_neighborhood_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_graph_neighborhood_not_found(monkeypatch))


async def _run_api_graph_neighborhood_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_graph_neighborhood(node_id: UUID, *, limit: int) -> GraphResponse:
        raise GraphEntityNotFoundError("missing")

    monkeypatch.setattr("app.api.graph.get_graph_neighborhood", fake_get_graph_neighborhood)

    with pytest.raises(HTTPException) as exc:
        await api_graph_neighborhood(CONCEPT_ID, limit=5)

    assert exc.value.status_code == 404
    assert "missing" in str(exc.value.detail)
