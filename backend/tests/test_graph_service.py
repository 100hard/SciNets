from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import pytest

from app.models.graph import GraphResponse
from app.services.graph import (
    GraphEntityNotFoundError,
    get_graph_neighborhood,
    get_graph_overview,
)


PAPER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
PAPER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
METHOD_ALPHA = UUID("11111111-1111-1111-1111-111111111111")
METHOD_BETA = UUID("22222222-2222-2222-2222-222222222222")
DATASET_X = UUID("33333333-3333-3333-3333-333333333333")
METRIC_F1 = UUID("44444444-4444-4444-4444-444444444444")
TASK_SUMMARY = UUID("55555555-5555-5555-5555-555555555555")


class FakeAcquireContext:
    def __init__(self, conn: "FakeGraphConnection") -> None:
        self._conn = conn

    async def __aenter__(self) -> "FakeGraphConnection":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - nothing to clean up
        return False


class FakePool:
    def __init__(self, conn: "FakeGraphConnection") -> None:
        self._conn = conn

    def acquire(self) -> FakeAcquireContext:
        return FakeAcquireContext(self._conn)


class FakeGraphConnection:
    def __init__(self) -> None:
        self.papers = {
            PAPER_A: {
                "id": PAPER_A,
                "title": "Structured Summarisation",
            },
            PAPER_B: {
                "id": PAPER_B,
                "title": "Benchmarking Methods",
            },
        }
        self.methods = {
            METHOD_ALPHA: {
                "id": METHOD_ALPHA,
                "name": "AlphaNet",
                "aliases": ["Alpha"],
                "description": "Primary approach",
            },
            METHOD_BETA: {
                "id": METHOD_BETA,
                "name": "BetaBaseline",
                "aliases": ["Beta"],
                "description": "Baseline comparison",
            },
        }
        self.datasets = {
            DATASET_X: {
                "id": DATASET_X,
                "name": "Dataset-X",
                "aliases": ["DX"],
                "description": "Evaluation corpus",
            }
        }
        self.metrics = {
            METRIC_F1: {
                "id": METRIC_F1,
                "name": "F1",
                "aliases": ["F1 Score"],
                "description": "Harmonic mean",
                "unit": "%",
            }
        }
        self.tasks = {
            TASK_SUMMARY: {
                "id": TASK_SUMMARY,
                "name": "Summarisation",
                "aliases": [],
                "description": "Automatic summarisation",
            }
        }
        self.concepts: list[dict[str, Any]] = []
        self.triple_candidates: list[dict[str, Any]] = []
        self.results = [
            {
                "id": uuid4(),
                "paper_id": PAPER_A,
                "method_id": METHOD_ALPHA,
                "dataset_id": DATASET_X,
                "metric_id": METRIC_F1,
                "task_id": TASK_SUMMARY,
                "confidence": 0.8,
                "evidence": [{"snippet": "Alpha performs well."}],
            },
            {
                "id": uuid4(),
                "paper_id": PAPER_B,
                "method_id": METHOD_ALPHA,
                "dataset_id": DATASET_X,
                "metric_id": METRIC_F1,
                "task_id": TASK_SUMMARY,
                "confidence": 0.6,
                "evidence": [{"snippet": "Alpha remains competitive."}],
            },
            {
                "id": uuid4(),
                "paper_id": PAPER_B,
                "method_id": METHOD_BETA,
                "dataset_id": DATASET_X,
                "metric_id": METRIC_F1,
                "task_id": TASK_SUMMARY,
                "confidence": 0.7,
                "evidence": [{"snippet": "Beta comparison."}],
            },
        ]

    # AsyncPG compatibility helpers -------------------------------------------------
    async def fetch(self, query: str, *params: Any) -> list[Mapping[str, Any]]:
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT r.id AS result_id"):
            paper_ids: Sequence[UUID] | None = None
            if "WHERE r.paper_id = ANY($1::uuid[])" in normalized:
                paper_ids = params[0]
            rows = [res for res in self.results if paper_ids is None or res["paper_id"] in paper_ids]
            return [self._build_result_row(res) for res in rows]

        if normalized.startswith("SELECT DISTINCT paper_id FROM results WHERE"):
            if "method_id" in normalized:
                target = params[0]
                return [
                    {"paper_id": res["paper_id"]}
                    for res in self.results
                    if res["method_id"] == target
                ]
            if "dataset_id" in normalized:
                target = params[0]
                return [
                    {"paper_id": res["paper_id"]}
                    for res in self.results
                    if res["dataset_id"] == target
                ]
            if "metric_id" in normalized:
                target = params[0]
                return [
                    {"paper_id": res["paper_id"]}
                    for res in self.results
                    if res["metric_id"] == target
                ]
            if "task_id" in normalized:
                target = params[0]
                return [
                    {"paper_id": res["paper_id"]}
                    for res in self.results
                    if res["task_id"] == target
                ]

        if normalized.startswith("SELECT c.id, c.paper_id, c.name, c.type, p.title AS paper_title"):
            concept_types = set(params[0]) if params else set()
            paper_filter = set(params[1]) if len(params) > 1 else None
            rows: list[Mapping[str, Any]] = []
            for concept in self.concepts:
                if concept_types and concept["type"] not in concept_types:
                    continue
                if paper_filter and concept["paper_id"] not in paper_filter:
                    continue
                paper = self.papers.get(concept["paper_id"]) or {}
                rows.append(
                    {
                        "id": concept["id"],
                        "paper_id": concept["paper_id"],
                        "name": concept["name"],
                        "type": concept["type"],
                        "paper_title": paper.get("title"),
                        "created_at": paper.get("created_at"),
                        "concept_created_at": concept.get("created_at"),
                    }
                )
            return rows

        if normalized.startswith("SELECT tc.paper_id, p.title AS paper_title, tc.graph_metadata"):
            paper_filter = set(params[0]) if params else None
            rows: list[Mapping[str, Any]] = []
            for candidate in self.triple_candidates:
                if paper_filter and candidate["paper_id"] not in paper_filter:
                    continue
                paper = self.papers.get(candidate["paper_id"]) or {}
                rows.append(
                    {
                        "paper_id": candidate["paper_id"],
                        "paper_title": paper.get("title"),
                        "graph_metadata": candidate.get("graph_metadata"),
                        "triple_conf": candidate.get("triple_conf"),
                        "evidence_text": candidate.get("evidence_text"),
                        "section_id": candidate.get("section_id"),
                        "created_at": candidate.get("created_at"),
                    }
                )
            return rows

        raise AssertionError(f"Unsupported fetch query: {normalized}")

    async def fetchrow(self, query: str, *params: Any) -> Mapping[str, Any] | None:
        normalized = " ".join(query.split())
        target = params[0]
        if normalized.startswith("SELECT id, name, aliases, description FROM methods"):
            return self.methods.get(target)
        if normalized.startswith("SELECT id, name, aliases, description FROM datasets"):
            return self.datasets.get(target)
        if normalized.startswith("SELECT id, name, aliases, description, unit FROM metrics"):
            return self.metrics.get(target)
        if normalized.startswith("SELECT id, name, aliases, description FROM tasks"):
            return self.tasks.get(target)
        return None

    # Helpers ----------------------------------------------------------------------
    def _build_result_row(self, result: Mapping[str, Any]) -> dict[str, Any]:
        paper = self.papers[result["paper_id"]]
        method = self.methods.get(result.get("method_id"))
        dataset = self.datasets.get(result.get("dataset_id"))
        metric = self.metrics.get(result.get("metric_id"))
        task = self.tasks.get(result.get("task_id"))
        return {
            "result_id": result["id"],
            "paper_id": result["paper_id"],
            "method_id": result.get("method_id"),
            "dataset_id": result.get("dataset_id"),
            "metric_id": result.get("metric_id"),
            "task_id": result.get("task_id"),
            "confidence": result.get("confidence"),
            "evidence": result.get("evidence"),
            "paper_title": paper.get("title"),
            "method_name": method.get("name") if method else None,
            "method_aliases": method.get("aliases") if method else None,
            "method_description": method.get("description") if method else None,
            "dataset_name": dataset.get("name") if dataset else None,
            "dataset_aliases": dataset.get("aliases") if dataset else None,
            "dataset_description": dataset.get("description") if dataset else None,
            "metric_name": metric.get("name") if metric else None,
            "metric_aliases": metric.get("aliases") if metric else None,
            "metric_description": metric.get("description") if metric else None,
            "metric_unit": metric.get("unit") if metric else None,
            "task_name": task.get("name") if task else None,
            "task_aliases": task.get("aliases") if task else None,
            "task_description": task.get("description") if task else None,
        }


def _setup_fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakeGraphConnection:
    conn = FakeGraphConnection()
    pool = FakePool(conn)
    monkeypatch.setattr("app.services.graph.get_pool", lambda: pool)
    return conn


def test_get_graph_overview_returns_typed_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_overview(monkeypatch))


async def _run_get_graph_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_overview(limit=10, min_conf=0.6)

    assert isinstance(response, GraphResponse)
    assert response.meta.limit == 10
    assert response.meta.node_count >= 4
    assert response.meta.edge_count >= 4
    assert response.meta.filters == {
        "types": ["method", "dataset", "metric", "task"],
        "relations": ["proposes", "evaluates_on", "reports", "compares"],
        "min_conf": 0.6,
    }

    node_types = {node.data.type for node in response.nodes}
    assert node_types == {"method", "dataset", "metric", "task"}

    method_node = next(node for node in response.nodes if node.data.type == "method" and node.data.entity_id == METHOD_ALPHA)
    assert method_node.data.paper_count == 2
    assert method_node.data.aliases == ["Alpha"]
    assert method_node.data.top_links

    edge_types = {edge.data.type for edge in response.edges}
    assert {"evaluates_on", "reports", "proposes", "compares"}.issubset(edge_types)

    compare_edge = next(edge for edge in response.edges if edge.data.type == "compares")
    assert compare_edge.data.weight == pytest.approx(0.65, rel=1e-5)
    assert compare_edge.data.paper_count == 1
    assert compare_edge.data.average_confidence == pytest.approx(0.65, rel=1e-5)


def test_graph_fallback_uses_triple_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_graph_fallback_overview(monkeypatch))


async def _run_graph_fallback_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _setup_fake_pool(monkeypatch)
    conn.results = []

    method_concept_id = uuid4()
    dataset_concept_id = uuid4()
    conn.concepts = [
        {
            "id": method_concept_id,
            "paper_id": PAPER_A,
            "name": "AlphaNet",
            "type": "method",
        },
        {
            "id": dataset_concept_id,
            "paper_id": PAPER_A,
            "name": "Dataset-X",
            "type": "dataset",
        },
    ]

    graph_metadata = {
        "entities": {
            "method": {"text": "AlphaNet", "normalized": "alphanet"},
            "dataset": {"text": "Dataset-X", "normalized": "dataset-x"},
        },
        "pairs": [
            {
                "source": {"type": "method", "text": "AlphaNet", "normalized": "alphanet"},
                "target": {"type": "dataset", "text": "Dataset-X", "normalized": "dataset-x"},
                "relation": "evaluates_on",
                "confidence": 0.72,
                "section_id": "section-1",
                "sentence_indices": [2],
                "evidence": "AlphaNet is evaluated on Dataset-X.",
                "source": "tier2_triple",
            }
        ],
    }
    conn.triple_candidates = [
        {
            "paper_id": PAPER_A,
            "graph_metadata": graph_metadata,
            "triple_conf": 0.72,
            "evidence_text": "AlphaNet is evaluated on Dataset-X.",
            "section_id": "section-1",
            "created_at": None,
        }
    ]

    response = await get_graph_overview(limit=10, min_conf=0.6)

    assert response.meta.node_count == 2
    assert response.meta.edge_count == 1

    method_node = next(node for node in response.nodes if node.data.label == "AlphaNet")
    dataset_node = next(node for node in response.nodes if node.data.label == "Dataset-X")

    assert method_node.data.metadata.get("concept") is True
    assert method_node.data.metadata.get("placeholder") is not True
    assert dataset_node.data.metadata.get("concept") is True
    assert dataset_node.data.metadata.get("placeholder") is not True

    edge = response.edges[0]
    assert edge.data.type == "evaluates_on"
    assert edge.data.metadata.get("evidence")
    snippets = [item.get("snippet") for item in edge.data.metadata["evidence"]]
    assert "AlphaNet is evaluated on Dataset-X." in snippets


def test_get_graph_overview_respects_min_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_overview_min_conf(monkeypatch))


async def _run_get_graph_overview_min_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_overview(limit=10, min_conf=0.75)

    assert response.nodes == []
    assert response.edges == []
    assert response.meta.edge_count == 0
    assert response.meta.node_count == 0


def test_get_graph_neighborhood_for_method(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_neighborhood(monkeypatch))


async def _run_get_graph_neighborhood(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_neighborhood(METHOD_ALPHA, limit=5, min_conf=0.6)

    assert response.meta.center_type == "method"
    assert response.meta.center_id.endswith(str(METHOD_ALPHA))
    assert response.meta.node_count >= 2
    assert response.meta.edge_count >= 3
    assert response.meta.filters["min_conf"] == 0.6

    node_ids = {node.data.entity_id for node in response.nodes}
    assert METHOD_ALPHA in node_ids
    assert DATASET_X in node_ids
    assert METRIC_F1 in node_ids
    assert TASK_SUMMARY in node_ids

    method_node = next(node for node in response.nodes if node.data.entity_id == METHOD_ALPHA)
    assert any(evidence.snippet for evidence in method_node.data.evidence)


def test_get_graph_neighborhood_missing_node(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_neighborhood_missing(monkeypatch))


async def _run_get_graph_neighborhood_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _setup_fake_pool(monkeypatch)
    missing_id = UUID("99999999-9999-9999-9999-999999999999")
    # Remove all entities to simulate missing identifier
    conn.methods.clear()
    conn.datasets.clear()
    conn.metrics.clear()
    conn.tasks.clear()

    with pytest.raises(GraphEntityNotFoundError):
        await get_graph_neighborhood(missing_id, limit=5)

