from __future__ import annotations

import asyncio
import copy
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
                "split": "test",
                "value_numeric": 0.81,
                "value_text": "0.81",
                "is_sota": True,
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
                "split": "validation",
                "value_numeric": 0.75,
                "value_text": "0.75",
                "is_sota": False,
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
                "split": "validation",
                "value_numeric": 0.7,
                "value_text": "0.70",
                "is_sota": False,
                "confidence": 0.7,
                "evidence": [{"snippet": "Beta comparison."}],
            },
        ]
        self.method_relations: list[dict[str, Any]] = []
        self.claims: list[dict[str, Any]] = [
            {
                "paper_id": PAPER_A,
                "category": "contribution",
                "text": "AlphaNet introduces a novel encoder.",
                "confidence": 0.9,
            },
            {
                "paper_id": PAPER_B,
                "category": "limitation",
                "text": "Dataset-X lacks diversity.",
                "confidence": 0.6,
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

        if normalized.startswith("SELECT mr.id AS result_id"):
            paper_ids: Sequence[UUID] | None = None
            if "WHERE mr.paper_id = ANY($1::uuid[])" in normalized:
                paper_ids = params[0]
            rows = [
                rel
                for rel in self.method_relations
                if paper_ids is None or rel["paper_id"] in paper_ids
            ]
            return [self._build_relation_row(rel) for rel in rows]

        if normalized.startswith("SELECT paper_id, category::text AS category, text, confidence FROM claims"):
            paper_ids_param: Sequence[UUID] = params[0]
            return [
                {
                    "paper_id": claim["paper_id"],
                    "category": claim["category"],
                    "text": claim["text"],
                    "confidence": claim.get("confidence"),
                }
                for claim in self.claims
                if claim["paper_id"] in paper_ids_param
            ]

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
            "split": result.get("split"),
            "value_numeric": result.get("value_numeric"),
            "value_text": result.get("value_text"),
            "is_sota": result.get("is_sota"),
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

    def _build_relation_row(self, relation: Mapping[str, Any]) -> dict[str, Any]:
        paper = self.papers[relation["paper_id"]]
        method = self.methods.get(relation.get("method_id"))
        dataset = self.datasets.get(relation.get("dataset_id"))
        task = self.tasks.get(relation.get("task_id"))
        return {
            "result_id": relation["id"],
            "paper_id": relation["paper_id"],
            "method_id": relation.get("method_id"),
            "dataset_id": relation.get("dataset_id"),
            "metric_id": None,
            "task_id": relation.get("task_id"),
            "split": None,
            "value_numeric": None,
            "value_text": None,
            "is_sota": None,
            "confidence": relation.get("confidence"),
            "evidence": relation.get("evidence"),
            "paper_title": paper["title"],
            "method_name": method.get("name") if method else None,
            "method_aliases": method.get("aliases") if method else None,
            "method_description": method.get("description") if method else None,
            "dataset_name": dataset.get("name") if dataset else None,
            "dataset_aliases": dataset.get("aliases") if dataset else None,
            "dataset_description": dataset.get("description") if dataset else None,
            "metric_name": None,
            "metric_aliases": None,
            "metric_description": None,
            "metric_unit": None,
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
    assert response.meta.filters["types"] == ["method", "dataset", "metric", "task"]
    assert response.meta.filters["relations"] == [
        "proposes",
        "evaluates_on",
        "reports",
        "compares",
        "uses",
        "causes",
        "part_of",
        "is_a",
        "outperforms",
        "assumes",
    ]
    assert response.meta.filters["min_conf"] == 0.6

    node_types = {node.data.type for node in response.nodes}
    assert node_types == {"method", "dataset", "metric", "task"}

    method_node = next(node for node in response.nodes if node.data.type == "method" and node.data.entity_id == METHOD_ALPHA)
    assert method_node.data.paper_count == 2
    assert method_node.data.aliases == ["Alpha"]
    assert method_node.data.top_links
    best_results = method_node.data.metadata.get("best_results")
    assert best_results
    assert any(pytest.approx(result.get("value"), rel=1e-6) == 0.81 for result in best_results if result.get("value") is not None)
    claims_summary = method_node.data.metadata.get("claims")
    assert claims_summary and claims_summary["by_category"].get("contribution") == 1

    dataset_node = next(node for node in response.nodes if node.data.type == "dataset")
    dataset_claims = dataset_node.data.metadata.get("claims")
    assert dataset_claims and dataset_claims["by_category"].get("limitation") == 1

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
                "provenance": "tier2_triple",
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


def test_graph_fallback_shared_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_graph_fallback_shared_nodes(monkeypatch))


async def _run_graph_fallback_shared_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _setup_fake_pool(monkeypatch)
    conn.results = []
    conn.methods.clear()
    conn.datasets.clear()
    conn.concepts = []

    shared_metadata = {
        "entities": {
            "method": {"text": "Shared Method", "normalized": "shared-method"},
            "dataset": {"text": "Shared Dataset", "normalized": "shared-dataset"},
        },
        "pairs": [
            {
                "source": {"type": "method", "text": "Shared Method", "normalized": "shared-method"},
                "target": {"type": "dataset", "text": "Shared Dataset", "normalized": "shared-dataset"},
                "relation": "evaluates_on",
                "confidence": 0.7,
                "section_id": "section-shared",
                "sentence_indices": [1],
                "evidence": "Shared Method evaluates Shared Dataset.",
                "provenance": "tier2_triple",
            }
        ],
    }

    conn.triple_candidates = [
        {
            "paper_id": PAPER_A,
            "graph_metadata": copy.deepcopy(shared_metadata),
            "triple_conf": 0.7,
            "evidence_text": "Shared Method evaluates Shared Dataset.",
            "section_id": "section-shared",
            "created_at": None,
        },
        {
            "paper_id": PAPER_B,
            "graph_metadata": copy.deepcopy(shared_metadata),
            "triple_conf": 0.68,
            "evidence_text": "Shared Method evaluates Shared Dataset in Paper B.",
            "section_id": "section-shared",
            "created_at": None,
        },
    ]

    response = await get_graph_overview(limit=10, min_conf=0.6)

    method_nodes = [node for node in response.nodes if node.data.type == "method"]
    dataset_nodes = [node for node in response.nodes if node.data.type == "dataset"]

    assert len(method_nodes) == 1, "Fallback methods should be deduplicated"
    assert len(dataset_nodes) == 1, "Fallback datasets should be deduplicated"

    method_node = method_nodes[0]
    dataset_node = dataset_nodes[0]

    assert method_node.data.paper_count == 2
    assert dataset_node.data.paper_count == 2

    def _paper_ids(metadata: Mapping[str, Any]) -> list[str]:
        papers = metadata.get("papers", []) if metadata else []
        return sorted({str(entry.get("paper_id")) for entry in papers if isinstance(entry, Mapping)})

    expected_ids = sorted({str(PAPER_A), str(PAPER_B)})
    assert _paper_ids(method_node.data.metadata) == expected_ids
    assert _paper_ids(dataset_node.data.metadata) == expected_ids

    assert response.edges, "Fallback edges should be generated"
    edge = response.edges[0]
    assert edge.data.paper_count == 2
    assert sorted(str(item.get("paper_id")) for item in edge.data.metadata.get("papers", [])) == expected_ids



def test_graph_fallback_extended_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_graph_fallback_extended_relations(monkeypatch))


async def _run_graph_fallback_extended_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _setup_fake_pool(monkeypatch)
    conn.results = []
    conn.methods.clear()
    conn.datasets.clear()
    conn.metrics.clear()
    conn.tasks.clear()

    conn.concepts = [
        {"id": uuid4(), "paper_id": PAPER_A, "name": "VisionFormer", "type": "model"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Graphite", "type": "material"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Stem cell", "type": "organism"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Higher capacity", "type": "finding"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Regeneration", "type": "process"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Transformer", "type": "concept"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Neural network", "type": "concept"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "Energy conservation", "type": "concept"},
        {"id": uuid4(), "paper_id": PAPER_A, "name": "ResNet", "type": "method"},
    ]

    graph_metadata = {
        "pairs": [
            {
                "source": {"type": "model", "text": "VisionFormer", "normalized": "visionformer"},
                "target": {"type": "material", "text": "Graphite", "normalized": "graphite"},
                "relation": "uses",
                "confidence": 0.9,
                "section_id": "sec-uses",
                "sentence_indices": [0],
                "evidence": "VisionFormer uses Graphite electrodes.",
                "provenance": "tier2_triple",
            },
            {
                "source": {"type": "material", "text": "Graphite", "normalized": "graphite"},
                "target": {"type": "finding", "text": "Higher capacity", "normalized": "higher capacity"},
                "relation": "causes",
                "confidence": 0.88,
                "section_id": "sec-causes",
                "sentence_indices": [1],
                "evidence": "Graphite causes higher capacity in cells.",
                "provenance": "tier2_triple",
            },
            {
                "source": {"type": "organism", "text": "Stem cell", "normalized": "stem cell"},
                "target": {"type": "process", "text": "Regeneration", "normalized": "regeneration"},
                "relation": "part_of",
                "confidence": 0.86,
                "section_id": "sec-part",
                "sentence_indices": [2],
                "evidence": "Stem cell activity is part of regeneration.",
                "provenance": "tier2_triple",
            },
            {
                "source": {"type": "concept", "text": "Transformer", "normalized": "transformer"},
                "target": {"type": "concept", "text": "Neural network", "normalized": "neural network"},
                "relation": "is_a",
                "confidence": 0.84,
                "section_id": "sec-isa",
                "sentence_indices": [3],
                "evidence": "Transformer is a neural network architecture.",
                "provenance": "tier2_triple",
            },
            {
                "source": {"type": "method", "text": "VisionFormer", "normalized": "visionformer"},
                "target": {"type": "method", "text": "ResNet", "normalized": "resnet"},
                "relation": "outperforms",
                "confidence": 0.93,
                "section_id": "sec-outperforms",
                "sentence_indices": [4],
                "evidence": "VisionFormer outperforms ResNet.",
                "provenance": "tier2_triple",
            },
            {
                "source": {"type": "process", "text": "Regeneration", "normalized": "regeneration"},
                "target": {"type": "concept", "text": "Energy conservation", "normalized": "energy conservation"},
                "relation": "assumes",
                "confidence": 0.82,
                "section_id": "sec-assumes",
                "sentence_indices": [5],
                "evidence": "Regeneration assumes energy conservation.",
                "provenance": "tier2_triple",
            },
        ]
    }

    conn.triple_candidates = [
        {
            "paper_id": PAPER_A,
            "graph_metadata": graph_metadata,
            "triple_conf": 0.9,
            "evidence_text": "Extended relations evidence.",
            "section_id": "section-extended",
            "created_at": None,
        }
    ]

    response = await get_graph_overview(limit=20, min_conf=0.6)

    node_types = {node.data.type for node in response.nodes}
    assert {
        "model",
        "material",
        "organism",
        "finding",
        "process",
        "concept",
        "method",
    }.issubset(node_types)

    relation_types = {edge.data.type for edge in response.edges}
    assert {"uses", "causes", "part_of", "is_a", "outperforms", "assumes"}.issubset(relation_types)

    uses_edge = next(edge for edge in response.edges if edge.data.type == "uses")
    statements = uses_edge.data.metadata.get("statements") or []
    assert any(statement.get("provenance") == "tier2_triple" for statement in statements)

    outperforms_edge = next(edge for edge in response.edges if edge.data.type == "outperforms")
    assert outperforms_edge.data.average_confidence >= 0.82


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


def test_get_graph_overview_uses_method_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_get_graph_overview_uses_method_relations(monkeypatch))


async def _run_get_graph_overview_uses_method_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = FakeGraphConnection()
    conn.results = []
    relation_id = uuid4()
    conn.method_relations = [
        {
            "id": relation_id,
            "paper_id": PAPER_A,
            "method_id": METHOD_ALPHA,
            "dataset_id": DATASET_X,
            "task_id": None,
            "confidence": 0.74,
            "evidence": [{"snippet": "QualNet is evaluated on GLUE."}],
        }
    ]

    pool = FakePool(conn)
    monkeypatch.setattr("app.services.graph.get_pool", lambda: pool)

    response = await get_graph_overview(limit=10, min_conf=0.6)

    assert response.nodes, "Nodes should be produced from qualitative relations"
    assert response.edges, "Edges should be produced from qualitative relations"
    snippets = []
    for edge in response.edges:
        assert edge.data.metadata, "Edge metadata should include evidence"
        evidence = edge.data.metadata.get("evidence", [])
        snippets.extend(item.get("snippet") for item in evidence if item)
    assert any("GLUE" in (snippet or "") for snippet in snippets)
    assert all(
        not node.data.metadata or not node.data.metadata.get("placeholder")
        for node in response.nodes
    ), "Placeholder nodes should not appear when relations exist"

