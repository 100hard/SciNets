from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Mapping
from uuid import UUID

import pytest

from app.models.graph import GraphResponse
from app.services.graph import (
    GraphEntityNotFoundError,
    get_graph_neighborhood,
    get_graph_overview,
)


PAPER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
PAPER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
CONCEPT_ALPHA = UUID("11111111-1111-1111-1111-111111111111")
CONCEPT_BETA = UUID("22222222-2222-2222-2222-222222222222")
CONCEPT_GAMMA = UUID("33333333-3333-3333-3333-333333333333")
RELATION_ALPHA = UUID("44444444-4444-4444-4444-444444444444")
RELATION_BETA = UUID("55555555-5555-5555-5555-555555555555")


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
        self._papers: Dict[UUID, Dict[str, Any]] = {
            PAPER_A: {
                "id": PAPER_A,
                "title": "Quantum Linking",
                "authors": "Ada Lovelace",
                "venue": "SciConf",
                "year": 2023,
            },
            PAPER_B: {
                "id": PAPER_B,
                "title": "Graph Learning",
                "authors": "Alan Turing",
                "venue": "MLSymposium",
                "year": 2024,
            },
        }
        self._concepts: Dict[UUID, Dict[str, Any]] = {
            CONCEPT_ALPHA: {
                "id": CONCEPT_ALPHA,
                "paper_id": PAPER_A,
                "name": "Quantum Fields",
                "type": "physics",
                "description": "Field theory overview",
                "created_at": 1,
            },
            CONCEPT_BETA: {
                "id": CONCEPT_BETA,
                "paper_id": PAPER_A,
                "name": "Gauge Symmetry",
                "type": "physics",
                "description": "Symmetry discussion",
                "created_at": 2,
            },
            CONCEPT_GAMMA: {
                "id": CONCEPT_GAMMA,
                "paper_id": PAPER_B,
                "name": "Graph Embeddings",
                "type": "ml",
                "description": "Embedding pipelines",
                "created_at": 3,
            },
        }
        self._relations: Dict[UUID, Dict[str, Any]] = {
            RELATION_ALPHA: {
                "id": RELATION_ALPHA,
                "paper_id": PAPER_A,
                "concept_id": CONCEPT_ALPHA,
                "related_concept_id": CONCEPT_BETA,
                "relation_type": "related_to",
                "description": "Alpha influences Beta",
                "created_at": 10,
            },
            RELATION_BETA: {
                "id": RELATION_BETA,
                "paper_id": PAPER_A,
                "concept_id": CONCEPT_BETA,
                "related_concept_id": CONCEPT_GAMMA,
                "relation_type": "extends",
                "description": "Cross paper relationship",
                "created_at": 8,
            },
        }

    # AsyncPG compatibility helpers -------------------------------------------------
    async def fetch(self, query: str, *params: Any) -> List[Mapping[str, Any]]:
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT c.id AS concept_id"):
            if "WHERE c.id = ANY" in normalized:
                concept_ids = set(params[0])
                return [self._build_concept_row(rec) for rec in self._filter_concepts(ids=concept_ids)]
            paper_id, concept_type, limit = self._parse_concept_filters(normalized, list(params))
            concepts = self._filter_concepts(paper_id=paper_id, concept_type=concept_type)
            return [self._build_concept_row(rec) for rec in concepts[:limit]]

        if normalized.startswith("SELECT id, paper_id, concept_id, related_concept_id, relation_type, description FROM relations"):
            if "concept_id = ANY($1::uuid[]) AND related_concept_id = ANY($1::uuid[])" in normalized:
                concept_ids = set(params[0])
                idx = 1
                paper_filter = None
                if "AND paper_id = $2" in normalized:
                    paper_filter = params[1]
                    idx = 2
                limit = params[idx]
                relations = self._relations_between(concept_ids, paper_filter)
                return [self._build_relation_row(rec) for rec in relations[:limit]]
            if "WHERE (concept_id = $1 OR related_concept_id = $1)" in normalized:
                concept_id = params[0]
                limit = params[1]
                relations = self._relations_touching(concept_id)
                return [self._build_relation_row(rec) for rec in relations[:limit]]
            if "WHERE paper_id = $1" in normalized:
                paper_id = params[0]
                limit = params[1]
                relations = self._relations_for_paper(paper_id)
                return [self._build_relation_row(rec) for rec in relations[:limit]]

        raise AssertionError(f"Unsupported fetch query: {normalized}")

    async def fetchrow(self, query: str, *params: Any) -> Mapping[str, Any] | None:
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT c.id AS concept_id") and "WHERE c.id = $1" in normalized:
            concept_id = params[0]
            concepts = self._filter_concepts(ids={concept_id})
            return self._build_concept_row(concepts[0]) if concepts else None
        if normalized.startswith("SELECT p.id AS paper_id") and "WHERE p.id = $1" in normalized:
            paper_id = params[0]
            paper = self._papers.get(paper_id)
            return self._build_paper_row(paper) if paper else None
        raise AssertionError(f"Unsupported fetchrow query: {normalized}")

    async def fetchval(self, query: str, *params: Any) -> Any:
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT COUNT(*) FROM concepts c"):
            paper_id, concept_type, _ = self._parse_concept_filters(normalized, list(params), include_limit=False)
            return len(self._filter_concepts(paper_id=paper_id, concept_type=concept_type))
        if normalized.startswith("SELECT COUNT(*) FROM concepts WHERE paper_id = $1"):
            paper_id = params[0]
            return len(self._filter_concepts(paper_id=paper_id))
        raise AssertionError(f"Unsupported fetchval query: {normalized}")

    # Helpers ----------------------------------------------------------------------
    def _filter_concepts(
        self,
        *,
        paper_id: UUID | None = None,
        concept_type: str | None = None,
        ids: Iterable[UUID] | None = None,
    ) -> List[Dict[str, Any]]:
        records = list(self._concepts.values())
        if ids is not None:
            target = set(ids)
            records = [rec for rec in records if rec["id"] in target]
        if paper_id is not None:
            records = [rec for rec in records if rec["paper_id"] == paper_id]
        if concept_type is not None:
            records = [rec for rec in records if rec["type"] == concept_type]
        records.sort(key=lambda rec: rec["created_at"], reverse=True)
        return records

    def _parse_concept_filters(
        self,
        normalized_query: str,
        params: List[Any],
        *,
        include_limit: bool = True,
    ) -> tuple[UUID | None, str | None, int]:
        paper_id = None
        concept_type = None
        limit = params[-1] if include_limit and params else 0

        if "WHERE c.paper_id = $1 AND c.type = $2" in normalized_query:
            paper_id = params[0]
            concept_type = params[1]
            if include_limit:
                limit = params[2]
        elif "WHERE c.paper_id = $1" in normalized_query and "AND" not in normalized_query.split("WHERE c.paper_id = $1", 1)[1][:5]:
            paper_id = params[0]
            if include_limit:
                limit = params[1]
        elif "WHERE c.type = $1" in normalized_query:
            concept_type = params[0]
            if include_limit:
                limit = params[1]
        elif include_limit:
            limit = params[0]

        return paper_id, concept_type, limit

    def _build_concept_row(self, concept: Mapping[str, Any]) -> Dict[str, Any]:
        paper = self._papers[concept["paper_id"]]
        return {
            "concept_id": concept["id"],
            "concept_name": concept["name"],
            "concept_type": concept["type"],
            "concept_description": concept["description"],
            "paper_id": concept["paper_id"],
            "paper_title": paper["title"],
            "paper_authors": paper["authors"],
            "paper_venue": paper["venue"],
            "paper_year": paper["year"],
        }

    def _build_paper_row(self, paper: Mapping[str, Any] | None) -> Dict[str, Any] | None:
        if not paper:
            return None
        return {
            "paper_id": paper["id"],
            "paper_title": paper["title"],
            "paper_authors": paper["authors"],
            "paper_venue": paper["venue"],
            "paper_year": paper["year"],
        }

    def _build_relation_row(self, relation: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": relation["id"],
            "paper_id": relation["paper_id"],
            "concept_id": relation["concept_id"],
            "related_concept_id": relation["related_concept_id"],
            "relation_type": relation["relation_type"],
            "description": relation["description"],
        }

    def _relations_between(
        self,
        concept_ids: Iterable[UUID],
        paper_filter: UUID | None,
    ) -> List[Dict[str, Any]]:
        targets = set(concept_ids)
        relations = [
            rel
            for rel in self._relations.values()
            if rel["concept_id"] in targets and rel["related_concept_id"] in targets
        ]
        if paper_filter is not None:
            relations = [rel for rel in relations if rel["paper_id"] == paper_filter]
        relations.sort(key=lambda rec: rec["created_at"], reverse=True)
        return relations

    def _relations_touching(self, concept_id: UUID) -> List[Dict[str, Any]]:
        relations = [
            rel
            for rel in self._relations.values()
            if rel["concept_id"] == concept_id or rel["related_concept_id"] == concept_id
        ]
        relations.sort(key=lambda rec: rec["created_at"], reverse=True)
        return relations

    def _relations_for_paper(self, paper_id: UUID) -> List[Dict[str, Any]]:
        relations = [rel for rel in self._relations.values() if rel["paper_id"] == paper_id]
        relations.sort(key=lambda rec: rec["created_at"], reverse=True)
        return relations


def _setup_fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    conn = FakeGraphConnection()
    pool = FakePool(conn)
    monkeypatch.setattr("app.services.graph.get_pool", lambda: pool)
    return pool


def test_get_graph_overview_returns_concepts_and_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_overview(monkeypatch))


async def _run_get_graph_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_overview(limit=5)

    assert isinstance(response, GraphResponse)
    assert response.meta.limit == 5
    assert response.meta.concept_count == 3
    assert response.meta.paper_count == 2
    assert not response.meta.has_more

    concept_ids = {node.data.concept_id for node in response.nodes if node.data.type == "concept"}
    assert concept_ids == {CONCEPT_ALPHA, CONCEPT_BETA, CONCEPT_GAMMA}

    relation_ids = {edge.data.relation_id for edge in response.edges if edge.data.relation_id}
    assert RELATION_ALPHA in relation_ids
    assert RELATION_BETA in relation_ids


def test_get_graph_neighborhood_for_concept(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_neighborhood_for_concept(monkeypatch))


async def _run_get_graph_neighborhood_for_concept(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_neighborhood(CONCEPT_BETA, limit=3)

    concept_ids = {node.data.concept_id for node in response.nodes if node.data.type == "concept"}
    assert {CONCEPT_ALPHA, CONCEPT_BETA, CONCEPT_GAMMA}.issubset(concept_ids)

    assert response.meta.center_type == "concept"
    assert response.meta.center_id.endswith(str(CONCEPT_BETA))
    assert response.meta.concept_count == 2  # two concepts share the primary paper
    assert response.meta.paper_count >= 1

    relation_pairs = {
        (edge.data.concept_id, edge.data.related_concept_id)
        for edge in response.edges
        if edge.data.relation_id == RELATION_BETA
    }
    assert (CONCEPT_BETA, CONCEPT_GAMMA) in relation_pairs


def test_get_graph_neighborhood_for_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_neighborhood_for_paper(monkeypatch))


async def _run_get_graph_neighborhood_for_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    response = await get_graph_neighborhood(PAPER_A, limit=5)

    assert response.meta.center_type == "paper"
    assert response.meta.center_id.endswith(str(PAPER_A))
    assert response.meta.concept_count == 2
    assert not response.meta.has_more

    concept_ids = {node.data.concept_id for node in response.nodes if node.data.type == "concept"}
    assert concept_ids == {CONCEPT_ALPHA, CONCEPT_BETA}

    relation_ids = {edge.data.relation_id for edge in response.edges if edge.data.relation_id}
    assert relation_ids == {RELATION_ALPHA}


def test_get_graph_neighborhood_missing_node(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_get_graph_neighborhood_missing_node(monkeypatch))


async def _run_get_graph_neighborhood_missing_node(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_fake_pool(monkeypatch)

    missing_id = UUID("99999999-9999-9999-9999-999999999999")
    with pytest.raises(GraphEntityNotFoundError):
        await get_graph_neighborhood(missing_id, limit=3)
