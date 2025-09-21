from __future__ import annotations

from datetime import datetime, timezone
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

import pytest

from app.models.concept import Concept, ConceptCreate
from app.models.section import SectionCreate
from app.services.concept_extraction import (
    extract_and_store_concepts,
    extract_concepts_from_sections,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_section(content: str, *, title: str | None = None) -> SectionCreate:
    return SectionCreate(
        paper_id=uuid4(),
        title=title,
        content=content,
        char_start=None,
        char_end=None,
        page_number=None,
        snippet=None,
    )


def test_extract_concepts_from_sections_deduplicates_variants() -> None:
    sections = [
        _make_section(
            "Graph neural networks (GNNs) are a class of deep learning models for graph "
            "structured data. Our graph neural network architecture extends message "
            "passing neural network techniques to temporal domains.",
            title="Model Overview",
        ),
        _make_section(
            "The Message Passing Neural Network approach is evaluated on the Cora dataset. "
            "We compare against baseline graph neural-network methods and traditional "
            "feature engineering baselines.",
            title="Experimental Setup",
        ),
    ]

    concepts = extract_concepts_from_sections(sections)
    assert concepts

    lowered = [concept.name.lower() for concept in concepts]
    graph_mentions = [name for name in lowered if "graph neural network" in name]
    assert len(graph_mentions) == 1

    mpnn_mentions = [name for name in lowered if "message passing neural network" in name]
    assert len(mpnn_mentions) == 1

    assert any("cora dataset" in name for name in lowered)


@pytest.mark.anyio("asyncio")
async def test_extract_and_store_concepts_persists_and_links(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    sections = [
        _make_section(
            "Deep learning pipelines rely on large datasets like ImageNet."
            " Modern deep-learning research explores continual learning.",
            title="Deep Learning Pipelines",
        ),
        _make_section(
            "Continual learning benchmarks include Split CIFAR-10 and Permuted MNIST.",
            title="Benchmarks",
        ),
    ]

    stored_models: List[Concept] = []
    captured_payloads: List[str] = []

    async def fake_replace_concepts(
        _: UUID, models: List[ConceptCreate]
    ) -> List[Concept]:
        nonlocal stored_models, captured_payloads
        captured_payloads = [model.name for model in models]
        now = datetime.now(timezone.utc)
        stored_models = [
            Concept(
                id=uuid4(),
                paper_id=paper_id,
                name=model.name,
                type=model.type,
                description=model.description,
                created_at=now,
                updated_at=now,
            )
            for model in models
        ]
        return stored_models

    relation_calls: List[List[str]] = []

    async def fake_replace_relations(
        _: UUID, concepts: List[Concept], relation_type: str = "mentions"
    ) -> None:
        relation_calls.append([concept.name for concept in concepts])
        assert relation_type == "mentions"

    monkeypatch.setattr(
        "app.services.concept_extraction.replace_concepts", fake_replace_concepts
    )
    monkeypatch.setattr(
        "app.services.concept_extraction.replace_paper_concept_relations",
        fake_replace_relations,
    )

    results = await extract_and_store_concepts(paper_id, sections)
    assert results == stored_models
    assert captured_payloads
    assert relation_calls

    lower_payloads = [name.lower() for name in captured_payloads]
    assert any("deep learning" in name for name in lower_payloads)
    assert any("cifar" in name for name in lower_payloads)
