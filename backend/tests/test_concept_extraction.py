from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

import pytest

from app.models.concept import Concept, ConceptCreate
from app.models.paper import Paper
from app.models.section import SectionCreate
from app.services.concept_extraction import (
    _Candidate,
    ConceptProvenance,
    _apply_method_post_filters,
    _infer_concept_type,
    extract_and_store_concepts,
    extract_concepts_from_sections,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_section(content: str, *, title: str | None = None) -> SectionCreate:
    return SectionCreate(
        id=uuid4(),
        paper_id=uuid4(),
        title=title,
        content=content,
        char_start=None,
        char_end=None,
        page_number=None,
        snippet=None,
    )


def _make_paper(
    *,
    title: str,
    venue: str,
    status: str = "parsed",
) -> Paper:
    now = datetime.now(timezone.utc)
    return Paper(
        id=uuid4(),
        title=title,
        authors=None,
        venue=venue,
        year=2024,
        file_path=None,
        file_name=None,
        file_size=None,
        file_content_type=None,
        status=status,
        created_at=now,
        updated_at=now,
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
    assert all(concept.provenance for concept in concepts)
    for concept in concepts:
        provenance = concept.provenance[0]
        assert provenance.provider in {"heuristic", "scispacy", "domain_ner"}
        assert provenance.snippet


def test_biology_domain_labels_organisms() -> None:
    paper = _make_paper(
        title="Metabolic regulation in Escherichia coli",
        venue="Journal of Molecular Biology",
    )
    sections = [
        _make_section(
            "Escherichia coli adapts to oxidative stress by inducing antioxidant enzymes.",
            title="Abstract",
        ),
        _make_section(
            "The bacterium Escherichia coli upregulates catalase and superoxide dismutase.",
            title="Results",
        ),
    ]

    concepts = extract_concepts_from_sections(sections, paper=paper)
    assert concepts

    organism_concepts = [
        concept for concept in concepts if "escherichia coli" in concept.name.lower()
    ]
    assert organism_concepts
    assert all(concept.type == "organism" for concept in organism_concepts)


def test_materials_domain_labels_materials() -> None:
    paper = _make_paper(
        title="Graphene oxide membranes for desalination",
        venue="Advanced Materials Research",
    )
    sections = [
        _make_section(
            "Graphene oxide membranes enabled rapid ion transport without sacrificing "
            "mechanical integrity.",
            title="Overview",
        ),
        _make_section(
            "The perovskite oxide thin film remained stable above 500 °C in repeated cycles.",
            title="Stability",
        ),
    ]

    concepts = extract_concepts_from_sections(sections, paper=paper)
    assert concepts

    material_concepts = {
        concept.name.lower(): concept.type for concept in concepts if concept.type
    }
    graphene_mentions = [
        name for name in material_concepts if "graphene oxide" in name
    ]
    assert graphene_mentions
    for name in graphene_mentions:
        assert material_concepts[name] == "material"


@pytest.mark.anyio("asyncio")
async def test_extract_and_store_concepts_persists_and_links(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    paper = _make_paper(title="Deep Learning Pipelines", venue="NeurIPS")
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
    captured_metadata: List[dict] = []

    async def fake_replace_concepts(
        _: UUID, models: List[ConceptCreate]
    ) -> List[Concept]:
        nonlocal stored_models, captured_payloads, captured_metadata
        captured_payloads = [model.name for model in models]
        captured_metadata = [model.metadata for model in models]
        now = datetime.now(timezone.utc)
        stored_models = [
            Concept(
                id=uuid4(),
                paper_id=paper_id,
                name=model.name,
                type=model.type,
                description=model.description,
                metadata=model.metadata,
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

    async def fake_get_paper(paper_lookup: UUID) -> Paper:
        assert paper_lookup == paper_id
        return paper

    monkeypatch.setattr("app.services.concept_extraction.get_paper", fake_get_paper)

    results = await extract_and_store_concepts(paper_id, sections)
    assert results == stored_models
    assert captured_payloads
    assert relation_calls

    lower_payloads = [name.lower() for name in captured_payloads]
    assert any("deep learning" in name for name in lower_payloads)
    assert any("cifar" in name for name in lower_payloads)
    assert captured_metadata
    for payload in captured_metadata:
        assert payload["provenance"]
        first = payload["provenance"][0]
        assert first["provider"]
        assert first["snippet"]
        assert first["section_id"]
        assert first["char_start"] is not None
        assert first["char_end"] is not None
        assert payload["occurrences"] >= 1


def test_infer_concept_type_uses_strong_method_cues() -> None:
    assert _infer_concept_type("BERT") == "method"
    assert _infer_concept_type("Cas9") == "method"
    assert _infer_concept_type("graph neural network") == "method"
    assert _infer_concept_type("multi-stage curriculum") == "keyword"


def test_method_post_filter_demotes_noisy_phrases() -> None:
    provenance = ConceptProvenance(
        section_id=None,
        char_start=0,
        char_end=10,
        snippet="sample snippet",
        provider="heuristic",
        provider_metadata={"strategy": "token_phrase"},
    )
    noisy = _Candidate(
        name="The Proposed Multi Stage Training Approach",
        normalized="proposed multi stage training approach",
        type="method",
        description=None,
        score=1.0,
        provenance=[provenance],
    )
    corroborated = _Candidate(
        name="Baseline Transformer Model",
        normalized="baseline transformer model",
        type="method",
        description=None,
        score=1.0,
        provenance=[provenance, provenance],
    )

    registry = {
        noisy.normalized: noisy,
        corroborated.normalized: corroborated,
    }

    _apply_method_post_filters(registry)

    assert registry[noisy.normalized].type == "keyword"
    assert registry[corroborated.normalized].type == "method"
