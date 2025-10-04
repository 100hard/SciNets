from __future__ import annotations

from datetime import datetime, timezone
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from uuid import UUID, uuid4

import pytest

from app.models.concept import Concept, ConceptCreate
from app.models.paper import Paper
from app.models.section import SectionCreate
from app.services.concept_extraction import (
    _Candidate,
    _apply_method_post_filters,
    _infer_concept_type,
    ConceptExtractionRuntimeConfig,
    FILLER_PREFIXES,
    FILLER_SUFFIXES,
    STOPWORDS,
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


def test_scispacy_candidates_capture_linker_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    sections = [
        _make_section(
            "We evaluate GNN models alongside their expanded form Graph Neural Network",
            title="Abbreviations",
        )
    ]

    class DummyEntity:
        def __init__(self, text: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = "METHOD"
            self.start_char = start
            self.end_char = end
            self._ = SimpleNamespace(
                kb_ents=[("UMLS:C12345", 0.87), ("WIKIDATA:Q1", 0.12)],
                long_form=SimpleNamespace(text="Graph Neural Network"),
            )

    class DummyModel:
        def __call__(self, text: str) -> SimpleNamespace:
            start = text.index("GNN")
            end = start + len("GNN")
            entity = DummyEntity(text[start:end], start, end)
            return SimpleNamespace(ents=[entity])

    dummy_model = DummyModel()

    monkeypatch.setattr(
        "app.services.concept_extraction._load_scispacy_model",
        lambda _: dummy_model,
    )

    config = ConceptExtractionRuntimeConfig(
        max_concepts=5,
        max_tokens=6,
        stopwords=set(STOPWORDS),
        filler_prefixes=set(FILLER_PREFIXES),
        filler_suffixes=set(FILLER_SUFFIXES),
        provider_priority=("scispacy",),
        scispacy_models=("dummy",),
        domain_model=None,
        llm_prompt=None,
        entity_hints={},
        domain_key=None,
    )

    concepts = extract_concepts_from_sections(sections, config=config)
    assert concepts
    linked = [concept for concept in concepts if concept.canonical_id == "UMLS:C12345"]
    assert linked
    resolved = linked[0]
    assert resolved.canonical_score is not None
    assert resolved.canonical_score == pytest.approx(0.87)
    assert resolved.name == "Graph Neural Network"


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


def test_infer_concept_type_uses_strong_method_cues() -> None:
    assert _infer_concept_type("BERT") == "method"
    assert _infer_concept_type("Cas9") == "method"
    assert _infer_concept_type("graph neural network") == "method"
    assert _infer_concept_type("multi-stage curriculum") == "keyword"


def test_method_post_filter_demotes_noisy_phrases() -> None:
    noisy = _Candidate(
        name="The Proposed Multi Stage Training Approach",
        normalized="proposed multi stage training approach",
        type="method",
        description=None,
        score=1.0,
    )
    corroborated = _Candidate(
        name="Baseline Transformer Model",
        normalized="baseline transformer model",
        type="method",
        description=None,
        score=1.0,
        occurrences=2,
    )

    registry = {
        noisy.normalized: noisy,
        corroborated.normalized: corroborated,
    }

    config = ConceptExtractionRuntimeConfig(
        max_concepts=5,
        max_tokens=6,
        stopwords=set(STOPWORDS),
        filler_prefixes=set(FILLER_PREFIXES),
        filler_suffixes=set(FILLER_SUFFIXES),
        provider_priority=("scispacy",),
        scispacy_models=("dummy",),
        domain_model=None,
        llm_prompt=None,
        entity_hints={},
        domain_key=None,
    )

    _apply_method_post_filters(registry, config)

    assert registry[noisy.normalized].type == "keyword"
    assert registry[corroborated.normalized].type == "method"
