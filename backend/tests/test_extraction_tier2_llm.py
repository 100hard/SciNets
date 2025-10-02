from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from app.core.config import settings
from app.schemas.tier2 import TripleExtractionResponse, TriplePayload
from app.services import extraction_tier2


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_section(section_id: str, title: str, sentences: list[str]) -> dict[str, object]:
    return {
        "section_id": section_id,
        "title": title,
        "section_hash": f"hash-{section_id}",
        "page_number": 1,
        "char_start": 0,
        "char_end": sum(len(sentence) for sentence in sentences),
        "sentence_spans": [
            {"start": idx * 10, "end": idx * 10 + len(sentence), "text": sentence}
            for idx, sentence in enumerate(sentences)
        ],
    }


def test_prepare_section_contexts_splits_long_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    section = _build_section(
        "sec-long",
        "Analysis",
        [
            "Sentence 0 provides background on the approach and includes multiple clauses to be long.",
            "Sentence 1 continues the discussion with more detail and elaboration for testing purposes.",
            "Sentence 2 introduces additional results that ensure chunking will require a third chunk eventually.",
            "Sentence 3 adds further elaboration so that there is enough material for more chunks in the dataset.",
            "Sentence 4 concludes the section but is still lengthy to challenge the chunking routine effectively.",
        ],
    )
    section["captions"] = [{"text": "Figure 1. Chunk demo."}]

    monkeypatch.setattr(settings, "tier2_llm_section_chunk_chars", 280)
    monkeypatch.setattr(settings, "tier2_llm_section_chunk_overlap_sentences", 1)
    monkeypatch.setattr(settings, "tier2_llm_max_chunks_per_section", 10)
    monkeypatch.setattr(settings, "tier2_llm_max_sections", 2)

    contexts = extraction_tier2._prepare_section_contexts([section])

    assert len(contexts) == 2
    assert contexts[0].chunk_id and contexts[1].chunk_id
    assert contexts[0].chunk_id.endswith("01")
    assert contexts[1].chunk_id.endswith("02")
    assert contexts[0].captions == ["Figure 1. Chunk demo."]
    assert contexts[1].captions == contexts[0].captions
    assert contexts[0].sentences[-1][0] == contexts[1].sentences[0][0]


@pytest.mark.anyio
async def test_run_tier2_structurer_parses_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Method",
                [
                    "We introduce AlphaNet, a transformer-based model.",
                    "AlphaNet is evaluated on the WMT14 En-Fr dataset.",
                ],
            ),
            _build_section(
                "sec-2",
                "Results",
                [
                    "AlphaNet achieves BLEU 41.8 on WMT14 En-Fr test set.",
                ],
            ),
        ],
        "tables": [],
    }

    fake_payload = {
        "triples": [
            {
                "subject": "AlphaNet",
                "relation": "achieves",
                "object": "BLEU 41.8 on WMT14 En-Fr test set",
                "evidence": "AlphaNet achieves BLEU 41.8 on WMT14 En-Fr test set.",
                "subject_span": [0, 8],
                "object_span": [18, 48],
                "subject_type_guess": "Method",
                "relation_type_guess": "MEASURES",
                "object_type_guess": "Metric",
                "triple_conf": 0.62,
                "schema_match_score": 0.97,
                "section_id": "sec-2",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    captured_messages: list[list[dict[str, str]]] = []
    persisted: dict[str, object] = {}
    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(messages: list[dict[str, str]]) -> str:
        captured_messages.append(messages)
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(paper: UUID, candidates: list) -> None:
        persisted["paper_id"] = paper
        persisted["candidates"] = list(candidates)

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    assert summary["tiers"] == [1, 2]
    candidates = summary["triple_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["subject"] == "AlphaNet"
    assert candidate["relation_type_guess"] == "MEASURES"
    assert candidate["subject_span"] == [0, 8]
    assert candidate["section_id"] == "sec-2"
    assert candidate["candidate_id"].startswith("tier2_llm_openie_")
    assert summary["metadata"]["tier2"]["triple_count"] == 1
    guardrails = summary["metadata"]["tier2"]["guardrails"]
    assert guardrails["deduplicated_triples"] == 0
    assert guardrails["unmatched_evidence"] == 0
    assert guardrails["metrics_inferred"] == 0
    assert len(captured_messages) == 2
    comparison_prompt = captured_messages[1][1]["content"]
    assert "comparison" in comparison_prompt.lower()

    assert persisted.get("paper_id") == paper_id
    stored = persisted.get("candidates") or []
    assert len(stored) == 1
    stored_candidate = stored[0]
    assert stored_candidate.subject == "AlphaNet"
    assert stored_candidate.section_id == "sec-2"
    assert stored_candidate.object_span == [18, 48]


@pytest.mark.anyio
async def test_run_tier2_structurer_populates_chunk_id(monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = uuid4()
    sentences = [
        "The introduction sentence elaborates on the motivation for the study and is intentionally long.",
        "The methods sentence contains specific phrasing that will be used as evidence for chunk two.",
        "The results sentence highlights the improvements achieved by the proposed system in extensive detail.",
    ]
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-chunk",
                "Methods",
                sentences,
            ),
        ],
        "tables": [],
    }
    base_summary["sections"][0]["captions"] = [{"text": "Table 1 summarizes results."}]

    monkeypatch.setattr(settings, "tier2_llm_section_chunk_chars", 170)
    monkeypatch.setattr(settings, "tier2_llm_section_chunk_overlap_sentences", 1)
    monkeypatch.setattr(settings, "tier2_llm_max_chunks_per_section", 5)
    monkeypatch.setattr(settings, "tier2_llm_max_sections", 5)

    contexts = extraction_tier2._prepare_section_contexts(base_summary["sections"])
    assert len(contexts) >= 2
    target_chunk = contexts[1]
    evidence_sentence = target_chunk.sentences[-1][1]

    fake_payload = {
        "triples": [
            {
                "subject": "ChunkNet",
                "relation": "uses",
                "object": "special phrasing",
                "evidence": evidence_sentence,
                "subject_span": [0, 8],
                "object_span": [25, 40],
                "subject_type_guess": "Method",
                "relation_type_guess": "USES",
                "object_type_guess": "OtherScientificTerm",
                "triple_conf": 0.7,
                "schema_match_score": 0.9,
                "section_id": "sec-chunk",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(_: UUID, __: list) -> None:
        return None

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    candidates = summary["triple_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["section_id"] == "sec-chunk"
    assert candidate["chunk_id"] == target_chunk.chunk_id
    evidence_spans = candidate["evidence_spans"]
    assert evidence_spans and evidence_spans[0]["chunk_id"] == target_chunk.chunk_id


@pytest.mark.anyio
async def test_run_tier2_structurer_omits_graph_metadata_for_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Results",
                [
                    "AlphaNet references Phys. Rev. Lett. 120, 123456 (2018).",
                ],
            ),
        ],
        "tables": [],
    }

    fake_payload = {
        "triples": [
            {
                "subject": "AlphaNet",
                "relation": "evaluated on",
                "object": "Phys. Rev. Lett. 120, 123456 (2018)",
                "evidence": "AlphaNet references Phys. Rev. Lett. 120, 123456 (2018).",
                "subject_span": [0, 8],
                "object_span": [21, 60],
                "subject_type_guess": "Method",
                "relation_type_guess": "EVALUATED_ON",
                "object_type_guess": "Dataset",
                "triple_conf": 0.6,
                "schema_match_score": 0.9,
                "section_id": "sec-1",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(_: UUID, __: list) -> None:
        return None

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    assert len(summary["triple_candidates"]) == 1
    candidate = summary["triple_candidates"][0]
    assert candidate["object"] == "Phys. Rev. Lett. 120, 123456 (2018)"
    assert "graph_metadata" not in candidate


@pytest.mark.anyio
async def test_run_tier2_structurer_skips_graph_metadata_for_non_eval_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Method",
                [
                    "We introduce AlphaNet, a transformer-based model.",
                    "AlphaNet references the WMT14 En-Fr dataset in passing.",
                ],
            ),
        ],
        "tables": [],
    }

    fake_payload = {
        "triples": [
            {
                "subject": "AlphaNet",
                "relation": "mentions",
                "object": "WMT14 En-Fr dataset",
                "evidence": "AlphaNet references the WMT14 En-Fr dataset in passing.",
                "subject_span": [0, 8],
                "object_span": [28, 48],
                "subject_type_guess": "Method",
                "relation_type_guess": "OTHER",
                "object_type_guess": "Dataset",
                "triple_conf": 0.64,
                "schema_match_score": 0.92,
                "section_id": "sec-1",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(_: UUID, __: list) -> None:
        return None

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    candidate = summary["triple_candidates"][0]
    assert candidate["object"] == "WMT14 En-Fr dataset"
    assert "graph_metadata" not in candidate


@pytest.mark.anyio
async def test_run_tier2_structurer_resolves_pronoun_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Results",
                [
                    "SuperNet is introduced as a transformer model.",
                    "It achieves BLEU 30.0 on WMT14 test set.",
                ],
            ),
        ],
        "tables": [],
    }

    fake_payload = {
        "triples": [
            {
                "subject": "It",
                "relation": "achieves",
                "object": "BLEU 30.0 on WMT14 test set",
                "evidence": "It achieves BLEU 30.0 on WMT14 test set.",
                "subject_span": [0, 2],
                "object_span": [13, 43],
                "subject_type_guess": "Method",
                "relation_type_guess": "MEASURES",
                "object_type_guess": "Metric",
                "triple_conf": 0.6,
                "schema_match_score": 0.9,
                "section_id": "sec-1",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(_: UUID, __: list) -> None:
        return None

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    candidate = summary["triple_candidates"][0]
    assert candidate["subject"] == "SuperNet"
    guardrails = summary["metadata"]["tier2"]["guardrails"]
    assert guardrails["pronoun_resolved_subjects"] == 1
    assert guardrails["low_info_subject_dropped"] == 0
    assert guardrails["unmatched_evidence"] == 0








@pytest.mark.anyio
async def test_run_tier2_structurer_dedupes_cross_pass_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Results",
                [
                    "Method Alpha compares favorably to Method Beta on Dataset-X.",
                ],
            ),
        ],
        "tables": [],
    }

    shared_triple = {
        "subject": "Method Alpha",
        "relation": "compares to",
        "object": "Method Beta on Dataset-X",
        "evidence": "Method Alpha compares favorably to Method Beta on Dataset-X.",
        "subject_span": [0, 12],
        "object_span": [33, 55],
        "subject_type_guess": "Method",
        "relation_type_guess": "COMPARED_TO",
        "object_type_guess": "Method",
        "triple_conf": 0.6,
        "schema_match_score": 0.9,
        "section_id": "sec-1",
    }

    primary_payload = {
        "triples": [shared_triple],
        "warnings": ["Duplicate triple warning", "Spacing issue "],
        "discarded": ["Row 1"],
    }
    comparison_payload = {
        "triples": [dict(shared_triple)],
        "warnings": ["duplicate triple warning", "Comparison only"],
        "discarded": ["row 1 ", "Row 2"],
    }

    payload_iter = iter([primary_payload, comparison_payload])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    persisted: dict[str, object] = {}

    async def fake_replace(paper: UUID, candidates: list) -> None:
        persisted["paper_id"] = paper
        persisted["count"] = len(candidates)

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    assert len(summary["triple_candidates"]) == 1
    guardrails = summary["metadata"]["tier2"]["guardrails"]
    assert guardrails["deduplicated_triples"] == 1
    tier2_meta = summary["metadata"]["tier2"]
    assert tier2_meta["warnings"] == ["Duplicate triple warning", "Spacing issue ", "Comparison only"]
    assert tier2_meta["discarded"] == ["Row 1", "Row 2"]
    assert persisted.get("paper_id") == paper_id
    assert persisted.get("count") == 1


@pytest.mark.anyio
async def test_run_tier2_structurer_infers_metric_from_synonym(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [
            _build_section(
                "sec-1",
                "Results",
                [
                    "Baseline model reports a misclassification rate of 5%.",
                ],
            ),
        ],
        "tables": [],
    }

    fake_payload = {
        "triples": [
            {
                "subject": "Baseline model",
                "relation": "reports",
                "object": "misclassification rate of 5%",
                "evidence": "Baseline model reports a misclassification rate of 5%.",
                "subject_span": [0, 14],
                "object_span": [29, 58],
                "subject_type_guess": "Method",
                "relation_type_guess": "MEASURES",
                "object_type_guess": "Concept",
                "triple_conf": 0.6,
                "schema_match_score": 0.9,
                "section_id": "sec-1",
            }
        ],
        "warnings": [],
        "discarded": [],
    }

    payload_iter = iter([
        fake_payload,
        {"triples": [], "warnings": [], "discarded": []},
    ])

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        payload = next(payload_iter, {"triples": [], "warnings": [], "discarded": []})
        return json.dumps(payload)

    async def fake_replace(_: UUID, __: list) -> None:
        return None

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    candidate = summary["triple_candidates"][0]
    assert candidate["metric_inference"]["normalized_metric"] == "Accuracy"
    assert candidate["metric_inference"]["confidence_penalty"] == pytest.approx(0.05)
    assert candidate["triple_conf"] == pytest.approx(0.55)

    guardrails = summary["metadata"]["tier2"]["guardrails"]
    assert guardrails["metrics_inferred"] == 1

@pytest.mark.anyio
async def test_run_tier2_structurer_needs_review_after_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = uuid4()
    base_summary = {
        "paper_id": str(paper_id),
        "tiers": [1],
        "sections": [_build_section("sec-1", "Intro", ["AlphaNet is proposed."])],
        "tables": [],
    }

    call_count = {"invocations": 0}

    async def fake_invoke_llm(_: list[dict[str, str]]) -> str:
        call_count["invocations"] += 1
        return "not-json"

    async def fake_replace(_: UUID, __: list) -> None:  # pragma: no cover - should not be called
        raise AssertionError("replace_triple_candidates should not be invoked on failure")

    monkeypatch.setattr(extraction_tier2, "_invoke_llm", fake_invoke_llm)
    monkeypatch.setattr(extraction_tier2, "replace_triple_candidates", fake_replace)

    monkeypatch.setattr(settings, "tier2_llm_model", "gpt-test")
    monkeypatch.setattr(settings, "tier2_llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "tier2_llm_completion_path", "/chat/completions")
    monkeypatch.setattr(settings, "tier2_llm_force_json", True)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_organization", None)

    summary = await extraction_tier2.run_tier2_structurer(paper_id, base_summary=base_summary)

    assert call_count["invocations"] == extraction_tier2.MAX_LLM_ATTEMPTS
    assert "needs_review" in summary
    entry = summary["needs_review"][0]
    assert entry["tier"] == extraction_tier2.TIER_NAME
    assert summary["metadata"]["tier2"]["status"] == "needs_review"
    assert summary.get("triple_candidates", []) == []


def test_merge_payloads_dedupes_entries() -> None:
    payload_a = TripleExtractionResponse(
        triples=[
            TriplePayload(
                subject="Method Alpha",
                relation="uses",
                object="Dataset Foo",
                evidence="Method Alpha uses Dataset Foo.",
                subject_span=[0, 12],
                object_span=[18, 29],
                subject_type_guess="Method",
                relation_type_guess="USES",
                object_type_guess="Dataset",
                triple_conf=0.6,
                schema_match_score=0.9,
                section_id="sec-1",
            )
        ],
        warnings=["Duplicate triple warning", "Spacing issue "],
        discarded=["Row 1", "Row 2"],
    )
    payload_b = TripleExtractionResponse(
        triples=[
            TriplePayload(
                subject="Method Beta",
                relation="reports",
                object="Accuracy 92%",
                evidence="Method Beta reports Accuracy 92%.",
                subject_span=[0, 11],
                object_span=[22, 33],
                subject_type_guess="Method",
                relation_type_guess="MEASURES",
                object_type_guess="Metric",
                triple_conf=0.7,
                schema_match_score=0.95,
                section_id="sec-2",
            )
        ],
        warnings=["duplicate triple warning", "Another note"],
        discarded=["row 1", "Row 3"],
    )

    merged = extraction_tier2._merge_payloads([payload_a, payload_b])

    assert len(merged.triples) == 2
    assert merged.warnings == [
        "Duplicate triple warning",
        "Spacing issue ",
        "Another note",
    ]
    assert merged.discarded == ["Row 1", "Row 2", "Row 3"]


def test_validate_payload_allows_single_value_spans() -> None:
    payload = {
        "triples": [
            {
                "subject": "AB",
                "relation": "improves",
                "object": "accuracy",
                "evidence": "AB improves accuracy on the benchmark dataset.",
                "subject_span": [11],
                "object_span": [22],
            }
        ]
    }

    validated = extraction_tier2._validate_payload(payload)
    triple = validated.triples[0]

    assert triple.subject_span == [11, 11]
    assert triple.object_span == [22, 22]
