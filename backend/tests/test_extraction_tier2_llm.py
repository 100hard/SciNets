import asyncio
import json
from uuid import uuid4

import pytest

from app.services import extraction_tier2


def _build_sections():
    return [
        {
            "id": "sec-1",
            "title": "Methodology",
            "content": "We evaluate Method Alpha on the Benchmark dataset.",
            "char_start": 0,
            "char_end": 55,
        },
        {
            "id": "sec-2",
            "title": "Results",
            "content": "Method Alpha reaches 92.4 accuracy on Benchmark (test split).",
            "char_start": 55,
            "char_end": 120,
        },
    ]


def test_normalise_llm_payload_coerces_structures():
    sections = _build_sections()
    raw_payload = json.dumps(
        {
            "paper_title": "Original Title",
            "methods": [
                {"name": "Method Alpha", "aliases": ["Method alpha", ""]},
                {"name": "method alpha"},
            ],
            "datasets": ["Benchmark", "benchmark"],
            "metrics": ["Accuracy"],
            "tasks": ["Image Classification"],
            "results": [
                {
                    "method": "Method Alpha",
                    "dataset": "Benchmark",
                    "metric": "Accuracy",
                    "value": "92.4",
                    "split": "test",
                    "task": None,
                    "evidence_span": {"section_id": "sec-2", "start": 7, "end": 20},
                }
            ],
            "claims": [
                {
                    "category": "sota",
                    "text": "Method Alpha sets a new state of the art on Benchmark",
                    "evidence_span": {"section_id": "sec-2", "start": 0, "end": 40},
                }
            ],
        }
    )

    normalised = extraction_tier2._normalise_llm_payload(
        raw_payload, paper_title="Overridden", sections=sections
    )

    assert normalised["paper_title"] == "Original Title"
    assert normalised["methods"] == [
        {"name": "Method Alpha", "aliases": ["Method alpha"]}
    ]
    assert normalised["datasets"] == ["Benchmark"]
    assert normalised["metrics"] == ["Accuracy"]
    assert normalised["tasks"] == ["Image Classification"]

    result = normalised["results"][0]
    assert result["method"] == "Method Alpha"
    assert result["dataset"] == "Benchmark"
    assert result["metric"] == "Accuracy"
    assert result["value"] == "92.4"
    assert result["split"] == "test"
    assert result["evidence_span"] == {"section_id": "sec-2", "start": 7, "end": 20}

    claim = normalised["claims"][0]
    assert claim["category"] == "sota"
    assert claim["text"].startswith("Method Alpha")
    assert claim["evidence_span"] == {"section_id": "sec-2", "start": 0, "end": 40}


@pytest.mark.parametrize(
    "raw_payload, expected",
    [
        (
            {
                "methods": [{"name": ""}, "invalid"],
                "datasets": "Benchmark",
                "metrics": [123, "F1"],
                "results": [
                    {
                        "method": "Method Alpha",
                        "dataset": "Benchmark",
                        "metric": "F1",
                        "value": True,
                        "evidence_span": {"section_id": "missing", "start": -5, "end": 999},
                    },
                    ["not", "a", "dict"],
                ],
                "claims": [
                    {
                        "category": "future_work",
                        "text": "We plan to release code.",
                        "evidence_span": {"section_id": "sec-1", "start": "0", "end": "10"},
                    },
                    "invalid",
                ],
            },
            {
                "methods": [],
                "datasets": [],
                "metrics": ["F1"],
                "results": [
                    {
                        "method": "Method Alpha",
                        "dataset": "Benchmark",
                        "metric": "F1",
                        "value": "true",
                        "split": None,
                        "task": None,
                        "evidence_span": {"section_id": "sec-1", "start": 0, "end": 50},
                    }
                ],
                "claims": [
                    {
                        "category": "future_work",
                        "text": "We plan to release code.",
                        "evidence_span": {"section_id": "sec-1", "start": 0, "end": 10},
                    }
                ],
            },
        ),
    ],
)
def test_normalise_llm_payload_discards_invalid_entries(raw_payload, expected):
    sections = _build_sections()
    payload = json.dumps(raw_payload)
    normalised = extraction_tier2._normalise_llm_payload(
        payload, paper_title="Paper", sections=sections
    )

    assert normalised["methods"] == expected["methods"]
    assert normalised["datasets"] == expected["datasets"]
    assert normalised["metrics"] == expected["metrics"]
    assert normalised["results"] == expected["results"]
    assert normalised["claims"] == expected["claims"]


def test_summary_inputs_with_null_collections_are_ignored():
    summary = {
        "methods": None,
        "datasets": None,
        "metrics": None,
        "tasks": None,
        "results": None,
        "claims": None,
        "tiers": None,
    }
    caches = extraction_tier2._Caches(methods={}, datasets={}, metrics={}, tasks={})

    async def exercise() -> tuple[list, list]:
        # Ensure catalog operations gracefully skip missing collections.
        await extraction_tier2._ensure_catalog_from_summary(summary, caches)

        # Conversion helpers should tolerate absent inputs and yield empty lists.
        converted_results = await extraction_tier2._convert_summary_results(
            uuid4(), summary, caches
        )
        converted_claims = await extraction_tier2._convert_summary_claims(
            uuid4(), summary
        )
        return converted_results, converted_claims

    converted_results, converted_claims = asyncio.run(exercise())

    assert converted_results == []
    assert converted_claims == []
    assert extraction_tier2._merge_tiers(summary.get("tiers"), [2]) == [2]
