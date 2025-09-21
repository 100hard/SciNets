import asyncio
from uuid import UUID

import pytest

from app.api.admin import api_canonicalize
from app.models.ontology import (
    CanonicalizationExample,
    CanonicalizationMergedItem,
    CanonicalizationReport,
    CanonicalizationTypeReport,
    ConceptResolutionType,
)


EXAMPLE_CANONICAL = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def test_api_canonicalize_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_canonicalize(monkeypatch))


async def _run_api_canonicalize(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = CanonicalizationReport(
        summary=[
            CanonicalizationTypeReport(
                resolution_type=ConceptResolutionType.METHOD,
                before=3,
                after=2,
                merges=1,
                examples=[
                    CanonicalizationExample(
                        canonical_id=EXAMPLE_CANONICAL,
                        canonical_name="RoBERTa",
                        merged=[
                            CanonicalizationMergedItem(
                                id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                                name="RoBERTa-Large",
                                score=0.91,
                            )
                        ],
                    )
                ],
            )
        ]
    )

    async def fake_canonicalize(types):  # noqa: ANN001
        assert types == [ConceptResolutionType.METHOD, ConceptResolutionType.DATASET]
        return expected

    monkeypatch.setattr("app.api.admin.canonicalize", fake_canonicalize)

    response = await api_canonicalize(
        types=[ConceptResolutionType.METHOD, ConceptResolutionType.DATASET]
    )
    assert response == expected


def test_api_canonicalize_defaults_to_all_types(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_api_canonicalize_defaults(monkeypatch))


async def _run_api_canonicalize_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_canonicalize(types):  # noqa: ANN001
        assert types == list(ConceptResolutionType)
        return CanonicalizationReport(summary=[])

    monkeypatch.setattr("app.api.admin.canonicalize", fake_canonicalize)

    response = await api_canonicalize(types=None)
    assert response.summary == []
