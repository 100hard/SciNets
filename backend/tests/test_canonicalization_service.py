import asyncio
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID, uuid4

import pytest

from app.models.ontology import ConceptResolutionType
from app.services import canonicalization as canonicalization_service


METHOD_A = UUID("11111111-1111-1111-1111-111111111111")
METHOD_B = UUID("22222222-2222-2222-2222-222222222222")
METHOD_C = UUID("33333333-3333-3333-3333-333333333333")


class FakeEmbeddingBackend:
    async def embed(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:  # noqa: ARG002
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.lower()
            if "roberta" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "bert" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeTransaction:
    def __init__(self, conn: "FakeCanonicalizationConnection") -> None:
        self._conn = conn
        self._snapshot: dict[str, Any] | None = None

    async def __aenter__(self) -> "FakeTransaction":
        self._snapshot = self._conn.snapshot()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - nothing to clean up
        if exc_type is not None and self._snapshot is not None:
            self._conn.restore_snapshot(self._snapshot)
        self._snapshot = None
        return False


class FakeAcquire:
    def __init__(self, conn: "FakeCanonicalizationConnection") -> None:
        self._conn = conn

    async def __aenter__(self) -> "FakeCanonicalizationConnection":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - nothing to clean up
        return False


class FakePool:
    def __init__(self, conn: "FakeCanonicalizationConnection") -> None:
        self._conn = conn

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._conn)


class FakeCanonicalizationConnection:
    def __init__(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        later = datetime(2024, 2, 1, tzinfo=timezone.utc)
        self.methods: dict[UUID, dict[str, Any]] = {
            METHOD_A: {
                "id": METHOD_A,
                "name": "RoBERTa",
                "aliases": ["Roberta Model"],
                "created_at": now,
            },
            METHOD_B: {
                "id": METHOD_B,
                "name": "RoBERTa-Large",
                "aliases": [],
                "created_at": later,
            },
            METHOD_C: {
                "id": METHOD_C,
                "name": "BERT",
                "aliases": [],
                "created_at": later,
            },
        }
        self.results: list[dict[str, Any]] = [
            {"id": uuid4(), "method_id": METHOD_A},
            {"id": uuid4(), "method_id": METHOD_B},
            {"id": uuid4(), "method_id": METHOD_C},
        ]
        self.concept_resolutions: list[dict[str, Any]] = []
        self.canonicalization_merge_decisions: list[dict[str, Any]] = []
        self.fail_on_audit_insert = False

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:  # noqa: ARG002
        normalized = " ".join(query.split())
        if normalized == "SELECT id, name, aliases, created_at FROM methods":
            return [dict(record) for record in self.methods.values()]
        raise AssertionError(f"Unsupported fetch query: {normalized}")

    async def execute(self, query: str, *params: Any) -> str:
        normalized = " ".join(query.split())
        if normalized.startswith("DELETE FROM concept_resolutions"):
            types = set(params[0])
            self.concept_resolutions = [
                row for row in self.concept_resolutions if row["resolution_type"] not in types
            ]
            return "DELETE"
        if normalized.startswith("DELETE FROM canonicalization_merge_decisions"):
            types = set(params[0])
            version = params[1]
            self.canonicalization_merge_decisions = [
                row
                for row in self.canonicalization_merge_decisions
                if row["resolution_type"] not in types or row["mapping_version"] != version
            ]
            return "DELETE"
        raise AssertionError(f"Unsupported execute query: {normalized}")

    async def executemany(self, query: str, param_sets: Sequence[Sequence[Any]]) -> None:
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO concept_resolutions"):
            for resolution_type, canonical_id, alias_text, score in param_sets:
                self.concept_resolutions.append(
                    {
                        "resolution_type": resolution_type,
                        "canonical_id": canonical_id,
                        "alias_text": alias_text,
                        "score": score,
                    }
                )
            return
        if normalized.startswith("INSERT INTO canonicalization_merge_decisions"):
            if self.fail_on_audit_insert:
                raise RuntimeError("Simulated failure")
            for (
                resolution_type,
                left_id,
                right_id,
                score,
                decision_source,
                verdict,
                rationale,
                mapping_version,
                adjudicator_metadata,
            ) in param_sets:
                self.canonicalization_merge_decisions.append(
                    {
                        "resolution_type": resolution_type,
                        "left_id": left_id,
                        "right_id": right_id,
                        "score": score,
                        "decision_source": decision_source,
                        "verdict": verdict,
                        "rationale": rationale,
                        "mapping_version": mapping_version,
                        "adjudicator_metadata": adjudicator_metadata,
                    }
                )
            return
        if normalized.startswith("UPDATE methods SET aliases"):
            for record_id, aliases in param_sets:
                self.methods[record_id]["aliases"] = list(aliases)
            return
        if normalized.startswith("UPDATE results SET method_id"):
            for old_id, new_id in param_sets:
                for row in self.results:
                    if row.get("method_id") == old_id:
                        row["method_id"] = new_id
            return
        raise AssertionError(f"Unsupported executemany query: {normalized}")

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    def snapshot(self) -> dict[str, Any]:
        return {
            "methods": deepcopy(self.methods),
            "results": deepcopy(self.results),
            "concept_resolutions": deepcopy(self.concept_resolutions),
            "canonicalization_merge_decisions": deepcopy(
                self.canonicalization_merge_decisions
            ),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.methods = deepcopy(snapshot["methods"])
        self.results = deepcopy(snapshot["results"])
        self.concept_resolutions = deepcopy(snapshot["concept_resolutions"])
        self.canonicalization_merge_decisions = deepcopy(
            snapshot["canonicalization_merge_decisions"]
        )


def test_canonicalize_service_merges_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_canonicalize_service_merges_methods(monkeypatch))


async def _run_canonicalize_service_merges_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeEmbeddingBackend()
    conn = FakeCanonicalizationConnection()
    pool = FakePool(conn)

    monkeypatch.setattr(canonicalization_service, "_embedding_backend", backend, raising=False)
    monkeypatch.setattr(canonicalization_service, "get_pool", lambda: pool)

    report = await canonicalization_service.canonicalize([ConceptResolutionType.METHOD])
    assert report.summary
    entry = report.summary[0]
    assert entry.resolution_type == ConceptResolutionType.METHOD
    assert entry.before == 3
    assert entry.after == 2
    assert entry.merges == 1
    assert entry.examples
    example = entry.examples[0]
    assert example.canonical_id == METHOD_A
    merged_ids = {item.id for item in example.merged}
    assert merged_ids == {METHOD_B}
    assert example.merged[0].score >= 0.85

    aliases_a = set(conn.methods[METHOD_A]["aliases"])
    aliases_b = set(conn.methods[METHOD_B]["aliases"])
    aliases_c = conn.methods[METHOD_C]["aliases"]
    assert aliases_a == {"Roberta Model", "RoBERTa-Large"}
    assert aliases_b == {"RoBERTa", "Roberta Model"}
    assert aliases_c == []

    alias_map = defaultdict(set)
    for row in conn.concept_resolutions:
        if row["resolution_type"] == ConceptResolutionType.METHOD.value:
            alias_map[row["canonical_id"]].add(row["alias_text"])

    assert alias_map[METHOD_A] == {"RoBERTa", "RoBERTa-Large", "Roberta Model"}
    assert alias_map[METHOD_C] == {"BERT"}

    method_ids = {row["method_id"] for row in conn.results}
    assert METHOD_B not in method_ids
    assert METHOD_A in method_ids

    audit_rows = [
        row
        for row in conn.canonicalization_merge_decisions
        if row["resolution_type"] == ConceptResolutionType.METHOD.value
    ]
    assert len(audit_rows) == 1
    audit = audit_rows[0]
    assert audit["left_id"] == METHOD_A
    assert audit["right_id"] == METHOD_B
    assert audit["decision_source"] == "llm"
    assert audit["verdict"] == "accepted"
    assert audit["mapping_version"] == canonicalization_service.settings.canonicalization_mapping_version
    assert audit["adjudicator_metadata"] is None
    assert audit["score"] >= 0.85
    assert "similarity" in audit["rationale"].lower()


def test_canonicalize_rolls_back_on_audit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_canonicalize_rolls_back_on_audit_failure(monkeypatch))


async def _run_canonicalize_rolls_back_on_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeEmbeddingBackend()
    conn = FakeCanonicalizationConnection()
    conn.fail_on_audit_insert = True
    pool = FakePool(conn)

    baseline = conn.snapshot()

    monkeypatch.setattr(canonicalization_service, "_embedding_backend", backend, raising=False)
    monkeypatch.setattr(canonicalization_service, "get_pool", lambda: pool)

    with pytest.raises(RuntimeError):
        await canonicalization_service.canonicalize([ConceptResolutionType.METHOD])

    assert conn.methods == baseline["methods"]
    assert conn.results == baseline["results"]
    assert conn.concept_resolutions == baseline["concept_resolutions"]
    assert (
        conn.canonicalization_merge_decisions
        == baseline["canonicalization_merge_decisions"]
    )
