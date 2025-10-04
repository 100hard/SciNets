from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID

import asyncpg

from app.db.pool import get_pool
from app.models.ontology import (
    Claim,
    ClaimCreate,
    Dataset,
    Method,
    MethodRelation,
    MethodRelationCreate,
    MethodRelationType,
    Metric,
    Result,
    ResultCreate,
    Task,
)

_METHOD_COLUMNS = "id, name, aliases, description, metadata, created_at, updated_at"
_DATASET_COLUMNS = "id, name, aliases, description, metadata, created_at, updated_at"
_METRIC_COLUMNS = "id, name, unit, aliases, description, metadata, created_at, updated_at"
_TASK_COLUMNS = "id, name, aliases, description, metadata, created_at, updated_at"


_RESULTS_VERIFICATION_SUPPORTED: Optional[bool] = None


_METHOD_RELATION_INSERT_SQL = """
    INSERT INTO method_relations (
        paper_id,
        method_id,
        dataset_id,
        task_id,
        relation_type,
        confidence,
        evidence
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
    RETURNING id, paper_id, method_id, dataset_id, task_id, relation_type,
              confidence, evidence, created_at, updated_at
"""


def _clean_aliases(raw_aliases: Optional[Union[Iterable[str], str]]) -> list[str]:
    if raw_aliases is None:
        candidates: list[Any] = []
    elif isinstance(raw_aliases, str):
        try:
            decoded = json.loads(raw_aliases)
        except json.JSONDecodeError:
            candidates = [raw_aliases]
        else:
            if isinstance(decoded, list):
                candidates = decoded
            elif decoded is None:
                candidates = []
            else:
                candidates = [decoded]
    else:
        candidates = list(raw_aliases)

    cleaned = [
        alias.strip()
        for alias in candidates
        if isinstance(alias, str) and alias.strip()
    ]
    return list(dict.fromkeys(cleaned))


def _clean_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, str):
        try:
            decoded = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return dict(decoded)
    return {}


def _clean_evidence(raw_evidence: Any) -> list[dict[str, Any]]:
    if raw_evidence is None:
        return []
    if isinstance(raw_evidence, list):
        return [item for item in raw_evidence if isinstance(item, dict)]
    if isinstance(raw_evidence, str):
        try:
            decoded = json.loads(raw_evidence)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
        return []
    return []


def _method_from_row(row: Any) -> Method:
    payload = dict(row)
    payload["aliases"] = _clean_aliases(payload.get("aliases"))
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    return Method(**payload)


def _dataset_from_row(row: Any) -> Dataset:
    payload = dict(row)
    payload["aliases"] = _clean_aliases(payload.get("aliases"))
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    return Dataset(**payload)


def _metric_from_row(row: Any) -> Metric:
    payload = dict(row)
    payload["aliases"] = _clean_aliases(payload.get("aliases"))
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    return Metric(**payload)


def _task_from_row(row: Any) -> Task:
    payload = dict(row)
    payload["aliases"] = _clean_aliases(payload.get("aliases"))
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    return Task(**payload)


def _method_relation_from_row(row: Any) -> MethodRelation:
    payload = dict(row)
    payload["evidence"] = _clean_evidence(payload.get("evidence"))
    payload["relation_type"] = MethodRelationType(payload["relation_type"])
    return MethodRelation(**payload)


_RESULT_INSERT_SQL = """
    INSERT INTO results (
        paper_id,
        method_id,
        dataset_id,
        metric_id,
        task_id,
        split,
        value_numeric,
        value_text,
        is_sota,
        confidence,
        evidence
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
    RETURNING id, paper_id, method_id, dataset_id, metric_id, task_id,
              split, value_numeric, value_text, is_sota, confidence,
              evidence, created_at, updated_at
"""


_RESULT_INSERT_SQL_WITH_VERIFICATION = """
    INSERT INTO results (
        paper_id,
        method_id,
        dataset_id,
        metric_id,
        task_id,
        split,
        value_numeric,
        value_text,
        is_sota,
        confidence,
        evidence,
        verified,
        verifier_notes
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13)
    RETURNING id, paper_id, method_id, dataset_id, metric_id, task_id,
              split, value_numeric, value_text, is_sota, confidence,
              evidence, verified, verifier_notes, created_at, updated_at
"""


async def _results_supports_verification(conn: Any) -> bool:
    global _RESULTS_VERIFICATION_SUPPORTED

    if _RESULTS_VERIFICATION_SUPPORTED is not None:
        return _RESULTS_VERIFICATION_SUPPORTED

    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'results'
          AND column_name IN ('verified', 'verifier_notes')
        """
    )
    columns = {row["column_name"] for row in rows}
    _RESULTS_VERIFICATION_SUPPORTED = {
        "verified",
        "verifier_notes",
    }.issubset(columns)
    return _RESULTS_VERIFICATION_SUPPORTED


async def ensure_method(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
) -> Method:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Method name cannot be empty")

    alias_list = _clean_aliases(aliases)

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_METHOD_COLUMNS} FROM methods WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return _method_from_row(row)

        row = await conn.fetchrow(
            """
            INSERT INTO methods (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, metadata, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return _method_from_row(row)


async def ensure_dataset(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
) -> Dataset:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Dataset name cannot be empty")

    alias_list = _clean_aliases(aliases)

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_DATASET_COLUMNS} FROM datasets WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return _dataset_from_row(row)

        row = await conn.fetchrow(
            """
            INSERT INTO datasets (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, metadata, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return _dataset_from_row(row)


async def ensure_metric(
    name: str,
    *,
    unit: Optional[str] = None,
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
) -> Metric:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Metric name cannot be empty")

    alias_list = _clean_aliases(aliases)

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_METRIC_COLUMNS} FROM metrics WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return _metric_from_row(row)

        row = await conn.fetchrow(
            """
            INSERT INTO metrics (name, unit, aliases, description)
            VALUES ($1, $2, $3::jsonb, $4)
            RETURNING id, name, unit, aliases, description, metadata, created_at, updated_at
            """,
            cleaned,
            unit,
            json.dumps(alias_list),
            description,
        )
    return _metric_from_row(row)


async def ensure_task(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
) -> Task:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Task name cannot be empty")

    alias_list = _clean_aliases(aliases)

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_TASK_COLUMNS} FROM tasks WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return _task_from_row(row)

        row = await conn.fetchrow(
            """
            INSERT INTO tasks (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, metadata, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return _task_from_row(row)


async def replace_results(
    paper_id: UUID,
    results: Sequence[ResultCreate],
) -> list[Result]:
    global _RESULTS_VERIFICATION_SUPPORTED
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM results WHERE paper_id = $1", paper_id)
            if not results:
                return []

            inserted: list[Result] = []
            include_verification = await _results_supports_verification(conn)
            insert_sql = (
                _RESULT_INSERT_SQL_WITH_VERIFICATION
                if include_verification
                else _RESULT_INSERT_SQL
            )

            for result in results:
                base_params: list[Any] = [
                    result.paper_id,
                    result.method_id,
                    result.dataset_id,
                    result.metric_id,
                    result.task_id,
                    result.split,
                    result.value_numeric,
                    result.value_text,
                    result.is_sota,
                    result.confidence,
                    json.dumps(result.evidence),
                ]

                while True:
                    params = (
                        [*base_params, result.verified, result.verifier_notes]
                        if include_verification
                        else base_params
                    )
                    try:
                        row = await conn.fetchrow(insert_sql, *params)
                    except asyncpg.UndefinedColumnError:
                        if not include_verification:
                            raise
                        include_verification = False
                        insert_sql = _RESULT_INSERT_SQL
                        _RESULTS_VERIFICATION_SUPPORTED = False
                        continue
                    break

                payload = dict(row)
                payload["evidence"] = _clean_evidence(payload.get("evidence"))
                if not include_verification:
                    payload.setdefault("verified", None)
                    payload.setdefault("verifier_notes", None)
                inserted.append(Result(**payload))
    return inserted


async def append_results(
    paper_id: UUID,
    results: Sequence[ResultCreate],
) -> list[Result]:
    pool = get_pool()
    async with pool.acquire() as conn:
        existing_keys = await _fetch_result_keys(conn, paper_id)
        supports_verification = await _results_supports_verification(conn)
        if not results:
            return []

        inserted: list[Result] = []
        for result in results:
            key = _result_key_from_create(result)
            if key in existing_keys:
                continue
            existing_keys.add(key)

            evidence_json = json.dumps(result.evidence)
            if supports_verification:
                row = await conn.fetchrow(
                    _RESULT_INSERT_SQL_WITH_VERIFICATION,
                    result.paper_id,
                    result.method_id,
                    result.dataset_id,
                    result.metric_id,
                    result.task_id,
                    result.split,
                    result.value_numeric,
                    result.value_text,
                    result.is_sota,
                    result.confidence,
                    evidence_json,
                    result.verified,
                    result.verifier_notes,
                )
            else:
                row = await conn.fetchrow(
                    _RESULT_INSERT_SQL,
                    result.paper_id,
                    result.method_id,
                    result.dataset_id,
                    result.metric_id,
                    result.task_id,
                    result.split,
                    result.value_numeric,
                    result.value_text,
                    result.is_sota,
                    result.confidence,
                    evidence_json,
                )
            payload = dict(row)
            payload["evidence"] = _clean_evidence(payload.get("evidence"))
            inserted.append(Result(**payload))
        return inserted


async def append_method_relations(
    paper_id: UUID,
    relations: Sequence[MethodRelationCreate],
) -> list[MethodRelation]:
    if not relations:
        return []

    pool = get_pool()
    async with pool.acquire() as conn:
        existing_keys = await _fetch_method_relation_keys(conn, paper_id)
        inserted: list[MethodRelation] = []
        for relation in relations:
            key = _method_relation_key_from_create(relation)
            if key in existing_keys:
                continue
            existing_keys.add(key)

            row = await conn.fetchrow(
                _METHOD_RELATION_INSERT_SQL,
                relation.paper_id,
                relation.method_id,
                relation.dataset_id,
                relation.task_id,
                relation.relation_type.value,
                relation.confidence,
                json.dumps(relation.evidence),
            )
            inserted.append(_method_relation_from_row(row))
        return inserted


async def _fetch_result_keys(conn: Any, paper_id: UUID) -> set[tuple[Any, ...]]:
    rows = await conn.fetch(
        """
        SELECT method_id, dataset_id, metric_id, task_id, split, value_text, value_numeric
        FROM results
        WHERE paper_id = $1
        """,
        paper_id,
    )
    return {
        (
            row["method_id"],
            row["dataset_id"],
            row["metric_id"],
            row["task_id"],
            row["split"],
            (row["value_text"] or "").strip(),
            str(row["value_numeric"]) if row["value_numeric"] is not None else None,
        )
        for row in rows
    }


def _result_key_from_create(result: ResultCreate) -> tuple[Any, ...]:
    return (
        result.method_id,
        result.dataset_id,
        result.metric_id,
        result.task_id,
        result.split,
        (result.value_text or "").strip(),
        str(result.value_numeric) if result.value_numeric is not None else None,
    )


async def _fetch_method_relation_keys(conn: Any, paper_id: UUID) -> set[tuple[Any, ...]]:
    rows = await conn.fetch(
        """
        SELECT method_id, dataset_id, task_id, relation_type
        FROM method_relations
        WHERE paper_id = $1
        """,
        paper_id,
    )
    return {
        (
            row["method_id"],
            row["dataset_id"],
            row["task_id"],
            MethodRelationType(row["relation_type"]),
        )
        for row in rows
    }


def _method_relation_key_from_create(
    relation: MethodRelationCreate,
) -> tuple[Any, ...]:
    return (
        relation.method_id,
        relation.dataset_id,
        relation.task_id,
        relation.relation_type,
    )


async def fetch_results_by_paper(paper_id: UUID) -> list[Result]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, paper_id, method_id, dataset_id, metric_id, task_id, split,
                   value_numeric, value_text, is_sota, confidence, evidence,
                   created_at, updated_at
            FROM results
            WHERE paper_id = $1
            """,
            paper_id,
        )
        inserted: list[Result] = []
        for row in rows:
            payload = dict(row)
            payload["evidence"] = _clean_evidence(payload.get("evidence"))
            payload.setdefault("verified", None)
            payload.setdefault("verifier_notes", None)
            inserted.append(Result(**payload))
        return inserted


async def replace_claims(
    paper_id: UUID,
    claims: Sequence[ClaimCreate],
) -> list[Claim]:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM claims WHERE paper_id = $1", paper_id)
            if not claims:
                return []

            inserted: list[Claim] = []
            for claim in claims:
                category_value = (
                    claim.category.value
                    if isinstance(claim.category, Enum)
                    else claim.category
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO claims (paper_id, category, text, confidence, evidence)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    RETURNING id, paper_id, category, text, confidence, evidence, created_at, updated_at
                    """,
                    claim.paper_id,
                    category_value,
                    claim.text,
                    claim.confidence,
                    json.dumps(claim.evidence),
                )
                payload = dict(row)
                payload["evidence"] = _clean_evidence(payload.get("evidence"))
                inserted.append(Claim(**payload))
            return inserted

