from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from uuid import UUID

from app.db.pool import get_pool
from app.models.ontology import (
    Claim,
    ClaimCreate,
    Dataset,
    Method,
    Metric,
    Result,
    ResultCreate,
    Task,
)


_METHOD_COLUMNS = "id, name, aliases, description, created_at, updated_at"
_DATASET_COLUMNS = "id, name, aliases, description, created_at, updated_at"
_METRIC_COLUMNS = "id, name, unit, aliases, description, created_at, updated_at"
_TASK_COLUMNS = "id, name, aliases, description, created_at, updated_at"


async def ensure_method(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
) -> Method:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Method name cannot be empty")

    alias_list = list(dict.fromkeys(alias.strip() for alias in (aliases or []) if alias))

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_METHOD_COLUMNS} FROM methods WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return Method(**dict(row))

        row = await conn.fetchrow(
            """
            INSERT INTO methods (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return Method(**dict(row))


async def ensure_dataset(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
) -> Dataset:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Dataset name cannot be empty")

    alias_list = list(dict.fromkeys(alias.strip() for alias in (aliases or []) if alias))

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_DATASET_COLUMNS} FROM datasets WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return Dataset(**dict(row))

        row = await conn.fetchrow(
            """
            INSERT INTO datasets (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return Dataset(**dict(row))


async def ensure_metric(
    name: str,
    *,
    unit: str | None = None,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
) -> Metric:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Metric name cannot be empty")

    alias_list = list(dict.fromkeys(alias.strip() for alias in (aliases or []) if alias))

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_METRIC_COLUMNS} FROM metrics WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return Metric(**dict(row))

        row = await conn.fetchrow(
            """
            INSERT INTO metrics (name, unit, aliases, description)
            VALUES ($1, $2, $3::jsonb, $4)
            RETURNING id, name, unit, aliases, description, created_at, updated_at
            """,
            cleaned,
            unit,
            json.dumps(alias_list),
            description,
        )
    return Metric(**dict(row))


async def ensure_task(
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
) -> Task:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Task name cannot be empty")

    alias_list = list(dict.fromkeys(alias.strip() for alias in (aliases or []) if alias))

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_TASK_COLUMNS} FROM tasks WHERE lower(name) = lower($1) LIMIT 1",
            cleaned,
        )
        if row:
            return Task(**dict(row))

        row = await conn.fetchrow(
            """
            INSERT INTO tasks (name, aliases, description)
            VALUES ($1, $2::jsonb, $3)
            RETURNING id, name, aliases, description, created_at, updated_at
            """,
            cleaned,
            json.dumps(alias_list),
            description,
        )
    return Task(**dict(row))


async def replace_results(
    paper_id: UUID,
    results: Sequence[ResultCreate],
) -> list[Result]:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM results WHERE paper_id = $1", paper_id)
            if not results:
                return []

            inserted: list[Result] = []
            for result in results:
                row = await conn.fetchrow(
                    """
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
                    """,
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
                    result.verified,
                    result.verifier_notes,
                )
                inserted.append(Result(**dict(row)))
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
                row = await conn.fetchrow(
                    """
                    INSERT INTO claims (paper_id, category, text, confidence, evidence)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    RETURNING id, paper_id, category, text, confidence, evidence, created_at, updated_at
                    """,
                    claim.paper_id,
                    claim.category,
                    claim.text,
                    claim.confidence,
                    json.dumps(claim.evidence),
                )
                inserted.append(Claim(**dict(row)))
    return inserted
