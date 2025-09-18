from __future__ import annotations

import io
import asyncio
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from app.core.config import settings


PDF_CONTENT_TYPES: Final[set[str]] = {"application/pdf"}
MAX_FILE_SIZE_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MB limit for MVP


@dataclass
class StorageUploadResult:
    bucket: str
    object_name: str
    file_name: str
    size: int
    content_type: str


def get_minio_client() -> Minio:
    endpoint, secure = _normalize_minio_endpoint(
        settings.minio_endpoint, settings.minio_secure
    )
    return Minio(
        endpoint=endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=secure,
    )


async def ensure_bucket_exists(
    bucket_name: str, max_attempts: int = 5, initial_delay_seconds: float = 1.0
) -> None:
    last_connection_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        client = get_minio_client()
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
            return
        except S3Error as exc:  # pragma: no cover - external dependency behaviour
            raise RuntimeError(
                f"Failed to ensure MinIO bucket '{bucket_name}' exists: {exc}"
            ) from exc
        except (Urllib3HTTPError, OSError) as exc:
            last_connection_error = exc
            if attempt < max_attempts:
                await asyncio.sleep(initial_delay_seconds * attempt)
            else:
                break

    if last_connection_error is not None:
        raise RuntimeError(
            "Unable to connect to MinIO after multiple attempts. "
            "Check MINIO_ENDPOINT and network connectivity."
        ) from last_connection_error


async def upload_pdf_to_storage(file: UploadFile) -> StorageUploadResult:
    if not file.filename:
        raise ValueError("Uploaded file must include a filename")

    if not _is_pdf(file):
        raise ValueError("Only PDF uploads are supported")

    data = await file.read()
    size = len(data)
    if size == 0:
        raise ValueError("Uploaded file is empty")
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError("Uploaded file exceeds maximum allowed size")

    client = get_minio_client()
    bucket = settings.minio_bucket_papers
    object_name = _build_object_name(file.filename)
    file_name = Path(file.filename).name

    content_type = _resolve_content_type(file)

    try:
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(data),
            length=size,
            content_type=content_type,
        )
    except S3Error as exc:  # pragma: no cover - external dependency behaviour
        raise RuntimeError(f"Failed to store file in MinIO: {exc}") from exc
    finally:
        await file.close()

    return StorageUploadResult(
        bucket=bucket,
        object_name=object_name,
        file_name=file_name,
        size=size,
        content_type=content_type,
    )


def create_presigned_download_url(object_name: str, expires_in: int = 3600) -> str:
    if expires_in <= 0:
        raise ValueError("expires_in must be a positive integer")

    client = get_minio_client()
    try:
        return client.presigned_get_object(
            bucket_name=settings.minio_bucket_papers,
            object_name=object_name,
            expires=timedelta(seconds=expires_in),
        )
    except S3Error as exc:  # pragma: no cover - external dependency behaviour
        raise RuntimeError(f"Failed to generate download URL: {exc}") from exc


def _is_pdf(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    if content_type in PDF_CONTENT_TYPES:
        return True
    filename = (file.filename or "").lower()
    return filename.endswith(".pdf")


def _build_object_name(filename: str) -> str:
    clean_name = Path(filename).name.replace(" ", "_")
    unique_prefix = uuid4().hex
    return f"{unique_prefix}/{clean_name}"


def _resolve_content_type(file: UploadFile) -> str:
    content_type = (file.content_type or "application/pdf").lower()
    if content_type not in PDF_CONTENT_TYPES:
        return "application/pdf"
    return content_type


def _normalize_minio_endpoint(endpoint: str, secure: bool) -> tuple[str, bool]:
    cleaned = endpoint.strip()
    if not cleaned:
        raise ValueError("MinIO endpoint cannot be empty")

    updated_secure = secure
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("MinIO endpoint must use http or https scheme")
        updated_secure = parsed.scheme == "https"
        cleaned = parsed.netloc or parsed.path

    cleaned = cleaned.rstrip("/")
    if not cleaned:
        raise ValueError("MinIO endpoint cannot be empty")

    return cleaned, updated_secure

