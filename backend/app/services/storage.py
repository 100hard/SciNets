from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final
from uuid import uuid4

from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error

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
    endpoint = settings.minio_endpoint
    # Minio SDK expects host:port without scheme
    if endpoint.startswith("http://"):
        endpoint = endpoint.replace("http://", "")
    if endpoint.startswith("https://"):
        endpoint = endpoint.replace("https://", "")
    return Minio(
        endpoint=endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket_exists(bucket_name: str) -> None:
    client = get_minio_client()
    exists = client.bucket_exists(bucket_name)
    if not exists:
        client.make_bucket(bucket_name)


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

