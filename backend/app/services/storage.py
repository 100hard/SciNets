from __future__ import annotations

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


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

