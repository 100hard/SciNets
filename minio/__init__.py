from __future__ import annotations


class Minio:  # pragma: no cover - simple compatibility stub
    def __init__(self, *args, **kwargs) -> None:
        pass

    def bucket_exists(self, *args, **kwargs) -> bool:
        return True

    def make_bucket(self, *args, **kwargs) -> None:
        return None

    def put_object(self, *args, **kwargs) -> None:
        return None

    def get_object(self, *args, **kwargs):
        raise RuntimeError("Minio stub does not support get_object")

    def presigned_get_object(self, *args, **kwargs) -> str:
        return ""
