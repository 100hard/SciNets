from __future__ import annotations

from typing import Any


class CORSMiddleware:
    def __init__(
        self,
        app: Any,
        *,
        allow_origins: list[str] | tuple[str, ...] | None = ...,
        allow_methods: list[str] | tuple[str, ...] | None = ...,
        allow_headers: list[str] | tuple[str, ...] | None = ...,
        allow_credentials: bool | None = ...,
        expose_headers: list[str] | tuple[str, ...] | None = ...,
        max_age: int | None = ...,
    ) -> None: ...
