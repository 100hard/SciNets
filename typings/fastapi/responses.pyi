from __future__ import annotations

from typing import Any, Mapping


class JSONResponse:
    status_code: int

    def __init__(
        self,
        content: Any = ...,
        *,
        status_code: int = ...,
        headers: Mapping[str, str] | None = ...,
        media_type: str | None = ...,
        background: Any | None = ...,
    ) -> None: ...

    def render(self, content: Any) -> bytes: ...
