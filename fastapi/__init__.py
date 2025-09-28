from __future__ import annotations

import io
from typing import Any, Optional


class UploadFile:
    def __init__(
        self,
        filename: Optional[str] = None,
        file: Any = None,
        content_type: Optional[str] = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.file = file if file is not None else io.BytesIO()

    async def read(self) -> bytes:
        if hasattr(self.file, "seek"):
            self.file.seek(0)
        data = self.file.read()
        if isinstance(data, str):
            return data.encode()
        return data or b""

    async def close(self) -> None:
        if hasattr(self.file, "close") and callable(self.file.close):
            self.file.close()
