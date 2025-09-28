from __future__ import annotations

from typing import Any


class BaseSettings:
    def __init__(self, **overrides: Any) -> None:
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            setattr(self, name, value)
        for key, value in overrides.items():
            setattr(self, key, value)
