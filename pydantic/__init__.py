from __future__ import annotations

from typing import Any, Callable, Optional


class ConfigDict(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def Field(
    default: Any = None,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    **kwargs: Any,
) -> Any:  # pragma: no cover - simple compatibility shim
    if default_factory is not None:
        return default_factory()
    if default is not Ellipsis:
        return default
    return None


def field_serializer(*args: Any, **kwargs: Any):  # pragma: no cover - compatibility shim
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


def field_validator(*args: Any, **kwargs: Any):  # pragma: no cover - compatibility shim
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator
