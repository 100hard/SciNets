from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _strip_quotes(value: str) -> str:
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        return value[1:-1]
    return value


def _load_env_file(path: Path, encoding: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        content = path.read_text(encoding=encoding)
    except FileNotFoundError:
        return data

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = _strip_quotes(value.strip())
    return data


def _resolve_annotation(annotation: Any) -> Any:
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
        if not args:
            return str
        return _resolve_annotation(args[0])
    return origin


def _coerce(value: str, annotation: Any) -> Any:
    annotation = _resolve_annotation(annotation)
    if annotation in (None, Any, str):
        return value
    if annotation is bool:
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        return bool(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation in (list, tuple, set):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            loaded = [item.strip() for item in value.split(",") if item.strip()]
        if annotation is tuple:
            return tuple(loaded)
        if annotation is set:
            return set(loaded)
        return list(loaded)
    return value


class BaseSettings:
    def __init__(self, **overrides: Any) -> None:
        annotations: Dict[str, Any] = {}
        defaults: Dict[str, Any] = {}

        for cls in reversed(self.__class__.mro()):
            annotations.update(getattr(cls, "__annotations__", {}))
            for name, value in cls.__dict__.items():
                if name.startswith("_") or name in {"Config", "model_config"}:
                    continue
                if isinstance(value, (classmethod, staticmethod)):
                    continue
                if callable(value):
                    continue
                defaults.setdefault(name, value)

        config = getattr(self.__class__, "Config", None)
        env_file = getattr(config, "env_file", None) if config else None
        env_encoding = getattr(config, "env_file_encoding", "utf-8") if config else "utf-8"
        env_prefix = getattr(config, "env_prefix", "") if config else ""

        env_file_data: Mapping[str, str]
        if env_file:
            env_path = Path(env_file)
            env_file_data = _load_env_file(env_path, env_encoding)
        else:
            env_file_data = {}

        resolved: MutableMapping[str, Any] = dict(defaults)
        for name, default in defaults.items():
            env_name = f"{env_prefix}{name}".upper()
            raw = os.getenv(env_name)
            if raw is None:
                raw = env_file_data.get(env_name)
            if raw is None:
                continue
            resolved[name] = _coerce(raw, annotations.get(name, type(default)))

        resolved.update(overrides)
        for key, value in resolved.items():
            setattr(self, key, value)
