from __future__ import annotations

from typing import Any, Callable, Dict, Type

_METHODS: Dict[str, Type] = {}


def register_method(name: str) -> Callable[[Type], Type]:
    """Register a continual method class under a CLI/config name."""

    def decorator(cls: Type) -> Type:
        key = name.lower().strip()
        if key in _METHODS and _METHODS[key] is not cls:
            raise KeyError(f"Method '{key}' is already registered by {_METHODS[key]}")
        _METHODS[key] = cls
        cls.method_name = key
        return cls

    return decorator


def build_method(name: str, **kwargs: Any):
    key = name.lower().strip()
    if key not in _METHODS:
        # Import modules lazily so registration side effects happen only when needed.
        from . import methods as _methods  # noqa: F401
    if key not in _METHODS:
        raise KeyError(f"Unknown method '{name}'. Available methods: {sorted(_METHODS)}")
    return _METHODS[key](**kwargs)


def list_methods() -> list[str]:
    from . import methods as _methods  # noqa: F401

    return sorted(_METHODS)
