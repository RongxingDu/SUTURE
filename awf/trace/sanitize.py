"""Bounded conversion of arbitrary runtime values into trace-safe data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from itertools import islice
import math
from pathlib import Path
from typing import Any

_MAX_DEPTH = 8
_MAX_ITEMS = 200
_MAX_STRING_LENGTH = 20_000


def to_trace_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Return a detached, JSON-compatible, size-bounded trace value."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "hex": value[: _MAX_STRING_LENGTH // 2].hex(),
            "truncated": len(value) > _MAX_STRING_LENGTH // 2,
        }
    if isinstance(value, Enum):
        return to_trace_value(value.value, _depth=_depth, _seen=_seen)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if _depth >= _MAX_DEPTH:
        return {"__truncated__": "maximum trace depth reached"}

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return {"__cycle__": type(value).__name__}

    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            result: dict[str, Any] = {}
            truncated = False
            for index, (key, item) in enumerate(
                islice(value.items(), _MAX_ITEMS + 1)
            ):
                if index == _MAX_ITEMS:
                    truncated = True
                    break
                result[_safe_key(key)] = to_trace_value(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                )
            if truncated:
                try:
                    omitted = max(1, len(value) - _MAX_ITEMS)
                except (TypeError, OverflowError):
                    omitted = 1
                result["__truncated_items__"] = omitted
            return result
        finally:
            seen.discard(identity)

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        seen.add(identity)
        try:
            bounded_items = list(islice(iter(value), _MAX_ITEMS + 1))
            truncated = len(bounded_items) > _MAX_ITEMS
            result = [
                to_trace_value(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                )
                for item in bounded_items[:_MAX_ITEMS]
            ]
            if truncated:
                try:
                    omitted = max(1, len(value) - _MAX_ITEMS)
                except (TypeError, OverflowError):
                    omitted = 1
                result.append(
                    {"__truncated_items__": omitted}
                )
            return result
        finally:
            seen.discard(identity)

    if isinstance(value, (set, frozenset)):
        items = list(islice(iter(value), _MAX_ITEMS + 1))
        truncated = len(items) > _MAX_ITEMS
        return {
            "__type__": type(value).__name__,
            "items": [
                to_trace_value(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                )
                for item in items[:_MAX_ITEMS]
            ],
            "truncated_items": 1 if truncated else 0,
        }

    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__repr__": _safe_repr(value),
    }


def _safe_key(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate(str(value))
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _safe_repr(value: Any) -> str:
    try:
        rendered = repr(value)
    except BaseException:
        rendered = object.__repr__(value)
    return _truncate(rendered)


def _truncate(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return value[: _MAX_STRING_LENGTH - 16] + "...[truncated]"
