"""Canonical hashing.

Plan hashes, spec hashes and idempotency payload hashes all go through ``canonical_json``
so two semantically equal payloads always hash the same (sorted keys, no whitespace,
no NaN, UTF-8).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def _default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=False)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_default
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """``sha256:<hex>`` of the canonical JSON form of ``value``."""
    return "sha256:" + sha256_hex(canonical_json(value))


def short(d: str, n: int = 12) -> str:
    """Short display form of a ``sha256:...`` digest."""
    return d.split(":", 1)[-1][:n]
