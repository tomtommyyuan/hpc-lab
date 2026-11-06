"""Common result envelope (design §2).

Every service call returns ``Envelope[T]``::

    {schema_version, request_id, ok, data, error, coverage, observed_at, next_cursor,
     warnings, summary}

* ``coverage`` says which sources were consulted and whether they were fresh, stale or
  unavailable, so an empty list can never be mistaken for "nothing exists".
* ``summary`` is a short human/LLM-readable line; ``data`` is the structured payload.
* On failure ``ok=False``, ``data`` may still hold confirmed partial data, and ``error``
  carries ``{code, message, retryable, operation_ref?, details}``.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab import SCHEMA_VERSION
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
DEFAULT_TEXT_BYTES = 32 * 1024
MAX_TEXT_BYTES = 64 * 1024


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    retryable: bool = False
    operation_ref: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exc(cls, exc: HpcLabError) -> ErrorInfo:
        return cls(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            operation_ref=exc.operation_ref,
            details=exc.details,
        )


SourceStatusValue = Literal[
    "ok", "cached", "stale", "partial", "unavailable", "unsupported", "unauthorized", "skipped"
]


class SourceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # e.g. "squeue", "sacct", "ledger", "log:stdout", "sinfo"
    status: SourceStatusValue
    observed_at: str | None = None
    detail: str | None = None


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: bool = True
    sources: list[SourceStatus] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def add(
        self, name: str, status: SourceStatusValue, observed_at: str | None = None, detail: str | None = None
    ) -> None:
        self.sources.append(SourceStatus(name=name, status=status, observed_at=observed_at, detail=detail))
        if status not in ("ok", "cached", "skipped"):
            self.complete = False


class Envelope[T](BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    request_id: str
    ok: bool = True
    data: T | None = None
    error: ErrorInfo | None = None
    coverage: Coverage | None = None
    observed_at: str | None = None
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    if limit < 1:
        raise InvalidArgument("limit must be >= 1")
    return min(limit, MAX_LIMIT)


def clamp_bytes(max_bytes: int | None) -> int:
    if max_bytes is None:
        return DEFAULT_TEXT_BYTES
    if max_bytes < 1:
        raise InvalidArgument("max_bytes must be >= 1")
    return min(max_bytes, MAX_TEXT_BYTES)


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> dict[str, Any]:
    if not cursor:
        return {}
    try:
        pad = "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(cursor + pad))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidArgument("malformed cursor") from exc
    if not isinstance(data, dict):
        raise InvalidArgument("malformed cursor")
    return data
