"""Error taxonomy (design §2).

Every failure that crosses a service boundary is an ``HpcLabError`` with a stable ``code``.
Frontends (MCP, CLI, socket) serialize it as ``{code, message, retryable, operation_ref?,
details}``. States such as ``unauthorized / unsupported / stale / truncated / ambiguous /
unavailable`` must never be swallowed into an empty success.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"  # identity/ownership check failed
    AUTH_REQUIRED = "auth_required"  # SSH needs interactive auth (MFA, expired ControlMaster)
    UNSUPPORTED = "unsupported"  # capability missing on this cluster/profile
    STALE = "stale"  # referenced content changed since the reference was made
    TRUNCATED = "truncated"
    AMBIGUOUS = "ambiguous"  # more than one plausible match; refuse to guess
    UNAVAILABLE = "unavailable"  # remote source down / timed out / accounting disabled
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    VERSION_CONFLICT = "version_conflict"  # expected_version mismatch
    NOT_READY = "not_ready"  # plan still preparing, probe pending, etc.
    PRECONDITION_FAILED = "precondition_failed"
    ADMISSION_DENIED = "admission_denied"  # execution claim / concurrency / dispatch closed
    BUDGET_EXCEEDED = "budget_exceeded"
    SUBMISSION_UNCERTAIN = "submission_uncertain"
    POLICY_VIOLATION = "policy_violation"  # profile policy forbids the request
    LIMIT_EXCEEDED = "limit_exceeded"  # matrix too large, snapshot too large, page too big
    CONTROLLER_UNAVAILABLE = "controller_unavailable"
    INTERNAL = "internal"


# Codes a caller may retry unchanged (possibly after waiting).
RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.UNAVAILABLE,
        ErrorCode.NOT_READY,
        ErrorCode.AUTH_REQUIRED,
        ErrorCode.CONTROLLER_UNAVAILABLE,
    }
)


class HpcLabError(Exception):
    """Base class for all anticipated failures."""

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        operation_ref: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.retryable = (self.code in RETRYABLE) if retryable is None else retryable
        self.operation_ref = operation_ref
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": str(self.code),
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
        if self.operation_ref:
            out["operation_ref"] = self.operation_ref
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HpcLabError:
        try:
            code = ErrorCode(data.get("code", "internal"))
        except ValueError:
            code = ErrorCode.INTERNAL
        return HpcLabError(
            data.get("message", "unknown error"),
            code=code,
            retryable=data.get("retryable"),
            operation_ref=data.get("operation_ref"),
            details=data.get("details") or {},
        )

    def __repr__(self) -> str:
        return f"HpcLabError({self.code}: {self.message})"


def _mk(name: str, c: ErrorCode) -> type[HpcLabError]:
    return type(name, (HpcLabError,), {"code": c})


InvalidArgument = _mk("InvalidArgument", ErrorCode.INVALID_ARGUMENT)
NotFound = _mk("NotFound", ErrorCode.NOT_FOUND)
Unauthorized = _mk("Unauthorized", ErrorCode.UNAUTHORIZED)
AuthRequired = _mk("AuthRequired", ErrorCode.AUTH_REQUIRED)
Unsupported = _mk("Unsupported", ErrorCode.UNSUPPORTED)
Stale = _mk("Stale", ErrorCode.STALE)
Ambiguous = _mk("Ambiguous", ErrorCode.AMBIGUOUS)
Unavailable = _mk("Unavailable", ErrorCode.UNAVAILABLE)
IdempotencyConflict = _mk("IdempotencyConflict", ErrorCode.IDEMPOTENCY_CONFLICT)
VersionConflict = _mk("VersionConflict", ErrorCode.VERSION_CONFLICT)
NotReady = _mk("NotReady", ErrorCode.NOT_READY)
PreconditionFailed = _mk("PreconditionFailed", ErrorCode.PRECONDITION_FAILED)
AdmissionDenied = _mk("AdmissionDenied", ErrorCode.ADMISSION_DENIED)
BudgetExceeded = _mk("BudgetExceeded", ErrorCode.BUDGET_EXCEEDED)
PolicyViolation = _mk("PolicyViolation", ErrorCode.POLICY_VIOLATION)
LimitExceeded = _mk("LimitExceeded", ErrorCode.LIMIT_EXCEEDED)
ControllerUnavailable = _mk("ControllerUnavailable", ErrorCode.CONTROLLER_UNAVAILABLE)
