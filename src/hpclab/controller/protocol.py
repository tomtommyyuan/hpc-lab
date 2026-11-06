"""Wire protocol between hpclab clients (CLI, MCP server, Python API) and the controller.

Framing
-------
Newline-delimited JSON over a Unix stream socket: one JSON-RPC 2.0 message per line, UTF-8,
at most ``MAX_MESSAGE_BYTES`` (8 MiB) per message excluding the newline. JSON never contains a
raw newline (they are escaped inside strings), so a line is always exactly one message. A
connection may carry any number of request/response pairs, strictly in order.

Request::

    {"jsonrpc": "2.0", "id": <str|int>, "method": <str>, "params": <object>?, "deadline_s": <number>?}

``deadline_s`` is an hpclab extension: the longest the caller is willing to wait. The controller
answers with a retryable Envelope error before that point instead of leaving the caller to time
out blind. A request without ``id`` is a JSON-RPC notification: it runs and gets no reply.

Response::

    {"jsonrpc": "2.0", "id": <same id>, "result": <Envelope dict>}
    {"jsonrpc": "2.0", "id": <same id or null>, "error": {"code": <int>, "message": <str>, "data": ...}}

Which failures are which (the contract the frontends rely on)
-------------------------------------------------------------
* Everything that reached the dispatcher is answered with ``result`` = an Envelope dict
  (``hpclab.api.envelope``), including domain failures (``ok=false`` + typed ``error``),
  **unknown methods** (``error.code = "invalid_argument"``, same as the method registry),
  unexpected exceptions (``internal``) and deadline overruns (``unavailable``, retryable).
* JSON-RPC ``error`` objects are reserved for failures of the protocol itself: unparsable JSON
  (-32700), malformed request objects or batches (-32600), non-object params (-32602), a message
  over the size limit (-32001; sent with ``id: null`` because the id could not be read — the rest
  of the line is discarded and the connection stays usable) and "controller shutting down"
  (-32002; the request was *not* executed, so clients may retry it against the next controller).
  An unexpected server failure after a request was accepted is answered with -32603 (never by
  hanging up, which would make clients re-send a request that may have run).
  ``METHOD_NOT_FOUND`` (-32601) is defined for completeness but never sent.

Non-finite floats (NaN/Infinity) are passed through as Python's JSON extension because both
ends are Python and training metrics legitimately contain NaN; frontends that emit strict JSON
must sanitize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from hpclab.api.envelope import Envelope, ErrorInfo
from hpclab.domain.errors import ErrorCode, HpcLabError

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = 1  # bump on incompatible wire changes; reported by controller.ping
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

# JSON-RPC 2.0 standard codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601  # reserved: unknown methods are Envelope errors (see module doc)
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# implementation-defined server errors (-32000..-32099)
MESSAGE_TOO_LARGE = -32001
SHUTTING_DOWN = -32002

# Process exit codes of ``python -m hpclab.controller`` (the client inspects them after spawning).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_ALREADY_RUNNING = 3

JsonRpcId = str | int | None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProtocolError(Exception):
    """A malformed or unacceptable message. Carries the JSON-RPC error code and, when it
    could be recovered, the request id so the error response can be correlated."""

    def __init__(self, code: int, message: str, *, request_id: JsonRpcId = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.data = data

    def to_rpc_error(self) -> RpcError:
        return RpcError(code=self.code, message=self.message, data=self.data)


class MessageTooLarge(ProtocolError):
    def __init__(self, size: int, max_bytes: int = MAX_MESSAGE_BYTES) -> None:
        super().__init__(
            MESSAGE_TOO_LARGE,
            f"message of {size} bytes exceeds the {max_bytes}-byte protocol limit",
            data={"size": size, "max_bytes": max_bytes},
        )
        self.size = size
        self.max_bytes = max_bytes


class MethodNotFound(HpcLabError):
    """Raised by an App's ``dispatch`` for a method it does not serve. The controller turns it
    into an Envelope error with code ``invalid_argument`` (never a JSON-RPC error)."""

    code = ErrorCode.INVALID_ARGUMENT

    def __init__(self, method: str, *, known: list[str] | None = None, reason: str | None = None) -> None:
        details: dict[str, Any] = {"method": method}
        if known is not None:
            details["known"] = sorted(known)[:200]
        if reason:
            details["reason"] = reason
        msg = f"unknown method {method!r}" + (f": {reason}" if reason else "")
        super().__init__(msg, details=details)
        self.method = method


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RpcError:
    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> RpcError:
        if not isinstance(obj, dict) or not isinstance(obj.get("code"), int) or isinstance(obj["code"], bool):
            raise ProtocolError(INVALID_REQUEST, "malformed JSON-RPC error object")
        return cls(code=obj["code"], message=str(obj.get("message", "")), data=obj.get("data"))


@dataclass(frozen=True)
class Request:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: JsonRpcId = None
    deadline_s: float | None = None
    notification: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method, "params": self.params}
        if not self.notification:
            out["id"] = self.id
        if self.deadline_s is not None:
            out["deadline_s"] = self.deadline_s
        return out


@dataclass(frozen=True)
class Response:
    id: JsonRpcId
    result: dict[str, Any] | None = None
    error: RpcError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("a Response carries exactly one of result / error")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            out["error"] = self.error.to_dict()
        else:
            out["result"] = self.result
        return out


def _valid_id(value: Any) -> bool:
    return value is None or isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def parse_request(obj: Any) -> Request:
    """Validate a decoded JSON value as a single JSON-RPC 2.0 request."""
    if isinstance(obj, list):
        raise ProtocolError(INVALID_REQUEST, "batch requests are not supported; send one request per line")
    if not isinstance(obj, dict):
        raise ProtocolError(INVALID_REQUEST, "request must be a JSON object")
    notification = "id" not in obj
    rid = obj.get("id")
    if not _valid_id(rid):
        raise ProtocolError(INVALID_REQUEST, "request id must be a string, an integer or null")
    if obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError(INVALID_REQUEST, 'request must carry "jsonrpc": "2.0"', request_id=rid)
    method = obj.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(INVALID_REQUEST, "request method must be a non-empty string", request_id=rid)
    params = obj.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        raise ProtocolError(
            INVALID_PARAMS, "params must be a JSON object (named parameters only)", request_id=rid
        )
    deadline = obj.get("deadline_s")
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, int | float) or not deadline > 0:
            raise ProtocolError(INVALID_REQUEST, "deadline_s must be a positive number", request_id=rid)
        deadline = float(deadline)
    return Request(method=method, params=params, id=rid, deadline_s=deadline, notification=notification)


def parse_response(obj: Any) -> Response:
    """Validate a decoded JSON value as a JSON-RPC 2.0 response (client side)."""
    if not isinstance(obj, dict) or obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError(INVALID_REQUEST, "response must be a JSON-RPC 2.0 object")
    rid = obj.get("id")
    if not _valid_id(rid):
        raise ProtocolError(INVALID_REQUEST, "response id must be a string, an integer or null")
    has_result, has_error = "result" in obj, "error" in obj
    if has_result == has_error:
        raise ProtocolError(INVALID_REQUEST, "response must carry exactly one of result / error")
    if has_error:
        return Response(id=rid, error=RpcError.from_dict(obj["error"]))
    result = obj["result"]
    if not isinstance(result, dict):
        raise ProtocolError(INVALID_REQUEST, "response result must be an Envelope object")
    return Response(id=rid, result=result)


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def encode_message(obj: Any, *, max_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    """Serialize one message to a newline-terminated line.

    Raises ``MessageTooLarge`` above ``max_bytes`` and ``TypeError``/``ValueError`` for values
    JSON cannot represent — including structures nested too deeply for the encoder (which
    ``json`` reports as ``RecursionError``) and integers above the int/str digit limit."""
    try:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except RecursionError as exc:
        raise ValueError("value is nested too deeply to encode as JSON") from exc
    body = text.encode("utf-8")
    if len(body) > max_bytes:
        raise MessageTooLarge(len(body), max_bytes)
    return body + b"\n"


def decode_message(data: bytes | str) -> Any:
    """Parse one line (with or without its newline).

    Every failure — bad UTF-8, invalid JSON, nesting too deep for the parser (``RecursionError``),
    integers above the int/str digit limit (plain ``ValueError``) — raises
    ``ProtocolError(PARSE_ERROR)``, so a hostile or buggy peer can never make the reader crash."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(PARSE_ERROR, f"message is not valid UTF-8: {exc.reason}") from exc
    else:
        text = data
    text = text.strip()
    if not text:
        raise ProtocolError(PARSE_ERROR, "empty message")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(PARSE_ERROR, f"invalid JSON: {exc.msg} at position {exc.pos}") from exc
    except RecursionError as exc:
        raise ProtocolError(PARSE_ERROR, "invalid JSON: nested too deeply") from exc
    except ValueError as exc:  # e.g. "Exceeds the limit (4300 digits) for integer string conversion"
        raise ProtocolError(PARSE_ERROR, f"invalid JSON: {exc}") from exc


class _LineReader(Protocol):
    def readline(self, size: int = -1, /) -> bytes: ...


def read_message(stream: _LineReader, *, max_bytes: int = MAX_MESSAGE_BYTES) -> bytes | None:
    """Read the next non-blank line from a binary stream, without its line terminator.

    Returns None at a clean EOF. A final line without a newline (peer shut down its write side)
    is still returned. Raises ``MessageTooLarge`` when a line exceeds ``max_bytes``; the stream is
    then out of sync and the connection must be closed."""
    while True:
        line = stream.readline(max_bytes + 1)
        if not line:
            return None
        if line.endswith(b"\n"):
            line = line[:-1]
        elif len(line) > max_bytes:
            raise MessageTooLarge(len(line), max_bytes)
        if line.endswith(b"\r"):
            line = line[:-1]
        if line.strip():
            return line


def skip_line(stream: _LineReader, *, max_skip: int, chunk: int = 64 * 1024) -> bool:
    """Discard the rest of the current line (after ``MessageTooLarge``) so the stream is back in
    sync at the next message. Returns False at EOF or when more than ``max_skip`` bytes would have
    to be discarded (the connection should then be closed)."""
    skipped = 0
    while skipped <= max_skip:
        part = stream.readline(chunk)
        if not part:
            return False
        if part.endswith(b"\n"):
            return True
        skipped += len(part)
    return False


# ---------------------------------------------------------------------------
# Envelope helpers (what every method answers with)
# ---------------------------------------------------------------------------


def ok_envelope(
    request_id: str, data: Any, *, summary: str = "", warnings: list[str] | None = None
) -> dict[str, Any]:
    env = Envelope[Any](request_id=request_id, ok=True, data=data, summary=summary, warnings=warnings or [])
    return env.model_dump(mode="json")


def error_envelope(
    request_id: str,
    code: ErrorCode | str,
    message: str,
    *,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
    operation_ref: str | None = None,
) -> dict[str, Any]:
    exc = HpcLabError(
        message,
        code=ErrorCode(code),
        retryable=retryable,
        operation_ref=operation_ref,
        details=details,
    )
    return envelope_from_exception(exc, request_id)


def envelope_from_exception(exc: HpcLabError, request_id: str) -> dict[str, Any]:
    env = Envelope[Any](
        request_id=request_id,
        ok=False,
        error=ErrorInfo.from_exc(exc),
        summary=f"{exc.code}: {exc.message}",
    )
    return env.model_dump(mode="json")


_RPC_TO_ENVELOPE_CODE: dict[int, ErrorCode] = {
    PARSE_ERROR: ErrorCode.INVALID_ARGUMENT,
    INVALID_REQUEST: ErrorCode.INVALID_ARGUMENT,
    METHOD_NOT_FOUND: ErrorCode.INVALID_ARGUMENT,
    INVALID_PARAMS: ErrorCode.INVALID_ARGUMENT,
    MESSAGE_TOO_LARGE: ErrorCode.LIMIT_EXCEEDED,
    SHUTTING_DOWN: ErrorCode.CONTROLLER_UNAVAILABLE,
}


def envelope_from_rpc_error(err: RpcError, request_id: str) -> dict[str, Any]:
    """Client-side view of a protocol-level failure, so frontends only ever handle Envelopes."""
    code = _RPC_TO_ENVELOPE_CODE.get(err.code, ErrorCode.INTERNAL)
    details: dict[str, Any] = {"jsonrpc_code": err.code}
    if err.data is not None:
        details["data"] = err.data
    return error_envelope(
        request_id, code, f"controller rejected the request: {err.message}", details=details
    )
