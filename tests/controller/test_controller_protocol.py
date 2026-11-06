"""Framing and message validation of the controller wire protocol."""

from __future__ import annotations

import io
import math
from typing import Any

import pytest

from hpclab.api.envelope import Envelope
from hpclab.controller import protocol as proto
from hpclab.domain.errors import ErrorCode


def test_request_round_trip_keeps_unicode_and_newlines_on_one_line() -> None:
    req = proto.Request(
        method="runs.list", params={"a": [1, "é", None], "text": "line1\nline2"}, id=7, deadline_s=1.5
    )
    line = proto.encode_message(req.to_dict())
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert proto.parse_request(proto.decode_message(line)) == req


def test_notification_and_default_params() -> None:
    req = proto.parse_request({"jsonrpc": "2.0", "method": "controller.ping"})
    assert req.notification and req.params == {} and req.id is None
    assert "id" not in req.to_dict()
    explicit_null = proto.parse_request({"jsonrpc": "2.0", "id": None, "method": "m"})
    assert not explicit_null.notification


def test_nan_passes_through_between_python_ends() -> None:
    line = proto.encode_message({"loss": float("nan")})
    assert math.isnan(proto.decode_message(line)["loss"])


@pytest.mark.parametrize(
    ("obj", "code"),
    [
        ([{"jsonrpc": "2.0", "id": 1, "method": "m"}], proto.INVALID_REQUEST),  # batch
        ("just a string", proto.INVALID_REQUEST),
        ({"jsonrpc": "1.0", "id": 1, "method": "m"}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": ""}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1.5, "method": "m"}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": True, "method": "m"}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": "m", "params": [1, 2]}, proto.INVALID_PARAMS),
        ({"jsonrpc": "2.0", "id": 1, "method": "m", "deadline_s": 0}, proto.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": "m", "deadline_s": "5"}, proto.INVALID_REQUEST),
    ],
)
def test_parse_request_rejects_malformed(obj: Any, code: int) -> None:
    with pytest.raises(proto.ProtocolError) as ei:
        proto.parse_request(obj)
    assert ei.value.code == code


def test_parse_error_keeps_request_id_for_correlation() -> None:
    with pytest.raises(proto.ProtocolError) as ei:
        proto.parse_request({"jsonrpc": "2.0", "id": "abc", "method": "m", "params": "x"})
    assert ei.value.request_id == "abc"
    assert ei.value.to_rpc_error().to_dict()["code"] == proto.INVALID_PARAMS


@pytest.mark.parametrize("raw", [b"{bad json", b"\xff\xfe{}", b"   ", ""])
def test_decode_errors_are_parse_errors(raw: bytes | str) -> None:
    with pytest.raises(proto.ProtocolError) as ei:
        proto.decode_message(raw)
    assert ei.value.code == proto.PARSE_ERROR


def test_encode_rejects_oversized_messages() -> None:
    with pytest.raises(proto.MessageTooLarge) as ei:
        proto.encode_message({"x": "a" * 100}, max_bytes=50)
    assert ei.value.code == proto.MESSAGE_TOO_LARGE
    assert ei.value.size > 50 and ei.value.max_bytes == 50
    assert proto.MAX_MESSAGE_BYTES == 8 * 1024 * 1024


def test_read_message_skips_blank_lines_and_handles_crlf_and_eof() -> None:
    stream = io.BytesIO(b'\n\r\n{"a":1}\r\n{"b":2}\npartial')
    assert proto.read_message(stream) == b'{"a":1}'
    assert proto.read_message(stream) == b'{"b":2}'
    assert proto.read_message(stream) == b"partial"  # peer shut down its write side
    assert proto.read_message(stream) is None


def test_read_message_limit_is_exact() -> None:
    assert proto.read_message(io.BytesIO(b"x" * 100 + b"\n"), max_bytes=100) == b"x" * 100
    with pytest.raises(proto.MessageTooLarge):
        proto.read_message(io.BytesIO(b"x" * 101 + b"\n"), max_bytes=100)


def test_response_round_trip_and_validation() -> None:
    ok = proto.Response(id=3, result={"ok": True, "request_id": "req_x"})
    assert proto.parse_response(proto.decode_message(proto.encode_message(ok.to_dict()))) == ok
    err = proto.Response(id=None, error=proto.RpcError(proto.PARSE_ERROR, "bad", data={"k": 1}))
    assert proto.parse_response(err.to_dict()) == err
    with pytest.raises(ValueError):
        proto.Response(id=1)
    for bad in (
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": 1, "message": ""}},
        {"jsonrpc": "2.0", "id": 1, "result": [1]},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": "x"}},
        {"id": 1, "result": {}},
    ):
        with pytest.raises(proto.ProtocolError):
            proto.parse_response(bad)


def test_envelope_helpers_produce_valid_envelopes() -> None:
    ok = proto.ok_envelope("req_1", {"a": 1}, summary="fine")
    assert Envelope[Any].model_validate(ok).ok and ok["summary"] == "fine"
    err = proto.error_envelope("req_2", ErrorCode.UNAVAILABLE, "down", details={"x": 1})
    parsed = Envelope[Any].model_validate(err)
    assert not parsed.ok and parsed.error is not None
    assert parsed.error.code == ErrorCode.UNAVAILABLE and parsed.error.retryable is True  # default by code
    assert err["summary"] == "unavailable: down"


def test_rpc_errors_map_to_envelope_codes() -> None:
    shutting = proto.envelope_from_rpc_error(proto.RpcError(proto.SHUTTING_DOWN, "bye"), "req_1")
    assert shutting["error"]["code"] == "controller_unavailable"
    assert shutting["error"]["details"]["jsonrpc_code"] == proto.SHUTTING_DOWN
    parse = proto.envelope_from_rpc_error(proto.RpcError(proto.PARSE_ERROR, "bad"), "req_2")
    assert parse["error"]["code"] == "invalid_argument"
    other = proto.envelope_from_rpc_error(proto.RpcError(-32099, "weird"), "req_3")
    assert other["error"]["code"] == "internal"


def test_method_not_found_is_an_invalid_argument_domain_error() -> None:
    exc = proto.MethodNotFound("a.b", known=["z", "x"], reason="not loaded")
    env = proto.envelope_from_exception(exc, "req_1")
    assert env["error"]["code"] == "invalid_argument"
    assert env["error"]["details"] == {"method": "a.b", "known": ["x", "z"], "reason": "not loaded"}
