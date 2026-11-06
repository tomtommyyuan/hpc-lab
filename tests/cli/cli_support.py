"""Helpers for the CLI tests: a scripted controller client and captured Envelopes.

``cli.client`` replaces the controller (``STATE._client``); it answers from canned Envelopes
(real ones captured from a controller + simulator run: ``fixtures/envelopes.json``) and
records every call. Sleeps and the monotonic clock are faked, so watch/wait loops run
instantly and deterministically.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from hpclab.cli.main import app
from hpclab.domain.errors import ControllerUnavailable

FIXTURES = Path(__file__).parent / "fixtures" / "envelopes.json"
_CORPUS: dict[str, dict[str, Any]] = json.loads(FIXTURES.read_text())

Responder = dict[str, Any] | list[dict[str, Any]] | Callable[[dict[str, Any]], dict[str, Any]]


def env(name: str) -> dict[str, Any]:
    """A deep copy of a captured Envelope (``method`` or ``method.variant``)."""
    return copy.deepcopy(_CORPUS[name])


def corpus_names() -> list[str]:
    return sorted(_CORPUS)


def ok(data: Any, summary: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": "hpclab/v1",
        "request_id": "req_test",
        "ok": True,
        "data": data,
        "error": None,
        "coverage": extra.pop("coverage", None),
        "observed_at": extra.pop("observed_at", "2026-09-28T06:00:00.000Z"),
        "next_cursor": extra.pop("next_cursor", None),
        "warnings": extra.pop("warnings", []),
        "summary": summary,
    }


def err(code: str, message: str = "boom", **details: Any) -> dict[str, Any]:
    return {
        "schema_version": "hpclab/v1",
        "request_id": "req_test",
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "operation_ref": None,
            "details": details,
        },
        "coverage": None,
        "observed_at": None,
        "next_cursor": None,
        "warnings": [],
        "summary": f"{code}: {message}",
    }


@dataclass
class FakeClient:
    """Answers ``call(method, params)`` from ``responses[method]``: an Envelope, a list consumed
    in order (the last one repeats) or a function of the params."""

    responses: dict[str, Responder] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    unavailable: bool = False

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        p = dict(params or {})
        self.calls.append((method, p))
        if self.unavailable:
            raise ControllerUnavailable(
                "the hpclab controller is not running", details={"hint": "hpclab controller start"}
            )
        r = self.responses.get(method)
        if r is None:
            return err("invalid_argument", f"unknown method {method!r} (not scripted)")
        if callable(r):
            return r(p)
        if isinstance(r, list):
            return copy.deepcopy(r.pop(0) if len(r) > 1 else r[0])
        return copy.deepcopy(r)

    def methods(self) -> list[str]:
        return [m for m, _ in self.calls]

    def params(self, method: str, index: int = -1) -> dict[str, Any]:
        found = [p for m, p in self.calls if m == method]
        assert found, f"{method} was not called; calls: {self.methods()}"
        return found[index]


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def json(self) -> Any:
        return json.loads(self.stdout)

    @property
    def text(self) -> str:
        return self.stdout + self.stderr


@dataclass
class Cli:
    client: FakeClient
    monkeypatch: pytest.MonkeyPatch
    interactive: bool = False
    sleeps: list[float] = field(default_factory=list)
    clock: list[float] = field(default_factory=lambda: [0.0])

    def run(self, *args: str, input: str | None = None, interactive: bool | None = None) -> Result:
        self.interactive = self.interactive if interactive is None else interactive
        res = CliRunner().invoke(app, list(args), input=input, catch_exceptions=False)
        return Result(res.exit_code, res.stdout, res.stderr)
