from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from hpclab.api.envelope import Coverage
from hpclab.config import Paths
from hpclab.domain.errors import NotFound
from hpclab.services.context import ServiceContext
from hpclab.services.registry import Registry


class EchoData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int
    label: str | None
    coverage: Coverage
    warnings: list[str] = []


def echo(ctx: ServiceContext, *, value: int, label: str | None = None) -> EchoData:
    """Echo a value."""
    cov = Coverage()
    cov.add("squeue", "stale", detail="rate limited")
    return EchoData(value=value, label=label, coverage=cov, warnings=["w1"])


def missing(ctx: ServiceContext, *, run_id: str) -> EchoData:
    raise NotFound(f"run {run_id} not found")


def boom(ctx: ServiceContext) -> EchoData:
    raise RuntimeError("secret internal detail")


def _ctx(paths: Paths) -> ServiceContext:
    return ServiceContext.create(paths)


def test_dispatch_wraps_and_lifts(paths: Paths) -> None:
    reg = Registry()
    reg.register("echo", echo, summarize=lambda d: f"echo {d.value}")
    env = reg.dispatch(_ctx(paths), "echo", {"value": "3"}, request_id="r1")
    assert env["ok"] is True and env["request_id"] == "r1"
    assert env["data"] == {"value": 3, "label": None}
    assert env["coverage"]["complete"] is False
    assert env["coverage"]["sources"][0]["status"] == "stale"
    assert env["warnings"] == ["w1"] and env["summary"] == "echo 3"


def test_validation_and_errors(paths: Paths) -> None:
    reg = Registry()
    reg.register("echo", echo)
    reg.register("missing", missing)
    reg.register("boom", boom)
    ctx = _ctx(paths)
    bad = reg.dispatch(ctx, "echo", {"value": "x", "extra": 1}, request_id="r")
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_argument"
    assert {e["loc"] for e in bad["error"]["details"]["errors"]} == {"value", "extra"}
    nf = reg.dispatch(ctx, "missing", {"run_id": "run_x"}, request_id="r")
    assert nf["error"]["code"] == "not_found" and nf["error"]["retryable"] is False
    crash = reg.dispatch(ctx, "boom", {}, request_id="r9")
    assert crash["error"]["code"] == "internal" and "secret" not in crash["error"]["message"]
    unknown = reg.dispatch(ctx, "nope", {}, request_id="r")
    assert unknown["error"]["code"] == "invalid_argument"
