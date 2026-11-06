"""MCP frontend tests with a fake controller client and a toy registry (no services needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from hpclab.api.envelope import Coverage
from hpclab.mcp.server import create_server, match_resource, normalize_arguments
from hpclab.services.context import ServiceContext
from hpclab.services.registry import Registry
from mcp import Client

pytestmark = pytest.mark.anyio


class InspectData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_id: str
    checks: list[str]
    coverage: Coverage


def cluster_inspect(ctx: ServiceContext, *, cluster_id: str, checks: list[str] | None = None) -> InspectData:
    raise NotImplementedError


class PlanScriptData(BaseModel):
    plan_id: str
    script: str


def plan_script(ctx: ServiceContext, *, plan_id: str, run_id: str | None = None) -> PlanScriptData:
    raise NotImplementedError


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, dict(params or {})))
        if method == "plan_script":
            return {
                "ok": True,
                "request_id": "r",
                "data": {"plan_id": params["plan_id"], "script": "#!/bin/bash\n"},
                "summary": "script",
            }
        if params and params.get("cluster_id") == "down":
            return {
                "ok": False,
                "request_id": "r",
                "data": None,
                "summary": "auth_required: run hpclab auth down",
                "error": {
                    "code": "auth_required",
                    "message": "run hpclab auth down",
                    "retryable": True,
                    "details": {},
                },
            }
        return {
            "ok": True,
            "request_id": "r",
            "data": {"cluster_id": params["cluster_id"], "checks": []},
            "coverage": {"complete": True, "sources": [], "missing": [], "notes": []},
            "summary": "cluster ok",
            "warnings": [],
        }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _registry() -> Registry:
    reg = Registry()
    reg.register("cluster_inspect", cluster_inspect)
    reg.register("plan_script", plan_script)
    return reg


async def test_tools_and_calls(tmp_path: Path) -> None:
    fake = FakeClient()
    server = create_server(client=fake, registry=_registry(), cwd=tmp_path)
    async with Client(server) as c:
        tools = (await c.list_tools()).tools
        assert [t.name for t in tools] == ["cluster_inspect"]  # only registered methods are exposed
        t = tools[0]
        assert t.annotations.read_only_hint is True
        assert "Registered cluster id" in t.input_schema["properties"]["cluster_id"]["description"]
        data_schema = t.output_schema["properties"]["data"]["anyOf"][0]
        assert "coverage" not in data_schema["properties"]  # lifted into the envelope
        r = await c.call_tool("cluster_inspect", {"cluster_id": "hai"})
        assert r.is_error is False and r.structured_content["data"]["cluster_id"] == "hai"
        assert r.content[0].text.startswith("cluster ok")
        bad = await c.call_tool("cluster_inspect", {"cluster_id": "down"})
        assert bad.is_error is True and bad.structured_content["error"]["code"] == "auth_required"
        rr = await c.read_resource("hpclab://plans/pln_1/script")
        assert rr.contents[0].text == "#!/bin/bash\n"
        prompts = (await c.list_prompts()).prompts
        assert {p.name for p in prompts} >= {"plan_experiment", "diagnose_run", "write_results"}
        p = await c.get_prompt("diagnose_run", {"run_id": "run_x"})
        assert "run_id: run_x" in p.messages[0].content.text
    assert fake.calls[0] == ("cluster_inspect", {"cluster_id": "hai"})


def test_normalize_paths(tmp_path: Path) -> None:
    out = normalize_arguments("experiment_prepare", {"spec_path": "exp/hpclab.yaml"}, tmp_path)
    assert out["spec_path"] == str((tmp_path / "exp/hpclab.yaml").resolve())
    out = normalize_arguments("experiment_prepare", {"spec": {"name": "x"}}, tmp_path)
    assert out["spec"]["code"]["project_root"] == str(tmp_path.resolve())
    out = normalize_arguments("evidence_verify", {"tex_paths": ["paper/main.tex"]}, tmp_path)
    assert out["tex_paths"] == [str((tmp_path / "paper/main.tex").resolve())]


def test_resource_routing() -> None:
    route, groups = match_resource("hpclab://evidence/evb_1/table.tex")
    assert route.method == "evidence_get" and groups == {"bundle_id": "evb_1", "file": "table.tex"}
    route, groups = match_resource("hpclab://workflows/wf_1/grid")
    assert route.params(groups) == {"workflow_id": "wf_1", "view": "grid"}
    assert match_resource("hpclab://nope/1") is None
