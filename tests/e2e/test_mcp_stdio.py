"""The real MCP stdio server (``python -m hpclab.mcp``) driven by the real MCP client.

What an agent does: inspect the cluster, prepare the example from its YAML (relative path, the
server's cwd is the agent's), dispatch the matrix, watch the grid, inspect a run, compare,
export evidence and read the LaTeX table and the reviewed sbatch script as resources. The
server autostarts the controller itself. Its stdout must carry nothing but JSON-RPC.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import anyio
import pytest
from e2e_support import Lab

from hpclab.controller.client import stop_controller
from mcp import Client, StdioServerParameters

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150), pytest.mark.anyio]

TOOLS = {
    "cluster_inspect",
    "experiment_prepare",
    "run_submit",
    "workflow_submit",
    "runs_list",
    "run_inspect",
    "run_logs",
    "run_cancel",
    "run_recover",
    "artifacts_list",
    "artifact_read",
    "evaluation_collect",
    "results_compare",
    "evidence_export",
    "evidence_verify",
    "workflow_cancel",
    "usage_report",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _text(item: Any) -> str:
    """Text of a content block / resource contents (fails on binary blocks)."""
    text = getattr(item, "text", None)
    assert isinstance(text, str), item
    return text


def _server(lab: Lab) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "hpclab.mcp"],
        env={"HPCLAB_HOME": str(lab.paths.home)},
        cwd=str(lab.root),
    )


async def test_agent_flow_over_stdio(lab: Lab) -> None:
    stop_controller(lab.paths, force=True)  # the MCP server must start its own controller
    transport_faults: list[Any] = []

    async def on_message(message: Any) -> None:
        if isinstance(message, Exception):  # e.g. a non-JSON-RPC line on the server's stdout
            transport_faults.append(message)

    async with Client(_server(lab), message_handler=on_message, read_timeout_seconds=60) as client:

        async def tool(name: str, **args: Any) -> dict[str, Any]:
            r = await client.call_tool(name, args)
            env = r.structured_content
            assert env is not None and not r.is_error and env["ok"], (name, _text(r.content[0])[:3000])
            assert r.content and r.content[0].type == "text" and _text(r.content[0])
            return env

        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == TOOLS and len(tools) == 17
        for t in tools:
            assert t.input_schema["type"] == "object" and t.input_schema.get("properties") is not None, t.name
            out = t.output_schema or {}
            assert {"ok", "data", "error", "summary"} <= set(out.get("properties", {})), t.name
            assert t.description and t.annotations is not None, t.name

        inspect = await tool("cluster_inspect", cluster_id="sim")
        assert inspect["data"]["cluster_id"] == "sim"
        assert lab.client.ping() is not None  # autostarted by the MCP server's client

        prep = (
            await tool("experiment_prepare", idempotency_key="mcp-prep", spec_path="project/hpclab.yaml")
        )["data"]
        assert prep["status"] == "ready" and len(prep["runs"]) == 7, prep["validation"]
        wf = (
            await tool(
                "workflow_submit",
                idempotency_key="mcp-wf",
                plan_id=prep["plan_id"],
                plan_hash=prep["plan_hash"],
            )
        )["data"]

        deadline = time.monotonic() + 90
        while True:
            grid = (await tool("runs_list", workflow_id=wf["workflow_id"], view="grid", metric="acc"))[
                "data"
            ]["grid"]
            codes = [c["code"] for row in grid["rows"] for c in row["cells"]]
            if len(codes) == 6 and set(codes) == {"V"}:
                break
            assert time.monotonic() < deadline, grid
            await anyio.sleep(1.0)
        assert all(isinstance(c["value"], float) for row in grid["rows"] for c in row["cells"])

        run_id = grid["rows"][0]["cells"][0]["run_id"]
        insp = (await tool("run_inspect", run_id=run_id))["data"]
        assert insp["run"]["code"] == "V" and insp["attempts"][0]["job"]["job_id"]

        cmp_env = await tool(
            "results_compare",
            idempotency_key="mcp-cmp",
            planned_run_set_ref=prep["experiment_id"],
            metric_names=["acc"],
        )
        assert cmp_env["data"]["counts"]["included"] == 6 and cmp_env["coverage"]["complete"]
        ev = (
            await tool(
                "evidence_export",
                purpose="MCP e2e",
                idempotency_key="mcp-ev",
                comparison_ref=cmp_env["data"]["comparison_ref"],
            )
        )["data"]

        tex = await client.read_resource(f"hpclab://evidence/{ev['bundle_id']}/table.tex")
        (content,) = tex.contents
        assert content.mime_type == "application/x-tex"
        table = _text(content)
        assert "\\begin{tabular}" in table and f"\\hpcval{{{ev['bundle_id']}:baseline.acc.mean}}" in table
        script = await client.read_resource(f"hpclab://plans/{prep['plan_id']}/script")
        text = _text(script.contents[0])
        assert text.startswith("#!") and "#SBATCH" in text and "--partition=gpu" in text, text[:500]

        # errors come back as tool errors with the typed envelope, not as protocol failures
        bad = await client.call_tool("run_inspect", {"run_id": "run_does_not_exist"})
        assert bad.is_error and bad.structured_content["error"]["code"] == "not_found"

    assert not transport_faults, transport_faults


def test_stdout_is_protocol_only(lab: Lab) -> None:
    """Raw check: every byte the server writes to stdout is a JSON-RPC message."""
    msgs: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "raw-e2e", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "runs_list", "arguments": {}}},
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "cluster_inspect", "arguments": {"cluster_id": "sim"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "hpclab://plans/pln_nope/script"},
        },
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("HPCLAB_")}
    env["HPCLAB_HOME"] = str(lab.paths.home)
    stderr_log = open(lab.root / "mcp-stderr.log", "wb")  # noqa: SIM115 - closed below
    proc = subprocess.Popen(
        [sys.executable, "-m", "hpclab.mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_log,  # a file, so a chatty stderr can never block the server
        env=env,
        cwd=lab.root,
    )
    assert proc.stdin is not None and proc.stdout is not None
    answered: dict[int, dict[str, Any]] = {}
    lines: list[bytes] = []
    try:
        for m in msgs:
            proc.stdin.write((json.dumps(m) + "\n").encode())
            proc.stdin.flush()
            if "id" not in m:
                continue
            while m["id"] not in answered:
                line = proc.stdout.readline()
                assert line, (lab.root / "mcp-stderr.log").read_text()
                lines.append(line)
                obj = json.loads(line)  # anything else on stdout fails here
                assert obj.get("jsonrpc") == "2.0", line
                if "id" in obj:
                    answered[obj["id"]] = obj
    finally:
        try:
            rest, _ = proc.communicate(timeout=15)  # closes stdin: the server exits at EOF
        except subprocess.TimeoutExpired:
            proc.kill()
            rest, _ = proc.communicate()
        stderr_log.close()
    for line in rest.splitlines():
        if line.strip():
            assert json.loads(line).get("jsonrpc") == "2.0", line
    assert len(answered[2]["result"]["tools"]) == 17
    assert answered[3]["result"]["isError"] is False
    assert answered[4]["result"]["structuredContent"]["ok"] is True
    assert "error" in answered[5] or answered[5]["result"]["contents"]  # an honest answer either way
