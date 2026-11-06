"""Drive the real HPC Lab MCP server the way an agent (Claude Code) does.

``Agent.call`` sends a tool call through an in-process ``mcp.Client``, then checks what every
call must satisfy: the structured result validates against the tool's declared output schema
(errors included), ``is_error`` matches the envelope, and the text block stays within the
agent-facing budget. Every call is recorded (tool, arguments, text size) so tests can assert on
the whole conversation.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from jsonschema import Draft202012Validator

from hpclab.config import Paths
from hpclab.controller.client import HpcLabClient
from hpclab.mcp.server import create_server
from hpclab.services.registry import Registry
from hpclab.sim.cluster import SimCluster
from mcp import Client

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_SPEC = REPO / "examples" / "tiny-classifier" / "hpclab.yaml"
TEXT_BUDGET = 8 * 1024  # a common call's text block must stay below this
CONTENT_TOOLS = frozenset({"run_logs", "artifact_read"})  # the text carries content the agent asked for
ALL_TOOLS = frozenset(
    {
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
)
CALLED_TOOLS: set[str] = set()  # every tool any live test called successfully (module-wide)


@dataclass
class Live:
    """A real controller subprocess on a fresh HPCLAB_HOME with the simulator as cluster 'sim'."""

    root: Path
    paths: Paths
    client: HpcLabClient
    sim: SimCluster
    registry: Registry


@dataclass
class Call:
    tool: str
    args: dict[str, Any]
    text: str
    env: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.env.get("ok"))

    @property
    def data(self) -> dict[str, Any]:
        return self.env.get("data") or {}

    @property
    def size(self) -> int:
        return len(self.text.encode())

    @property
    def next_steps(self) -> str:
        """The 'Next:' section of the text ('' when there is none)."""
        _, _, tail = self.text.partition("\nNext:\n")
        return tail


@dataclass
class Agent:
    client: Client
    tools: dict[str, Any]
    calls: list[Call] = field(default_factory=list)

    def _validator(self, tool: str) -> Draft202012Validator:
        schema = self.tools[tool].output_schema
        return Draft202012Validator(schema)

    async def call(self, tool: str, args: dict[str, Any], *, expect_error: bool | None = False) -> Call:
        """``expect_error``: False = must succeed, True = must fail, None = either."""
        result = await self.client.call_tool(tool, args)
        assert result.content and result.content[0].type == "text"
        text = result.content[0].text
        env = result.structured_content
        assert isinstance(env, dict), f"{tool}: no structuredContent"
        errors = sorted(self._validator(tool).iter_errors(env), key=lambda e: list(e.path))
        assert not errors, (
            f"{tool}: structuredContent does not match outputSchema: {errors[0].message} at {list(errors[0].path)}"
        )
        assert result.is_error is (not env.get("ok")), f"{tool}: is_error disagrees with envelope.ok"
        call = Call(tool, dict(args), text, env)
        self.calls.append(call)
        if expect_error is True:
            assert not call.ok, f"{tool} unexpectedly succeeded:\n{text}"
        elif expect_error is False:
            assert call.ok, f"{tool} failed:\n{text}"
        if call.ok:
            CALLED_TOOLS.add(tool)
        if tool not in CONTENT_TOOLS:
            assert call.size <= TEXT_BUDGET, f"{tool}: text block is {call.size} bytes:\n{text[:3000]}"
        assert text.splitlines()[0].strip(), f"{tool}: empty first line"
        return call

    async def resource(self, uri: str) -> tuple[str, str | None]:
        rr = await self.client.read_resource(uri)
        content = rr.contents[0]
        return content.text, content.mime_type  # type: ignore[union-attr]

    async def wait(
        self,
        tool: str,
        args: dict[str, Any],
        until: Callable[[Call], bool],
        *,
        timeout: float = 90.0,
        interval: float = 1.0,
    ) -> Call:
        """Poll a read tool (as an agent would, but faster) until ``until(call)``."""
        deadline = time.monotonic() + timeout
        while True:
            call = await self.call(tool, args)
            if until(call):
                return call
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting on {tool}({args}); last text:\n{call.text}")
            await anyio.sleep(interval)


@asynccontextmanager
async def agent_session(live: Live, *, cwd: Path | None = None) -> AsyncIterator[Agent]:
    """A fresh MCP session (a new chat) against the shared controller."""
    server = create_server(client=live.client, registry=live.registry, cwd=cwd or REPO)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        yield Agent(client, tools)


def grid_counts(call: Call) -> dict[str, int]:
    return dict(call.data.get("counts") or {})


def grid_settled(call: Call) -> bool:
    """No main cell is waiting or running any more (only V/E/F/X/C!/U remain)."""
    counts = grid_counts(call)
    return bool(counts) and not any(counts.get(k) for k in ("R", "Q", "S", "P", "?"))


def make_writable(root: Path) -> None:
    """Snapshots are read-only on purpose; tests must still be able to delete their tmp dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            try:
                mode = os.lstat(p).st_mode
                if not stat.S_ISLNK(mode):
                    os.chmod(p, mode | stat.S_IWUSR | (stat.S_IXUSR if stat.S_ISDIR(mode) else 0))
            except OSError:
                pass
