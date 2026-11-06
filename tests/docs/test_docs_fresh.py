"""The generated reference docs (docs/tools.md, docs/cli.md) must match the code.

They are rendered by ``scripts/gen_docs.py`` from the MCP catalog, the service registry
schemas and the Typer command tree. When this test fails, regenerate them::

    uv run python scripts/gen_docs.py
"""

from __future__ import annotations

import difflib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gen_docs.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hpclab_gen_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load_generator()


def _assert_fresh(path: Path, expected: str) -> None:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    if current == expected:
        return
    diff = "".join(
        list(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=f"{path.relative_to(REPO)} (on disk)",
                tofile=f"{path.relative_to(REPO)} (generated)",
                n=1,
            )
        )[:60]
    )
    pytest.fail(
        f"{path.relative_to(REPO)} is stale; regenerate it with `uv run python scripts/gen_docs.py`.\n{diff}",
        pytrace=False,
    )


def test_tools_md_is_fresh(gen: ModuleType) -> None:
    _assert_fresh(gen.TOOLS_MD, gen.render_tools())


def test_cli_md_is_fresh(gen: ModuleType) -> None:
    _assert_fresh(gen.CLI_MD, gen.render_cli())


def test_every_tool_resource_and_command_is_documented(gen: ModuleType) -> None:
    """Guards the generator itself: nothing in the catalog or the command tree is skipped."""
    from typer.main import get_command

    from hpclab.cli.main import app
    from hpclab.mcp.catalog import TOOLS
    from hpclab.mcp.server import PROMPTS, RESOURCES

    tools_md = gen.render_tools()
    for tool in TOOLS:
        assert f"### {tool.name}\n" in tools_md
    for route in RESOURCES:
        assert f"`{route.template}`" in tools_md
    for name in PROMPTS:
        assert f"`{name}`" in tools_md

    cli_md = gen.render_cli()

    def walk(cmd: object, path: list[str]) -> None:
        for name, sub in getattr(cmd, "commands", {}).items():
            if getattr(sub, "hidden", False):
                continue
            title = " ".join([*path, name])
            assert f" {title}\n" in cli_md, f"{title} is missing from docs/cli.md"
            walk(sub, [*path, name])

    walk(get_command(app), ["hpclab"])
