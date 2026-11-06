"""Fixtures for the MCP tests: anyio backend and a live controller + simulator (module scope)."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator

import pytest
from mcp_support import Live, make_writable

from hpclab.config import get_paths
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.services.app import build_registry
from hpclab.sim.cluster import SimCluster


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Live]:
    """A real controller subprocess (autostarted by the client, exactly as the MCP server does)
    on a fresh home, with the local Slurm simulator registered as cluster 'sim'. Stopped, the
    simulator shut down and the (partly read-only) tree removed at the end of the module."""
    root = tmp_path_factory.mktemp("mcp-live")
    paths = get_paths(root / "home")
    client = HpcLabClient(paths, timeout=60)
    try:
        env = client.call("init_sim", {})
        assert env["ok"], env
        yield Live(
            root=root,
            paths=paths,
            client=client,
            sim=SimCluster(paths.sim_root("sim")),
            registry=build_registry(strict=True),
        )
    finally:
        client.close()
        with contextlib.suppress(Exception):
            stop_controller(paths, timeout=20, force=True)
        with contextlib.suppress(Exception):
            SimCluster(paths.sim_root("sim")).shutdown(grace_s=2.0)
        make_writable(root)
        shutil.rmtree(root, ignore_errors=True)
