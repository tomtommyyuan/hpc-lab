"""Fixtures for cluster setup tests (init_sim, auth_resume): service contexts on homes whose
paths job scripts cannot carry, and cleanup of the simulator directories that ``init_sim``
relocates to ``/var/tmp`` for such homes."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from setup_support import make_writable, relocated_sim_root, remove_sim_root

from hpclab.config import get_paths
from hpclab.domain.clock import Clock
from hpclab.services.context import ServiceContext


@pytest.fixture
def make_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ServiceContext]]:
    """``make_ctx("my home")`` -> a ServiceContext on ``tmp_path/<name>/.hpclab`` (HPCLAB_HOME set).
    Every simulator it may have created — under the home or relocated to /var/tmp — is shut down
    and removed afterwards."""
    ctxs: list[ServiceContext] = []

    def make(name: str = "home", *, clock: Clock | None = None) -> ServiceContext:
        home = tmp_path / name / ".hpclab"
        monkeypatch.setenv("HPCLAB_HOME", str(home))
        ctx = ServiceContext.create(get_paths(home), clock=clock)
        ctxs.append(ctx)
        return ctx

    yield make
    for ctx in ctxs:
        with contextlib.suppress(Exception):
            ctx.clusters.close()
        with contextlib.suppress(Exception):
            ctx.db.close_all()
        remove_sim_root(relocated_sim_root(ctx.paths.home))
        remove_sim_root(ctx.paths.sim_root("sim"))
    make_writable(tmp_path)
