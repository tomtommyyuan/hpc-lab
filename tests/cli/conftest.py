"""CLI test harness: the real typer app against a scripted controller client.

``cli.client`` replaces the controller (``STATE._client``); it answers from canned Envelopes
(real ones captured from a controller + simulator run: ``fixtures/envelopes.json``) and
records every call. Sleeps and the monotonic clock are faked, so watch/wait loops run
instantly and deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from cli_support import Cli, FakeClient

from hpclab.cli import commands
from hpclab.cli.common import STATE, console, err_console


@pytest.fixture
def cli(
    monkeypatch: pytest.MonkeyPatch, hpclab_home: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Cli]:
    # commands with a cwd-relative default (`export` -> ./evidence, `examples copy` -> ./NAME)
    # must never write into the checkout when a test forgets --out
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    client = FakeClient()
    harness = Cli(client=client, monkeypatch=monkeypatch)
    monkeypatch.setattr(STATE, "_client", client)
    monkeypatch.setattr(commands, "_interactive", lambda: harness.interactive)

    def fake_sleep(seconds: float) -> None:
        harness.sleeps.append(seconds)
        harness.clock[0] += seconds

    monkeypatch.setattr(commands, "_sleep", fake_sleep)
    monkeypatch.setattr(commands, "_now", lambda: harness.clock[0])
    # wide, deterministic rendering (no wrapping of ids in assertions)
    monkeypatch.setattr(console, "_width", 200)
    monkeypatch.setattr(err_console, "_width", 200)
    yield harness
    STATE.json_output = False
