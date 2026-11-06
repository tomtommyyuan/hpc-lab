"""``hpclab demo``: the guided tour runs end to end (``fast``) and leaves nothing behind.

The demo runs in its own throwaway home under the user's home; after it returns, its
controller must be stopped, its simulated jobs gone and its directory removed (unless keep).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from e2e_support import make_writable
from rich.console import Console

from hpclab import demo as demo_mod
from hpclab.config import get_paths
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.controller.files import lock_is_held
from hpclab.demo import run_demo

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=140, force_terminal=False, color_system=None), buf


def _processes_mentioning(path: Path) -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=False).stdout
    return [line for line in out.splitlines() if str(path) in line]


def test_demo_fast_tour_and_no_leftovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A home whose path job scripts cannot use (a space, as in ~/Library/Application Support):
    # the simulator's "remote" paths live below the home, so the demo must run elsewhere.
    home = get_paths(tmp_path / "Application Support" / "hpclab")
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    console, buf = _console()
    rc = run_demo(home, console=console, fast=True, keep=False)
    text = buf.getvalue()
    assert rc == 0, text
    assert str(temp) in text.replace("\n", "")  # the intro says where it runs
    for expected in (
        "[1/8]",
        "smoke gate passed",
        "injected",
        "diagnosis node_fail",
        "n=2/3",
        "retry (diagnosis node_fail)",
        "\\begin{tabular}",
        "\\hpcval{evb_",
        "| mixup |",
        "Trace of one number",
        "← Slurm job",
        "claude mcp add hpclab",
        "E completed, evaluation pending (not a failure)",  # the grid legend
        "hpclab examples copy tiny-classifier my-project",  # works from an installed wheel too
        "demo complete",
    ):
        assert expected in text, f"{expected!r} missing from the demo output:\n{text}"
    # the final grid line is all valid
    final = [ln for ln in text.splitlines() if "complete " in ln and "baseline:" in ln]
    assert final and re.search(r"baseline: V V V\s+mixup: V V V", final[-1]), final
    # no leftovers: directory, controller, simulated jobs
    assert not (home.home / "demo").exists(), list((home.home / "demo").rglob("*"))[:10]
    assert not list(temp.iterdir()), list(temp.rglob("*"))[:10]
    assert not _processes_mentioning(temp), _processes_mentioning(temp)
    assert "cleanup" not in text


def test_demo_keep_leaves_an_explorable_home(tmp_path: Path) -> None:
    home = get_paths(tmp_path / "home")
    console, buf = _console()
    rc = run_demo(home, console=console, fast=True, keep=True)
    text = buf.getvalue()
    assert rc == 0, text
    (root,) = list((home.home / "demo").iterdir())
    demo_home = get_paths(root / "home")
    try:
        assert not lock_is_held(demo_home.lock)  # its controller was stopped ...
        assert not _processes_mentioning(root)
        assert f"--home {demo_home.home}" in text.replace("\n", "")  # ... and the output says how to explore
        with HpcLabClient(demo_home, timeout=60) as client:  # ... which autostarts it again
            env = client.call("experiments_list", {})
            assert env["ok"] and len(env["data"]["experiments"]) == 1, env
            exp = env["data"]["experiments"][0]["experiment_id"]
            grid = client.call("runs_list", {"experiment_id": exp, "view": "grid"})["data"]["grid"]
            assert {c["code"] for row in grid["rows"] for c in row["cells"]} == {"V"}
    finally:
        stop_controller(demo_home, force=True)
        make_writable(root)


def test_demo_without_a_script_safe_directory_relocates_the_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Home and temp dir both unusable in job scripts (spaces): `init_sim` places the simulator
    in /var/tmp; the tour must drive and clean up THAT simulator (no jobs or directory left)."""
    from hpclab.services.clusters import relocated_sim_root

    home = get_paths(tmp_path / "my home")
    temp = tmp_path / "my tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    console, buf = _console()
    rc = run_demo(home, console=console, fast=True, keep=False)
    text = buf.getvalue()
    flat = " ".join(text.split())
    assert rc == 0, text
    assert "demo complete" in text and "cleanup" not in text
    root = re.search(r"throwaway hpclab home \((.+?)\);", flat)
    assert root is not None, flat
    sim_root = relocated_sim_root(Path(root.group(1)) / "home")
    assert not sim_root.exists(), list(sim_root.rglob("*"))[:10]
    assert not _processes_mentioning(sim_root) and not _processes_mentioning(temp)
    assert not list(temp.iterdir()) and not (home.home / "demo").exists()


def test_demo_invalid_plan_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan that cannot run: the tour stops at prepare and prints the blocking finding itself,
    not just its check name."""
    examples = tmp_path / "examples"
    src = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"
    shutil.copytree(src, examples / "tiny-classifier", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    spec = examples / "tiny-classifier" / "hpclab.yaml"
    spec.write_text(spec.read_text().replace("partition: gpu", "partition: no-such-partition"))
    monkeypatch.setattr(demo_mod, "bundled_path", lambda kind: examples)
    home = get_paths(tmp_path / "home")
    console, buf = _console()
    rc = run_demo(home, console=console, fast=True, keep=False)
    text = " ".join(buf.getvalue().split())
    assert rc == 1, text
    assert "demo stopped" in text and "partition_exists" in text and "no-such-partition" in text, text
    assert not (home.home / "demo").exists()
    assert not _processes_mentioning(home.home)


def test_ctrl_c_cleans_up_and_says_so(tmp_path: Path) -> None:
    """Ctrl-C in a terminal (SIGINT to the foreground process group), then an impatient second
    Ctrl-C while the demo cleans up: exit 130, its controller, simulated jobs and directory are
    gone, and the message does not send the user to reset their own simulator."""
    home = tmp_path / "home"
    env = {k: v for k, v in os.environ.items() if not k.startswith("HPCLAB_")}
    env.update(HPCLAB_HOME=str(home), COLUMNS="200", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "hpclab.cli", "demo", "--fast"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    try:
        deadline = time.monotonic() + 90
        for line in proc.stdout:  # wait until jobs are on the (simulated) cluster
            lines.append(line)
            if "injected" in line:
                break
            assert time.monotonic() < deadline, "".join(lines)
        time.sleep(1.0)
        os.killpg(proc.pid, signal.SIGINT)
        for line in proc.stdout:
            lines.append(line)
            if "interrupted" in line:  # cleanup has started: Ctrl-C again
                os.killpg(proc.pid, signal.SIGINT)
                break
        rest, _ = proc.communicate(timeout=60)
        lines.append(rest)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    text = "".join(lines)
    assert proc.returncode == 130, text
    assert "sim reset" not in text and "cleaned up" in text, text
    assert not (home / "demo").exists(), list((home / "demo").rglob("*"))[:10]
    assert not _processes_mentioning(home), _processes_mentioning(home)


def test_unwritable_home_is_a_clear_error(tmp_path: Path) -> None:
    """A mistyped HPCLAB_HOME on a read-only place: a message and exit 1, not a traceback."""
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o500)
    try:
        console, buf = _console()
        rc = run_demo(get_paths(ro / "home"), console=console, fast=True, keep=False)
    finally:
        ro.chmod(0o700)
    text = " ".join(buf.getvalue().split())
    assert rc == 1, text
    assert "demo stopped" in text and "must be a directory you own and can write" in text, text
    assert not list(ro.iterdir())


def test_demo_failure_is_explained_and_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A step that cannot complete ends the tour with the reason; nothing is left running."""
    monkeypatch.setattr(demo_mod, "bundled_path", lambda kind: tmp_path / "no-such-dir")
    home = get_paths(tmp_path / "home")
    console, buf = _console()
    rc = run_demo(home, console=console, fast=True, keep=False)
    text = buf.getvalue()
    assert rc == 1, text
    assert "demo stopped" in text and "example is not bundled" in text, text
    assert not (home.home / "demo").exists()
    assert not _processes_mentioning(home.home / "demo")
