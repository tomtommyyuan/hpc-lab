"""benchmarks/fault_matrix.py keeps working: a two-scenario run writes a complete report."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from e2e_support import REPO

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]


def test_fault_matrix_subset_writes_json_and_markdown(tmp_path: Path) -> None:
    out = tmp_path / "results"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / "fault_matrix.py"),
            "--out",
            str(out),
            "--only",
            "lost_reply,cancel_foreign_owner",
            "--reps",
            "1",
            "--jobs",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=140,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout == ""  # progress goes to stderr; stdout stays clean for piping
    # a subset never overwrites (or passes itself off as) the full report
    assert sorted(p.name for p in out.iterdir()) == ["fault_matrix_custom.json", "fault_matrix_custom.md"]
    data = json.loads((out / "fault_matrix_custom.json").read_text())
    assert data["meta"]["mode"] == "custom" and data["meta"]["reps"] == 1
    assert {r["scenario"] for r in data["records"]} == {"lost_reply", "cancel_foreign_owner"}
    assert all(r["passed"] for r in data["records"]), data["records"]
    lost = next(r for r in data["records"] if r["scenario"] == "lost_reply")
    assert (lost["jobs_accepted"], lost["duplicates"], lost["dispatches"]) == (1, 0, 1)
    foreign = next(r for r in data["records"] if r["scenario"] == "cancel_foreign_owner")
    assert foreign["cancels"] == 0 and foreign["outcome"] == "cancel refused"
    assert data["meta"]["policy_overrides"] and data["meta"]["random_seed"] is not None
    md = (out / "fault_matrix_custom.md").read_text()
    assert "mode **custom** (2 of 14 scenarios)" in md
    assert "## Summary" in md and "| `lost_reply` |" in md and "## Failure cases (0)" in md


def _processes_mentioning(path: Path) -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=False).stdout
    return [line for line in out.splitlines() if str(path) in line]


def test_ctrl_c_stops_the_matrix_and_cleans_up(tmp_path: Path) -> None:
    """Ctrl-C (SIGINT to the process group, as a terminal sends it) while a repetition runs: the
    script stops within seconds instead of finishing the matrix, stops the repetitions'
    controllers and simulated jobs, removes its work directory and writes no partial report."""
    temp = tmp_path / "tmp"  # the script's work directory goes here (tempfile honours TMPDIR)
    temp.mkdir()
    out = tmp_path / "results"
    env = {**os.environ, "TMPDIR": str(temp)}
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "benchmarks" / "fault_matrix.py"),
            "--out",
            str(out),
            "--only",
            "auth_required_window,lost_reply",  # auth_required_window takes > 2 min when left alone
            "--reps",
            "1",
            "--jobs",
            "2",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not list(temp.glob("hpclab-fault-matrix-*/auth_required_window-1/home/run/controller.pid")):
            assert proc.poll() is None and time.monotonic() < deadline, proc.communicate()
            time.sleep(0.2)
        time.sleep(2.0)
        t0 = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        _, err = proc.communicate(timeout=45)
        took = time.monotonic() - t0
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
    assert proc.returncode == 130, err[-3000:]
    assert "interrupted" in err and took < 40, (took, err[-3000:])
    assert not list(temp.iterdir()), list(temp.rglob("*"))[:10]
    assert not _processes_mentioning(temp), _processes_mentioning(temp)
    assert not out.exists() or not list(out.iterdir())  # no partial report passed off as a result
