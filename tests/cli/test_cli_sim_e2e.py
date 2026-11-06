"""The real `hpclab` CLI (subprocess) against a real controller and the Slurm simulator.

init --sim → examples copy tiny-classifier → prepare → submit the plan → grid until V=6 → compare →
export (and export --bundle again) → paper check (clean, through \\input, then a hand-edited number)
→ paper strip → controller stop.
Everything runs in a throwaway HPCLAB_HOME; the controller is stopped and simulated jobs are
killed even when an assertion fails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hpclab.cli.commands import make_writable

pytestmark = [pytest.mark.sim, pytest.mark.slow]


class HpcLab:
    def __init__(self, home: Path, cwd: Path) -> None:
        self.home = home
        self.cwd = cwd
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("HPCLAB_", "SLURM_"))}
        self.env.update(HPCLAB_HOME=str(home), COLUMNS="200", PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")

    def run(self, *args: str, timeout: float = 90, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [sys.executable, "-m", "hpclab.cli", *args],
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and proc.returncode != 0:
            raise AssertionError(
                f"hpclab {' '.join(args)} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
            )
        return proc

    def json(self, *args: str, timeout: float = 90) -> dict[str, Any]:
        proc = self.run("--json", *args, timeout=timeout)
        doc: dict[str, Any] = json.loads(proc.stdout)
        return doc


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[HpcLab]:
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    h = HpcLab(home, work)
    try:
        yield h
    finally:
        h.run("controller", "stop", "--force", check=False, timeout=60)
        sim_root = home / "sim" / "sim"
        if sim_root.exists():
            from hpclab.sim.cluster import SimCluster

            SimCluster(sim_root).shutdown(grace_s=2.0)
        if home.exists():
            make_writable(home)  # published snapshots are read-only by design


def test_cli_end_to_end_on_the_simulator(lab: HpcLab, tmp_path: Path) -> None:
    r = lab.run("init", "--sim")
    assert "cluster sim" in r.stdout
    assert "hpclab examples copy tiny-classifier" in r.stdout

    r = lab.run("examples", "copy", "tiny-classifier")
    assert "hpclab prepare tiny-classifier/hpclab.yaml" in r.stdout
    assert (lab.cwd / "tiny-classifier" / "train.py").is_file()
    assert not list((lab.cwd / "tiny-classifier").rglob("*.pyc"))

    r = lab.run("prepare", "tiny-classifier/hpclab.yaml")
    plan_id = re.search(r"plan (pln_\w+) ready", r.stdout).group(1)  # type: ignore[union-attr]
    assert f"hpclab submit {plan_id}" in r.stdout
    assert "unchecked" in r.stdout  # unknowns are reported as unchecked, not as passes

    refused = lab.run("submit", plan_id, check=False)  # no terminal and no --yes: nothing is spent
    assert refused.returncode == 2 and "refusing without --yes" in refused.stderr

    sub = lab.json("submit", plan_id, "--yes")
    assert sub["ok"], sub
    wf = sub["data"]["workflow_id"]
    exp = sub["data"]["experiment_id"]

    watch = lab.run(
        "grid", wf, "--watch", "--interval", "1", "--timeout", "120", "--metric", "acc", timeout=150
    )
    assert "done:" in watch.stderr
    deadline = time.monotonic() + 60
    while True:
        g = lab.json("grid", wf, "--metric", "acc")
        if g["data"]["counts"].get("V") == 6 or time.monotonic() > deadline:
            break
        time.sleep(1)
    assert g["data"]["counts"].get("V") == 6, g["data"]["counts"]
    values = [c["value"] for row in g["data"]["grid"]["rows"] for c in row["cells"]]
    assert all(isinstance(v, float) for v in values)

    human = lab.run("grid", exp, "--metric", "acc")
    assert "cells: 6/6 valid" in human.stdout

    cmp_human = lab.run("compare", exp, "-m", "acc", "-m", "loss")
    assert "(3/3)" in cmp_human.stdout and "coverage: 6/6 planned runs included" in cmp_human.stdout
    cmp_id = re.search(r"comparison (cmp_\w+)", cmp_human.stdout).group(1)  # type: ignore[union-attr]

    out_dir = tmp_path / "paper" / "evidence"
    exp_out = lab.run("export", cmp_id, "--out", str(out_dir), "--purpose", "Table 1")
    table = out_dir / "table.tex"
    assert table.is_file() and (out_dir / "summary.md").is_file() and (out_dir / "manifest.json").is_file()
    assert str(table) in exp_out.stdout

    paper = tmp_path / "paper" / "main.tex"
    paper.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n"
        + table.read_text()
        + "\n\\end{document}\n"
    )
    ok = lab.run("paper", "check", str(paper))
    assert "no errors" in ok.stdout and str(paper) in ok.stdout  # the files scanned are listed

    # the same bundle written again (no new bundle), then a paper that only \\inputs the table
    bundle_id = re.search(r"\\hpcval\{(evb_\w+):", table.read_text()).group(1)  # type: ignore[union-attr]
    again = tmp_path / "paper" / "again"
    lab.run("export", "--bundle", bundle_id, "--out", str(again))
    assert (again / "table.tex").read_text() == table.read_text()
    wrapper = tmp_path / "paper" / "wrapper.tex"
    wrapper.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{evidence/table}\n\\end{document}\n"
    )
    unanchored = lab.run("paper", "check", str(wrapper), check=False)
    assert unanchored.returncode == 1 and "--bundle" in unanchored.stderr
    via_input = lab.run("paper", "check", str(wrapper), "--bundle", bundle_id)
    assert "no errors" in via_input.stdout and str(table) in via_input.stdout, via_input.stdout
    plain = tmp_path / "paper" / "plain.tex"
    plain.write_text("\\documentclass{article}\n\\begin{document}\nAccuracy 0.91.\n\\end{document}\n")
    not_checked = lab.run("paper", "check", str(plain), "--bundle", bundle_id, check=False)
    assert not_checked.returncode == 1 and "NO_TAGS_FOUND" in not_checked.stdout

    text = paper.read_text()
    tag = re.search(r"\\hpcval\{(evb_\w+):mixup\.acc\.mean\}\{([^}]*)\}", text)
    assert tag is not None
    edited = tmp_path / "paper" / "edited.tex"
    edited.write_text(text.replace(tag.group(0), "\\hpcval{" + tag.group(1) + ":mixup.acc.mean}{0.123}"))
    bad = lab.run("paper", "check", str(edited), check=False)
    assert bad.returncode == 1
    assert "VALUE_MISMATCH" in bad.stdout and "0.123" in bad.stdout

    stripped = lab.run("paper", "strip", str(paper))
    assert "\\hpcval" not in stripped.stdout
    assert tag.group(2) in stripped.stdout  # the number itself stays
    assert "stripped" in stripped.stderr

    stop = lab.run("controller", "stop")
    assert "controller stopped" in stop.stdout
