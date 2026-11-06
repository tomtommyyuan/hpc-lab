"""End to end with the vendorable example (examples/tiny-classifier): its real logs, metrics and
checkpoints go through diagnosis, progress and checkpoint validation exactly as the services will."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ops_support import EXAMPLE_DIR

from hpclab.adapters.checkpoint import MANIFEST_NAME, validate_checkpoint_manifest
from hpclab.services.diagnosis import AttemptFacts, diagnose
from hpclab.services.progress import estimate_progress

TRAIN = EXAMPLE_DIR / "train.py"
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="example not present")


@dataclass
class Run:
    rc: int
    stdout: str
    stderr: str
    out: Path
    ckpt: Path

    def metrics_lines(self) -> list[str]:
        return (self.out / "metrics.jsonl").read_text().splitlines()


def train(tmp_path: Path, *args: str) -> Run:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HPCLAB_", "SLURM_"))}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        HPCLAB_OUTPUT_DIR=str(tmp_path / "outputs"),
        HPCLAB_CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
    )
    proc = subprocess.run(
        [sys.executable, str(TRAIN), "--steps", "60", "--log-every", "10", "--eval-every", "20", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return Run(proc.returncode, proc.stdout, proc.stderr, tmp_path / "outputs", tmp_path / "checkpoints")


def disk_stats(ckpt_dir: Path) -> dict[str, dict[str, Any]]:
    stats = {}
    for path in ckpt_dir.iterdir():
        if path.is_file() and path.name != MANIFEST_NAME:
            data = path.read_bytes()
            stats[path.name] = {
                "exists": True,
                "size": len(data),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
    return stats


def test_successful_run_progress_and_checkpoints(tmp_path: Path) -> None:
    run = train(tmp_path, "--ckpt-every", "20")
    assert run.rc == 0, run.stderr

    progress = estimate_progress(run.metrics_lines(), elapsed_s=30, time_limit_s=3600)
    assert progress.final_seen and progress.step == 60 and progress.total_steps == 60
    assert progress.timelimit_risk == "none" and progress.fraction == 1.0
    assert isinstance(progress.latest_metrics["val/acc"], float)

    facts = AttemptFacts(
        scheduler_state="COMPLETED", exit_code=0, app_state="exited_ok", result_state="valid"
    )
    assert diagnose(facts, run.stdout, run.stderr).items == []

    ckpts = sorted(run.ckpt.iterdir())
    assert len(ckpts) == 3
    latest = ckpts[-1]
    manifest = json.loads((latest / MANIFEST_NAME).read_text())
    check = validate_checkpoint_manifest(
        manifest, disk_stats(latest), expected_run_spec_hash=None, expected_config=manifest["config"]
    )
    assert check.status == "valid" and check.step == 60
    assert check.digests_verified == len(manifest["files"])
    # no run_spec_hash outside hpclab, but every declared semantic key matches -> exact
    assert "lr" in manifest["semantic_keys"] and manifest["resumable"] is True
    assert check.continuation == "exact"

    changed = dict(manifest["config"], lr=manifest["config"]["lr"] * 2)
    other = validate_checkpoint_manifest(
        manifest, disk_stats(latest), expected_run_spec_hash=None, expected_config=changed
    )
    assert other.continuation == "incompatible" and "lr" in other.config_diffs

    victim = next(p for p in latest.iterdir() if p.name != MANIFEST_NAME)
    victim.write_bytes(victim.read_bytes()[:-10])  # simulate a write cut off by a kill
    broken = validate_checkpoint_manifest(manifest, disk_stats(latest), expected_run_spec_hash=None)
    assert broken.status == "invalid" and "size_mismatch" in broken.issue_codes()


def test_injected_cuda_oom(tmp_path: Path) -> None:
    run = train(tmp_path, "--oom-at-step", "15")
    assert run.rc != 0
    d = diagnose(AttemptFacts(scheduler_state="FAILED", exit_code=run.rc), run.stdout, run.stderr)
    assert d.primary_code == "cuda_oom"
    assert d.items[0].suggested_strategies[0] == "derive_run"


def test_injected_nan(tmp_path: Path) -> None:
    run = train(tmp_path, "--nan-at-step", "25")
    assert run.rc != 0
    d = diagnose(AttemptFacts(scheduler_state="FAILED", exit_code=run.rc), run.stdout, run.stderr)
    assert d.primary_code == "nan_loss" and "retry" not in d.items[0].suggested_strategies
    progress = estimate_progress(run.metrics_lines(), elapsed_s=10, time_limit_s=3600)
    assert progress.step == 25 and progress.latest_metrics["train/loss"] == "NaN"


def test_injected_python_error(tmp_path: Path) -> None:
    run = train(tmp_path, "--fail-at-step", "7")
    assert run.rc != 0
    d = diagnose(AttemptFacts(scheduler_state="FAILED", exit_code=run.rc), run.stdout, run.stderr)
    assert d.primary_code == "python_exception"
    assert d.last_error_line == "RuntimeError: injected failure at step 7 (--fail-at-step)"
    assert d.items[0].resource_hint["location"].endswith(" in inject_failures")
