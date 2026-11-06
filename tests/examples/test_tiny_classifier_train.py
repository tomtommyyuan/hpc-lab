"""train.py behaves like a well-mannered cluster job: formats, determinism, resume, failures."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from tiny_classifier_support import TRAIN, RunTrain, read_jsonl, strip_times, train_env

FAST = ["--steps", "60", "--eval-every", "20", "--log-every", "10", "--ckpt-every", "20"]


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


def _check_record(rec: dict[str, Any]) -> None:
    assert rec["schema"] == "hpclab.metrics/v1"
    assert isinstance(rec["step"], int) and rec["step"] >= 1
    assert rec["split"] in {"train", "val", "test"}
    assert isinstance(rec["time"], float) and rec["time"] > 1e9
    assert isinstance(rec["metrics"], dict) and rec["metrics"]
    for value in rec["metrics"].values():
        assert value is None or (isinstance(value, float) and math.isfinite(value))


def test_default_run_uses_hpclab_env_and_writes_valid_metrics(run_train: RunTrain, tmp_path: Path) -> None:
    work = tmp_path / "run"
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "hpclab-plan.json").write_text(json.dumps({"execution_plan": {"run_spec_hash": "sha256:abc"}}))
    env = train_env(work, HPCLAB_ATTEMPT_DIR=str(attempt), HPCLAB_SEED="7")
    res = run_train("--method", "mixup", env=env)  # every path and the seed come from HPCLAB_*

    assert res.returncode == 0, res.stderr
    records = res.records()
    for rec in records:
        _check_record(rec)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    assert [r["step"] for r in train] == list(range(10, 301, 10))
    assert all(r["total_steps"] == 300 and "loss" in r["metrics"] for r in train)
    assert [r["step"] for r in val] == [50, 100, 150, 200, 250, 300]
    assert all(r["dataset"] == "synthetic-v1" and r["seed"] == 7 for r in val)

    final = res.final()
    assert final["split"] == "val" and final["step"] == 300 and final["total_steps"] == 300
    assert set(final["metrics"]) == {"acc", "loss"}
    assert 0.5 < final["metrics"]["acc"] <= 1.0
    assert final["units"] == {"acc": "fraction", "loss": "nats"}
    assert final["higher_is_better"] == {"acc": True, "loss": False}
    assert final["method"] == "mixup"

    summary = res.summary()
    _check_record(summary)
    assert summary["split"] == "test" and summary["final"] is True and summary["step"] == 300
    assert summary["dataset"] == "synthetic-v1" and summary["seed"] == 7

    ckpts = sorted(p.name for p in res.ckpt.iterdir())
    assert ckpts == ["ckpt-000000100", "ckpt-000000200", "ckpt-000000300"]
    manifest = json.loads((res.ckpt / "ckpt-000000300" / "hpclab-ckpt.json").read_text())
    assert manifest["schema"] == "hpclab.checkpoint/v1" and manifest["complete"] is True
    assert manifest["step"] == 300 and manifest["resumable"] is True
    assert manifest["run_id"] == "run_test" and manifest["attempt_id"] == "att_test"
    assert manifest["run_spec_hash"] == "sha256:abc"
    assert set(manifest["semantic_keys"]) >= {"method", "lr", "batch_size"}
    assert set(manifest["semantic_keys"]) <= set(manifest["config"])
    for entry in manifest["files"]:
        data = (res.ckpt / "ckpt-000000300" / entry["path"]).read_bytes()
        assert entry["size"] == len(data)
        assert entry["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()


def test_default_run_is_fast(run_train: RunTrain) -> None:
    start = time.monotonic()
    res = run_train("--method", "baseline", "--seed", "1")
    assert res.returncode == 0, res.stderr
    assert time.monotonic() - start < 3.0


def test_without_ckpt_dir_no_checkpoints_are_written(run_train: RunTrain, tmp_path: Path) -> None:
    work = tmp_path / "noc"
    res = run_train(*FAST, name="noc", env=train_env(work, HPCLAB_CHECKPOINT_DIR=""))
    assert res.returncode == 0, res.stderr
    assert not res.ckpt.exists()


# ---------------------------------------------------------------------------
# Determinism and treatment effects
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_metrics(run_train: RunTrain) -> None:
    a = run_train(*FAST, "--method", "mixup", "--seed", "3", name="a")
    b = run_train(*FAST, "--method", "mixup", "--seed", "3", name="b")
    assert a.returncode == b.returncode == 0
    assert strip_times(a.records()) == strip_times(b.records())
    assert strip_times([a.summary()]) == strip_times([b.summary()])


def test_methods_and_seeds_change_results(run_train: RunTrain) -> None:
    finals = {}
    for method in ("baseline", "mixup", "label_smoothing"):
        res = run_train(*FAST, "--method", method, "--seed", "1", name=method)
        assert res.returncode == 0, res.stderr
        finals[method] = res.final()["metrics"]
    assert len({json.dumps(m, sort_keys=True) for m in finals.values()}) == 3
    other_seed = run_train(*FAST, "--method", "baseline", "--seed", "2", name="seed2")
    assert other_seed.final()["metrics"] != finals["baseline"]
    assert other_seed.final()["dataset"] == "synthetic-v1"  # the dataset is fixed across seeds


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["baseline", "mixup"])
def test_resumed_run_matches_uninterrupted_run(run_train: RunTrain, method: str) -> None:
    common = [*FAST, "--method", method, "--seed", "2"]
    full = run_train(*common, name="full")
    crashed = run_train(*common, "--fail-at-step", "55", name="crashed")
    assert full.returncode == 0 and crashed.returncode == 1
    assert (crashed.ckpt / "ckpt-000000040").is_dir()
    assert not (crashed.ckpt / "ckpt-000000060").exists()

    resumed = run_train(*common, "--resume", str(crashed.ckpt), name="resumed")
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed from" in resumed.stdout and "step 40" in resumed.stdout
    assert resumed.final()["metrics"] == full.final()["metrics"]
    assert resumed.summary()["metrics"] == full.summary()["metrics"]
    after = [r for r in strip_times(full.records()) if r["step"] > 40]
    assert strip_times(resumed.records()) == after


def test_resume_auto_starts_fresh_without_checkpoint(run_train: RunTrain) -> None:
    res = run_train(*FAST, "--resume", "auto")
    assert res.returncode == 0, res.stderr
    assert "resumed" not in res.stdout


def test_resume_auto_picks_latest_in_ckpt_dir(run_train: RunTrain, tmp_path: Path) -> None:
    first = run_train(*FAST, "--fail-at-step", "50", name="shared")
    assert first.returncode == 1
    again = run_train(*FAST, "--resume", "auto", name="shared")
    assert again.returncode == 0, again.stderr
    assert "step 40" in again.stdout


def test_explicit_resume_without_checkpoint_is_an_error(run_train: RunTrain, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    res = run_train(*FAST, "--resume", str(empty))
    assert res.returncode == 2
    assert "no complete checkpoint" in res.stderr
    assert not (res.out / "metrics.jsonl").exists()


def test_resume_refuses_incompatible_checkpoint(run_train: RunTrain) -> None:
    base = run_train(*FAST, "--method", "baseline", "--seed", "1", name="base")
    assert base.returncode == 0
    res = run_train(*FAST, "--method", "mixup", "--seed", "1", "--resume", str(base.ckpt), name="other")
    assert res.returncode == 2
    assert "incompatible" in res.stderr and "method" in res.stderr


def test_resume_refuses_corrupted_checkpoint(run_train: RunTrain) -> None:
    base = run_train(*FAST, name="base")
    assert base.returncode == 0
    model = base.ckpt / "ckpt-000000060" / "model.json"
    model.write_bytes(model.read_bytes().replace(b"0", b"1", 1))
    res = run_train(*FAST, "--resume", str(base.ckpt / "ckpt-000000060"), name="again")
    assert res.returncode == 2
    assert "digest mismatch" in res.stderr


def test_fresh_run_refuses_a_ckpt_dir_holding_another_runs_checkpoints(run_train: RunTrain) -> None:
    """Silently mixing runs in one --ckpt-dir lets ``--resume auto`` pick up a stale checkpoint."""
    first = run_train(*FAST, name="shared")
    assert first.returncode == 0
    before = {p.name: (p / "hpclab-ckpt.json").read_bytes() for p in first.ckpt.iterdir()}

    again = run_train(*FAST, "--method", "mixup", name="shared")
    assert again.returncode == 2
    assert "already holds checkpoints" in again.stderr and "--resume auto" in again.stderr
    assert {p.name: (p / "hpclab-ckpt.json").read_bytes() for p in again.ckpt.iterdir()} == before


def test_fresh_run_never_inherits_a_previous_runs_results(run_train: RunTrain, tmp_path: Path) -> None:
    """A crashed rerun into the same --out must not leave the old final record/summary behind.

    Otherwise the evaluator reports the *previous* run's number for the run that crashed.
    """
    work = tmp_path / "same-out"
    env = train_env(work, HPCLAB_CHECKPOINT_DIR="")
    done = run_train(*FAST, "--method", "baseline", name="same-out", env=env)
    assert done.returncode == 0 and done.summary()
    crashed = run_train(*FAST, "--method", "mixup", "--fail-at-step", "35", name="same-out", env=env)
    assert crashed.returncode == 1
    records = crashed.records()
    assert not any(r.get("final") for r in records)
    assert all(r.get("method", "mixup") == "mixup" for r in records)
    assert max(r["step"] for r in records) == 30
    assert not (crashed.out / "metrics.json").exists()


def test_requeued_job_resumes_from_its_own_checkpoints(run_train: RunTrain, tmp_path: Path) -> None:
    """Slurm requeue (preemption, node failure) reruns the same script in the same attempt dir.

    ``SLURM_RESTART_COUNT`` > 0 means "this is a restart": continue from the job's own latest
    checkpoint instead of starting over (or refusing the non-empty checkpoint dir).
    """
    common = [*FAST, "--method", "mixup", "--seed", "4"]
    full = run_train(*common, name="full")
    preempted = run_train(*common, "--fail-at-step", "55", name="job")
    assert preempted.returncode == 1

    requeued = run_train(*common, name="job", env=train_env(tmp_path / "job", SLURM_RESTART_COUNT="1"))
    assert requeued.returncode == 0, requeued.stderr
    assert "resumed from" in requeued.stdout and "ckpt-000000040 at step 40" in requeued.stdout
    assert requeued.final()["metrics"] == full.final()["metrics"]  # exactly one final record
    assert requeued.summary()["metrics"] == full.summary()["metrics"]


def test_requeue_prefers_own_newer_checkpoint_over_the_resume_source(
    run_train: RunTrain, tmp_path: Path
) -> None:
    source = run_train(*FAST, "--seed", "6", name="source")
    assert source.returncode == 0
    env = train_env(tmp_path / "attempt2", HPCLAB_RESUME_FROM=str(source.ckpt / "ckpt-000000020"))
    first = run_train(*FAST, "--seed", "6", "--fail-at-step", "55", name="attempt2", env=env)
    assert first.returncode == 1 and "at step 20" in first.stdout

    env["SLURM_RESTART_COUNT"] = "2"
    requeued = run_train(*FAST, "--seed", "6", name="attempt2", env=env)
    assert requeued.returncode == 0, requeued.stderr
    assert str(requeued.ckpt / "ckpt-000000040") in requeued.stdout  # its own progress, not step 20
    assert requeued.final()["metrics"] == source.final()["metrics"]


def test_resume_from_a_checkpoint_beyond_steps_is_refused(run_train: RunTrain) -> None:
    """Reporting step-60 weights as a 20-step result would silently mislabel the number."""
    base = run_train(*FAST, name="base")
    assert base.returncode == 0
    res = run_train("--steps", "20", "--resume", str(base.ckpt / "ckpt-000000060"), name="short")
    assert res.returncode == 2
    assert "step 60" in res.stderr and "--steps 20" in res.stderr
    assert not (res.out / "metrics.jsonl").exists()


def test_malformed_seed_env_is_a_usage_error(run_train: RunTrain, tmp_path: Path) -> None:
    res = run_train(*FAST, env=train_env(tmp_path / "run", HPCLAB_SEED="one"))
    assert res.returncode == 2
    assert "invalid int value" in res.stderr and "Traceback" not in res.stderr


def test_stop_file_checkpoints_and_exits_like_a_signal(run_train: RunTrain, tmp_path: Path) -> None:
    """The wrapper's HPCLAB_STOP_FILE relays Slurm's warning signal; it is a clean stop, not exit 1."""
    work = tmp_path / "stopfile"
    work.mkdir()
    stop_file = work / "STOP"
    proc = subprocess.Popen(
        [sys.executable, str(TRAIN), "--steps", "400", "--ckpt-every", "1000", "--sleep-per-step", "0.02"],
        cwd=work,
        env=train_env(work, HPCLAB_STOP_FILE=str(stop_file)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    metrics = work / "outputs" / "metrics.jsonl"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (not metrics.exists() or len(metrics.read_text().splitlines()) < 2):
        time.sleep(0.02)
    stop_file.write_text("")
    stdout, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 128 + signal.SIGUSR1, stderr
    assert "received STOP_FILE" in stdout
    assert len(list((work / "checkpoints").iterdir())) == 1


def _oldest_python() -> str | None:
    for name in ("python3.8", "python3.9"):
        found = shutil.which(name) or shutil.which(name, path=str(Path.home() / ".local" / "bin"))
        if found:
            return found
    return None


def test_old_cluster_python_gives_identical_results_and_compatible_checkpoints(run_train: RunTrain) -> None:
    """Cluster python3 is often 3.8/3.9: same numbers, and checkpoints resume across versions."""
    old = _oldest_python()
    if old is None:
        pytest.skip("no python3.8/python3.9 interpreter available")
    common = [*FAST, "--method", "mixup", "--seed", "2"]
    current = run_train(*common, name="current")
    legacy = run_train(*common, name="legacy", python=old)
    assert legacy.returncode == 0, legacy.stderr
    assert strip_times(legacy.records()) == strip_times(current.records())

    crashed = run_train(*common, "--fail-at-step", "45", name="crashed", python=old)
    assert crashed.returncode == 1
    resumed = run_train(*common, "--resume", str(crashed.ckpt), name="resumed")  # newer python resumes
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.final()["metrics"] == current.final()["metrics"]


def test_sigterm_checkpoints_and_resume_completes_identically(run_train: RunTrain, tmp_path: Path) -> None:
    common = ["--steps", "400", "--eval-every", "100", "--ckpt-every", "1000", "--seed", "5"]
    full = run_train(*common, name="full")
    assert full.returncode == 0

    work = tmp_path / "stopped"
    work.mkdir()
    proc = subprocess.Popen(
        [sys.executable, str(TRAIN), *common, "--sleep-per-step", "0.02"],
        cwd=work,
        env=train_env(work),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    metrics = work / "outputs" / "metrics.jsonl"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (not metrics.exists() or len(metrics.read_text().splitlines()) < 2):
        time.sleep(0.02)
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=20)

    assert proc.returncode == 128 + signal.SIGTERM, stderr
    assert "received SIGTERM" in stdout
    saved = sorted((work / "checkpoints").iterdir())
    assert len(saved) == 1
    stopped_at = json.loads((saved[0] / "hpclab-ckpt.json").read_text())["step"]
    assert 20 <= stopped_at < 400

    resumed = run_train(*common, "--resume", str(saved[0]), name="resumed")
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.final()["metrics"] == full.final()["metrics"]


# ---------------------------------------------------------------------------
# Failure injection: exit codes and messages hpclab's diagnosis relies on
# ---------------------------------------------------------------------------


def test_fail_at_step_raises_with_traceback(run_train: RunTrain) -> None:
    res = run_train(*FAST, "--fail-at-step", "25")
    assert res.returncode == 1
    assert "Traceback (most recent call last)" in res.stderr
    assert "RuntimeError: injected failure at step 25" in res.stderr
    assert max(r["step"] for r in res.records()) == 20


def test_oom_at_step_prints_realistic_cuda_oom(run_train: RunTrain) -> None:
    res = run_train(*FAST, "--oom-at-step", "30")
    assert res.returncode == 1
    lines = res.stderr.strip().splitlines()
    assert lines[0] == "Traceback (most recent call last):"
    assert lines[-1].startswith("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.")
    assert "GPU 0 has a total capacity of 79.19 GiB" in lines[-1]
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in lines[-1]
    assert max(r["step"] for r in res.records()) == 20
    assert not (res.out / "metrics.json").exists()


def test_nan_at_step_logs_nonfinite_and_exits(run_train: RunTrain) -> None:
    res = run_train(*FAST, "--nan-at-step", "33")
    assert res.returncode == 1
    assert "Loss is NaN at step 33" in res.stderr
    records = res.records()  # strict parse: no NaN tokens in the file
    last = records[-1]
    assert last["step"] == 33 and last["split"] == "train"
    assert last["metrics"]["loss"] is None and last["nonfinite"] == {"loss": "nan"}
    assert not any(r.get("final") for r in records)
    assert not (res.out / "metrics.json").exists()


def test_sleep_per_step_slows_the_job(run_train: RunTrain) -> None:
    start = time.monotonic()
    res = run_train("--steps", "10", "--sleep-per-step", "0.05")
    assert res.returncode == 0
    assert time.monotonic() - start >= 0.5


@pytest.mark.parametrize(
    "args",
    [["--steps", "0"], ["--batch-size", "0"], ["--lr", "-1"], ["--lr", "nan"], ["--method", "dropout"]],
)
def test_invalid_arguments_exit_2(run_train: RunTrain, args: list[str]) -> None:
    res = run_train(*args)
    assert res.returncode == 2
    assert "error" in res.stderr


def test_metrics_file_is_strict_json_even_after_failures(run_train: RunTrain) -> None:
    res = run_train(*FAST, "--nan-at-step", "11")
    text = (res.out / "metrics.jsonl").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert len(read_jsonl(res.out / "metrics.jsonl")) == 2
