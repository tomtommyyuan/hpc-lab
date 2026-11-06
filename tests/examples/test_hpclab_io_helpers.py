"""hpclab_io: the vendorable metrics/checkpoint/env helper researchers copy into projects."""

from __future__ import annotations

import json
import math
import os
import signal
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from tiny_classifier_support import read_jsonl

CONFIG = {"lr": 0.1, "batch_size": 32, "method": "mixup", "world_size": 1}
KEYS = ["lr", "batch_size", "method"]


# ---------------------------------------------------------------------------
# MetricsWriter
# ---------------------------------------------------------------------------


def test_metrics_writer_appends_schema_records(hpclab_io: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "deep" / "dir" / "metrics.jsonl"
    with hpclab_io.MetricsWriter(str(path)) as w:
        w.log(10, "train", {"loss": 0.5}, total_steps=100, epoch=0)
        w.log(
            100,
            "val",
            {"acc": 0.9},
            total_steps=100,
            final=True,
            dataset="d",
            seed=1,
            units={"acc": "fraction"},
        )
    with hpclab_io.MetricsWriter(str(path)) as w:  # reopening appends, never truncates
        w.log(5, "test", {"acc": 1})
    first, final, third = read_jsonl(path)
    assert first["schema"] == "hpclab.metrics/v1" and first["step"] == 10 and first["total_steps"] == 100
    assert first["epoch"] == 0 and "final" not in first
    assert final["final"] is True and final["dataset"] == "d" and final["units"] == {"acc": "fraction"}
    assert third["metrics"] == {"acc": 1.0} and isinstance(third["metrics"]["acc"], float)
    assert "total_steps" not in third and "epoch" not in third


@pytest.mark.parametrize(("value", "label"), [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")])
def test_metrics_writer_encodes_nonfinite_as_null(
    hpclab_io: ModuleType, tmp_path: Path, value: float, label: str
) -> None:
    path = tmp_path / "m.jsonl"
    with hpclab_io.MetricsWriter(str(path)) as w:
        record = w.log(1, "train", {"loss": value, "lr": 0.1})
    assert record["metrics"] == {"loss": None, "lr": 0.1}
    assert record["nonfinite"] == {"loss": label}
    assert read_jsonl(path)[0]["nonfinite"] == {"loss": label}


def test_metrics_writer_overwrite_starts_a_new_file(hpclab_io: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    with hpclab_io.MetricsWriter(str(path)) as w:
        w.log(100, "val", {"acc": 0.9}, final=True)
    with hpclab_io.MetricsWriter(str(path), overwrite=True) as w:
        w.log(1, "train", {"loss": 0.7})
    assert [r["step"] for r in read_jsonl(path)] == [1]


def test_atomic_write_json_leaves_no_temp_and_rejects_nan(hpclab_io: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "s" / "summary.json"
    hpclab_io.atomic_write_json(str(target), {"a": 1})
    hpclab_io.atomic_write_json(str(target), {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    with pytest.raises(ValueError):
        hpclab_io.atomic_write_json(str(target), {"a": math.nan})
    assert json.loads(target.read_text()) == {"a": 2}  # the old file survives a failed write
    assert sorted(p.name for p in target.parent.iterdir()) == ["summary.json"]  # no temp left behind


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _write(io: ModuleType, root: Path, step: int, **kw: object) -> Path:
    files = {"model.bin": bytes([step % 256]) * 64, "sub/opt.json": b'{"m": 0.9}'}
    return Path(io.write_checkpoint(str(root), files, step=step, config=CONFIG, semantic_keys=KEYS, **kw))


def test_write_checkpoint_layout_and_manifest(
    hpclab_io: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPCLAB_RUN_ID", "run_x")
    monkeypatch.setenv("HPCLAB_ATTEMPT_ID", "att_y")
    monkeypatch.delenv("HPCLAB_ATTEMPT_DIR", raising=False)
    path = _write(hpclab_io, tmp_path, 500, epoch=2, framework={"name": "pytorch", "version": "2.8.0"})
    assert path == tmp_path / "ckpt-000000500"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt-000000500"]  # no staging leftovers
    manifest = json.loads((path / "hpclab-ckpt.json").read_text())
    assert manifest["schema"] == "hpclab.checkpoint/v1" and manifest["complete"] is True
    assert manifest["step"] == 500 and manifest["epoch"] == 2 and manifest["resumable"] is True
    assert manifest["run_id"] == "run_x" and manifest["attempt_id"] == "att_y"
    assert manifest["run_spec_hash"] is None
    assert manifest["config"] == CONFIG and manifest["semantic_keys"] == KEYS
    assert manifest["framework"] == {"name": "pytorch", "version": "2.8.0"}
    assert {f["path"] for f in manifest["files"]} == {"model.bin", "sub/opt.json"}
    assert all(f["sha256"].startswith("sha256:") and f["size"] > 0 for f in manifest["files"])


def test_write_checkpoint_validates_inputs(hpclab_io: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="semantic_keys"):
        hpclab_io.write_checkpoint(
            str(tmp_path), {"m": b"x"}, step=1, config={"lr": 1}, semantic_keys=["seed"]
        )
    for bad in ["../escape", "/abs", "hpclab-ckpt.json", ""]:
        with pytest.raises(ValueError, match="member name"):
            hpclab_io.write_checkpoint(str(tmp_path), {bad: b"x"}, step=1, config={}, semantic_keys=[])
    assert hpclab_io.latest_checkpoint(str(tmp_path)) is None


def test_rewriting_a_step_replaces_it(hpclab_io: ModuleType, tmp_path: Path) -> None:
    _write(hpclab_io, tmp_path, 7)
    path = Path(
        hpclab_io.write_checkpoint(
            str(tmp_path), {"new.bin": b"n"}, step=7, config=CONFIG, semantic_keys=KEYS
        )
    )
    assert sorted(p.name for p in path.iterdir()) == ["hpclab-ckpt.json", "new.bin"]


def test_rewriting_a_step_keeps_the_old_one_if_installing_the_new_one_fails(
    hpclab_io: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old checkpoint is never deleted before its replacement is in place.

    (rmtree-then-rename could leave a half-deleted dir whose manifest still says complete
    if the job is SIGKILLed mid-delete, e.g. Slurm KillWait while a big checkpoint is removed.)
    """
    old = _write(hpclab_io, tmp_path, 7)
    old_manifest = (old / "hpclab-ckpt.json").read_text()
    real_rename = os.rename

    def failing_rename(src: str, dst: str) -> None:
        if os.path.basename(src).startswith(".staging-"):
            raise OSError(28, "No space left on device")
        real_rename(src, dst)

    monkeypatch.setattr(hpclab_io.os, "rename", failing_rename)
    with pytest.raises(OSError, match="No space"):
        hpclab_io.write_checkpoint(
            str(tmp_path), {"new.bin": b"n"}, step=7, config=CONFIG, semantic_keys=KEYS
        )
    assert (old / "hpclab-ckpt.json").read_text() == old_manifest
    assert hpclab_io.read_checkpoint(str(old))[0]["step"] == 7
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt-000000007"]  # staging cleaned up


def test_keep_last_prunes_oldest_complete(hpclab_io: ModuleType, tmp_path: Path) -> None:
    for step in (10, 20, 30, 40):
        _write(hpclab_io, tmp_path, step, keep_last=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt-000000030", "ckpt-000000040"]


def test_keep_last_never_deletes_the_new_or_newer_checkpoints(hpclab_io: ModuleType, tmp_path: Path) -> None:
    """keep_last prunes *older* progress; it must not delete what it just wrote."""
    _write(hpclab_io, tmp_path, 500)  # e.g. left behind by a longer run
    _write(hpclab_io, tmp_path, 50)
    new = _write(hpclab_io, tmp_path, 100, keep_last=1)
    assert new.is_dir() and hpclab_io.read_checkpoint(str(new))[0]["step"] == 100
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt-000000100", "ckpt-000000500"]


def test_latest_checkpoint_ignores_incomplete_ones(hpclab_io: ModuleType, tmp_path: Path) -> None:
    good = _write(hpclab_io, tmp_path, 100)
    # A newer directory without a manifest: a job killed while writing files.
    (tmp_path / "ckpt-000000300").mkdir()
    (tmp_path / "ckpt-000000300" / "model.bin").write_bytes(b"partial")
    # Newer, manifest present but not complete / unparsable / wrong schema / hidden staging.
    for name, text in [
        ("ckpt-000000400", json.dumps({"schema": "hpclab.checkpoint/v1", "step": 400, "complete": False})),
        ("ckpt-000000500", "{not json"),
        ("ckpt-000000600", json.dumps({"schema": "other/v1", "step": 600, "complete": True})),
        (
            ".staging-000000700-1",
            json.dumps({"schema": "hpclab.checkpoint/v1", "step": 700, "complete": True}),
        ),
    ]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "hpclab-ckpt.json").write_text(text)
    assert hpclab_io.latest_checkpoint(str(tmp_path)) == str(good)
    newer = _write(hpclab_io, tmp_path, 200)
    assert hpclab_io.latest_checkpoint(str(tmp_path)) == str(newer)


def test_latest_checkpoint_accepts_a_checkpoint_dir_and_missing_dirs(
    hpclab_io: ModuleType, tmp_path: Path
) -> None:
    path = _write(hpclab_io, tmp_path, 3)
    assert hpclab_io.latest_checkpoint(str(path)) == str(path)
    assert hpclab_io.latest_checkpoint(str(tmp_path / "nope")) is None
    assert hpclab_io.latest_checkpoint("") is None


def test_read_checkpoint_verifies_files(hpclab_io: ModuleType, tmp_path: Path) -> None:
    path = _write(hpclab_io, tmp_path, 9)
    manifest, blobs = hpclab_io.read_checkpoint(str(path))
    assert manifest["step"] == 9 and blobs["sub/opt.json"] == b'{"m": 0.9}'

    (path / "model.bin").write_bytes(b"\x00" * 64)  # same size, different content
    with pytest.raises(hpclab_io.CheckpointError, match="digest mismatch"):
        hpclab_io.read_checkpoint(str(path))
    _, unverified = hpclab_io.read_checkpoint(str(path), verify=False)
    assert unverified["model.bin"] == b"\x00" * 64

    (path / "model.bin").write_bytes(b"short")
    with pytest.raises(hpclab_io.CheckpointError, match="size"):
        hpclab_io.read_checkpoint(str(path))
    (path / "model.bin").unlink()
    with pytest.raises(hpclab_io.CheckpointError, match="unreadable"):
        hpclab_io.read_checkpoint(str(path))
    with pytest.raises(hpclab_io.CheckpointError, match="not a complete"):
        hpclab_io.read_checkpoint(str(tmp_path))


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "../../etc/passwd", "size": 1, "sha256": "sha256:x"}],
        [{"size": 1}],
        ["model.bin"],
        "model.bin",
    ],
)
def test_read_checkpoint_rejects_malformed_manifests_as_checkpoint_errors(
    hpclab_io: ModuleType, tmp_path: Path, files: object
) -> None:
    """A hand-edited or foreign manifest is an unusable checkpoint (exit 2), not a crash."""
    path = _write(hpclab_io, tmp_path, 3)
    manifest = json.loads((path / "hpclab-ckpt.json").read_text())
    (path / "hpclab-ckpt.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(hpclab_io.CheckpointError):
        hpclab_io.read_checkpoint(str(path))


@pytest.mark.parametrize("step", [None, "300", 3.5, True])
def test_manifest_without_an_integer_step_is_not_a_checkpoint(
    hpclab_io: ModuleType, tmp_path: Path, step: object
) -> None:
    good = _write(hpclab_io, tmp_path, 100)
    bad = tmp_path / "ckpt-000000300"
    bad.mkdir()
    (bad / "hpclab-ckpt.json").write_text(
        json.dumps({"schema": "hpclab.checkpoint/v1", "step": step, "complete": True, "files": []})
    )
    assert hpclab_io.latest_checkpoint(str(tmp_path)) == str(good)
    _write(hpclab_io, tmp_path, 200, keep_last=1)  # pruning must not trip over it either
    assert bad.is_dir()


def test_check_compatible_reports_semantic_differences(hpclab_io: ModuleType, tmp_path: Path) -> None:
    manifest, _ = hpclab_io.read_checkpoint(str(_write(hpclab_io, tmp_path, 1)))
    assert hpclab_io.check_compatible(manifest, CONFIG) == []
    assert hpclab_io.check_compatible(manifest, {**CONFIG, "world_size": 8}) == []  # not a semantic key
    assert hpclab_io.check_compatible(manifest, {**CONFIG, "lr": 0.2, "method": "baseline"}) == [
        "lr",
        "method",
    ]


# ---------------------------------------------------------------------------
# Environment and graceful stop
# ---------------------------------------------------------------------------


def test_hpclab_env_parses_wrapper_variables(hpclab_io: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "hpclab-plan.json").write_text(json.dumps({"execution_plan": {"run_spec_hash": "sha256:r"}}))
    env = {
        "HPCLAB_RUN_ID": "run_1",
        "HPCLAB_OUTPUT_DIR": "/o",
        "HPCLAB_RESUME_FROM": "",
        "HPCLAB_SEED": "-3",
        "HPCLAB_PARAMS_JSON": '{"method": "mixup"}',
        "HPCLAB_ATTEMPT_DIR": str(tmp_path),
        "SLURM_JOB_ID": "1234",
    }
    ctx = hpclab_io.hpclab_env(env)
    assert ctx["run_id"] == "run_1" and ctx["output_dir"] == "/o" and ctx["job_id"] == "1234"
    assert ctx["resume_from"] is None and ctx["checkpoint_dir"] is None
    assert ctx["seed"] == -3 and ctx["params"] == {"method": "mixup"}
    assert ctx["run_spec_hash"] == "sha256:r" and ctx["managed"] is True


def test_hpclab_env_outside_hpclab_is_all_none(hpclab_io: ModuleType) -> None:
    ctx = hpclab_io.hpclab_env(
        {"HPCLAB_PARAMS_JSON": "not json", "HPCLAB_SEED": "x", "HPCLAB_ATTEMPT_DIR": "/nope"}
    )
    assert ctx["managed"] is False and ctx["seed"] is None and ctx["params"] == {}
    assert ctx["run_spec_hash"] is None


@pytest.mark.parametrize("seed", ["--3", "1-2", "²", " "])
def test_hpclab_env_never_crashes_on_a_malformed_seed(hpclab_io: ModuleType, seed: str) -> None:
    assert hpclab_io.hpclab_env({"HPCLAB_SEED": seed})["seed"] is None


@pytest.fixture
def restore_signals() -> Iterator[None]:
    saved = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


@pytest.mark.usefixtures("restore_signals")
def test_graceful_stop_records_signal(hpclab_io: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HPCLAB_STOP_FILE", raising=False)
    stop = hpclab_io.GracefulStop()
    assert not stop.requested
    os.kill(os.getpid(), signal.SIGUSR1)
    assert stop.requested and stop.signal_name == "SIGUSR1"
    assert stop.exit_code == 128 + signal.SIGUSR1


@pytest.mark.usefixtures("restore_signals")
def test_graceful_stop_honours_stop_file(
    hpclab_io: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop file relays the wrapper's warning signal: exit 128+N like a signal, never 1.

    Exit 1 is this program's "training failure" code; a clean checkpoint-and-stop must not look
    like one, and must not depend on whether the file or the forwarded signal was seen first.
    """
    flag = tmp_path / "STOP"
    monkeypatch.setenv("HPCLAB_STOP_FILE", str(flag))
    stop = hpclab_io.GracefulStop(signals=())
    assert not stop.requested
    flag.write_text("")
    assert stop.requested and stop.signal_name == "STOP_FILE"
    assert stop.exit_code == 128 + signal.SIGUSR1  # unnamed: Slurm's conventional warning signal


@pytest.mark.usefixtures("restore_signals")
@pytest.mark.parametrize(("content", "expected"), [("TERM\n", signal.SIGTERM), ("SIGUSR2", signal.SIGUSR2)])
def test_graceful_stop_file_may_name_the_signal(
    hpclab_io: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str, expected: int
) -> None:
    flag = tmp_path / "STOP"
    flag.write_text(content)
    monkeypatch.setenv("HPCLAB_STOP_FILE", str(flag))
    stop = hpclab_io.GracefulStop(signals=())
    assert stop.requested and stop.exit_code == 128 + expected


@pytest.mark.usefixtures("restore_signals")
def test_graceful_stop_prefers_a_real_signal_over_the_stop_file(
    hpclab_io: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flag = tmp_path / "STOP"
    flag.write_text("")
    monkeypatch.setenv("HPCLAB_STOP_FILE", str(flag))
    stop = hpclab_io.GracefulStop()
    assert stop.requested and stop.signal_name == "STOP_FILE"
    os.kill(os.getpid(), signal.SIGTERM)  # the forwarded signal lands while checkpointing
    assert stop.signal_name == "SIGTERM" and stop.exit_code == 128 + signal.SIGTERM
