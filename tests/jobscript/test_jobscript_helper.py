"""Unit tests for the compute-node helper embedded in job.sbatch (loaded in-process)."""

from __future__ import annotations

import ast
import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

from hpclab.services.jobscript import HELPER_SOURCE


@pytest.fixture(scope="module")
def helper() -> types.ModuleType:
    module = types.ModuleType("hpclab_wrapper_helper")
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), module.__dict__)
    return module


def test_helper_is_python36_syntax() -> None:
    """Clusters may only have an old system python3 before a venv is activated."""
    ast.parse(HELPER_SOURCE, feature_version=(3, 6))


def test_helper_rejects_unknown_commands(helper: types.ModuleType) -> None:
    assert helper.main(["helper.py"]) == 2
    assert helper.main(["helper.py", "nope"]) == 2


def test_helper_failure_is_reported_not_raised(helper: types.ModuleType, tmp_path: Path) -> None:
    assert helper.main(["helper.py", "env", str(tmp_path / "env.json"), str(tmp_path / "missing.json")]) == 1


# ---------------------------------------------------------------------------
# GPU architecture compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "arch_list", "ok"),
    [
        ((9, 0), ["sm_80", "sm_90"], True),
        ((8, 6), ["sm_80", "sm_90"], True),  # same-major cubin compatibility
        ((8, 0), ["sm_86"], False),  # newer minor cubin does not run on an older GPU
        ((10, 0), ["sm_80", "sm_90"], False),  # B200 with a cu124 build
        ((10, 0), ["sm_90", "compute_90"], True),  # PTX can be JIT-compiled forward
        ((10, 0), ["sm_90a"], False),  # arch-specific variants are exact-only
        ((9, 0), ["sm_90a"], True),
        ((12, 0), ["sm_100", "sm_120"], True),
        ((9, 0), [], False),
        ((9, 0), ["garbage"], False),
    ],
)
def test_arch_compatible(
    helper: types.ModuleType, cap: tuple[int, int], arch_list: list[str], ok: bool
) -> None:
    assert helper.arch_compatible(cap, arch_list) is ok


# ---------------------------------------------------------------------------
# guard evaluation
# ---------------------------------------------------------------------------

NVIDIA_OK = {"driver_version": "570.86.15", "cuda_version": "12.8", "gpus": [{"index": 0, "name": "H100"}]}


def torch_info(**kw: Any) -> dict[str, Any]:
    base = {
        "installed": True,
        "version": "2.8.0+cu128",
        "cuda": "12.8",
        "cuda_available": True,
        "arch_list": ["sm_90"],
        "device_count": 1,
        "device_names": ["H100"],
        "device_capabilities": [[9, 0]],
    }
    base.update(kw)
    return base


def test_guard_skipped_without_gpus(helper: types.ModuleType) -> None:
    guard = helper.evaluate_guard({"gpus_per_node": 0, "guard": True}, None, "missing", None, "absent")
    assert guard["status"] == "skipped" and guard["checks"] == [] and guard["failed"] is None


def test_guard_pass(helper: types.ModuleType) -> None:
    guard = helper.evaluate_guard({"gpus_per_node": 1, "guard": True}, NVIDIA_OK, "ok", torch_info(), "ok")
    assert guard["status"] == "pass" and guard["enabled"] is True
    assert [c["check"] for c in guard["checks"]] == ["nvidia_smi", "torch_cuda_available", "torch_gpu_arch"]


def test_guard_reports_the_first_failure_in_check_order(helper: types.ModuleType) -> None:
    guard = helper.evaluate_guard(
        {"gpus_per_node": 1, "guard": True}, None, "missing", torch_info(cuda_available=False), "ok"
    )
    assert guard["status"] == "fail" and guard["failed"] == "nvidia_smi"


def test_guard_timeouts_are_unchecked_never_failures(helper: types.ModuleType) -> None:
    guard = helper.evaluate_guard({"gpus_per_node": 2, "guard": True}, None, "timeout", None, "timeout")
    assert guard["status"] == "unchecked" and guard["failed"] is None


def test_cpu_only_torch_build_is_explained(helper: types.ModuleType) -> None:
    info = torch_info(version="2.8.0+cpu", cuda=None, cuda_available=False)
    guard = helper.evaluate_guard({"gpus_per_node": 1, "guard": True}, NVIDIA_OK, "ok", info, "ok")
    assert guard["failed"] == "torch_cuda_available" and "CPU-only build" in guard["message"]


def test_cuda_hint_only_when_torch_is_newer_than_the_driver(helper: types.ModuleType) -> None:
    newer = helper.cuda_unavailable_message(torch_info(cuda="13.0", cuda_available=False), NVIDIA_OK)
    assert "whl/cu128" in newer
    older = helper.cuda_unavailable_message(torch_info(cuda="12.4", cuda_available=False), NVIDIA_OK)
    assert "whl/" not in older and "driver supports CUDA 12.8" in older


# ---------------------------------------------------------------------------
# status / packages / outputs / resume / linkfarm
# ---------------------------------------------------------------------------


def test_status_command_types_fields(helper: types.ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    args = ["state=killed", "exit_code=143", "signal=TERM", "start_time=100", "end_time=", "error="]
    args += ["signals_received=USR1,TERM", "stop_requested=1", "restart_count=2", "job_id=77"]
    assert helper.main(["helper.py", "status", str(path), *args]) == 0
    doc = json.loads(path.read_text())
    assert doc["schema"] == "hpclab.status/v1"
    assert doc["exit_code"] == 143 and doc["start_time"] == 100 and doc["end_time"] is None
    assert doc["signals_received"] == ["USR1", "TERM"] and doc["stop_requested"] is True
    assert doc["error"] is None and doc["job_id"] == "77" and doc["restart_count"] == 2
    assert not list(tmp_path.glob(".*.tmp"))  # atomic write leaves no temp file


def test_normalize_packages_is_sorted_and_deduplicated(helper: types.ModuleType) -> None:
    assert helper.normalize_packages("b==2\n# comment\na==1\n\nb==2\n") == "a==1\nb==2\n"
    assert helper.normalize_packages("") == ""


def write_config(tmp_path: Path, outputs: list[dict[str, Any]]) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"attempt_id": "att_1", "code_dir": str(tmp_path / "work"), "outputs": outputs})
    )
    return cfg


def test_outputs_quick_pass_is_pending_then_full_pass_hashes(
    helper: types.ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "m.json").write_text("{}")
    cfg = write_config(tmp_path, [{"name": "m", "path": str(tmp_path / "out" / "m.json"), "kind": "file"}])
    target = tmp_path / "outputs.json"

    helper.main(["helper.py", "outputs", str(target), str(cfg), "quick"])
    quick = json.loads(target.read_text())
    assert quick["hash_state"] == "pending" and quick["files"][0]["sha256"] is None
    assert quick["files"][0]["size"] == 2

    helper.main(["helper.py", "outputs", str(target), str(cfg), "full"])
    full = json.loads(target.read_text())
    assert full["hash_state"] == "complete" and full["files"][0]["sha256"].startswith("sha256:")


def test_outputs_bounds_members_and_large_files(
    helper: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "many"
    d.mkdir()
    for i in range(5):
        (d / f"f{i}.txt").write_text("x" * (i + 1))
    (tmp_path / "big.bin").write_bytes(b"\x00" * 64)
    monkeypatch.setitem(helper.__dict__, "MAX_MEMBERS", 3)
    monkeypatch.setitem(helper.__dict__, "MAX_HASH_BYTES", 32)
    cfg = write_config(
        tmp_path,
        [
            {"name": "many", "path": str(d), "kind": "dir"},
            {"name": "big", "path": str(tmp_path / "big.bin"), "kind": "file"},
            {"name": "wrongkind", "path": str(d), "kind": "file"},
        ],
    )
    target = tmp_path / "outputs.json"
    helper.main(["helper.py", "outputs", str(target), str(cfg), "full"])
    doc = json.loads(target.read_text())
    files = {f["name"]: f for f in doc["files"]}

    many = files["many"]
    assert len(many["members"]) == 3 and many["member_count"] == 5 and many["truncated"] is True
    assert many["size"] == 15 and many["sha256"] is None
    assert files["big"]["sha256"] is None and files["big"]["skipped"] == "too_large"
    assert "error" in files["wrongkind"]
    assert doc["hash_state"] == "partial"


def test_outputs_relative_and_job_id_paths(
    helper: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "res-99.txt").write_text("r")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    cfg = write_config(
        tmp_path, [{"name": "r", "path": "res-{job_id}.txt", "kind": "file", "required": True}]
    )
    target = tmp_path / "outputs.json"
    helper.main(["helper.py", "outputs", str(target), str(cfg), "full"])
    entry = json.loads(target.read_text())["files"][0]
    assert entry["exists"] is True and entry["path"] == str(tmp_path / "work" / "res-99.txt")


def make_ckpt(
    root: Path, name: str, *, step: int, complete: bool = True, size: int = 3, actual: int = 3
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"\x00" * actual)
    manifest = {
        "schema": "hpclab.checkpoint/v1",
        "step": step,
        "complete": complete,
        "resumable": True,
        "files": [{"path": "model.pt", "size": size}],
    }
    (d / "hpclab-ckpt.json").write_text(json.dumps(manifest))
    return d


def test_resume_picks_the_latest_complete_valid_checkpoint(
    helper: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "checkpoints"
    make_ckpt(root, "step-10", step=10)
    best = make_ckpt(root, "step-20", step=20)
    make_ckpt(root, "step-30-partial", step=30, complete=False)
    make_ckpt(root, "step-40-truncated", step=40, size=3, actual=1)
    (root / "junk").mkdir()
    assert helper.main(["helper.py", "resume", str(root)]) == 0
    assert capsys.readouterr().out.strip() == str(best)


def test_resume_with_no_checkpoints_prints_nothing(
    helper: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert helper.main(["helper.py", "resume", str(tmp_path / "absent")]) == 0
    assert capsys.readouterr().out == ""


def test_linkfarm_hardlinks_files_and_drops_snapshot_markers(
    helper: types.ModuleType, tmp_path: Path
) -> None:
    src = tmp_path / "snap"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("a")
    (src / ".hpclab-complete").write_text("m")
    (src / ".hpclab-manifest.json").write_text("{}")
    os.chmod(src / "pkg" / "a.py", 0o444)
    dst = tmp_path / "work"
    assert helper.main(["helper.py", "linkfarm", str(src), str(dst)]) == 0
    assert (dst / "pkg" / "a.py").stat().st_ino == (src / "pkg" / "a.py").stat().st_ino
    assert not (dst / ".hpclab-complete").exists() and not (dst / ".hpclab-manifest.json").exists()
    assert os.access(dst / "pkg", os.W_OK)


def test_secretish_environment_variables_are_not_recorded(
    helper: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    recorded = helper.interesting_env()
    assert recorded["NCCL_DEBUG"] == "INFO" and recorded["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "WANDB_API_KEY" not in recorded and "HF_TOKEN" not in recorded
