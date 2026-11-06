"""Diagnosis: realistic log fixtures, precedence rules, resource hints, redaction, bounds."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ops_support import LOG_DIR, read_log

from hpclab.domain.models import ResourceSpec
from hpclab.domain.states import AppState, SchedulerState
from hpclab.services.diagnosis import (
    CODE_CATEGORY,
    DIAGNOSIS_CODES,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_LINES,
    STRATEGIES,
    AttemptFacts,
    Diagnosis,
    DiagnosisItem,
    diagnose,
    normalize_error_line,
    redact_secrets,
    resource_patch_from_hint,
)

GIB = 1024**3
FAILED = {"scheduler_state": "FAILED", "exit_code": 1}


def run(stderr: str = "", stdout: str = "", **facts: Any) -> Diagnosis:
    return diagnose(AttemptFacts(**facts), stdout, stderr)


def all_fixture_texts() -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(LOG_DIR.glob("*.log")):
        text = path.read_text(encoding="utf-8")
        out.append(
            (path.name, "" if ".stdout." in path.name else text, text if ".stdout." in path.name else "")
        )
    return out


# ---------------------------------------------------------------------------
# Every fixture maps to the right primary cause and first strategy
# ---------------------------------------------------------------------------

FIXTURE_CASES: list[tuple[str, dict[str, Any], str, str]] = [
    # (stderr fixture, facts, primary code, first strategy)
    ("cuda_oom_torchrun.stderr.log", {**FAILED, "gpus": 4}, "cuda_oom", "derive_run"),
    ("cuda_oom_shared_gpu.stderr.log", FAILED, "cuda_oom", "derive_run"),
    ("nccl_watchdog.stderr.log", FAILED, "nccl_error", "fix_code"),
    ("ddp_rank_keyerror.stderr.log", FAILED, "python_exception", "fix_code"),
    ("slurm_oom_kill.stderr.log", {"scheduler_state": "OUT_OF_MEMORY"}, "host_oom", "adjust_resources"),
    ("slurm_oom_kill.stderr.log", FAILED, "host_oom", "adjust_resources"),
    ("dataloader_shm.stderr.log", FAILED, "host_oom", "adjust_resources"),
    ("time_limit.stderr.log", {"scheduler_state": "TIMEOUT"}, "timeout", "adjust_resources"),
    ("preemption.stderr.log", {"scheduler_state": "PREEMPTED"}, "preempted", "retry"),
    ("node_failure.stderr.log", {"scheduler_state": "NODE_FAIL"}, "node_fail", "retry"),
    ("stepd_unkillable.stderr.log", FAILED, "node_fail", "retry"),
    ("driver_too_old.stderr.log", FAILED, "cuda_driver_mismatch", "fix_environment"),
    ("sm100_incompatible.stderr.log", FAILED, "cuda_driver_mismatch", "fix_environment"),
    ("module_not_found.stderr.log", FAILED, "import_error", "fix_environment"),
    ("disk_quota.stderr.log", FAILED, "disk_quota", "fix_environment"),
    ("torch_save_disk_full.stderr.log", FAILED, "disk_quota", "fix_environment"),
    ("checkpoint_corrupt.stderr.log", FAILED, "checkpoint_corrupt", "resume"),
    ("weights_only.stderr.log", FAILED, "weights_only_load", "fix_code"),
    ("hf_offline.stderr.log", FAILED, "network_unreachable", "fix_environment"),
    ("port_in_use.stderr.log", FAILED, "port_in_use", "fix_code"),
    ("segfault.stderr.log", {"scheduler_state": "FAILED", "exit_code": 139}, "segfault", "fix_environment"),
    ("device_assert.stderr.log", FAILED, "cuda_device_assert", "fix_code"),
    ("stale_nfs.stderr.log", FAILED, "nfs_stale", "retry"),
    (
        "command_not_found.stderr.log",
        {"scheduler_state": "FAILED", "exit_code": 127},
        "command_not_found",
        "fix_environment",
    ),
    (
        "venv_missing.stderr.log",
        {"scheduler_state": "FAILED", "exit_code": 127},
        "command_not_found",
        "fix_environment",
    ),
    ("state_dict_mismatch.stderr.log", FAILED, "state_dict_mismatch", "fix_code"),
    (
        "guard_sm_arch.stderr.log",
        {"scheduler_state": "FAILED", "exit_code": 3},
        "env_guard_failed",
        "fix_environment",
    ),
    ("file_not_found_relative.stderr.log", FAILED, "file_not_found", "fix_code"),
    ("read_only_snapshot.stderr.log", FAILED, "permission_denied", "fix_code"),
    ("secrets.stderr.log", FAILED, "python_exception", "fix_code"),
]


@pytest.mark.parametrize(("fixture", "facts", "code", "strategy"), FIXTURE_CASES)
def test_fixture_primary_cause(fixture: str, facts: dict[str, Any], code: str, strategy: str) -> None:
    d = run(read_log(fixture), **facts)
    assert d.primary_code == code, [(i.code, i.confidence) for i in d.items]
    assert d.items[0].suggested_strategies[0] == strategy
    assert d.items[0].evidence, "every diagnosis must cite evidence"
    assert d.summary.startswith(d.items[0].title)


@pytest.mark.parametrize(("name", "stderr", "stdout"), all_fixture_texts())
def test_every_fixture_respects_output_contract(name: str, stderr: str, stdout: str) -> None:
    d = diagnose(AttemptFacts(scheduler_state="FAILED", exit_code=1, node_list="node07"), stdout, stderr)
    assert d.items, name
    for item in d.items:
        assert item.code in DIAGNOSIS_CODES
        assert item.confidence in ("high", "medium", "low")
        assert item.suggested_strategies and set(item.suggested_strategies) <= STRATEGIES
        assert len(item.evidence) <= MAX_EVIDENCE_LINES
        assert all(len(e) <= MAX_EVIDENCE_CHARS for e in item.evidence)
        assert all(isinstance(v, str) for v in item.resource_hint.values())
    assert len({i.code for i in d.items}) == len(d.items), "one item per code"
    # lossless persistence (attempts.diagnosis_json)
    assert Diagnosis.model_validate_json(d.model_dump_json()) == d


def test_taxonomy_tables_are_consistent() -> None:
    assert set(CODE_CATEGORY) == DIAGNOSIS_CODES
    required = {
        "cuda_oom", "host_oom", "timeout", "node_fail", "preempted", "cancelled", "nan_loss", "nccl_error",
        "import_error", "cuda_driver_mismatch", "disk_quota", "file_not_found", "permission_denied", "segfault",
        "python_exception", "nfs_stale", "port_in_use", "checkpoint_corrupt", "killed_signal",
        "success_no_outputs", "unknown_failure",
    }  # fmt: skip
    assert required <= DIAGNOSIS_CODES


# ---------------------------------------------------------------------------
# Precedence: scheduler truth first, specific beats generic
# ---------------------------------------------------------------------------


def test_scheduler_oom_beats_generic_traceback(log: Callable[[str], str]) -> None:
    d = run(log("ddp_rank_keyerror.stderr.log"), scheduler_state="OUT_OF_MEMORY", mem_request_bytes=64 * GIB)
    assert d.codes()[0] == "host_oom"
    assert "python_exception" in d.codes()  # still reported, just not primary


def test_timeout_with_sigterm_traceback_is_timeout_only(log: Callable[[str], str]) -> None:
    d = run(log("time_limit.stderr.log"), log("time_limit.stdout.log"), scheduler_state="TIMEOUT")
    assert d.codes() == ["timeout"]  # SignalException is a consequence, not a cause
    assert d.items[0].confidence == "high"
    assert "DUE TO TIME LIMIT" in " ".join(d.items[0].evidence)


@pytest.mark.parametrize(
    ("fixture", "state", "code"),
    [
        ("time_limit.stderr.log", "CANCELLED", "timeout"),
        ("preemption.stderr.log", "CANCELLED", "preempted"),
        ("node_failure.stderr.log", "CANCELLED", "node_fail"),
        ("time_limit.stderr.log", "UNKNOWN", "timeout"),
    ],
)
def test_slurmstepd_markers_override_cancelled(fixture: str, state: str, code: str) -> None:
    d = run(read_log(fixture), scheduler_state=state)
    assert d.primary_code == code


def test_marker_on_stale_running_state_is_medium(log: Callable[[str], str]) -> None:
    d = run(log("time_limit.stderr.log"), scheduler_state="RUNNING")
    assert d.primary_code == "timeout"
    assert d.items[0].confidence == "medium"


def test_plain_cancel_is_intentional_by_default() -> None:
    d = run(
        "slurmstepd: error: *** JOB 77 ON node3 CANCELLED AT 2026-09-27T10:00:00 ***\n",
        scheduler_state="CANCELLED",
        raw_state="CANCELLED by 51234",
    )
    item = d.items[0]
    assert item.code == "cancelled"
    assert item.suggested_strategies[0] == "none"
    assert "uid 51234" in item.explanation


def test_cancelled_while_pending_is_called_out() -> None:
    d = run(scheduler_state="CANCELLED", elapsed_s=0)
    assert "before it started" in d.items[0].explanation


def test_crashed_rank_beats_nccl_cascade(log: Callable[[str], str]) -> None:
    d = run(log("ddp_rank_keyerror.stderr.log"), **FAILED)
    assert d.codes() == ["python_exception", "nccl_error"]
    assert d.items[0].resource_hint["exception"] == "KeyError"
    assert "data.py:57" in d.items[0].resource_hint["location"]
    assert d.last_error_line == "KeyError: 'img_000918'"


def test_launcher_wrapper_exception_is_never_the_cause(log: Callable[[str], str]) -> None:
    d = run(log("cuda_oom_torchrun.stderr.log"), **FAILED)
    assert d.codes() == ["cuda_oom"]
    assert d.last_error_line is not None and d.last_error_line.startswith("torch.OutOfMemoryError")


def test_traceback_source_lines_are_not_errors() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/w/train.py", line 3, in <module>\n'
        '    raise FileNotFoundError("CUDA out of memory in a string")\n'
        "        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        "ValueError: bad config value\n"
    )
    d = run(stderr, **FAILED)
    assert d.codes() == ["python_exception"]
    assert d.items[0].resource_hint["exception"] == "ValueError"


def test_generic_error_line_suppresses_redundant_python_exception(log: Callable[[str], str]) -> None:
    d = run(log("file_not_found_relative.stderr.log"), **FAILED)
    assert d.codes() == ["file_not_found"]


def test_timeout_after_fatal_error_prefers_fixing_the_error(log: Callable[[str], str]) -> None:
    d = run(
        log("cuda_oom_shared_gpu.stderr.log"), scheduler_state="TIMEOUT", elapsed_s=3600, time_limit_s=3600
    )
    timeout = d.items[0]
    assert timeout.code == "timeout"
    assert timeout.confidence == "medium"
    assert timeout.suggested_strategies[0] == "fix_code"
    assert "cuda_oom" in timeout.explanation
    assert "cuda_oom" in d.codes()


def test_nccl_abort_at_kill_does_not_downgrade_timeout() -> None:
    stderr = (
        "slurmstepd: error: *** JOB 9 ON n1 CANCELLED AT 2026-09-27T10:00:00 DUE TO TIME LIMIT ***\n"
        "[rank0]:[E927 10:00:01 ProcessGroupNCCL.cpp:1] NCCL communicator was aborted on rank 0.\n"
    )
    d = run(stderr, scheduler_state="TIMEOUT", elapsed_s=3600, time_limit_s=3600)
    assert d.items[0].code == "timeout" and d.items[0].confidence == "high"


def test_host_oom_from_exit_137_and_high_maxrss(log: Callable[[str], str]) -> None:
    killed_only = "\n".join(
        line for line in log("slurm_oom_kill.stderr.log").splitlines() if "oom-kill" not in line
    )
    d = run(
        killed_only,
        scheduler_state="FAILED",
        exit_code=137,
        max_rss_bytes=63 * GIB,
        mem_request_bytes=64 * GIB,
    )
    item = d.items[0]
    assert item.code == "host_oom" and item.confidence == "high"
    assert item.resource_hint["mem"] == "96G"  # max(64*1.5, 63*1.3) = 96


def test_exit_137_without_memory_evidence_is_sigkill() -> None:
    d = run(scheduler_state="FAILED", exit_code=137)
    assert d.primary_code == "killed_signal"
    assert d.items[0].resource_hint["signal"] == "SIGKILL"
    assert d.items[0].title == "Killed by SIGKILL"


@pytest.mark.parametrize(
    ("exit_code", "signal", "code"),
    [(139, None, "segfault"), (0, 11, "segfault"), (134, None, "killed_signal")],
)
def test_signals_from_exit_codes(exit_code: int, signal: int | None, code: str) -> None:
    d = run(scheduler_state="FAILED", exit_code=exit_code, exit_signal=signal)
    assert d.primary_code == code


def test_specific_log_error_beats_abort_signal(log: Callable[[str], str]) -> None:
    d = run(log("nccl_watchdog.stderr.log"), scheduler_state="FAILED", exit_code=134)
    assert d.codes() == ["nccl_error"]


# ---------------------------------------------------------------------------
# Resource hints and strategies
# ---------------------------------------------------------------------------


def test_timeout_hint_is_1_5x_rounded_and_patch_is_a_valid_resource_spec() -> None:
    d = run(scheduler_state="TIMEOUT", elapsed_s=14_412, time_limit_s=14_400, checkpoint_available=True)
    item = d.items[0]
    assert item.suggested_strategies[:2] == ["resume", "adjust_resources"]
    assert item.resource_hint["time"] == "06:05:00"  # ceil(14412 * 1.5) rounded up to 5 min
    assert item.resource_hint["extra_sbatch"] == "--signal=B:USR1@600"
    spec = ResourceSpec(**resource_patch_from_hint(item.resource_hint))
    assert spec.time_seconds == 6 * 3600 + 300
    assert spec.extra_sbatch == ["--signal=B:USR1@600"]


def test_timeout_hint_capped_at_partition_max() -> None:
    d = run(scheduler_state="TIMEOUT", elapsed_s=36_000, time_limit_s=36_000, partition_max_time_s=43_200)
    assert d.items[0].resource_hint["time"] == "12:00:00"
    assert "capped" in d.items[0].explanation


def test_timeout_already_at_partition_max_requires_checkpointing() -> None:
    d = run(
        scheduler_state="TIMEOUT",
        elapsed_s=172_800,
        time_limit_s=172_800,
        partition_max_time_s=172_800,
        checkpoint_available=False,
    )
    item = d.items[0]
    assert "time" not in item.resource_hint
    assert item.suggested_strategies == ["fix_code"]
    assert item.resource_hint["extra_sbatch"].startswith("--signal=B:USR1@")


def test_timeout_uses_training_progress_for_projection() -> None:
    d = run(
        scheduler_state="TIMEOUT",
        elapsed_s=3600,
        time_limit_s=3600,
        progress_fraction=0.25,
        checkpoint_available=True,
    )
    hint = d.items[0].resource_hint
    assert hint["time"] == "04:40:00"  # 3600/0.25*1.15 = 16560 s -> 04:36:00 -> 04:40:00
    assert hint["resume_time"] == "03:55:00"  # remaining 10800 s * 1.25 + 600 s = 14100 s


def test_timeout_without_any_times_still_gives_advice() -> None:
    d = run(scheduler_state="TIMEOUT")
    assert d.items[0].code == "timeout" and "1.5x" in d.items[0].explanation


def test_host_oom_hint_uses_request_when_rss_unknown() -> None:
    d = run(scheduler_state="OUT_OF_MEMORY", mem_request_bytes=32 * GIB)
    assert d.items[0].resource_hint == {"mem": "48G"}


def test_cuda_oom_never_proposes_a_silent_batch_change(log: Callable[[str], str]) -> None:
    item = run(log("cuda_oom_torchrun.stderr.log"), **FAILED, gpus=4).items[0]
    assert item.suggested_strategies == ["derive_run", "fix_code"]
    assert "adjust_resources" not in item.suggested_strategies
    assert "gradient accumulation" in item.explanation
    assert "FSDP" in item.explanation  # multi-GPU sharding keeps the math
    assert item.resource_hint["env"] == "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    assert item.resource_hint["tried_to_allocate"] == "4.00 GiB"


def test_cuda_oom_detects_ranks_sharing_one_gpu(log: Callable[[str], str]) -> None:
    item = run(log("cuda_oom_shared_gpu.stderr.log"), **FAILED).items[0]
    assert "set_device(local_rank)" in item.explanation
    assert "env" not in item.resource_hint  # 312 MiB reserved is not fragmentation


def test_nan_loss_is_never_answered_with_retry(log: Callable[[str], str]) -> None:
    item = run(stdout=log("nan_loss.stdout.log"), **FAILED).items[0]
    assert item.code == "nan_loss" and item.confidence == "high"
    assert item.suggested_strategies == ["fix_code", "derive_run"]
    assert "stayed non-finite" in item.explanation


def test_nan_loss_from_hpclab_io_nonfinite_record() -> None:
    line = '{"schema":"hpclab.metrics/v1","step":7,"split":"train","metrics":{"loss":null},"nonfinite":{"loss":"nan"}}\n'
    assert run(stdout=line, **FAILED).primary_code == "nan_loss"


def test_transient_fp16_overflow_is_low_confidence(log: Callable[[str], str]) -> None:
    item = run(stdout=log("fp16_transient_nan.stdout.log"), scheduler_state="COMPLETED", exit_code=0).items[0]
    assert item.code == "nan_loss" and item.confidence == "low"
    assert item.suggested_strategies[0] == "none"


@pytest.mark.parametrize(
    "text", ["the nanosecond timer", "Nancy's loss function 0.5", "loss_scale: 65536", "loss 0.53"]
)
def test_nan_loss_has_no_false_positives(text: str) -> None:
    assert "nan_loss" not in run(stdout=text + "\n", **FAILED).codes()


def test_preempted_on_preemptible_partition_with_checkpoint(log: Callable[[str], str]) -> None:
    item = run(
        log("preemption.stderr.log"),
        scheduler_state="PREEMPTED",
        partition="hai-lo",
        preemptible=True,
        checkpoint_available=True,
        restart_count=2,
    ).items[0]
    assert item.suggested_strategies[:2] == ["resume", "retry"]
    assert item.resource_hint["avoid_partition"] == "hai-lo"
    assert "non-preemptible" in item.explanation and "2 time(s)" in item.explanation


def test_node_fail_excludes_node_and_escalates_when_repeated(log: Callable[[str], str]) -> None:
    first = run(
        log("node_failure.stderr.log"), scheduler_state="NODE_FAIL", checkpoint_available=False
    ).items[0]
    assert first.suggested_strategies == ["retry", "contact_support"]
    assert resource_patch_from_hint(first.resource_hint) == {"exclude_nodes": ["haic-hgx-2"]}
    again = run(
        log("node_failure.stderr.log"),
        scheduler_state="NODE_FAIL",
        checkpoint_available=False,
        previous_codes=("node_fail",),
    ).items[0]
    assert again.suggested_strategies == ["contact_support", "retry"]
    assert "already failed the same way 1 time(s)" in again.explanation


def test_node_fail_without_any_log_output_points_at_filesystem_client() -> None:
    item = run(scheduler_state="NODE_FAIL", node_list="haic-hgx-2").items[0]
    assert "could not even create the log files" in item.explanation
    assert item.resource_hint["exclude_nodes"] == "haic-hgx-2"


@pytest.mark.parametrize("node_list", ["None assigned", "(null)", "", None])
def test_placeholder_node_lists_are_not_excluded(node_list: str | None) -> None:
    item = run(scheduler_state="NODE_FAIL", node_list=node_list).items[0]
    assert "exclude_nodes" not in item.resource_hint


def test_boot_fail_is_node_fail_variant() -> None:
    item = run(scheduler_state="BOOT_FAIL", node_list="n3").items[0]
    assert (item.code, item.title) == ("node_fail", "Node failed to boot")


def test_driver_mismatch_names_the_right_wheel(log: Callable[[str], str]) -> None:
    item = run(log("driver_too_old.stderr.log"), **FAILED).items[0]
    assert item.resource_hint["driver_cuda"] == "12.8"
    assert item.resource_hint["torch_index_url"] == "https://download.pytorch.org/whl/cu128"
    assert "no_gpu" not in run(log("driver_too_old.stderr.log"), **FAILED).codes()


def test_sm_arch_mismatch_allows_changing_gpu_type(log: Callable[[str], str]) -> None:
    item = run(log("sm100_incompatible.stderr.log"), **FAILED).items[0]
    assert item.resource_hint["gpu_arch"] == "sm_100"
    assert "adjust_resources" in item.suggested_strategies
    assert "sm_90" in item.explanation  # lists what the build supports


def test_guard_failure_reports_check_name(log: Callable[[str], str]) -> None:
    item = run(log("guard_sm_arch.stderr.log"), scheduler_state="FAILED", exit_code=3).items[0]
    assert item.resource_hint["guard_check"] == "sm_arch"
    assert CODE_CATEGORY[item.code] == "environment"


def test_import_error_mentions_snapshot_for_local_modules(log: Callable[[str], str]) -> None:
    item = run(log("module_not_found.stderr.log"), **FAILED).items[0]
    assert item.resource_hint["missing_module"] == "flash_attn"
    assert "include_untracked" in item.explanation


@pytest.mark.parametrize(
    ("line", "needle"),
    [
        (
            "ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama'",
            "version",
        ),
        (
            "ImportError: libcudart.so.12: cannot open shared object file: No such file or directory",
            "libcudart.so.12",
        ),
        (
            "ImportError: /x/flash_attn_2_cuda.so: undefined symbol: _ZN3c104cuda9SetDeviceEi",
            "compiled extension",
        ),
        ("ImportError: /lib64/libc.so.6: version `GLIBC_2.34' not found", "GLIBC_2.34"),
    ],
)
def test_import_error_variants(line: str, needle: str) -> None:
    d = run(line + "\n", **FAILED)
    assert d.primary_code == "import_error"
    assert needle in d.items[0].explanation
    assert "file_not_found" not in d.codes()


def test_command_not_found_for_module_and_missing_venv(log: Callable[[str], str]) -> None:
    module = run(log("command_not_found.stderr.log"), scheduler_state="FAILED", exit_code=127).items[0]
    assert module.resource_hint["command"] == "module"
    assert "modules.sh" in module.explanation
    venv = run(log("venv_missing.stderr.log"), scheduler_state="FAILED", exit_code=127)
    assert ".venv/bin/activate" in venv.items[0].explanation
    assert venv.item("file_not_found") is not None


def test_disk_quota_names_the_path(log: Callable[[str], str]) -> None:
    item = run(log("disk_quota.stderr.log"), **FAILED).items[0]
    assert item.resource_hint["path"].startswith("/hai/users/alice/.cache/huggingface")
    assert "HF_HOME" in item.explanation


def test_shm_bus_error_is_not_disk_quota() -> None:
    line = "OSError: [Errno 28] No space left on device: '/dev/shm/torch_1234_56789'\n"
    codes = run(line, **FAILED).codes()
    assert "disk_quota" not in codes and "host_oom" in codes


def test_relative_path_explains_snapshot_layout(log: Callable[[str], str]) -> None:
    item = run(log("file_not_found_relative.stderr.log"), **FAILED).items[0]
    assert item.resource_hint["path"] == "data/train.csv"
    assert "excluded from snapshots" in item.explanation


def test_read_only_snapshot_points_at_output_dir(log: Callable[[str], str]) -> None:
    assert "$HPCLAB_OUTPUT_DIR" in run(log("read_only_snapshot.stderr.log"), **FAILED).items[0].explanation


def test_success_without_outputs() -> None:
    d = run(
        scheduler_state="COMPLETED", exit_code=0, app_state="exited_ok", missing_outputs=("metrics", "model")
    )
    item = d.items[0]
    assert item.code == "success_no_outputs" and item.confidence == "high"
    assert item.resource_hint["missing_outputs"] == "metrics,model"


def test_completed_but_app_not_exited_ok() -> None:
    d = run(scheduler_state="COMPLETED", exit_code=0, app_state="killed")
    assert d.primary_code == "success_no_outputs"
    assert "app_state=killed" in d.items[0].explanation


def test_clean_success_has_no_items() -> None:
    d = run(
        stdout="done\n", scheduler_state="COMPLETED", exit_code=0, app_state="exited_ok", result_state="valid"
    )
    assert d.items == [] and d.primary_code is None
    assert d.summary.startswith("Completed")


def test_logged_error_in_successful_job_is_low_confidence(log: Callable[[str], str]) -> None:
    d = run(log("hf_offline.stderr.log"), scheduler_state="COMPLETED", exit_code=0)
    item = d.items[0]
    assert item.confidence == "low"
    assert item.explanation.startswith("The job completed; this was logged but did not stop it.")


def test_running_job_errors_are_capped_at_medium(log: Callable[[str], str]) -> None:
    d = run(stdout=log("nan_loss.stdout.log"), scheduler_state="RUNNING")
    assert d.primary_code == "nan_loss" and d.items[0].confidence == "medium"
    assert d.items[0].explanation.startswith("The job is still running.")


def test_pending_job_is_not_a_failure() -> None:
    d = run(scheduler_state="PENDING", reason="Priority")
    assert d.items == [] and "Priority" in d.summary


def test_unknown_failure_quotes_the_log_tail() -> None:
    d = run("warming up\nsomething odd happened\nexiting\n", **FAILED)
    item = d.items[0]
    assert item.code == "unknown_failure" and item.confidence == "low"
    assert item.evidence[-1] == "stderr: exiting"


def test_unknown_failure_with_empty_logs() -> None:
    d = run(**FAILED)
    assert d.primary_code == "unknown_failure" and "empty" in d.items[0].explanation


def test_transient_network_exception_allows_retry() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/w/upload.py", line 3, in <module>\n'
        "    push()\n"
        "ConnectionResetError: [Errno 104] Connection reset by peer\n"
    )
    item = run(stderr, **FAILED).items[0]
    assert item.suggested_strategies[0] == "retry"


# ---------------------------------------------------------------------------
# Log normalisation, redaction, bounds, performance
# ---------------------------------------------------------------------------


def test_ansi_codes_rank_prefixes_and_progress_bars_are_normalised() -> None:
    stderr = (
        "\x1b[31m[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.\x1b[0m\n"
        "3: slurmstepd: error: *** JOB 5 ON n1 CANCELLED AT 2026-09-27T10:00:00 DUE TO TIME LIMIT ***\r\n"
    )
    d = run(stderr, scheduler_state="TIMEOUT")
    assert d.codes() == ["timeout", "cuda_oom"]
    assert all("\x1b" not in e for i in d.items for e in i.evidence)


def test_secrets_never_leave_the_function(log: Callable[[str], str]) -> None:
    dumped = run(log("secrets.stderr.log"), **FAILED).model_dump_json()
    assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789" not in dumped
    assert "ghp_1234567890abcdefghijABCDEFGHIJ123456" not in dumped
    assert "[REDACTED]" in dumped


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123", "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123"),
        (
            "export WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567",
            "0123456789abcdef0123456789abcdef01234567",
        ),
        ("git clone https://alice:s3cr3tpass@github.com/lab/repo.git", "s3cr3tpass"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("openai key sk-proj-abcdefghijklmnopqrstuvwx", "sk-proj-abcdefghijklmnopqrstuvwx"),
        ("aws AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        ("db password=hunter2", "hunter2"),
        ('{"api_key": "abcd1234efgh5678"}', "abcd1234efgh5678"),
        ("xoxb-123456789012-abcdefghij", "xoxb-123456789012-abcdefghij"),
        ("-----BEGIN RSA PRIVATE KEY----- MIIEow", "MIIEow"),
    ],
)
def test_redact_secrets(text: str, secret: str) -> None:
    assert secret not in redact_secrets(text)


@pytest.mark.parametrize(
    "text", ["max_tokens=2048", "num_tokens: 512", "loss 0.53 step 100", "git commit 3f2a9c1d0b"]
)
def test_redaction_keeps_ordinary_text(text: str) -> None:
    assert redact_secrets(text) == text


def test_evidence_is_bounded_even_for_huge_lines() -> None:
    line = "torch.OutOfMemoryError: CUDA out of memory. " + "x" * 5000
    item = run(line, **FAILED).items[0]
    assert all(len(e) <= MAX_EVIDENCE_CHARS for e in item.evidence)
    manual = DiagnosisItem(
        code="cuda_oom", title="t", confidence="high", explanation="e", evidence=["y" * 999] * 9
    )
    assert len(manual.evidence) == MAX_EVIDENCE_LINES and all(
        len(e) <= MAX_EVIDENCE_CHARS for e in manual.evidence
    )


def test_evidence_window_keeps_late_match_visible() -> None:
    line = "x" * 600 + " RuntimeError: CUDA error: out of memory"
    evidence = run(line, **FAILED).items[0].evidence[0]
    assert "out of memory" in evidence and len(evidence) <= MAX_EVIDENCE_CHARS


def test_large_logs_are_bounded_and_fast() -> None:
    noise = "step 100 | loss 0.53 | lr 3e-4 | grad_norm 1.2 | tokens/s 51234\n" * 20_000
    start = time.perf_counter()
    d = run(noise + "ModuleNotFoundError: No module named 'wandb'\n", stdout=noise, **FAILED)
    assert time.perf_counter() - start < 3.0
    assert d.primary_code == "import_error"


def test_last_error_line_normalisation_groups_similar_failures() -> None:
    a = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB on att_01ABC rank 3 at 0x7f3a"
    b = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.50 GiB on att_09XYZ rank 5 at 0x1b2c"
    assert normalize_error_line(a) == normalize_error_line(b)
    assert normalize_error_line(None) == ""
    assert normalize_error_line("KeyError: 'a'") != normalize_error_line("ValueError: 'a'")


# ---------------------------------------------------------------------------
# Facts coercion and hint -> patch conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "state", "app"),
    [
        ({"scheduler_state": "CANCELLED by 1234"}, SchedulerState.CANCELLED, AppState.UNKNOWN),
        ({"scheduler_state": "OUT_OF_ME+"}, SchedulerState.OUT_OF_MEMORY, AppState.UNKNOWN),
        ({"raw_state": "TIMEOUT", "app_state": "killed"}, SchedulerState.TIMEOUT, AppState.KILLED),
        (
            {"scheduler_state": "CD", "app_state": "guard_failed"},
            SchedulerState.COMPLETED,
            AppState.GUARD_FAILED,
        ),
        ({"scheduler_state": "CD", "app_state": "not_a_state"}, SchedulerState.COMPLETED, AppState.UNKNOWN),
        (
            {"scheduler_state": None, "app_state": None, "restart_count": None},
            SchedulerState.UNKNOWN,
            AppState.UNKNOWN,
        ),
    ],
)
def test_attempt_facts_accept_ledger_strings(
    kwargs: dict[str, Any], state: SchedulerState, app: AppState
) -> None:
    facts = AttemptFacts(**kwargs)
    assert facts.scheduler_state is state and facts.app_state is app


def test_attempt_facts_round_trip_and_reject_bad_fraction() -> None:
    facts = AttemptFacts(scheduler_state="TIMEOUT", elapsed_s="3600", previous_codes=["timeout"])  # type: ignore[arg-type]
    assert facts.elapsed_s == 3600 and facts.previous_codes == ("timeout",)
    assert AttemptFacts.model_validate(json.loads(facts.model_dump_json())) == facts
    with pytest.raises(ValueError):
        AttemptFacts(progress_fraction=1.5)


def test_resource_patch_from_hint() -> None:
    hint = {
        "time": "06:00:00",
        "mem": "96G",
        "gpus": "1",
        "exclude_nodes": "haic-hgx-[2,5],node7",
        "extra_sbatch": "--signal=B:USR1@600",
        "torch_index_url": "https://download.pytorch.org/whl/cu128",
    }
    patch = resource_patch_from_hint(hint)
    assert patch == {
        "time": "06:00:00",
        "mem": "96G",
        "gpus": 1,
        "exclude_nodes": ["haic-hgx-[2,5]", "node7"],
        "extra_sbatch": ["--signal=B:USR1@600"],
    }
    ResourceSpec(**patch)  # every resource key is a valid ResourceSpec value
    assert resource_patch_from_hint({"gpus": "many", "time": ""}) == {}


def test_fixture_directory_is_complete() -> None:
    names = {name for name, _, _ in all_fixture_texts()}
    referenced = {fixture for fixture, *_ in FIXTURE_CASES}
    assert referenced <= names
    assert Path(LOG_DIR).is_dir()


def test_deadline_is_a_timeout_variant() -> None:
    item = run(scheduler_state="DEADLINE", elapsed_s=100, time_limit_s=3600).items[0]
    assert item.code == "timeout" and "--deadline" in item.explanation


@pytest.mark.parametrize("state", ["SUSPENDED", "REQUEUED", "CONFIGURING", "COMPLETING"])
def test_active_states_without_log_signals_have_no_items(state: str) -> None:
    d = run("loading data\n", scheduler_state=state)
    assert d.items == [] and d.summary.startswith("Running")


def test_oom_kill_inside_completed_job_is_reported_not_primary_failure(log: Callable[[str], str]) -> None:
    d = run(log("slurm_oom_kill.stderr.log"), scheduler_state="COMPLETED", exit_code=0)
    item = d.items[0]
    assert item.code == "host_oom" and item.confidence == "medium"


def test_sigterm_without_scheduler_cause() -> None:
    item = run(scheduler_state="FAILED", exit_code=143).items[0]
    assert (item.code, item.resource_hint["signal"], item.suggested_strategies) == (
        "killed_signal",
        "SIGTERM",
        ["retry"],
    )


# ---------------------------------------------------------------------------
# Regressions found in review (real Slurm/wrapper behaviour)
# ---------------------------------------------------------------------------

TRACEBACK_VALUE_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "/w/train.py", line 10, in <module>\n'
    "    main()\n"
    "ValueError: shapes do not match\n"
)


def test_srun_step_abort_marker_does_not_turn_a_failed_job_into_cancelled() -> None:
    # srun --kill-on-bad-exit prints a *plain* STEP cancel marker for the surviving tasks.
    stderr = TRACEBACK_VALUE_ERROR + (
        "srun: error: node1: task 0: Exited with exit code 1\n"
        "srun: Job step aborted: Waiting up to 32 seconds for job step to finish.\n"
        "slurmstepd: error: *** STEP 123.0 ON node1 CANCELLED AT 2026-09-27T10:00:00 ***\n"
    )
    d = run(stderr, **FAILED)
    assert d.primary_code == "python_exception"
    assert "cancelled" not in d.codes()


def test_plain_step_marker_with_unknown_state_is_not_a_cancel() -> None:
    stderr = (
        TRACEBACK_VALUE_ERROR + "slurmstepd: error: *** STEP 9.0 ON n1 CANCELLED AT 2026-09-27T10:00:00 ***\n"
    )
    assert run(stderr, exit_code=1).primary_code == "python_exception"


def test_plain_job_marker_with_stale_running_state_is_still_a_cancel() -> None:
    d = run(
        "slurmstepd: error: *** JOB 9 ON n1 CANCELLED AT 2026-09-27T10:00:00 ***\n", scheduler_state="RUNNING"
    )
    assert d.primary_code == "cancelled" and d.items[0].confidence == "medium"


WARNING_SIGNAL_LOG = (
    "[hpclab] received SIGUSR1; forwarded to the program (pid 4242)\n"
    "saving checkpoint to /scratch/ckpt/ckpt-000012000 ...\n"
    "[hpclab] finished: state=killed exit_code=138 signal=USR1\n"
)


def test_checkpoint_and_exit_on_warning_signal_is_a_planned_stop_to_resume() -> None:
    # --signal=B:USR1@600 + GracefulStop: the program checkpoints and exits 128+SIGUSR1.
    d = run(
        WARNING_SIGNAL_LOG,
        scheduler_state="FAILED",
        exit_code=138,
        app_state="killed",
        elapsed_s=13_800,
        time_limit_s=14_400,
        checkpoint_available=True,
    )
    item = d.items[0]
    assert item.code == "timeout", d.codes()
    assert item.suggested_strategies[0] == "resume"
    assert "retry" not in item.suggested_strategies
    assert "SIGUSR1" in item.explanation
    assert "killed_signal" not in d.codes()
    ResourceSpec(**resource_patch_from_hint(item.resource_hint))


def test_warning_signal_from_exit_code_alone() -> None:
    d = run(scheduler_state="FAILED", exit_code=140, elapsed_s=3000, time_limit_s=3600)
    assert d.primary_code == "timeout"
    assert d.items[0].suggested_strategies[:2] == ["resume", "adjust_resources"]


def test_warning_signal_without_checkpoint_says_the_program_did_not_save() -> None:
    d = run(WARNING_SIGNAL_LOG, scheduler_state="FAILED", exit_code=138, checkpoint_available=False)
    item = d.items[0]
    assert item.code == "timeout" and item.suggested_strategies[0] == "fix_code"
    assert "no checkpoint" in item.explanation.lower()


def test_warning_signal_then_exit_0_with_missing_outputs_is_not_blamed_on_output_paths() -> None:
    d = run(
        WARNING_SIGNAL_LOG.replace("exit_code=138 signal=USR1", "exit_code=0"),
        scheduler_state="COMPLETED",
        exit_code=0,
        app_state="exited_ok",
        missing_outputs=("final_metrics",),
    )
    assert d.primary_code == "timeout"
    assert "success_no_outputs" in d.codes()


def test_warning_signal_in_a_clean_success_is_not_a_failure() -> None:
    d = run(
        WARNING_SIGNAL_LOG.replace("exit_code=138 signal=USR1", "exit_code=0"),
        scheduler_state="COMPLETED",
        exit_code=0,
        app_state="exited_ok",
        result_state="valid",
    )
    assert d.items == []


@pytest.mark.parametrize(
    ("check", "first", "needle"),
    [
        ("torch_gpu_arch", "fix_environment", "architecture"),
        ("torch_cuda_available", "fix_environment", "CUDA"),
        ("venv", "fix_environment", "runtime.venv"),
        ("nvidia_smi", "retry", "nvidia-smi"),
    ],
)
def test_guard_checks_use_the_wrapper_check_names(check: str, first: str, needle: str) -> None:
    # jobscript.GUARD_CHECKS = ("venv", "nvidia_smi", "torch_cuda_available", "torch_gpu_arch")
    stderr = (
        f"[hpclab] guard check '{check}' failed: something specific about {check}\n"
        f"HPCLAB_GUARD_FAILED:{check}\n"
    )
    item = run(stderr, scheduler_state="FAILED", exit_code=3, node_list="n7").items[0]
    assert item.code == "env_guard_failed"
    assert item.suggested_strategies[0] == first
    assert needle in item.explanation
    assert "see the wrapper's stderr" not in item.explanation
    assert any(f"something specific about {check}" in e for e in item.evidence)
    if check == "torch_gpu_arch":
        assert "adjust_resources" in item.suggested_strategies


def test_wrapper_setup_phase_failure_is_an_environment_failure() -> None:
    stderr = (
        "[hpclab] setup[1]: conda activate llm\n"
        "CondaError: Run 'conda init' before 'conda activate'\n"
        "[hpclab] setup[1] failed with exit code 1\n"
        "[hpclab] finished: state=exited exit_code=97 failed_phase=setup (setup line 1 failed with exit code 1)\n"
    )
    d = run(stderr, scheduler_state="FAILED", exit_code=97)
    item = d.items[0]
    assert item.code == "env_guard_failed" and item.resource_hint["guard_check"] == "setup"
    assert item.suggested_strategies[0] == "fix_environment"
    assert "runtime.setup" in item.explanation


def test_specific_cause_of_a_setup_failure_ranks_first() -> None:
    stderr = (
        "/var/spool/slurmd/job1/slurm_script: line 40: module: command not found\n"
        "[hpclab] setup[1] failed with exit code 127\n"
        "[hpclab] finished: state=exited exit_code=97 failed_phase=setup (setup line 1 failed with exit code 127)\n"
    )
    d = run(stderr, scheduler_state="FAILED", exit_code=97)
    assert d.codes()[:2] == ["command_not_found", "env_guard_failed"]


def test_wrapper_prepare_failure_for_missing_entry_workdir_is_a_code_fix() -> None:
    stderr = (
        "[hpclab] finished: state=exited exit_code=96 failed_phase=prepare "
        "(entrypoint workdir 'src/train' does not exist in the code snapshot)\n"
    )
    item = run(stderr, scheduler_state="FAILED", exit_code=96).items[0]
    assert item.code == "env_guard_failed" and item.resource_hint["guard_check"] == "prepare"
    assert item.suggested_strategies[0] == "fix_code"


def test_harmless_shell_noise_does_not_outrank_the_fatal_traceback() -> None:
    stderr = "ls: cannot access '/scratch/foo/cache': No such file or directory\n" + TRACEBACK_VALUE_ERROR
    d = run(stderr, **FAILED)
    assert d.primary_code == "python_exception"
    assert d.item("file_not_found") is not None and d.item("file_not_found").confidence == "medium"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("found", "driver", "index"),
    [("12020", "12.2", "cu121"), ("12080", "12.8", "cu128"), ("12050", "12.5", "cu124"),
     ("13000", "13.0", "cu130"), ("11040", "11.4", "cu113"), ("12090", "12.9", "cu129")],
)  # fmt: skip
def test_driver_mismatch_suggests_an_existing_wheel_index(found: str, driver: str, index: str) -> None:
    line = f"RuntimeError: The NVIDIA driver on your system is too old (found version {found}).\n"
    item = run(line, **FAILED).items[0]
    assert item.resource_hint["driver_cuda"] == driver
    assert item.resource_hint["torch_index_url"] == f"https://download.pytorch.org/whl/{index}"
    assert index in item.explanation


def test_ancient_driver_gets_no_wheel_url() -> None:
    item = run("RuntimeError: ... driver is too old (found version 9020).\n", **FAILED).items[0]
    assert "torch_index_url" not in item.resource_hint


def test_incomplete_results_are_not_reported_as_a_clean_success() -> None:
    d = run(scheduler_state="COMPLETED", exit_code=0, app_state="exited_ok", result_state="incomplete")
    assert d.primary_code == "success_no_outputs"
    assert not d.summary.startswith("Completed; no problems")


def test_multi_node_node_fail_does_not_exclude_the_healthy_batch_host() -> None:
    marker = (
        "slurmstepd: error: *** JOB 1 ON haic-hgx-1 CANCELLED AT 2026-09-27T10:00:00 "
        "DUE TO NODE FAILURE, SEE SLURMCTLD LOG FOR DETAILS ***\n"
    )
    item = run(marker, scheduler_state="NODE_FAIL", node_list="haic-hgx-[1-4]").items[0]
    assert "exclude_nodes" not in item.resource_hint
    assert item.resource_hint["suspect_nodes"] == "haic-hgx-[1-4]"
    assert "which" in item.explanation


def test_single_node_node_fail_still_excludes_the_node() -> None:
    item = run(scheduler_state="NODE_FAIL", node_list="haic-hgx-2").items[0]
    assert resource_patch_from_hint(item.resource_hint) == {"exclude_nodes": ["haic-hgx-2"]}


def test_unkillable_stepd_names_the_stuck_node_even_in_multi_node_jobs(log: Callable[[str], str]) -> None:
    item = run(log("stepd_unkillable.stderr.log"), **FAILED, node_list="haic-hgx-[1-2]").items[0]
    assert item.resource_hint["exclude_nodes"] == "haic-hgx-2"


@pytest.mark.parametrize("node_list", ["haic-hgx-[1-4]", "n1,n2"])
def test_multi_node_nccl_and_nfs_errors_do_not_exclude_whole_allocation(node_list: str) -> None:
    nccl = run(
        "torch.distributed.DistBackendError: NCCL error in: x.cpp:1, unhandled system error\n",
        **FAILED,
        node_list=node_list,
    ).item("nccl_error")
    nfs = run("OSError: [Errno 116] Stale file handle\n", **FAILED, node_list=node_list).item("nfs_stale")
    for item in (nccl, nfs):
        assert item is not None
        assert "exclude_nodes" not in item.resource_hint
        assert item.resource_hint["suspect_nodes"] == node_list


def test_transient_nan_in_a_failed_job_is_not_the_primary_cause() -> None:
    d = run(stdout="step 10 loss 0.5\nstep 20 loss nan\nstep 30 loss 0.4\n", **FAILED)
    assert d.primary_code == "unknown_failure"
    assert d.item("nan_loss") is not None and d.item("nan_loss").confidence == "low"  # type: ignore[union-attr]
    assert d.items[0].suggested_strategies[0] != "none"


def test_nonfinite_map_with_split_prefixed_loss_name() -> None:
    line = '{"schema":"hpclab.metrics/v1","step":7,"split":"train","metrics":{"train/loss":null},"nonfinite":{"train/loss":"nan"}}\n'
    assert run(stdout=line, **FAILED).primary_code == "nan_loss"


@pytest.mark.parametrize(
    ("stderr", "secret"),
    [
        ("FileNotFoundError: [Errno 2] No such file or directory: 'https://alice:s3cr3tpass@host/d.csv'\n", "s3cr3tpass"),
        ("OSError: [Errno 122] Disk quota exceeded: '/tmp/token=abcd1234efgh5678/x'\n", "abcd1234efgh5678"),
        ("PermissionError: [Errno 13] Permission denied: '/x/password=hunter2/y'\n", "hunter2"),
        ("ModuleNotFoundError: No module named 'hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123'\n", "hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123"),
    ],
)  # fmt: skip
def test_secrets_never_reach_explanations_summaries_or_hints(stderr: str, secret: str) -> None:
    dumped = run(stderr, **FAILED).model_dump_json()
    assert secret not in dumped


def test_private_key_body_lines_are_redacted() -> None:
    body = "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun"
    stderr = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n{body[::-1]}\n-----END RSA PRIVATE KEY-----\n"
    dumped = run(stderr, **FAILED).model_dump_json()
    assert body not in dumped and body[::-1] not in dumped
    assert body not in redact_secrets(stderr)


def test_explanations_of_every_fixture_survive_redaction_unchanged() -> None:
    # redaction is applied to prose too, so it must not mangle ordinary advice text
    for name, stderr, stdout in all_fixture_texts():
        if name.startswith("secrets"):
            continue
        for item in diagnose(AttemptFacts(**FAILED), stdout, stderr).items:
            assert "[REDACTED" not in item.explanation, (name, item.code)


def test_timeout_that_cannot_finish_within_partition_max_needs_resume_chain() -> None:
    d = run(
        scheduler_state="TIMEOUT",
        elapsed_s=86_400,
        time_limit_s=86_400,
        partition_max_time_s=172_800,
        progress_fraction=0.3,
        checkpoint_available=True,
    )
    item = d.items[0]
    assert item.resource_hint["time"] == "2-00:00:00"
    assert item.resource_hint["extra_sbatch"].startswith("--signal=B:USR1@")
    assert item.suggested_strategies[0] == "resume"
    assert "more than one attempt" in item.explanation


def test_repeated_timeouts_of_a_resume_chain_are_not_called_hopeless() -> None:
    item = run(
        scheduler_state="TIMEOUT",
        elapsed_s=3600,
        time_limit_s=3600,
        checkpoint_available=True,
        previous_codes=("timeout", "timeout"),
    ).items[0]
    assert item.suggested_strategies[0] == "resume"
    assert "unlikely to help" not in item.explanation


def test_timeout_after_the_warning_signal_points_at_the_lead_time() -> None:
    stderr = (
        "[hpclab] received SIGUSR1; forwarded to the program (pid 4242)\n"
        "saving checkpoint ...\n"
        "slurmstepd: error: *** JOB 9 ON n1 CANCELLED AT 2026-09-27T10:00:00 DUE TO TIME LIMIT ***\n"
    )
    item = run(stderr, scheduler_state="TIMEOUT", elapsed_s=3600, time_limit_s=3600).items[0]
    assert item.code == "timeout" and item.confidence == "high"
    assert "lead_s" in item.explanation
    assert any("received SIGUSR1" in e for e in item.evidence)


def test_out_of_memory_state_while_the_program_exited_ok_is_not_a_hard_failure() -> None:
    # Slurm sets OUT_OF_MEMORY when *any* process in the cgroup was OOM-killed (e.g. a DataLoader
    # worker that got restarted), even if the program itself exited 0 and wrote valid results.
    d = run(
        "slurmstepd: error: Detected 1 oom_kill event in StepId=77.batch. Some of the step tasks have been OOM Killed.\n",
        scheduler_state="OUT_OF_MEMORY",
        exit_code=0,
        exit_signal=125,
        app_state="exited_ok",
        result_state="valid",
        mem_request_bytes=64 * GIB,
    )
    item = d.items[0]
    assert item.code == "host_oom" and item.confidence == "medium"
    assert item.suggested_strategies[0] == "none"
    assert "exited 0" in item.explanation
    assert item.resource_hint["mem"] == "96G"


def test_out_of_memory_with_the_program_killed_stays_high() -> None:
    d = run(scheduler_state="OUT_OF_MEMORY", exit_code=0, exit_signal=125, app_state="killed")
    assert d.items[0].confidence == "high" and d.items[0].suggested_strategies[0] == "adjust_resources"
