"""The recovery eligibility table (pure ``assess``) and resource validation."""

from __future__ import annotations

from typing import Any

import pytest
from wf_support import local_profile

from hpclab.domain.models import ResourceSpec
from hpclab.profiles.schema import ResolvedCluster, UserOverlay
from hpclab.services.diagnosis import AttemptFacts, Diagnosis, DiagnosisItem, diagnose
from hpclab.services.recovery import assess, validate_resources


def dg(code: str, strategies: list[str] | None = None, hint: dict[str, str] | None = None) -> Diagnosis:
    item = DiagnosisItem(
        code=code,
        title=code,
        confidence="high",
        explanation="x",
        suggested_strategies=strategies or ["retry"],
        resource_hint=hint or {},
    )
    return Diagnosis(items=[item], primary_code=code, summary=code)


def run(**kw: Any):
    base: dict[str, Any] = {
        "phase": "terminal",
        "scheduler_state": "FAILED",
        "result_state": "unchecked",
        "diagnosis": None,
        "requested": None,
        "resumable": False,
        "has_checkpoint": False,
        "has_hint_patch": False,
        "run_id": "run_x",
    }
    base.update(kw)
    return assess(**base)


# (case, kwargs, eligible, strategy, blocking substring)
TABLE = [
    ("staging is active", {"phase": "staging"}, False, None, "still active"),
    ("in flight is active", {"phase": "submission_in_flight"}, False, None, "still active"),
    (
        "observing is active",
        {"phase": "observing", "scheduler_state": "RUNNING"},
        False,
        None,
        "still active",
    ),
    ("uncertain -> reconcile", {"phase": "submission_uncertain"}, False, None, "reconciliation"),
    ("duplicate -> reconcile", {"phase": "duplicate_detected"}, False, None, "never a blind resubmit"),
    (
        "newer attempt",
        {"newer_attempt_id": "att_new", "diagnosis": dg("node_fail")},
        False,
        None,
        "newer attempt",
    ),
    (
        "claim held elsewhere",
        {"claim_attempt_id": "att_other", "diagnosis": dg("node_fail")},
        False,
        None,
        "claim",
    ),
    (
        "success is final",
        {"scheduler_state": "COMPLETED", "result_state": "valid"},
        False,
        None,
        "nothing to recover",
    ),
    (
        "unvalidated completion",
        {"scheduler_state": "COMPLETED", "result_state": "unchecked"},
        False,
        None,
        "not validated",
    ),
    (
        "node_fail auto retry",
        {"scheduler_state": "NODE_FAIL", "diagnosis": dg("node_fail", ["retry", "resume"])},
        True,
        "retry",
        None,
    ),
    (
        "node_fail auto resume with checkpoint",
        {"diagnosis": dg("node_fail", ["resume", "retry"]), "resumable": True, "has_checkpoint": True},
        True,
        "resume",
        None,
    ),
    ("preempted retry", {"diagnosis": dg("preempted", ["retry"]), "requested": "retry"}, True, "retry", None),
    ("cancelled retry", {"diagnosis": dg("cancelled", ["retry"]), "requested": "retry"}, True, "retry", None),
    (
        "timeout retry refused",
        {"diagnosis": dg("timeout", ["adjust_resources"]), "requested": "retry"},
        False,
        "retry",
        "times out again",
    ),
    (
        "timeout auto adjust",
        {
            "diagnosis": dg("timeout", ["adjust_resources", "resume"], {"time": "02:00:00"}),
            "has_hint_patch": True,
        },
        True,
        "adjust_resources",
        None,
    ),
    (
        "timeout resume needs resumable",
        {"diagnosis": dg("timeout", ["resume"]), "requested": "resume"},
        False,
        "resume",
        "resumable",
    ),
    (
        "timeout resume",
        {"diagnosis": dg("timeout", ["resume"]), "requested": "resume", "resumable": True},
        True,
        "resume",
        None,
    ),
    (
        "host_oom retry refused",
        {"diagnosis": dg("host_oom", ["adjust_resources"]), "requested": "retry"},
        False,
        "retry",
        "memory",
    ),
    (
        "cuda_oom retry refused",
        {"diagnosis": dg("cuda_oom", ["derive_run", "fix_code"]), "requested": "retry"},
        False,
        "retry",
        "GPU memory",
    ),
    (
        "cuda_oom adjust allowed (patch checked later)",
        {"diagnosis": dg("cuda_oom", ["derive_run"]), "requested": "adjust_resources"},
        True,
        "adjust_resources",
        None,
    ),
    (
        "nan is semantic",
        {"diagnosis": dg("nan_loss", ["fix_code", "derive_run"])},
        False,
        None,
        "diverges again",
    ),
    ("code bug is semantic", {"diagnosis": dg("python_exception", ["fix_code"])}, False, None, "same code"),
    (
        "transient exception",
        {"diagnosis": dg("python_exception", ["retry", "fix_code"])},
        True,
        "retry",
        None,
    ),
    (
        "missing outputs",
        {
            "scheduler_state": "COMPLETED",
            "result_state": "incomplete",
            "diagnosis": dg("success_no_outputs", ["fix_code"]),
        },
        False,
        None,
        "same code",
    ),
    (
        "state dict mismatch",
        {"diagnosis": dg("state_dict_mismatch", ["fix_code", "derive_run"])},
        False,
        None,
        "same code",
    ),
    (
        "environment retry with warning",
        {"diagnosis": dg("import_error", ["fix_environment"]), "requested": "retry"},
        True,
        "retry",
        None,
    ),
    (
        "unknown failure retry",
        {"diagnosis": dg("unknown_failure", ["fix_code", "retry"])},
        True,
        "retry",
        None,
    ),
    ("no signature at all", {"diagnosis": None}, True, "retry", None),
    (
        "rejected -> retry",
        {"phase": "submission_rejected", "scheduler_state": "UNKNOWN"},
        True,
        "retry",
        None,
    ),
    (
        "rejected cannot resume",
        {"phase": "submission_rejected", "requested": "resume", "resumable": True},
        False,
        "resume",
        "nothing ran",
    ),
    ("cancelled before submit -> retry", {"phase": "cancelled_before_submit"}, True, "retry", None),
    ("abandoned -> retry with warning", {"phase": "abandoned"}, True, "retry", None),
]


@pytest.mark.parametrize(
    ("case", "kwargs", "eligible", "strategy", "blocking"), TABLE, ids=[t[0] for t in TABLE]
)
def test_eligibility_table(
    case: str, kwargs: dict[str, Any], eligible: bool, strategy: str | None, blocking: str | None
) -> None:
    a = run(**kwargs)
    assert a.eligible is eligible, (case, a)
    assert a.strategy == strategy
    if blocking:
        assert any(blocking in b for b in a.blocking), a.blocking
    else:
        assert not a.blocking


def test_semantic_failures_come_with_a_derive_run_hint() -> None:
    a = run(diagnosis=dg("nan_loss", ["fix_code", "derive_run"]))
    hint = next(h for h in a.hints if h.strategy == "derive_run")
    assert hint.tool == "experiment_prepare" and hint.arguments["derive_from"]["run_id"] == "run_x"
    code_bug = run(diagnosis=dg("python_exception", ["fix_code"]))
    assert any(h.arguments.get("derive_from", {}).get("recapture_code") for h in code_bug.hints)


def test_environment_failures_warn_that_the_environment_must_be_fixed() -> None:
    a = run(diagnosis=dg("cuda_driver_mismatch", ["fix_environment"]), requested="retry")
    assert a.eligible and a.warnings and any(h.strategy == "fix_environment" for h in a.hints)


def test_real_diagnoses_drive_the_table() -> None:
    # a real CUDA OOM log: a plain retry is refused, a derived run is suggested
    oom = diagnose(
        AttemptFacts(scheduler_state="FAILED", exit_code=1, gpus=1),
        "",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.10 GiB\n",
    )
    assert oom.primary_code == "cuda_oom"
    a = run(diagnosis=oom, requested="retry")
    assert not a.eligible and any(h.strategy == "derive_run" for h in a.hints)
    nf = diagnose(AttemptFacts(scheduler_state="NODE_FAIL", node_list="n1"), "", "")
    assert run(scheduler_state="NODE_FAIL", diagnosis=nf).strategy == "retry"


def _cluster() -> ResolvedCluster:
    return ResolvedCluster(profile=local_profile(), overlay=UserOverlay(cluster_id="sim", remote_root="/r"))


@pytest.mark.parametrize(
    ("res", "resumable", "failing"),
    [
        ({"partition": "gpu", "gpus": 1, "time": "12:00:00", "gpu_type": "a100"}, False, set()),
        ({"partition": "gpu", "gpus": 1, "time": "3-00:00:00"}, False, {"time_within_partition_max"}),
        ({"partition": "gpu", "gpus": 1, "gpu_type": "b200"}, False, {"gpu_type_in_partition"}),
        ({"partition": "gpu", "gpus": 8, "nodes": 4}, False, {"gpus_limits"}),
        ({"partition": "interactive", "gpus": 1}, False, {"interactive_partition_batch"}),
        ({"partition": "scavenge", "gpus": 1, "gpu_type": "h100"}, False, {"preemptible_gate"}),
        ({"partition": "scavenge", "gpus": 1, "gpu_type": "h100"}, True, set()),
        ({"partition": "nope", "gpus": 1}, False, {"partition_exists"}),
    ],
)
def test_partition_limits(res: dict[str, Any], resumable: bool, failing: set[str]) -> None:
    checks = validate_resources(_cluster(), ResourceSpec(**res), resumable=resumable)
    assert {c.check for c in checks if c.status == "fail"} == failing
