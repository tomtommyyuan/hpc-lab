"""Table-driven tests of every static prepare check (pass / warn / fail / unchecked)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from prepare_support import (
    BASE_SPEC,
    Lab,
    lab_overlay,
    lab_profile,
    manifest_of,
    plan_context,
)

from hpclab.profiles.schema import PartitionRule, ResolvedCluster, SitePolicy, StorageRole
from hpclab.services import validation as vd

Spec = dict[str, Any]


def base_spec(lab: Lab) -> Spec:
    spec = copy.deepcopy(BASE_SPEC)
    spec["code"] = {"project_root": str(lab.project)}
    return spec


def cluster(lab: Lab, *, overlay: dict[str, Any] | None = None, **profile: Any) -> ResolvedCluster:
    return ResolvedCluster(profile=lab_profile(lab.root, **profile), overlay=lab_overlay(**(overlay or {})))


def res(**kw: Any) -> Callable[[Spec], None]:
    def patch(s: Spec) -> None:
        s["resources"].update(kw)

    return patch


def setk(key: str, value: Any) -> Callable[[Spec], None]:
    def patch(s: Spec) -> None:
        if value is None:
            s.pop(key, None)
        else:
            s[key] = value

    return patch


def chain(*patches: Callable[[Spec], None]) -> Callable[[Spec], None]:
    def patch(s: Spec) -> None:
        for p in patches:
            p(s)

    return patch


@dataclass
class Case:
    id: str
    check: Callable[[vd.PlanContext], vd.Finding]
    status: str
    severity: Any = None
    patch: Callable[[Spec], None] | None = None
    cluster: Callable[[Lab], ResolvedCluster | None] | None = None  # default: the lab cluster
    pc: dict[str, Any] = field(default_factory=dict)
    contains: str | None = None

    def __post_init__(self) -> None:
        # table shorthand: Case(id, check, status, patch) when the severity is not asserted
        if callable(self.severity):
            self.patch, self.severity = self.severity, None


NO_CLUSTER: Callable[[Lab], ResolvedCluster | None] = lambda lab: None  # noqa: E731
HOME_ROOT: Callable[[Lab], ResolvedCluster | None] = lambda lab: cluster(  # noqa: E731
    lab, overlay={"remote_root": str(lab.root / "home" / "alice" / "hpclab")}
)

CASES: list[Case] = [
    # spec_schema
    Case("schema-pass", vd.check_spec_schema, "pass"),
    Case("schema-no-question", vd.check_spec_schema, "warn", "info", setk("question", None)),
    Case(
        "schema-runtime-clash", vd.check_spec_schema, "fail", "block", setk("parameters", {"output_dir": "x"})
    ),
    Case("schema-seed-param", vd.check_spec_schema, "fail", "block", setk("parameters", {"seed": 1})),
    Case(
        "schema-bad-treatment",
        vd.check_spec_schema,
        "warn",
        "info",
        setk("comparison", {"treatment": ["lr"], "baseline": {"method": "nope"}}),
        contains="not matrix axes",
    ),
    # placeholders
    Case("ph-pass", vd.check_placeholders, "pass"),
    Case(
        "ph-unknown",
        vd.check_placeholders,
        "fail",
        "block",
        setk("entrypoint", {"argv": ["python", "train.py", "{lr}", "{method}"]}),
    ),
    Case(
        "ph-output-param",
        vd.check_placeholders,
        "fail",
        "block",
        setk("outputs", [{"name": "m", "path": "{output_dir}/{method}.json"}]),
    ),
    Case(
        "ph-unused-axis",
        vd.check_placeholders,
        "warn",
        "info",
        setk("entrypoint", {"argv": ["python", "train.py", "{seed}"]}),
        contains="HPCLAB_PARAMS_JSON",
    ),
    Case(
        "ph-null",
        vd.check_placeholders,
        "warn",
        "warn",
        chain(setk("matrix", {"method": [None, "a"]}), setk("smoke", None)),
    ),
    Case("ph-seed-without-seeds", vd.check_placeholders, "fail", "block", setk("seeds", None)),
    # matrix_duplicates
    Case("dup-pass", vd.check_matrix_duplicates, "pass"),
    Case("dup-type-only", vd.check_matrix_duplicates, "fail", "block", setk("matrix", {"method": [1, "1"]})),
    # cluster_registered / remote_root
    Case("cluster-pass", vd.check_cluster_registered, "pass"),
    Case("cluster-missing", vd.check_cluster_registered, "fail", "block", cluster=NO_CLUSTER),
    Case("rr-pass", vd.check_remote_root, "pass"),
    Case("rr-unresolved", vd.check_remote_root, "fail", "block", pc={"_remote_root": None}),
    Case("rr-no-cluster", vd.check_remote_root, "unchecked", cluster=NO_CLUSTER),
    # partition_exists
    Case("part-pass", vd.check_partition_exists, "pass"),
    Case("part-unknown", vd.check_partition_exists, "fail", "block", res(partition="nope")),
    Case(
        "part-none",
        vd.check_partition_exists,
        "warn",
        "warn",
        cluster=lambda lab: cluster(lab, default_partition=None),
    ),
    Case(
        "part-no-rules",
        vd.check_partition_exists,
        "unchecked",
        cluster=lambda lab: cluster(lab, partitions={}, default_partition=None),
    ),
    Case("part-no-cluster", vd.check_partition_exists, "unchecked", cluster=NO_CLUSTER),
    # time_within_partition_max
    Case("time-pass", vd.check_time_within_partition_max, "pass"),
    Case("time-over", vd.check_time_within_partition_max, "fail", "block", res(time="3-00:00:00")),
    Case(
        "time-unknown",
        vd.check_time_within_partition_max,
        "unchecked",
        chain(res(partition="nomax", gpus=0, gpu_type=None)),
        cluster=lambda lab: cluster(
            lab, partitions={"nomax": PartitionRule(), "gpu": PartitionRule(max_time="1:00:00")}
        ),
    ),
    # gpu_type_in_partition
    Case("gtype-pass", vd.check_gpu_type_in_partition, "pass"),
    Case("gtype-wrong", vd.check_gpu_type_in_partition, "fail", "block", res(gpu_type="h100")),
    Case(
        "gtype-any",
        vd.check_gpu_type_in_partition,
        "warn",
        "warn",
        res(gpu_type=None),
        contains="heterogeneous",
    ),
    Case("gtype-unknown", vd.check_gpu_type_in_partition, "unchecked", res(partition="cpu", gpu_type=None)),
    Case(
        "gtype-cpu-job", vd.check_gpu_type_in_partition, "pass", res(partition="cpu", gpus=0, gpu_type=None)
    ),
    # gpus_limits
    Case("gpus-pass", vd.check_gpus_limits, "pass"),
    Case("gpus-per-job", vd.check_gpus_limits, "fail", "block", res(gpus=16)),
    Case(
        "gpus-per-user",
        vd.check_gpus_limits,
        "fail",
        "block",
        res(gpus=2),
        cluster=lambda lab: cluster(
            lab,
            partitions={"gpu": PartitionRule(max_time="1-00:00:00", gpu_types=["a100"], max_gpus_per_user=1)},
        ),
        contains="never starts",
    ),
    Case("gpus-unknown", vd.check_gpus_limits, "unchecked", res(partition="scavenge", gpu_type="h100")),
    # account_present / qos_required
    Case("acct-pass", vd.check_account_present, "pass"),
    Case(
        "acct-missing",
        vd.check_account_present,
        "fail",
        "block",
        cluster=lambda lab: cluster(lab, overlay={"account": None}),
    ),
    Case(
        "acct-optional",
        vd.check_account_present,
        "pass",
        cluster=lambda lab: cluster(lab, overlay={"account": None}, policy=SitePolicy(require_account=False)),
    ),
    Case("acct-no-cluster", vd.check_account_present, "unchecked", cluster=NO_CLUSTER),
    Case("qos-pass", vd.check_qos_required, "pass"),
    Case(
        "qos-missing",
        vd.check_qos_required,
        "fail",
        "block",
        cluster=lambda lab: cluster(lab, policy=SitePolicy(require_qos=True)),
    ),
    Case(
        "qos-given",
        vd.check_qos_required,
        "pass",
        res(qos="normal"),
        cluster=lambda lab: cluster(lab, policy=SitePolicy(require_qos=True)),
    ),
    Case("qos-no-cluster", vd.check_qos_required, "unchecked", cluster=NO_CLUSTER),
    # interactive_partition_batch
    Case("inter-pass", vd.check_interactive_partition_batch, "pass"),
    Case("inter-fail", vd.check_interactive_partition_batch, "fail", "block", res(partition="interactive")),
    Case("inter-unknown", vd.check_interactive_partition_batch, "unchecked", res(partition="nope")),
    # storage_home_avoidance
    Case("store-pass", vd.check_storage_home_avoidance, "pass"),
    Case("store-home-root", vd.check_storage_home_avoidance, "fail", "block", cluster=HOME_ROOT),
    Case(
        "store-venv-home",
        vd.check_storage_home_avoidance,
        "warn",
        "warn",
        setk("runtime", {"venv": "~/venvs/x"}),
    ),
    Case(
        "store-input-home",
        vd.check_storage_home_avoidance,
        "warn",
        "warn",
        setk("inputs", [{"name": "d", "uri": "{home}/data/train"}]),
        cluster=lambda lab: cluster(
            lab,
            storage={
                "home": StorageRole(path=str(lab.root / "home" / "{user}"), avoid_for=["datasets"]),
                "scratch": StorageRole(path=str(lab.root / "scratch" / "{user}")),
            },
        ),
    ),
    Case("store-unresolved", vd.check_storage_home_avoidance, "unchecked", pc={"_remote_root": None}),
    # cache_locations
    Case(
        "cache-pass",
        vd.check_cache_locations,
        "pass",
        cluster=lambda lab: cluster(
            lab,
            storage={
                "home": StorageRole(path=str(lab.root / "home" / "{user}")),
                "scratch": StorageRole(path=str(lab.root / "scratch" / "{user}")),
            },
        ),
    ),
    Case(
        "cache-home",
        vd.check_cache_locations,
        "warn",
        "warn",
        setk("runtime", {"env": {"HF_HOME": "$HOME/.cache/hf"}}),
    ),
    Case(
        "cache-setup-home",
        vd.check_cache_locations,
        "warn",
        "warn",
        setk("runtime", {"setup": ["export TORCH_HOME=~/.cache/torch"]}),
    ),
    Case("cache-unset", vd.check_cache_locations, "warn", "info", contains="unset"),
    Case("cache-no-cluster", vd.check_cache_locations, "unchecked", cluster=NO_CLUSTER),
    # outputs_declared
    Case("out-pass", vd.check_outputs_declared, "pass"),
    Case(
        "out-none",
        vd.check_outputs_declared,
        "warn",
        "warn",
        chain(setk("outputs", []), setk("smoke", None), setk("evaluation", None)),
    ),
    Case(
        "out-optional",
        vd.check_outputs_declared,
        "warn",
        "warn",
        setk(
            "outputs",
            [{"name": "metrics", "path": "{output_dir}/m.json", "role": "metrics", "required": False}],
        ),
    ),
    Case(
        "out-no-metrics-role",
        vd.check_outputs_declared,
        "warn",
        "info",
        setk("outputs", [{"name": "metrics", "path": "{output_dir}/m.json"}]),
    ),
    # evaluation_output_declared
    Case("eval-pass", vd.check_evaluation_output_declared, "pass"),
    Case(
        "eval-bad-output",
        vd.check_evaluation_output_declared,
        "fail",
        "block",
        setk("evaluation", {"output": "nope"}),
    ),
    Case(
        "eval-unknown-evaluator",
        vd.check_evaluation_output_declared,
        "fail",
        "block",
        setk("evaluation", {"output": "metrics", "evaluator_id": "nope"}),
    ),
    Case(
        "eval-bad-config",
        vd.check_evaluation_output_declared,
        "fail",
        "block",
        setk("evaluation", {"output": "metrics", "config": {"no_such_option": 1}}),
    ),
    Case("eval-none", vd.check_evaluation_output_declared, "warn", "info", setk("evaluation", None)),
    Case(
        "eval-none-with-comparison",
        vd.check_evaluation_output_declared,
        "warn",
        "warn",
        chain(setk("evaluation", None), setk("comparison", {"treatment": ["method"]})),
    ),
    # smoke_gate
    Case("smoke-pass", vd.check_smoke_gate, "pass"),
    Case("smoke-none", vd.check_smoke_gate, "warn", "info", setk("smoke", None)),
    Case("smoke-exit-only", vd.check_smoke_gate, "warn", "info", setk("smoke", {"argv_append": []})),
    Case(
        "smoke-undeclared", vd.check_smoke_gate, "fail", "block", setk("smoke", {"require_outputs": ["nope"]})
    ),
    Case("smoke-bad-cell", vd.check_smoke_gate, "fail", "block", setk("smoke", {"cell": {"method": "zzz"}})),
    # preemptible_gate
    Case("preempt-pass", vd.check_preemptible_gate, "pass"),
    Case(
        "preempt-fail", vd.check_preemptible_gate, "fail", "block", res(partition="scavenge", gpu_type="h100")
    ),
    Case(
        "preempt-resumable",
        vd.check_preemptible_gate,
        "pass",
        chain(res(partition="scavenge", gpu_type="h100"), setk("resilience", {"resumable": True})),
    ),
    Case("preempt-unknown", vd.check_preemptible_gate, "unchecked", res(partition="nope")),
    Case("preempt-no-cluster", vd.check_preemptible_gate, "unchecked", cluster=NO_CLUSTER),
    # walltime_signal
    Case("wall-pass", vd.check_walltime_signal, "pass"),
    Case(
        "wall-ckpt-ok",
        vd.check_walltime_signal,
        "pass",
        setk("resilience", {"stop_behavior": "checkpoint_and_exit", "lead_s": 300}),
    ),
    Case(
        "wall-lead-too-long",
        vd.check_walltime_signal,
        "fail",
        "block",
        chain(
            setk("resilience", {"stop_behavior": "checkpoint_and_exit", "lead_s": 3600}), setk("smoke", None)
        ),
    ),
    Case(
        "wall-signal-conflict",
        vd.check_walltime_signal,
        "fail",
        "block",
        chain(
            setk("resilience", {"stop_behavior": "checkpoint_and_exit", "lead_s": 300}),
            res(extra_sbatch=["--signal=B:USR1@60"]),
        ),
    ),
    Case(
        "wall-resumable-no-signal",
        vd.check_walltime_signal,
        "warn",
        "warn",
        setk("resilience", {"resumable": True}),
    ),
    Case(
        "wall-raw-signal", vd.check_walltime_signal, "warn", "warn", res(extra_sbatch=["--signal=B:USR1@60"])
    ),
    Case("wall-long-job", vd.check_walltime_signal, "warn", "info", res(time="10:00:00")),
    # budget
    Case("budget-pass", vd.check_budget, "pass"),
    Case("budget-none", vd.check_budget, "pass", setk("budget", {}), contains="no hpclab budget"),
    Case("budget-over", vd.check_budget, "fail", "block", setk("budget", {"gpu_hours": 3})),
    Case("budget-tight", vd.check_budget, "warn", "warn", setk("budget", {"gpu_hours": 7})),
    Case("budget-existing", vd.check_budget, "fail", "block", pc={"existing_gpu_hours": 45.0}),
    # code_snapshot
    Case("snap-pass", vd.check_code_snapshot, "pass"),
    Case("snap-empty", vd.check_code_snapshot, "fail", "block", pc={"_manifest": manifest_of({})}),
    Case(
        "snap-secret",
        vd.check_code_snapshot,
        "warn",
        "warn",
        pc={"snapshot_warnings": ["SECRET: excluded '.env'"]},
    ),
    Case(
        "snap-other-warning",
        vd.check_code_snapshot,
        "warn",
        "info",
        pc={"snapshot_warnings": ["not a git repository"]},
    ),
    Case("snap-none", vd.check_code_snapshot, "unchecked", pc={"_manifest": None}),
    # entrypoint_in_snapshot
    Case("entry-pass", vd.check_entrypoint_in_snapshot, "pass"),
    Case(
        "entry-missing",
        vd.check_entrypoint_in_snapshot,
        "fail",
        "block",
        pc={"_manifest": manifest_of({"other.py": 0o644})},
    ),
    Case(
        "entry-not-exec",
        vd.check_entrypoint_in_snapshot,
        "fail",
        "block",
        setk("entrypoint", {"argv": ["./run.sh", "{method}", "{seed}"]}),
        pc={"_manifest": manifest_of({"run.sh": 0o644})},
    ),
    Case(
        "entry-exec-ok",
        vd.check_entrypoint_in_snapshot,
        "pass",
        setk("entrypoint", {"argv": ["./run.sh", "{method}", "{seed}"]}),
        pc={"_manifest": manifest_of({"run.sh": 0o755})},
    ),
    Case(
        "entry-module-missing",
        vd.check_entrypoint_in_snapshot,
        "warn",
        "warn",
        setk("entrypoint", {"argv": ["python", "-m", "pkg.train", "{method}", "{seed}"]}),
    ),
    Case(
        "entry-module-found",
        vd.check_entrypoint_in_snapshot,
        "pass",
        setk("entrypoint", {"argv": ["python", "-m", "pkg.train", "{method}", "{seed}"]}),
        pc={"_manifest": manifest_of({"src/pkg/train.py": 0o644})},
    ),
    Case(
        "entry-workdir",
        vd.check_entrypoint_in_snapshot,
        "pass",
        setk("entrypoint", {"argv": ["python", "train.py", "{method}", "{seed}"], "workdir": "sub"}),
        pc={"_manifest": manifest_of({"sub/train.py": 0o644})},
    ),
    Case(
        "entry-opaque",
        vd.check_entrypoint_in_snapshot,
        "unchecked",
        setk("entrypoint", {"argv": ["bash", "-c", "echo {method} {seed}"]}),
    ),
    Case("entry-no-manifest", vd.check_entrypoint_in_snapshot, "unchecked", pc={"_manifest": None}),
    # lockfiles_present
    Case("lock-auto", vd.check_lockfiles_present, "pass", contains="uv.lock"),
    Case("lock-declared", vd.check_lockfiles_present, "pass", setk("runtime", {"lockfiles": ["uv.lock"]})),
    Case(
        "lock-declared-missing",
        vd.check_lockfiles_present,
        "fail",
        "block",
        setk("runtime", {"lockfiles": ["poetry.lock"]}),
    ),
    Case(
        "lock-none",
        vd.check_lockfiles_present,
        "warn",
        "info",
        pc={"_manifest": manifest_of({"train.py": 0o644})},
    ),
    Case("lock-no-manifest", vd.check_lockfiles_present, "unchecked", pc={"_manifest": None}),
    # inputs_pinning
    Case("inputs-none", vd.check_inputs_pinning, "pass"),
    Case(
        "inputs-pinned",
        vd.check_inputs_pinning,
        "pass",
        setk(
            "inputs",
            [
                {"name": "d", "uri": "/data/x", "digest": "sha256:" + "1" * 64},
                {"name": "m", "uri": "hf://m", "version": "v2"},
            ],
        ),
    ),
    Case(
        "inputs-unpinned",
        vd.check_inputs_pinning,
        "warn",
        "info",
        setk("inputs", [{"name": "d", "uri": "/data/x"}]),
    ),
    Case(
        "inputs-overclaimed",
        vd.check_inputs_pinning,
        "warn",
        "warn",
        setk("inputs", [{"name": "d", "uri": "/data/x", "pin": "pinned"}]),
    ),
    # script_render
    Case("render-pass", vd.check_script_render, "pass"),
    Case(
        "render-bad-env", vd.check_script_render, "fail", "block", setk("runtime", {"env": {"BASH_ENV": "x"}})
    ),
    Case(
        "render-container", vd.check_script_render, "fail", "block", setk("runtime", {"container": "img.sif"})
    ),
    Case("render-no-cluster", vd.check_script_render, "unchecked", cluster=NO_CLUSTER),
    # code_drift
    Case("drift-same", vd.check_code_drift, "pass", pc={"base_digests": [manifest_of()["digest"]]}),
    Case(
        "drift-changed",
        vd.check_code_drift,
        "warn",
        "warn",
        pc={
            "base_digests": ["sha256:" + "cd" * 32],
            "base_manifests": {
                "sha256:" + "cd" * 32: manifest_of(
                    {"train.py": 0o755, "old.py": 0o644}, digest="sha256:" + "cd" * 32
                )
            },
        },
    ),
    Case(
        "drift-base-missing",
        vd.check_code_drift,
        "warn",
        "warn",
        pc={"base_digests": ["sha256:" + "cd" * 32], "base_manifests": {"sha256:" + "cd" * 32: None}},
    ),
    Case(
        "drift-no-manifest", vd.check_code_drift, "unchecked", pc={"_manifest": None, "base_digests": ["x"]}
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_check(case: Case, lab: Lab) -> None:
    spec = base_spec(lab)
    if case.patch:
        case.patch(spec)
    cl = case.cluster(lab) if case.cluster else cluster(lab)
    kw = dict(case.pc)
    manifest = kw.pop("_manifest", "default")
    remote_root = kw.pop("_remote_root", True)
    pc = plan_context(spec, cl, manifest=manifest, remote_root=remote_root, **kw)
    f = case.check(pc)
    assert f.status == case.status, f
    if case.severity is not None:
        assert f.severity == case.severity, f
    if case.contains:
        assert case.contains in f.message, f
    if f.status == "fail" and f.severity == "block":
        assert f.blocking


def test_every_static_check_is_table_tested() -> None:
    tested = {c.check.__name__ for c in CASES}
    for fn in vd.STATIC_CHECKS:
        assert fn.__name__ in tested, fn.__name__
        statuses = {c.status for c in CASES if c.check is fn}
        assert "pass" in statuses and len(statuses) >= 2, fn.__name__


def test_code_drift_lists_changed_files(lab: Lab) -> None:
    other = "sha256:" + "cd" * 32
    pc = plan_context(
        base_spec(lab),
        cluster(lab),
        base_digests=[other],
        base_manifests={other: manifest_of({"train.py": 0o755, "old.py": 0o644}, digest=other)},
    )
    f = vd.check_code_drift(pc)
    diff = f.evidence["diff"]["cdcdcdcdcdcd"]
    assert diff["changed"] == ["train.py"] and diff["removed"] == ["old.py"] and diff["added"] == ["uv.lock"]


def test_evaluator_registry_failure_is_unchecked(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpclab.services.evaluators as ev

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("registry broken")

    monkeypatch.setattr(ev, "validate_config", boom)
    f = vd.check_evaluation_output_declared(plan_context(base_spec(lab), cluster(lab)))
    assert f.status == "unchecked"


def test_crashing_check_becomes_unchecked(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(pc: vd.PlanContext) -> vd.Finding:
        raise RuntimeError("bug")

    boom.__name__ = "check_budget"
    monkeypatch.setattr(vd, "STATIC_CHECKS", (boom,))
    out = vd.run_static_checks(plan_context(base_spec(lab), cluster(lab)))
    assert [(f.check, f.status) for f in out] == [("budget", "unchecked")]


# ---------------------------------------------------------------------------
# Report + overrides
# ---------------------------------------------------------------------------


def test_build_report_statuses() -> None:
    ok = vd.finding("a", "pass", "fine")
    un = vd.finding("b", "unchecked", "?")
    bad = vd.finding("c", "fail", "no")
    soft = vd.finding("d", "fail", "meh", severity="warn")
    assert vd.build_report([ok]).status == "pass"
    assert vd.build_report([ok, un]).status == "warn"
    assert vd.build_report([ok, soft]).status == "warn"
    r = vd.build_report([ok, bad])
    assert r.status == "fail" and r.blocking == ["c"] and r.counts["fail"] == 1
    assert vd.build_report([], pending=True).status == "pending"


def test_apply_overrides() -> None:
    fs = [
        vd.finding("budget", "fail", "over"),
        vd.finding("placeholders", "fail", "unknown"),
        vd.finding("outputs_declared", "warn", "none"),
    ]
    out, outcomes = vd.apply_overrides(
        fs,
        [
            vd.ValidationOverride(check="budget", reason="approved"),
            vd.ValidationOverride(check="placeholders", reason="please"),
            vd.ValidationOverride(check="outputs_declared", reason="n/a"),
            vd.ValidationOverride(check="ghost", reason="n/a"),
        ],
    )
    assert out[0].status == "warn" and out[0].severity == "warn" and out[0].override_reason == "approved"
    assert out[1].blocking and out[2].status == "warn"
    by = {o.check: o for o in outcomes}
    assert by["budget"].applied and by["budget"].original_status == "fail"
    assert not by["placeholders"].applied and "cannot be overridden" in by["placeholders"].note
    assert not by["outputs_declared"].applied and "not blocking" in by["outputs_declared"].note
    assert not by["ghost"].applied and "no finding" in by["ghost"].note
    assert vd.build_report(out).overridden == ["budget"]


# ---------------------------------------------------------------------------
# Resource advice
# ---------------------------------------------------------------------------


def _history(pc: vd.PlanContext, elapsed: list[int], rss: int | None = None) -> list[vd.HistoryRow]:
    run = next(r for r in pc.runs if r.kind == "main")
    fam = vd.family_key(run.run_spec, run.resources)
    return [vd.HistoryRow(f"att_{i}", fam, e, rss) for i, e in enumerate(elapsed)]


def test_resource_advice_insufficient_history(lab: Lab) -> None:
    pc = plan_context(base_spec(lab), cluster(lab))
    advice = vd.advise_resources(pc.runs, _history(pc, [100, 200]))
    assert advice.status == "insufficient_history"
    assert vd.check_resource_advice(advice).status == "unchecked"


def test_resource_advice_under_request(lab: Lab) -> None:
    pc = plan_context(base_spec(lab), cluster(lab))
    advice = vd.advise_resources(pc.runs, _history(pc, [3000, 3200, 3400, 3500], rss=4 * 1024**3))
    item = next(i for i in advice.items if i.verdict != "insufficient_history")
    # p90 = 3500 -> 3500 * 1.15 + 300 = 4325 -> rounded up to 4380 s = 01:13:00 > 01:00:00
    assert item.verdict == "under_request" and item.p90_elapsed_s == 3500
    assert item.suggested_time == "01:13:00" and item.suggested_mem == "5G" and item.n == 4
    f = vd.check_resource_advice(advice)
    assert f.status == "warn" and f.severity == "warn" and "TIMEOUT" in f.message


def test_resource_advice_over_request_and_ok(lab: Lab) -> None:
    pc = plan_context(base_spec(lab), cluster(lab))
    over = vd.advise_resources(pc.runs, _history(pc, [600, 700, 800]))
    assert vd.check_resource_advice(over).status == "warn"
    assert vd.check_resource_advice(over).severity == "info"
    ok = vd.advise_resources(pc.runs, _history(pc, [1800, 2000, 2400]))
    assert vd.check_resource_advice(ok).status == "pass"


def test_resource_advice_families_ignore_seed_but_not_params(lab: Lab) -> None:
    pc = plan_context(base_spec(lab), cluster(lab))
    mains = [r for r in pc.runs if r.kind == "main"]
    fams = {(r.run_spec.params["method"], vd.family_key(r.run_spec, r.resources)) for r in mains}
    assert len(fams) == 2  # 2 methods x 3 seeds -> 2 families
    smoke = next(r for r in pc.runs if r.kind == "smoke")
    assert vd.family_key(smoke.run_spec, smoke.resources) not in {f for _, f in fams}


def test_resource_advice_unavailable_is_unchecked() -> None:
    f = vd.check_resource_advice(vd.ResourceAdvice(status="unavailable", note="db down"))
    assert f.status == "unchecked"


def test_p90_nearest_rank() -> None:
    assert vd.p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert vd.p90([5]) == 5
    assert vd.p90([3, 1, 2]) == 3


def test_resilience_lead_s_is_capped_to_what_sbatch_accepts() -> None:
    """Slurm stores --signal's sig_time as a uint16: sbatch rejects a lead above 65535 s."""
    import pydantic

    from hpclab.domain.models import EntrypointSpec, ResilienceSpec

    assert ResilienceSpec(stop_behavior="checkpoint_and_exit", lead_s=65535).lead_s == 65535
    with pytest.raises(pydantic.ValidationError, match="65535"):
        ResilienceSpec(stop_behavior="checkpoint_and_exit", lead_s=65536)
    doc = EntrypointSpec.__doc__ or ""
    assert "{job_id}" in doc
    field = ResilienceSpec.model_fields["max_checkpoint_interval_min"]
    assert "not enforced in v0.1" in (field.description or "")
