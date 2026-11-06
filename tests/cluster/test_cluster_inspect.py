"""cluster_inspect: every check, honest statuses, auth classification, identity, candidates.

Unit tests drive a scripted login node + scheduler (conftest ``env``); ``@pytest.mark.sim``
tests run the real SlurmScheduler against the simulator.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from cluster_fakes import Env, job

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    CommandResult,
    ConnectionLost,
    TransportTimeout,
)
from hpclab.adapters.slurm import SlurmScheduler
from hpclab.domain.clock import FakeClock, iso
from hpclab.domain.errors import ErrorCode, InvalidArgument
from hpclab.profiles.schema import SiteObservation
from hpclab.services.cluster import ALL_CHECKS, CandidateShape, ClusterInspectData, cluster_inspect
from hpclab.testing import factories as f
from hpclab.testing.simctx import sim_context


def by_name(data: ClusterInspectData) -> dict[str, Any]:
    return {c.name: c for c in data.checks}


def codes(check: Any) -> set[str]:
    return {x.code for x in check.findings}


# ---------------------------------------------------------------------------
# unit: happy path over every check
# ---------------------------------------------------------------------------


def test_all_checks_report_with_sources(env: Env) -> None:
    env.sched.my_jobs = [job("100", "RUNNING", gpus=2), job("101", "PENDING", gpus=4, reason="Priority")]
    with env.ctx.db.tx() as tx:
        e = f.make_experiment(tx, cluster_id="fake")
        r = f.make_run(tx, e, f.make_plan(tx, e))
        f.make_attempt(tx, r, cluster_id="fake", job_id="100")
    data = cluster_inspect(env.ctx, cluster_id="fake")
    checks = by_name(data)
    assert [c.name for c in data.checks] == list(ALL_CHECKS)
    assert all(c.status in ("ok", "warn") for c in data.checks), {c.name: c.status for c in data.checks}
    assert checks["auth"].data["state"] == "ok" and checks["auth"].data["master_alive"] is True
    assert checks["identity"].status == "ok"
    sched = checks["scheduler"].data
    assert sched["capabilities"]["cluster_name"] == "fakecluster"
    assert sched["config"]["PrivateData"]["value"] == "none"
    assert sched["config"]["PrivateData"]["observed_at"] is not None
    storage = checks["storage"]
    assert {p["role"] for p in storage.data["paths"]} == {"remote_root", "scratch", "home"}
    assert all(p["capacity_pct"] == 50 for p in storage.data["paths"])
    assert checks["runtime"].data["tools"]["rsync"]["path"] == "/usr/bin/rsync"
    assert "uv_missing" in codes(checks["runtime"])
    mine = checks["queue"].data["mine"]
    assert (mine["running_gpus"], mine["pending_gpus"], mine["managed_jobs"], mine["foreign_jobs"]) == (
        2,
        4,
        1,
        1,
    )
    assert checks["queue"].data["account_now"]["visibility"] == "full"
    lo = next(p for p in checks["capacity"].data["partitions"] if p["name"] == "lo")
    assert lo["preempt_mode"] == "REQUEUE" and lo["grace_time_s"] == 120 and lo["preemptible"] is True
    assert "profile_partition_missing" in codes(checks["capacity"])  # "interactive" is not in sinfo
    assert checks["limits"].data["accounts"] == ["lab"] and "per_user_gpu_limit" in codes(checks["limits"])
    assert checks["nodes"].data["unavailable_total"] == 1
    assert checks["nodes"].data["unavailable"][0]["reason"] == "bad gpu"
    assert checks["profile_facts"].source == "site profile"
    assert data.coverage.complete and len(data.coverage.sources) == len(ALL_CHECKS)
    assert all(c.observed_at for c in data.checks)


def test_checks_subset_runs_only_what_was_asked(env: Env) -> None:
    data = cluster_inspect(env.ctx, cluster_id="fake", checks=["profile_facts", "capacity"])
    assert [c.name for c in data.checks] == ["profile_facts", "capacity"]
    assert "association_limits" not in env.sched.calls and "query_queue" not in env.sched.calls
    with pytest.raises(InvalidArgument):
        cluster_inspect(env.ctx, cluster_id="fake", candidate_shapes=[CandidateShape(partition="gpu")] * 4)


# ---------------------------------------------------------------------------
# auth classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "state", "remediation_part"),
    [
        (
            AuthRequiredError(
                "fake: SSH needs interactive authentication",
                details={"reason": "auth", "login_command_str": "ssh -o ControlMaster=auto login true"},
            ),
            "auth_required",
            "hpclab auth fake",
        ),
        (
            AuthRequiredError("host key", details={"reason": "host_key"}),
            "host_key_changed",
            "cluster admins",
        ),
        (
            ConnectionLost("ssh: connect to host login port 22: Operation timed out", maybe_executed=False),
            "vpn_unreachable",
            "VPN",
        ),
        (TransportTimeout("timed out"), "unknown", "did not answer"),
    ],
)
def test_auth_states_and_remediation(env: Env, error: Exception, state: str, remediation_part: str) -> None:
    env.login.id_error = error
    data = cluster_inspect(env.ctx, cluster_id="fake")
    checks = by_name(data)
    assert checks["auth"].data["state"] == state
    assert checks["auth"].status == ("unavailable" if state == "unknown" else "fail")
    assert data.status == "fail"
    assert any(remediation_part in r for r in data.remediation), data.remediation
    for name in ALL_CHECKS:
        if name not in ("auth", "profile_facts"):
            assert checks[name].status == "skipped" and "blocked_by_auth" in codes(checks[name])
    assert checks["profile_facts"].status == "ok"  # local, still useful
    assert not data.coverage.complete
    assert env.sched.calls == {}  # nothing remote was attempted after the failed round trip
    if state == "auth_required":
        assert "ssh -o ControlMaster=auto login true" in checks["auth"].findings[0].remediation


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_mismatch_is_a_blocking_finding(env: Env) -> None:
    env.login.user = "bob"
    data = cluster_inspect(env.ctx, cluster_id="fake", checks=["identity"])
    check = by_name(data)["identity"]
    assert check.status == "fail" and data.status == "fail"
    finding = check.findings[0]
    assert finding.code == "identity_mismatch" and finding.severity == "block"
    assert "--user bob" in (finding.remediation or "")
    assert finding.evidence == {"remote_user": "bob", "expected_user": "alice"}


def test_identity_without_overlay_user_warns(env: Env) -> None:
    env.register(user=None)
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["identity"]))["identity"]
    assert check.status == "warn" and "user_not_set" in codes(check)


# ---------------------------------------------------------------------------
# unavailable / unsupported / hidden
# ---------------------------------------------------------------------------


def _failed(msg: str, code: ErrorCode, **details: Any) -> CommandFailed:
    return CommandFailed(msg, CommandResult([], 1, "", msg, 0.0), code=code, details=details)


def test_hidden_qos_limits_are_unavailable_never_zero(env: Env) -> None:
    env.sched.errors["qos_limits"] = _failed(
        "sacctmgr: Access denied: PrivateData", ErrorCode.UNAVAILABLE, reason="access_denied"
    )
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["limits"]))["limits"]
    assert check.status == "warn" and "qos_limits_unavailable" in codes(check)
    assert check.data["qos"] is None  # unknown, not "no limits"
    assert "0" not in str(check.data["qos"])


def test_source_failures_map_to_unavailable_and_unsupported(env: Env) -> None:
    env.sched.errors["partitions"] = _failed(
        "sinfo: Unable to contact slurm controller", ErrorCode.UNAVAILABLE
    )
    env.sched.errors["association_limits"] = _failed(
        "not running a supported accounting_storage plugin", ErrorCode.UNSUPPORTED
    )
    env.sched.errors["nodes"] = _failed("sinfo failed", ErrorCode.UNAVAILABLE)
    env.sched.errors["cluster_config"] = _failed("scontrol failed", ErrorCode.UNAVAILABLE)
    data = cluster_inspect(env.ctx, cluster_id="fake")
    checks = by_name(data)
    assert checks["capacity"].status == "unavailable"
    assert checks["limits"].status == "unsupported"
    assert checks["nodes"].status == "unavailable" and checks["nodes"].data["cluster_nodes"] is None
    assert checks["scheduler"].status == "warn" and "config_unavailable" in codes(checks["scheduler"])
    assert checks["scheduler"].data["config"] is None
    assert not data.coverage.complete and data.status == "warn"
    statuses = {s.name: s.status for s in data.coverage.sources}
    assert statuses["capacity"] == "unavailable" and statuses["limits"] == "unsupported"


def test_accounting_off_and_gpu_tres_untracked_are_warnings(env: Env) -> None:
    env.sched.caps.sacct_available = False
    env.sched.caps.notes = ["AccountingStorageTRES does not track gres/gpu: job GPU counts are unknown"]
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["scheduler"]))["scheduler"]
    assert check.status == "warn"
    assert {"accounting_unavailable", "gpu_accounting_untracked"} <= codes(check)


def test_queue_gpu_counts_unknown_are_not_summed(env: Env) -> None:
    env.sched.my_jobs = [job("1", "RUNNING", gpus=None), job("2", "RUNNING", gpus=2)]
    env.sched.caps.private_data = "jobs"
    env.sched.account_jobs = [job("1", "RUNNING", gpus=None)]
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["queue"]))["queue"]
    mine = check.data["mine"]
    assert mine["running_gpus"] is None and mine["gpu_counts_complete"] is False
    acct = check.data["account_now"]
    assert acct["visibility"] == "self_only" and acct["running_gpus"] is None


def test_never_starting_pending_job_is_flagged(env: Env) -> None:
    env.sched.my_jobs = [job("7", "PENDING", reason="PartitionTimeLimit")]
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["queue"]))["queue"]
    assert check.status == "warn" and "pending_never_starts" in codes(check)


def test_storage_full_rsync_missing_and_avoided_remote_root(env: Env) -> None:
    env.login.capacity_pct = 99
    env.login.tools["rsync"] = ("", "")
    env.register(remote_root="/home/alice/hpclab")
    checks = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["storage", "runtime"]))
    storage = checks["storage"]
    assert storage.status == "fail" and "filesystem_full" in codes(storage)
    assert sum(1 for x in storage.findings if x.code == "filesystem_full") == 1  # one per filesystem
    assert "remote_root_on_avoided_storage" in codes(storage)
    assert checks["runtime"].status == "fail" and "rsync_missing" in codes(checks["runtime"])


def test_unresolvable_remote_root(env: Env) -> None:
    env.register(user=None, remote_root=None)
    check = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["storage"]))["storage"]
    assert check.status == "fail" and "remote_root_unresolved" in codes(check)


def test_nodes_and_facts_include_local_observations_and_expiry(env: Env) -> None:
    now = env.clock.now()
    with env.ctx.db.tx() as tx:
        tx.insert(
            "observations",
            {
                "cluster_id": "fake",
                "obs_key": "node:n1:nfs_stale:att_x",
                "value_json": '{"class": "nfs_stale"}',
                "source": "observe.finalize",
                "observed_at": iso(now),
                "expires_at": iso(now.replace(year=2027)),
            },
        )
    prof = env.ctx.clusters.get("fake").profile.model_copy(
        update={
            "facts": [
                SiteObservation(
                    key="driver_cuda", value="12.8", source="nvidia-smi", observed_at="2026-08-01T00:00:00Z"
                ),
                SiteObservation(
                    key="old",
                    value="x",
                    source="doc",
                    observed_at="2025-01-01T00:00:00Z",
                    expires_at="2026-01-01T00:00:00Z",
                ),
            ]
        }
    )
    env.ctx.clusters.register(prof, env.ctx.clusters.get("fake").overlay)
    checks = by_name(cluster_inspect(env.ctx, cluster_id="fake", checks=["nodes", "profile_facts"]))
    assert checks["nodes"].status == "warn" and "node_suspect" in codes(checks["nodes"])
    facts = {x["key"]: x for x in checks["profile_facts"].data["facts"]}
    assert facts["driver_cuda"]["expired"] is False and facts["driver_cuda"]["age_s"] > 0
    assert facts["old"]["expired"] is True and "fact_expired" in codes(checks["profile_facts"])


def test_deadline_skips_remaining_checks(env: Env) -> None:
    class SlowClock(FakeClock):
        def monotonic(self) -> float:
            self.advance(4.0)  # every check "takes" seconds
            return super().monotonic()

    env.ctx.clock = SlowClock()
    env.ctx.settings.request_deadline_s = 10.0
    data = cluster_inspect(env.ctx, cluster_id="fake")
    skipped = [c for c in data.checks if c.status == "skipped"]
    assert skipped and all("deadline" in codes(c) for c in skipped)
    assert not data.coverage.complete and {c.name for c in skipped} <= set(data.coverage.missing)
    assert by_name(data)["auth"].status == "ok"
    assert data.status != "ok"  # checks that never ran cannot vouch for the cluster


# ---------------------------------------------------------------------------
# candidate shapes
# ---------------------------------------------------------------------------


def test_candidate_shapes_are_cached_and_blocked_shapes_never_hit_sbatch(env: Env) -> None:
    shapes = [
        CandidateShape(partition="gpu", gpu_type="a100", gpus=2, time="04:00:00"),
        CandidateShape(partition="lo", gpu_type="h100", gpus=4, time="2-00:00:00"),  # > max time
        CandidateShape(partition="interactive", gpus=1),  # interactive-only
    ]
    for _ in range(5):
        data = cluster_inspect(env.ctx, cluster_id="fake", checks=["profile_facts"], candidate_shapes=shapes)
        env.clock.advance(10)
    assert env.sched.calls["test_only"] == 1  # 5 calls within a minute -> one scheduler RPC
    gpu, lo, inter = data.candidates
    assert gpu.status == "ok" and gpu.est_source == "test_only" and gpu.cached and gpu.confidence == "low"
    assert gpu.free_now == 8 and gpu.preempt_mode == "OFF"
    assert lo.status == "blocked" and any("exceeds" in b for b in lo.blockers) and lo.preemptible is True
    assert inter.status == "blocked" and any("interactive-only" in b for b in inter.blockers)
    env.clock.advance(400)  # past the 5 minute cache
    cluster_inspect(env.ctx, cluster_id="fake", checks=["profile_facts"], candidate_shapes=shapes[:1])
    assert env.sched.calls["test_only"] == 2


def test_candidate_refused_by_sbatch_is_blocked(env: Env) -> None:
    from hpclab.adapters.base import SubmissionRejected

    env.sched.errors["test_only"] = SubmissionRejected(
        "sbatch refused", details={"reason_code": "node_config_unavailable"}
    )
    data = cluster_inspect(
        env.ctx,
        cluster_id="fake",
        checks=["auth"],
        candidate_shapes=[CandidateShape(partition="gpu", gpus=64)],
    )
    (cand,) = data.candidates
    assert cand.status == "blocked" and cand.est_start is None and cand.est_source == "none"
    assert any("node_config_unavailable" in b for b in cand.blockers)


# ---------------------------------------------------------------------------
# sim-backed
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_all_checks_against_the_simulator(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with sim_context(tmp_path) as sim_env:
        sim = sim_env.sim
        now = time.time()
        managed = sim.inject_job(
            submit_time=now - 300, start_time=now - 200, gpus=2, state="RUNNING", name="hl.a.b"
        )
        sim.inject_job(submit_time=now - 300, start_time=now - 200, gpus=1, state="RUNNING", name="notebook")
        sim.inject_job(user="carol", submit_time=now - 300, start_time=now - 200, gpus=1, state="RUNNING")
        sim.set_node_state("sim-a100-04", "drain", "bad gpu")
        with sim_env.ctx.db.tx() as tx:
            e = f.make_experiment(tx, cluster_id="sim")
            r = f.make_run(tx, e, f.make_plan(tx, e))
            f.make_attempt(tx, r, cluster_id="sim", job_id=str(managed))
        shapes = [CandidateShape(partition="gpu", gpu_type="a100", gpus=2, time="01:00:00"),
                  CandidateShape(partition="gpu", gpus=64)]  # fmt: skip
        data = cluster_inspect(sim_env.ctx, cluster_id="sim", candidate_shapes=shapes)
        checks = by_name(data)
        assert [c.name for c in data.checks] == list(ALL_CHECKS)
        for c in data.checks:
            assert c.status in ("ok", "warn", "fail"), (c.name, c.status, c.summary)  # all answered
        assert checks["auth"].data["state"] == "ok" and checks["auth"].data["remote_user"] == sim.user
        assert checks["identity"].status == "ok"
        config = checks["scheduler"].data["config"]
        assert config["PrivateData"]["value"] == "none" and config["ClusterName"]["value"] == "simcluster"
        assert checks["scheduler"].data["capabilities"]["min_job_age_s"] == 1
        assert all(p.get("capacity_pct") is not None for p in checks["storage"].data["paths"])
        assert checks["runtime"].data["tools"]["python3"]["path"]
        mine = checks["queue"].data["mine"]
        assert (mine["running_jobs"], mine["managed_jobs"], mine["foreign_jobs"], mine["running_gpus"]) == (
            2,
            1,
            1,
            3,
        )
        acct = checks["queue"].data["account_now"]
        assert acct["visibility"] == "full" and acct["running_gpus"] == 4 and acct["users_visible"] == 2
        parts = {p["name"]: p for p in checks["capacity"].data["partitions"]}
        assert (
            set(parts) == {"gpu", "interactive", "scavenge"} and parts["scavenge"]["preempt_mode"] == "CANCEL"
        )
        assert parts["gpu"]["gpus"]["a100"]["unavailable"] == 4  # the drained node
        assert checks["limits"].data["accounts"] == ["lab", "other"]
        assert checks["nodes"].data["unavailable"][0]["name"] == "sim-a100-04"
        ok_shape, refused = data.candidates
        assert (
            ok_shape.status == "ok" and ok_shape.est_start is not None and ok_shape.est_source == "test_only"
        )
        assert refused.status == "blocked" and any("refused" in b for b in refused.blockers)
        assert len(sim.jobs()) == 3  # --test-only queued nothing


@pytest.mark.sim
def test_auth_required_fault_against_the_simulator(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with sim_context(tmp_path) as sim_env:
        sim_env.sim.set_faults(auth_required=True)
        data = cluster_inspect(sim_env.ctx, cluster_id="sim")
        checks = by_name(data)
        assert checks["auth"].status == "fail" and checks["auth"].data["state"] == "auth_required"
        assert data.status == "fail" and any("hpclab auth sim" in r for r in data.remediation)
        assert checks["capacity"].status == "skipped" and checks["queue"].status == "skipped"
        assert not data.coverage.complete
        sim_env.sim.clear_faults()
        again = cluster_inspect(sim_env.ctx, cluster_id="sim", checks=["auth", "limits"])
        assert by_name(again)["auth"].status == "ok" and by_name(again)["limits"].status == "ok"


@pytest.mark.sim
def test_accounting_disabled_simulator_is_reported_not_ok(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with sim_context(tmp_path, accounting_enabled=False) as sim_env:
        checks = by_name(cluster_inspect(sim_env.ctx, cluster_id="sim", checks=["scheduler", "limits"]))
        assert checks["scheduler"].status == "warn" and "accounting_unavailable" in codes(checks["scheduler"])
        assert checks["limits"].status == "unsupported"
        assert isinstance(sim_env.scheduler, SlurmScheduler)


# ---------------------------------------------------------------------------
# time budget: a hung slurmctld must not stretch the call past the request deadline
# ---------------------------------------------------------------------------

_LOCAL_ONLY = (["id", "-un"], ["sbatch", "--version"], ["squeue", "--help"], ["sacct", "--helpformat"],
               ["sstat", "--helpformat"], ["srun", "--version"], ["quota", "-s"])  # fmt: skip


class _HungController:
    """Login node answers; everything that needs slurmctld/slurmdbd hangs until the timeout."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.timeouts: list[tuple[str, float, float]] = []  # (command, timeout, budget left)

    def remote_user(self) -> str:
        return "alice"

    def _cmd(self, argv: list[str]) -> list[str]:
        return argv[2:] if argv[:1] == ["env"] else argv

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        cmd = self._cmd(list(argv))
        self.timeouts.append((" ".join(cmd[:2]), timeout, self.clock.monotonic()))
        if cmd == ["id", "-un"]:
            return CommandResult(argv, 0, "alice\n", "", 0.01)
        if any(cmd[: len(x)] == x for x in _LOCAL_ONLY):
            return CommandResult(argv, 0, "slurm 23.02.7\n", "", 0.01)
        self.clock.advance(timeout)  # hangs until the transport gives up
        raise TransportTimeout(f"{cmd[0]} timed out after {timeout:.0f}s")

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        return CommandResult(["bash"], 0, "", "", 0.01)  # storage/runtime snippets: quick, empty

    def close(self) -> None:
        return None


def test_hung_controller_cannot_stretch_the_call_past_the_deadline(paths) -> None:  # type: ignore[no-untyped-def]
    from cluster_fakes import OVERLAY, profile

    from hpclab.services.context import ServiceContext, Settings

    clock = FakeClock()
    login = _HungController(clock)
    ctx = ServiceContext.create(
        paths,
        clock=clock,
        transport_factory=lambda cluster, p: login,  # type: ignore[arg-type,return-value]
        scheduler_factory=lambda t, c, clk: SlurmScheduler(t, c, clock=clk),
        settings=Settings(request_deadline_s=20.0),
    )
    try:
        from hpclab.profiles.schema import UserOverlay

        ctx.clusters.register(profile(), UserOverlay(cluster_id="fake", **OVERLAY))
        start = clock.monotonic()
        data = cluster_inspect(
            ctx, cluster_id="fake", candidate_shapes=[CandidateShape(partition="gpu", gpus=2)]
        )
        elapsed = clock.monotonic() - start
        assert elapsed <= 20.0, (elapsed, login.timeouts)  # bounded by settings.request_deadline_s
        for c in data.checks:  # nothing that needed the controller is reported as fine
            if c.name in ("queue", "capacity", "limits", "nodes"):
                assert c.status in ("unavailable", "skipped"), (c.name, c.status)
        assert not data.coverage.complete
        (cand,) = data.candidates
        assert cand.status == "unavailable" and cand.est_start is None
    finally:
        ctx.db.close()


@pytest.mark.sim
def test_inspect_refresh_reaches_the_shared_scheduler(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with sim_context(tmp_path) as sim_env:
        shared = sim_env.scheduler
        assert isinstance(shared, SlurmScheduler)
        before = shared.probe_capabilities()
        cluster_inspect(sim_env.ctx, cluster_id="sim", checks=["scheduler"])
        after = shared.probe_capabilities()
        assert after is not before and after.cluster_name == "simcluster"  # the doctor's re-probe is kept


def test_hidden_sub_sources_make_coverage_partial_not_complete(env: Env) -> None:
    env.sched.errors["qos_limits"] = _failed(
        "sacctmgr: Access denied: PrivateData", ErrorCode.UNAVAILABLE, reason="access_denied"
    )
    env.sched.errors["cluster_config"] = _failed("scontrol failed", ErrorCode.UNAVAILABLE)
    env.sched.errors["query_account_queue"] = _failed("squeue --account failed", ErrorCode.UNAVAILABLE)
    data = cluster_inspect(env.ctx, cluster_id="fake", checks=["scheduler", "queue", "limits"])
    statuses = {s.name: s.status for s in data.coverage.sources}
    assert statuses == {"scheduler": "partial", "queue": "partial", "limits": "partial"}, statuses
    assert not data.coverage.complete
    queue = by_name(data)["queue"]
    assert queue.data["account_now"]["visibility"] == "unavailable"
    assert "account_queue_unavailable" in codes(queue)


def test_slurm_commands_missing_on_path_is_a_failure_with_remediation(env: Env) -> None:
    env.sched.errors["probe_capabilities"] = _failed(
        "sbatch --version failed: bash: sbatch: command not found",
        ErrorCode.UNSUPPORTED,
        command="sbatch --version",
        exit_code=127,
    )
    data = cluster_inspect(env.ctx, cluster_id="fake", checks=["auth", "scheduler"])
    check = by_name(data)["scheduler"]
    assert check.status == "fail" and data.status == "fail"
    assert "slurm_commands_missing" in codes(check)
    assert any("module load" in r for r in data.remediation), data.remediation
    assert not data.coverage.complete
