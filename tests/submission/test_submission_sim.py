"""Fault matrix against the Slurm simulator.

Every assertion about "how many jobs exist" uses the simulator's ground truth
(``SimCluster.jobs()`` = what sbatch really accepted), never hpclab's own ledger. The service
runs unmodified: real ``SlurmScheduler`` + ``SimTransport`` via ``hpclab.testing.simctx``; a
``FakeClock`` synced to wall time drives reconcile windows without sleeping.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from submission_support import (
    Seeded,
    SimulatedCrash,
    attempt,
    drain,
    expire_leases,
    operation,
    reservation,
    run_row,
    seed_run,
    write_project,
)

from hpclab.domain.clock import FakeClock
from hpclab.domain.models import CodeOptions, CodeSnapshotRef
from hpclab.services import submission
from hpclab.services.snapshot import SnapshotStore, capture_snapshot
from hpclab.services.submission import attempt_resolve, run_cancel, run_submit
from hpclab.testing.simctx import SimEnv, sim_context, wait_until

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]

SIM_WINDOW_S = 31.0  # sim profile: reconcile_window_s=30 + accounting_lag_s=max(1, lag)


@dataclass
class Harness:
    env: SimEnv
    clock: FakeClock
    code: CodeSnapshotRef
    tmp: Path

    @property
    def ctx(self) -> Any:
        return self.env.ctx

    def seed(self, *, sleep_s: float = 0.0, **kw: Any) -> Seeded:
        return seed_run(
            self.ctx,
            cluster_id=self.env.cluster_id,
            remote_root=self.env.remote_root,
            code=self.code,
            argv=[sys.executable, "job.py", str(sleep_s)],
            profile_revision=self.env.cluster.revision,
            time="00:10:00",
            **kw,
        )

    def submit(self, s: Seeded, key: str | None = None) -> str:
        r = run_submit(
            self.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key=key or s.run_id
        )
        return r.attempt_id

    def step(self, seconds: float = 0.0) -> None:
        if seconds:
            self.clock.advance(seconds)
        drain(self.ctx)

    def jobs(self) -> list[dict[str, Any]]:
        return self.env.sim.jobs()

    def inject_job(self, name: str, comment: str | None = None, sleep_s: int = 30) -> str:
        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            "-A",
            "lab",
            "-p",
            "gpu",
            f"--wrap=sleep {sleep_s}",
        ]
        if comment:
            argv.insert(2, f"--comment={comment}")
        res = self.env.sim.run(*argv, cwd=self.tmp)
        assert res.returncode == 0, res.stderr
        return res.stdout.strip()

    def count_scancel(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []
        sched = self.env.scheduler
        real = sched.cancel

        def counting(job_id: str) -> Any:
            calls.append(job_id)
            return real(job_id)

        monkeypatch.setattr(sched, "cancel", counting)
        return calls


@contextmanager
def harness(tmp_path: Path, **sim_config: Any) -> Iterator[Harness]:
    clock = FakeClock(start=datetime.now(UTC))
    with sim_context(tmp_path, clock=clock, **sim_config) as env:
        store = SnapshotStore(env.ctx.paths.snapshots)
        ref = capture_snapshot(write_project(tmp_path / "project"), CodeOptions(), store).ref
        yield Harness(env=env, clock=clock, code=ref, tmp=tmp_path)


def test_happy_path_submits_exactly_one_traceable_job(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        s = h.seed()
        aid = h.submit(s)
        h.step()
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing"
        jobs = h.jobs()
        assert len(jobs) == 1
        assert str(jobs[0]["job_id"]) == a["job_id"]
        assert jobs[0]["name"] == a["job_name"] and jobs[0]["comment"] == f"hpclab:{a['submit_token']}"
        assert jobs[0]["user"] == a["job_owner"]
        plan_doc = json.loads(Path(a["remote_dir"], "hpclab-plan.json").read_text())
        assert plan_doc["plan_hash"] == s.exec_plan_hash
        assert Path(a["remote_dir"], "submit.receipt").read_text().splitlines()[0] == "0"
        assert Path(s.plan.snapshot_path, ".hpclab-complete").exists()
        done = h.env.sim.wait_for(a["job_id"], {"COMPLETED", "FAILED"}, 20)
        assert done is not None and done["state"] == "COMPLETED", Path(
            a["remote_dir"], "logs/stderr.log"
        ).read_text()


def test_lost_reply_after_accept_is_bound_from_the_receipt(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_lose_reply=1)
        aid = h.submit(h.seed(sleep_s=5))
        h.step()
        assert attempt(h.ctx, aid)["phase"] == "submission_uncertain"
        assert len(h.jobs()) == 1  # the job WAS accepted
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing" and a["job_id"] == str(h.jobs()[0]["job_id"])
        op = operation(h.ctx, a["operation_id"])
        assert json.loads(op["result_json"])["source"] == "reconcile:receipt+squeue"
        assert len(h.jobs()) == 1


def test_lost_reply_without_receipt_is_found_in_squeue(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_lose_reply=1)
        aid = h.submit(h.seed(sleep_s=5))
        h.step()
        a = attempt(h.ctx, aid)
        os.remove(Path(a["remote_dir"], "submit.receipt"))  # journal lost too
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing" and a["job_id"] == str(h.jobs()[0]["job_id"])
        assert json.loads(operation(h.ctx, a["operation_id"])["result_json"])["source"] == "reconcile:squeue"
        assert len(h.jobs()) == 1


def test_ambiguous_sbatch_error_that_queued_a_job_is_reconciled(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_ambiguous_error=1)  # "Socket timed out" after queuing
        aid = h.submit(h.seed(sleep_s=5))
        h.step()
        assert attempt(h.ctx, aid)["phase"] == "submission_uncertain"
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing" and len(h.jobs()) == 1
        assert a["job_id"] == str(h.jobs()[0]["job_id"])


def test_ambiguous_error_without_a_job_ends_unresolved_and_is_never_resubmitted(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_ambiguous_error=1, sbatch_ambiguous_creates_job=False)
        s = h.seed()
        aid = h.submit(s)
        h.step()
        h.step(3)
        assert attempt(h.ctx, aid)["phase"] == "submission_uncertain"
        h.step(SIM_WINDOW_S + 10)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "submission_uncertain"
        op = operation(h.ctx, a["operation_id"])
        assert op["state"] == "blocked" and op["stage"] == "unresolved"
        assert json.loads(op["error_json"])["details"]["sources"]["squeue"] == "ok"
        assert h.jobs() == []
        assert run_row(h.ctx, s.run_id)["claim_attempt_id"] == aid
        assert reservation(h.ctx, aid)["state"] == "reserved"
        h.step(120)
        assert h.jobs() == []  # still nothing: never resubmitted
        data = attempt_resolve(
            h.ctx,
            attempt_id=aid,
            expected_version=attempt(h.ctx, aid)["version"],
            resolution="abandon",
            reason="squeue/sacct show nothing",
            idempotency_key="abandon",
        )
        assert data.phase == "abandoned" and run_row(h.ctx, s.run_id)["claim_attempt_id"] is None


def test_duplicate_job_with_the_same_token_is_detected_not_cancelled(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_lose_reply=1)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        a = attempt(h.ctx, aid)
        dup = h.inject_job(a["job_name"], comment=f"hpclab:{a['submit_token']}")
        assert len(h.jobs()) == 2
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "duplicate_detected"
        op = operation(h.ctx, a["operation_id"])
        cands = json.loads(op["error_json"])["details"]["candidates"]
        assert {c["job_id"] for c in cands} == {str(j["job_id"]) for j in h.jobs()}
        assert dup in {c["job_id"] for c in cands}
        assert all(j["state"] in ("PENDING", "RUNNING") for j in h.jobs())  # nothing auto-cancelled


@pytest.mark.parametrize(
    ("stderr", "reason_code", "retry_later"),
    [
        (
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
            "invalid_account",
            False,
        ),
        ("sbatch: error: Batch job submission failed: Invalid qos specification", "invalid_qos", False),
        (
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
            "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
            "(job submit limit, user's size and/or time limits)",
            "submit_limit",
            True,
        ),
    ],
    ids=["invalid_account", "invalid_qos", "submit_limit"],
)
def test_definite_rejection_releases_claim_and_budget(
    tmp_path: Path, stderr: str, reason_code: str, retry_later: bool
) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_reject=stderr)
        s = h.seed()
        aid = h.submit(s)
        h.step()
        a = attempt(h.ctx, aid)
        assert a["phase"] == "submission_rejected"
        assert h.jobs() == []
        assert run_row(h.ctx, s.run_id)["claim_attempt_id"] is None
        assert reservation(h.ctx, aid)["state"] == "released"
        details = json.loads(operation(h.ctx, a["operation_id"])["error_json"])["details"]
        assert details["reason_code"] == reason_code and details["retry_later"] is retry_later


def test_crash_after_in_flight_commit_before_sbatch_never_resubmits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        sched = h.env.scheduler
        real_submit = sched.submit
        calls: list[str] = []

        def crash_before_call(script: str, **kw: Any) -> Any:
            calls.append(script)
            raise SimulatedCrash("controller killed right before sbatch")

        monkeypatch.setattr(sched, "submit", crash_before_call)
        aid = h.submit(h.seed())
        with pytest.raises(SimulatedCrash):
            h.step()
        assert attempt(h.ctx, aid)["phase"] == "submission_in_flight"
        monkeypatch.setattr(sched, "submit", real_submit)
        expire_leases(h.ctx)  # controller restart
        h.step()
        assert attempt(h.ctx, aid)["phase"] == "submission_uncertain"
        h.step(3)
        h.step(SIM_WINDOW_S + 10)
        assert operation(h.ctx, attempt(h.ctx, aid)["operation_id"])["stage"] == "unresolved"
        assert h.jobs() == []  # the re-entry never called sbatch
        assert len(calls) == 1


def test_crash_after_sbatch_accepted_before_outcome_commit_binds_the_one_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        sched = h.env.scheduler
        real_submit = sched.submit

        def crash_after_accept(script: str, **kw: Any) -> Any:
            real_submit(script, **kw)
            raise SimulatedCrash("controller killed before recording the job id")

        monkeypatch.setattr(sched, "submit", crash_after_accept)
        aid = h.submit(h.seed(sleep_s=5))
        with pytest.raises(SimulatedCrash):
            h.step()
        assert len(h.jobs()) == 1
        expire_leases(h.ctx)
        h.step()
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing" and a["job_id"] == str(h.jobs()[0]["job_id"])
        assert len(h.jobs()) == 1


def test_crash_before_in_flight_commit_submits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        real = submission._transition
        crashed: list[bool] = []

        def crashing(tx: Any, row: Any, dst: Any, *a: Any, **kw: Any) -> Any:
            if dst == submission.AttemptPhase.SUBMISSION_IN_FLIGHT and not crashed:
                crashed.append(True)
                raise SimulatedCrash("killed before the intent commit")
            return real(tx, row, dst, *a, **kw)

        monkeypatch.setattr(submission, "_transition", crashing)
        aid = h.submit(h.seed())
        with pytest.raises(SimulatedCrash):
            h.step()
        assert attempt(h.ctx, aid)["phase"] == "staging" and h.jobs() == []
        expire_leases(h.ctx)
        h.step()
        assert attempt(h.ctx, aid)["phase"] == "observing"
        assert len(h.jobs()) == 1


def test_concurrent_run_submit_with_different_keys_makes_one_sim_job(tmp_path: Path) -> None:
    import threading

    from hpclab.domain.errors import AdmissionDenied

    with harness(tmp_path) as h:
        s = h.seed()
        from submission_support import add_plan

        plans = [s.exec_plan_hash, *(add_plan(h.ctx, s, nonce=f"n{i}") for i in range(3))]
        barrier = threading.Barrier(len(plans))
        outcomes: list[object] = []

        def worker(i: int) -> None:
            barrier.wait()
            try:
                outcomes.append(
                    run_submit(h.ctx, run_id=s.run_id, execution_plan_hash=plans[i], idempotency_key=f"k{i}")
                )
            except AdmissionDenied as exc:
                outcomes.append(exc)
            finally:
                h.ctx.db.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(plans))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert sum(isinstance(o, submission.SubmitReceiptData) for o in outcomes) == 1
        h.step()
        assert len(h.jobs()) == 1


def test_same_key_replay_returns_the_same_receipt_and_one_job(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        s = h.seed()
        first = run_submit(
            h.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="same"
        )
        h.step()
        again = run_submit(
            h.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="same"
        )
        h.step()
        assert again == first and len(h.jobs()) == 1


def test_cancel_verified_job_is_cancelled_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with harness(tmp_path) as h:
        scancels = h.count_scancel(monkeypatch)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        a = attempt(h.ctx, aid)
        assert h.env.sim.wait_for(a["job_id"], {"RUNNING"}, 10) is not None
        run_cancel(h.ctx, attempt_id=aid, expected_version=a["version"], idempotency_key="c", reason="bad lr")
        h.step()
        assert scancels == [a["job_id"]]
        assert attempt(h.ctx, aid)["cancel_state"] == "sent"
        done = h.env.sim.wait_for(a["job_id"], {"CANCELLED"}, 10)
        assert done is not None


def test_cancel_refused_when_the_cluster_user_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        scancels = h.count_scancel(monkeypatch)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        a = attempt(h.ctx, aid)
        assert h.env.sim.wait_for(a["job_id"], {"RUNNING"}, 10) is not None
        monkeypatch.setattr(h.env.transport, "remote_user", lambda: "someone-else")
        run_cancel(h.ctx, attempt_id=aid, expected_version=a["version"], idempotency_key="c")
        h.step()
        assert scancels == []
        assert attempt(h.ctx, aid)["cancel_state"] == "refused"
        assert h.env.sim.job(a["job_id"])["state"] == "RUNNING"  # type: ignore[index]


def test_cancel_refused_when_the_job_id_now_names_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        scancels = h.count_scancel(monkeypatch)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        foreign = h.inject_job("train-baseline")  # the user's own non-hpclab job
        wait_until(lambda: h.env.sim.job(foreign)["state"] == "RUNNING", timeout=10)  # type: ignore[index]
        with h.ctx.db.tx() as tx:  # the ledger's job id now points at a different job
            tx.execute("UPDATE attempts SET job_id=? WHERE attempt_id=?", (foreign, aid))
        a = attempt(h.ctx, aid)
        run_cancel(h.ctx, attempt_id=aid, expected_version=a["version"], idempotency_key="c")
        h.step()
        assert scancels == []
        assert attempt(h.ctx, aid)["cancel_state"] == "refused"
        assert h.env.sim.job(foreign)["state"] == "RUNNING"  # type: ignore[index]
        ev = h.ctx.db.one(
            "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.cancel_refused'",
            (aid,),
        )
        assert ev is not None
        problems = json.loads(ev["payload_json"])["problems"]
        assert any("token" in p for p in problems)


def test_cancel_before_submit_sends_nothing(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        s = h.seed()
        aid = h.submit(s)
        receipt = run_cancel(h.ctx, attempt_id=aid, expected_version=1, idempotency_key="c")
        assert receipt.phase == "cancelled_before_submit"
        h.step()
        assert h.jobs() == []
        assert reservation(h.ctx, aid)["state"] == "released"


def test_cancel_while_uncertain_waits_for_reconcile_then_cancels_the_one_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with harness(tmp_path) as h:
        scancels = h.count_scancel(monkeypatch)
        h.env.sim.set_faults(sbatch_lose_reply=1)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        a = attempt(h.ctx, aid)
        assert a["phase"] == "submission_uncertain"
        receipt = run_cancel(h.ctx, attempt_id=aid, expected_version=a["version"], idempotency_key="c")
        assert receipt.cancel_state == "requested" and scancels == []
        h.step(3)
        a = attempt(h.ctx, aid)
        assert a["phase"] == "observing" and a["cancel_state"] == "sent"
        assert scancels == [a["job_id"]]
        assert len(h.jobs()) == 1
        assert h.env.sim.wait_for(a["job_id"], {"CANCELLED"}, 10) is not None


def test_auth_required_before_dispatch_blocks_then_submits_once(tmp_path: Path) -> None:
    with harness(tmp_path) as h:
        h.env.sim.set_faults(auth_required=True)
        aid = h.submit(h.seed())
        h.step()
        a = attempt(h.ctx, aid)
        assert a["phase"] == "staging"
        op = operation(h.ctx, a["operation_id"])
        assert (
            op["state"] == "blocked"
            and json.loads(op["error_json"])["details"]["remediation"] == "hpclab auth sim"
        )
        assert h.jobs() == []
        h.env.sim.clear_faults()
        h.step(121)
        assert attempt(h.ctx, aid)["phase"] == "observing"
        assert len(h.jobs()) == 1


def test_a_finished_duplicate_that_left_squeue_is_still_detected(tmp_path: Path) -> None:
    """Review regression (invariant 4.3): a same-token job that already finished and aged out of
    squeue is only visible in sacct. Reconcile must see both jobs, not bind the pending one."""
    with harness(tmp_path) as h:
        h.env.sim.set_faults(sbatch_lose_reply=1)
        aid = h.submit(h.seed(sleep_s=30))
        h.step()
        a = attempt(h.ctx, aid)
        assert a["phase"] == "submission_uncertain"
        dup = h.inject_job(a["job_name"], comment=f"hpclab:{a['submit_token']}", sleep_s=0)
        assert h.env.sim.wait_for(dup, {"COMPLETED"}, 15) is not None
        h.env.scheduler.probe_capabilities()
        wait_until(
            lambda: dup not in {o.job_id for o in h.env.scheduler.query_queue(user=a["job_owner"]).jobs},
            timeout=15,
            what="the finished duplicate to leave squeue",
        )
        h.step(3)
        b = attempt(h.ctx, aid)
        assert b["phase"] == "duplicate_detected", b["phase"]
        cands = json.loads(operation(h.ctx, b["operation_id"])["error_json"])["details"]["candidates"]
        assert {c["job_id"] for c in cands} == {str(j["job_id"]) for j in h.jobs()}
        assert len(h.jobs()) == 2
