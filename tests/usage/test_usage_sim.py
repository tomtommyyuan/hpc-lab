"""usage_report through the real SlurmScheduler against the simulator (sacct -X -D)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hpclab.services.usage import usage_report
from hpclab.testing.simctx import sim_context

pytestmark = pytest.mark.sim
H = 3600.0


def test_requeue_segments_typed_tres_and_running_job(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        now = time.time()
        # AllocTRES in the simulator is "gres/gpu:h100=4,gres/gpu=4": 4 GPUs, never 8.
        env.sim.inject_job(
            submit_time=now - 4 * H,
            start_time=now - 2 * H,
            end_time=now - 1 * H,
            gpus=4,
            gpu_type="h100",
            partition="scavenge",
            state="COMPLETED",
            segments=[
                {
                    "start": now - 3 * H,
                    "end": now - 2.5 * H,
                    "state": "PREEMPTED",
                    "node_list": ["sim-h100-02"],
                }
            ],
        )
        env.sim.inject_job(
            user="bob", submit_time=now - 1.5 * H, start_time=now - 1 * H, gpus=2, state="RUNNING"
        )
        data = usage_report(
            env.ctx, cluster_id="sim", scope="account", since="1d", group_by=["user", "state"]
        )
        total = data.totals["account"]
        # 4 GPUs x 0.5 h (preempted segment) + 4 x 1 h (completed) + 2 x ~1 h (running, charged to now)
        assert total.gpu_hours == pytest.approx(8.0, abs=0.05)
        assert total.gpu_hours_by_state["PREEMPTED"] == pytest.approx(2.0, abs=0.01)
        assert total.gpu_hours_by_state["COMPLETED"] == pytest.approx(4.0, abs=0.01)
        assert total.gpu_hours_by_state["RUNNING"] == pytest.approx(2.0, abs=0.05)
        assert total.wasted_gpu_hours_by_cause == {"preempted": pytest.approx(2.0, abs=0.01)}
        assert total.jobs == 2 and data.visibility == "full"
        rows = {(r.group["user"], r.group["state"]): r for r in data.rows}
        assert set(rows) == {(env.sim.user, "PREEMPTED"), (env.sim.user, "COMPLETED"), ("bob", "RUNNING")}


def test_private_data_jobs_hides_others_and_drops_the_account_total(tmp_path: Path) -> None:
    with sim_context(tmp_path, private_data="jobs") as env:
        now = time.time()
        env.sim.inject_job(user="alice", submit_time=now - 3 * H, start_time=now - 2 * H, end_time=now - H, gpus=8,
                           gpu_type="h100", partition="scavenge")  # fmt: skip
        env.sim.inject_job(submit_time=now - 3 * H, start_time=now - 2 * H, end_time=now - H, gpus=1)
        data = usage_report(env.ctx, cluster_id="sim", scope="account", since="1d")
        assert data.visibility == "self_only" and data.private_data == "jobs"
        assert set(data.totals) == {"self"}  # no account-total field at all
        assert data.totals["self"].gpu_hours == pytest.approx(1.0, abs=0.01)
        assert not data.coverage.complete
        assert {r.group["user"] for r in data.rows} == {env.sim.user}


def test_second_call_reads_only_the_live_tail(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        now = time.time()
        for days_ago, user in ((2, None), (3, "bob")):  # another user visible: an account total
            start = now - days_ago * 24 * H
            env.sim.inject_job(
                user=user, submit_time=start - 60, start_time=start, end_time=start + 2 * H, gpus=2
            )
        first = usage_report(env.ctx, cluster_id="sim", scope="account", since="4d")
        assert first.details.sacct_calls == 2 and first.details.cached_days == 0
        second = usage_report(env.ctx, cluster_id="sim", scope="account", since="4d")
        assert second.details.sacct_calls == 1 and second.details.cached_days >= 4
        assert len(second.details.queried_windows) == 1  # 0 sacct calls for finalized days
        assert second.totals["account"].gpu_hours == pytest.approx(first.totals["account"].gpu_hours)
        assert second.totals["account"].gpu_hours == pytest.approx(8.0, abs=0.01)
