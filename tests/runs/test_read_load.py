"""Load the read side puts on the login node and the controller.

run_inspect re-reads an attempt's files (metrics/telemetry tails, heartbeat, mtimes: several ssh
execs per call on a real cluster) at most once per poll interval unless asked to refresh, and
concurrent inspects of one attempt share a read. runs_list counts its whole scope in bounded
batches instead of holding every run's attempt rows (with their JSON blobs) at once.
"""

from __future__ import annotations

import threading
import tracemalloc
from typing import Any

import pytest
from observe_support import CLUSTER, Env, observation, seed

from hpclab.adapters.base import CommandFailed, CommandResult, RangeRead
from hpclab.services import runs as runs_mod
from hpclab.services.runs import run_inspect, runs_list
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_attempt, make_evaluation, make_experiment, make_plan, make_run

POLL_S = 30.0  # observe_support.profile(): poll_min_interval_s


def _remote_ops(env: Env) -> list[tuple[str, str]]:
    return [c for c in env.transport.calls if c[0] in ("read_range", "read_tail", "stat")]


def _running(env: Env) -> tuple[str, str]:
    run_id, aid = seed(env, elapsed_s=500, time_limit_s=3600, last_observed_at=env.now(), gpus=2)
    d = env.attempt(aid)["remote_dir"]
    now = env.clock.now().timestamp()
    lines = "".join(
        jdump({"schema": "hpclab.metrics/v1", "step": s, "total_steps": 1000, "time": now - 500 + s}) + "\n"
        for s in range(0, 450, 50)
    )
    env.transport.put(f"{d}/outputs/metrics.jsonl", lines, mtime=now - 5)
    env.transport.put(f"{d}/logs/stdout.log", "ok\n", mtime=now - 5)
    env.transport.put(
        f"{d}/telemetry/gpu.csv", "t, 0, G0, 99, 1000, 81920, 300\nt, 1, G1, 0, 10, 81920, 60\n" * 5
    )
    env.transport.put(f"{d}/telemetry/heartbeat", f"{int(now) - 20}\n")
    return run_id, aid


def _source(data: Any, name: str) -> Any:
    found = [s for s in data.coverage.sources if s.name == name]
    assert len(found) == 1, data.coverage.sources
    return found[0]


# ---------------------------------------------------------------------------
# run_inspect: live reads are cached for a poll interval
# ---------------------------------------------------------------------------


def test_inspect_without_refresh_reuses_a_recent_live_read(env: Env) -> None:
    run_id, _ = _running(env)
    first = run_inspect(env.ctx, run_id=run_id)
    assert len(_remote_ops(env)) == 4
    assert _source(first, "metrics").status == "ok"
    read_at = env.now()
    for _ in range(5):  # an agent "waiting for the job" by polling inspect
        env.clock.advance(2)
        env.transport.calls.clear()
        again = run_inspect(env.ctx, run_id=run_id)
        assert _remote_ops(env) == []
        assert again.progress is not None and again.progress.step == 400
        assert again.health is not None and again.health.flags == ["partial_gpu_use"]
        metrics, telemetry = _source(again, "metrics"), _source(again, "telemetry")
        assert metrics.status == "cached" and metrics.observed_at == read_at
        assert telemetry.status == "cached" and telemetry.observed_at == read_at
        assert metrics.detail is not None and "refresh=true" in metrics.detail


def test_the_live_read_is_repeated_after_a_poll_interval(env: Env) -> None:
    run_id, _ = _running(env)
    run_inspect(env.ctx, run_id=run_id)
    env.clock.advance(POLL_S + 1)
    env.transport.calls.clear()
    data = run_inspect(env.ctx, run_id=run_id)
    assert len(_remote_ops(env)) == 4
    assert _source(data, "metrics").status == "ok" and _source(data, "metrics").observed_at == env.now()


def test_refresh_reads_the_files_now(env: Env) -> None:
    run_id, aid = _running(env)
    run_inspect(env.ctx, run_id=run_id)
    env.clock.advance(1)
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=501, time_limit_s=3600)]
    env.transport.calls.clear()
    data = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert len(_remote_ops(env)) == 4
    assert _source(data, "metrics").status == "ok" and _source(data, "metrics").observed_at == env.now()


def test_a_state_change_invalidates_the_recent_live_read(env: Env) -> None:
    """The job finished since the last read: its final progress is read, not the running one."""
    run_id, aid = _running(env)
    run_inspect(env.ctx, run_id=run_id)
    d = env.attempt(aid)["remote_dir"]
    env.transport.append(
        f"{d}/outputs/metrics.jsonl",
        jdump({"schema": "hpclab.metrics/v1", "step": 1000, "total_steps": 1000, "final": True}) + "\n",
    )
    with env.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET phase='terminal', scheduler_state='COMPLETED', node_list='n1' WHERE attempt_id=?",
            (aid,),
        )
    env.clock.advance(2)
    env.transport.calls.clear()
    data = run_inspect(env.ctx, run_id=run_id)
    assert len(_remote_ops(env)) == 1  # metrics only: health is for running jobs
    assert data.progress is not None and data.progress.step == 1000


def test_a_recent_read_replays_what_it_found(env: Env) -> None:
    """A missing metrics file / an unreadable one stays disclosed while the read is reused."""
    run_id, aid = seed(env, elapsed_s=50, time_limit_s=3600, last_observed_at=env.now())
    run_inspect(env.ctx, run_id=run_id)
    env.clock.advance(2)
    env.transport.calls.clear()
    data = run_inspect(env.ctx, run_id=run_id)
    assert _remote_ops(env) == []
    metrics = _source(data, "metrics")
    assert (
        metrics.status == "cached" and metrics.detail is not None and "no metrics file yet" in metrics.detail
    )

    env.clock.advance(POLL_S + 1)
    env.transport.put(f"{env.attempt(aid)['remote_dir']}/outputs/metrics.jsonl", '{"step": 1}\n')
    env.transport.fail["read_range"] = CommandFailed("down", CommandResult(["x"], 1, "", "", 0.1))
    first = run_inspect(env.ctx, run_id=run_id)
    assert _source(first, "metrics").status == "unavailable"
    env.clock.advance(2)
    env.transport.calls.clear()
    again = run_inspect(env.ctx, run_id=run_id)
    assert _remote_ops(env) == []
    assert _source(again, "metrics").status == "unavailable" and not again.coverage.complete


def test_an_unreachable_cluster_is_not_cached_as_a_read(env: Env) -> None:
    from hpclab.adapters.base import TransportTimeout

    run_id, _ = _running(env)
    env.transport.fail["read_tail"] = TransportTimeout("read timed out")
    run_inspect(env.ctx, run_id=run_id)
    env.transport.fail.clear()
    env.clock.advance(2)
    env.transport.calls.clear()
    data = run_inspect(env.ctx, run_id=run_id)
    assert len(_remote_ops(env)) == 4 and _source(data, "metrics").status == "ok"


def test_expired_live_read_records_are_dropped(env: Env) -> None:
    run_a, _ = _running(env)
    run_inspect(env.ctx, run_id=run_a)
    env.clock.advance(POLL_S + 1)
    run_b, aid_b = _running(env)
    run_inspect(env.ctx, run_id=run_b)
    keys = [
        r["obs_key"]
        for r in env.ctx.db.all(
            "SELECT obs_key FROM observations WHERE cluster_id=? AND obs_key LIKE 'live:%'", (CLUSTER,)
        )
    ]
    assert keys == [f"live:{aid_b}"]


def test_concurrent_inspects_of_one_attempt_share_the_live_read(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, _ = _running(env)
    entered, release = threading.Event(), threading.Event()
    real_tail = env.transport.read_tail

    def slow_tail(path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        entered.set()
        release.wait(10)
        return real_tail(path, length, timeout=timeout)

    monkeypatch.setattr(env.transport, "read_tail", slow_tail)
    results: dict[str, Any] = {}

    def first() -> None:
        results["first"] = run_inspect(env.ctx, run_id=run_id)

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(10)
    second: dict[str, Any] = {}

    def other() -> None:
        second["data"] = run_inspect(env.ctx, run_id=run_id)

    t2 = threading.Thread(target=other)
    t2.start()
    threading.Event().wait(0.2)  # the second request is waiting for the first read
    release.set()
    t.join(10)
    t2.join(10)
    assert len(_remote_ops(env)) == 4, env.transport.calls  # one live read, not two
    data = second["data"]
    assert data.progress is not None and data.progress.step == 400
    assert _source(data, "metrics").status == "cached"
    assert _source(results["first"], "metrics").status == "ok"


# ---------------------------------------------------------------------------
# runs_list: counts over the scope in bounded batches
# ---------------------------------------------------------------------------


def _ledger(env: Env, n: int, *, env_kib: int = 0) -> None:
    blob = jdump({"schema": "hpclab.env/v1", "pad": "x" * (env_kib * 1024)}) if env_kib else None
    with env.ctx.db.tx() as tx:
        for e in range(2):
            eid = make_experiment(tx, cluster_id=CLUSTER, name=f"e{e}")
            pid = make_plan(tx, eid)
            for i in range(n // 2):
                rid = make_run(tx, eid, pid, cell={"m": i}, seed=1)
                if i % 4 == 0:
                    continue  # prepared, never dispatched: P
                extra: dict[str, Any] = {"env_json": blob} if blob else {}
                if i % 4 == 1:
                    make_attempt(
                        tx, rid, phase="terminal", scheduler_state="FAILED", cluster_id=CLUSTER, **extra
                    )
                elif i % 4 == 2:
                    aid = make_attempt(
                        tx,
                        rid,
                        phase="terminal",
                        scheduler_state="COMPLETED",
                        result_state="valid",
                        cluster_id=CLUSTER,
                        **extra,
                    )
                    make_evaluation(tx, aid, status="valid")
                else:
                    make_attempt(
                        tx,
                        rid,
                        phase="terminal",
                        scheduler_state="COMPLETED",
                        result_state="valid",
                        cluster_id=CLUSTER,
                        **extra,
                    )


def test_runs_list_counts_the_whole_scope_in_bounded_batches(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ledger(env, 24)
    sizes: list[int] = []
    real = runs_mod.cell_statuses

    def spy(reader: Any, run_ids: list[str], **kw: Any) -> Any:
        sizes.append(len(run_ids))
        return real(reader, run_ids, **kw)

    monkeypatch.setattr(runs_mod, "cell_statuses", spy)
    monkeypatch.setattr(runs_mod, "COUNT_BATCH_RUNS", 5)
    data = runs_list(env.ctx, limit=3)
    assert data.total == 24 and len(data.runs) == 3
    assert data.counts == {"P": 6, "F": 6, "V": 6, "E": 6, "planned": 24, "valid": 6}
    assert max(sizes) <= 5, sizes


def test_runs_list_memory_does_not_grow_with_the_ledgers_blobs(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default (unfiltered) listing holds one batch of attempt rows at a time, not the
    whole ledger's (env_json alone may be up to 512 KiB per attempt)."""
    _ledger(env, 400, env_kib=32)  # ~300 attempts x 32 KiB ~ 10 MB of env_json
    monkeypatch.setattr(runs_mod, "COUNT_BATCH_RUNS", 20)
    tracemalloc.start()
    try:
        data = runs_list(env.ctx)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert data.total == 400 and data.counts["planned"] == 400
    assert peak < 5_000_000, f"peak {peak / 1e6:.1f} MB"
