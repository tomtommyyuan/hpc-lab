"""End to end: an expired SSH login while a sweep is finishing does not hammer the login node.

Real ``OpenSSHTransport`` + ``SlurmScheduler`` + ``observe.finalize`` items; ``ssh`` is a
script that fails authentication like BatchMode against a password+Duo login node and logs
every connection attempt.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from observe_support import make_env, observation, seed, stub_budget_and_notify

from hpclab.adapters.openssh import OpenSSHTransport
from hpclab.adapters.slurm import SlurmScheduler
from hpclab.services.observe import finalize_attempt, poll_cluster
from hpclab.services.outbox import OutboxRunner
from hpclab.testing.factories import make_experiment

N_JOBS = 24


def _fake_ssh(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "ssh.log"
    script = tmp_path / "ssh"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{log}"\n'
        'echo "alice@login.example: Permission denied (gssapi-with-mic,keyboard-interactive)." >&2\n'
        "exit 255\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log


def test_finalize_backlog_makes_one_login_attempt_per_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path)
    stub_budget_and_notify(monkeypatch)
    with env.ctx.db.tx() as tx:
        eid = make_experiment(tx, cluster_id="c1")
    aids = [seed(env, experiment_id=eid, job_id=str(7000 + i), cell={"i": i})[1] for i in range(N_JOBS)]
    env.scheduler.queue = [observation(env, a, "COMPLETED") for a in aids]
    poll_cluster(env.ctx, cluster_id="c1")  # auth still fine: N jobs seen terminal -> N finalize items
    assert len(env.outbox("observe.finalize")) == N_JOBS

    # the ControlMaster is gone: from now on every ssh needs Duo
    ssh, log = _fake_ssh(tmp_path)
    transport = OpenSSHTransport(
        "c1", "login.example", user="alice", control_dir=tmp_path / "run", ssh_bin=str(ssh)
    )
    env.ctx.clusters._transports["c1"] = transport
    env.ctx.clusters._schedulers["c1"] = SlurmScheduler(
        transport, env.ctx.clusters.get("c1"), clock=env.clock
    )
    runner = OutboxRunner(env.ctx, lease_s=60)
    runner.register("observe.finalize", finalize_attempt)

    for _ in range(600 // 5):  # ten minutes of controller time
        while runner.run_once(limit=1000):
            pass
        env.clock.advance(5)

    spawned = log.read_text().splitlines() if log.exists() else []
    # before: every item logged in on its own every 120 s (24 x 5 = 120 connections)
    assert len(spawned) <= 6, f"{len(spawned)} ssh connections in 10 minutes"
    rows = env.outbox("observe.finalize")
    assert {r["state"] for r in rows} == {"pending"}
    assert all(str(r["last_error"]).startswith("auth_required") for r in rows)
