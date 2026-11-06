"""init_sim on homes whose path job scripts cannot carry (spaces, non-ASCII, shell metacharacters).

The simulator's "remote" filesystem lived under ``$HPCLAB_HOME/sim/sim/fs``, so a home such as
``/Users/张三/.hpclab`` or ``~/My Drive/.hpclab`` produced a remote root the job-script renderer
refuses: every ``experiment_prepare`` came back ``invalid`` (script_render failed) and nothing
could run. init_sim now places such a home's simulator at
``/var/tmp/hpclab-sim-<uid>-<sha256(home)[:10]>`` (persistent, per home, 0700), records it in
the overlay and says why in ``next_steps``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from setup_support import relocated_sim_root

from hpclab.domain.errors import HpcLabError
from hpclab.profiles.loader import load_overlay
from hpclab.services.app import create_app
from hpclab.services.clusters import effective_sim_root, init_sim
from hpclab.services.context import ServiceContext
from hpclab.services.experiments import experiment_prepare
from hpclab.services.jobscript import attempt_layout
from hpclab.services.outbox import OutboxRunner
from hpclab.services.submission import run_submit
from hpclab.sim.cluster import SimCluster
from hpclab.sim.config import SimConfig
from hpclab.sim.testing import FAST_TIMINGS
from hpclab.testing.simctx import wait_until

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "tiny-classifier"

MakeCtx = Callable[..., ServiceContext]
UNSAFE_NAMES = ["my home", "张三", "lab;rm -rf $HOME"]


def _next_steps_text(steps: list[str]) -> str:
    return "\n".join(steps)


# -- next steps (plain homes keep their simulator inside the home) -------------------------------


def test_plain_home_keeps_the_simulator_inside_the_home(make_ctx: MakeCtx) -> None:
    ctx = make_ctx("plain")
    data = init_sim(ctx)
    root = ctx.paths.sim_root("sim")
    overlay = load_overlay("sim", ctx.paths)
    assert overlay is not None and overlay.sim_root == str(root)
    assert data.cluster.remote_root.startswith(str(root / "fs"))
    assert not relocated_sim_root(ctx.paths.home).exists()
    assert effective_sim_root(ctx.paths) == root
    steps = data.next_steps
    copy = next(
        i for i, s in enumerate(steps) if "hpclab examples copy tiny-classifier ./tiny-classifier" in s
    )
    prep = next(i for i, s in enumerate(steps) if "hpclab prepare tiny-classifier/hpclab.yaml" in s)
    assert copy < prep
    assert "/var/tmp" not in _next_steps_text(steps)


# -- relocation -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_unsafe_home_gets_a_private_persistent_simulator_root(make_ctx: MakeCtx, name: str) -> None:
    ctx = make_ctx(name)
    with pytest.raises(HpcLabError):  # the renderer refuses the old location: the bug being fixed
        attempt_layout(str(ctx.paths.sim_root("sim") / "fs" / "scratch" / "u" / "hpclab"), "att_x")

    data = init_sim(ctx)
    root = relocated_sim_root(ctx.paths.home)
    assert root.is_dir() and not root.is_symlink()
    st = root.stat()
    assert stat.S_IMODE(st.st_mode) == 0o700 and st.st_uid == os.getuid()
    assert (root / "bin" / "sbatch").exists()  # the simulator itself lives there
    # recorded in the overlay (the user's editable source of truth) and in the ledger
    overlay = load_overlay("sim", ctx.paths)
    assert overlay is not None
    assert overlay.sim_root == str(root)
    assert overlay.remote_root == data.cluster.remote_root
    assert data.cluster.remote_root.startswith(str(root / "fs") + "/")
    attempt_layout(data.cluster.remote_root, "att_x")  # the renderer accepts it
    assert ctx.clusters.get("sim").overlay.sim_root == str(root)
    assert effective_sim_root(ctx.paths) == root
    # nothing of the simulator under the unusable home path
    assert not (ctx.paths.sim_root("sim") / "state.db").exists()
    text = _next_steps_text(data.next_steps)
    assert str(root) in text and "HPCLAB_HOME" in text, text
    assert "hpclab examples copy tiny-classifier ./tiny-classifier" in text


def test_init_sim_is_idempotent_for_a_relocated_simulator(make_ctx: MakeCtx) -> None:
    ctx = make_ctx("my home")
    first = init_sim(ctx)
    root = relocated_sim_root(ctx.paths.home)
    root.chmod(0o755)  # loosened in between: tightened again
    marker = root / "fs" / "keep.txt"
    marker.write_text("state survives a second init_sim\n")
    second = init_sim(ctx)
    assert second.created is False
    assert second.cluster == first.cluster  # same remote root, same profile revision
    assert second.next_steps == first.next_steps
    assert marker.read_text() == "state survives a second init_sim\n"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_relocated_root_differs_per_home(make_ctx: MakeCtx) -> None:
    a = make_ctx("home one")
    b = make_ctx("home two")
    ra = init_sim(a).cluster.remote_root
    rb = init_sim(b).cluster.remote_root
    assert ra != rb
    assert relocated_sim_root(a.paths.home) != relocated_sim_root(b.paths.home)


def test_a_planted_symlink_is_refused(make_ctx: MakeCtx, tmp_path: Path) -> None:
    """The /var/tmp name is predictable: a symlink (or another user's directory) planted there
    must never be used as the simulator root."""
    ctx = make_ctx("my home")
    root = relocated_sim_root(ctx.paths.home)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root.symlink_to(elsewhere)
    try:
        with pytest.raises(HpcLabError, match="symlink") as ei:
            init_sim(ctx)
        # another user's entry in sticky /var/tmp cannot be removed: the way out is a plain home
        assert "HPCLAB_HOME" in ei.value.details["hint"], ei.value.details
        assert list(elsewhere.iterdir()) == []
        assert load_overlay("sim", ctx.paths) is None
    finally:
        root.unlink()


def test_a_home_with_a_colon_gets_a_relocated_simulator(make_ctx: MakeCtx) -> None:
    """The renderer accepts ':' in a remote root, but the simulator puts ``<sim_root>/bin`` (and
    ``gpu-bin`` for jobs) on PATH, where ':' separates entries: ``sbatch`` was then not found
    and prepare failed its scheduler_test. Such a home's simulator is relocated as well."""
    ctx = make_ctx("lab:2024")
    attempt_layout(str(ctx.paths.sim_root("sim") / "fs" / "scratch" / "u" / "hpclab"), "att_x")  # accepted

    data = init_sim(ctx)
    root = relocated_sim_root(ctx.paths.home)
    assert effective_sim_root(ctx.paths) == root
    assert data.cluster.remote_root.startswith(str(root) + "/")
    assert ":" not in str(root) and ":" not in data.cluster.remote_root
    assert str(root) in _next_steps_text(data.next_steps)


def test_an_earlier_registration_under_the_unsafe_home_is_repaired(make_ctx: MakeCtx) -> None:
    """Homes registered before the fix recorded sim_root/remote_root under the home: init_sim
    moves the registration to the relocated root."""
    from hpclab.services.clusters import cluster_register

    ctx = make_ctx("张三")
    old_root = ctx.paths.sim_root("sim")
    cluster_register(
        ctx,
        profile="sim",
        user="me",
        account="lab",
        sim_root=str(old_root),
        remote_root=str(old_root / "fs" / "scratch" / "me" / "hpclab"),
        default_partition="gpu",
    )
    data = init_sim(ctx)
    root = relocated_sim_root(ctx.paths.home)
    assert data.cluster.remote_root.startswith(str(root))
    overlay = load_overlay("sim", ctx.paths)
    assert overlay is not None and overlay.sim_root == str(root)


# -- the whole path: prepare the example and run a job ------------------------------------------------


def _handlers(ctx: ServiceContext) -> OutboxRunner:
    runner = OutboxRunner(ctx, lease_s=60.0)
    runner.register_all(create_app(ctx).outbox_handlers())
    return runner


@pytest.mark.sim
@pytest.mark.timeout(120)
@pytest.mark.parametrize("name", ["my home", "张三", "lab:2024"])
def test_example_prepares_and_runs_on_a_relocated_simulator(make_ctx: MakeCtx, name: str) -> None:
    ctx = make_ctx(name)
    init_sim(ctx)
    root = relocated_sim_root(ctx.paths.home)
    SimCluster.ensure(root, SimConfig().replace(**FAST_TIMINGS))  # short job lifecycles
    # the user's project lives next to the home, with the same kind of path
    project = ctx.paths.home.parent / "tiny-classifier"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("__pycache__", "outputs"))

    prep = experiment_prepare(ctx, idempotency_key="prep-1", spec_path=str(project / "hpclab.yaml"))
    findings = {f.check: f for f in prep.validation.findings}
    assert findings["script_render"].status == "pass", findings["script_render"]
    assert findings["remote_root"].status == "pass", findings["remote_root"]
    assert prep.status == "ready", [(f.check, f.status, f.message) for f in prep.validation.findings]

    smoke = next(r for r in prep.runs if r.kind == "smoke")
    assert smoke.run_id and smoke.exec_plan_hash
    receipt = run_submit(
        ctx, run_id=smoke.run_id, execution_plan_hash=smoke.exec_plan_hash, idempotency_key="submit-1"
    )
    runner = _handlers(ctx)
    sim = SimCluster(root)

    def job_done() -> dict[str, Any] | None:
        runner.drain(max_rounds=20)
        jobs = sim.jobs()
        return (
            jobs[0] if jobs and jobs[0]["state"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT") else None
        )

    job = wait_until(job_done, timeout=60, interval=0.2, what="the smoke job to finish on the simulator")
    assert len(sim.jobs()) == 1
    assert job["state"] == "COMPLETED", job
    attempt = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (receipt.attempt_id,))
    assert attempt is not None and attempt["remote_dir"].startswith(str(root) + "/")
    status = json.loads((Path(attempt["remote_dir"]) / "status.json").read_text())
    assert status.get("exit_code") == 0, status
    metrics = Path(attempt["remote_dir"]) / "outputs" / "metrics.json"
    assert metrics.is_file()
