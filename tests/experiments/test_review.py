"""Adversarial review tests for the prepare area: races, crash safety, honest receipts, bounded and
well-formed paging, spec §4 invariants (no network I/O inside a transaction, atomic idempotency,
events for every state change)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from prepare_support import Lab, lab_overlay, lab_profile, make_venv, plan_context

from hpclab.adapters.base import CommandResult
from hpclab.adapters.local import LocalTransport
from hpclab.api.envelope import encode_cursor
from hpclab.domain.errors import InvalidArgument, Stale, VersionConflict
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services import experiments as ex
from hpclab.services import validation as vd
from hpclab.services.outbox import OutboxRunner
from hpclab.storage.db import Database, jload


def prepare(lab: Lab, key: str = "k1", **kw: Any) -> ex.PrepareData:
    kw.setdefault("validate_remote", False)
    if "spec" not in kw and "spec_path" not in kw and "extends" not in kw and "derive_from" not in kw:
        kw["spec"] = lab.spec()
    return ex.experiment_prepare(lab.ctx, idempotency_key=key, **kw)


def count(lab: Lab, table: str, where: str = "1=1", params: tuple[Any, ...] = ()) -> int:
    row = lab.ctx.db.one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params)
    assert row is not None
    return int(row["n"])


def spec_seeds(lab: Lab, experiment_id: str) -> list[int]:
    return list(jload(lab.ctx.db.get("experiments", "experiment_id", experiment_id)["spec_json"])["seeds"])


def run_seeds(lab: Lab, experiment_id: str) -> set[int]:
    rows = lab.ctx.db.all("SELECT seed FROM runs WHERE experiment_id=? AND kind='main'", (experiment_id,))
    return {int(r["seed"]) for r in rows}


def drive_outbox(lab: Lab, rounds: int = 12) -> None:
    """Run the prepare handler like the controller does, letting retry delays elapse."""
    runner = OutboxRunner(lab.ctx)
    runner.register_all(ex.outbox_handlers())
    for _ in range(rounds):
        runner.run_once()
        lab.clock.advance(600)


# ---------------------------------------------------------------------------
# Races between concurrent extensions
# ---------------------------------------------------------------------------


def test_concurrent_extensions_never_lose_spec_additions(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two extensions validated against the same spec version: the second commit must not
    overwrite the first one's added seeds (runs would exist that the spec no longer lists)."""
    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    real = ex._materialize
    raced: list[str] = []

    def racing(ctx: Any, req: Any, target: Any, env: Any, **kw: Any) -> Any:
        if not raced:
            raced.append("b")
            other = prepare(lab, key="ext-b", extends=base.experiment_id, add_seeds=[3])
            assert other.status == "ready"
        return real(ctx, req, target, env, **kw)

    monkeypatch.setattr(ex, "_materialize", racing)
    with pytest.raises(VersionConflict):
        prepare(lab, key="ext-a", extends=base.experiment_id, add_seeds=[2])
    assert set(spec_seeds(lab, base.experiment_id)) == run_seeds(lab, base.experiment_id) == {1, 3}
    # nothing of the losing request was stored: a retry with the same key re-validates and succeeds
    assert count(lab, "idempotency", "idem_key='ext-a'") == 0
    again = prepare(lab, key="ext-a", extends=base.experiment_id, add_seeds=[2])
    assert again.status == "ready"
    assert spec_seeds(lab, base.experiment_id) == [1, 3, 2]
    assert run_seeds(lab, base.experiment_id) == {1, 2, 3}


def test_background_extension_losing_a_race_is_not_left_preparing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker's final write can conflict with a concurrent extension. The plan must end
    ready/invalid (never 'preparing' forever, which would block every later extension)."""
    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    lab.ctx.settings.prepare_sync_max_bytes = 10
    bg = prepare(lab, key="bg", extends=base.experiment_id, add_seeds=[2])
    assert bg.status == "preparing"
    lab.ctx.settings.prepare_sync_max_bytes = 1 << 30

    # A client whose extendability check passed before the background plan existed commits the
    # same cells while the worker is capturing/validating.
    real_check = ex._require_extendable
    bypass = {"on": False}

    def check(ctx: Any, row: Any, plan_id: Any) -> None:
        if not bypass["on"]:
            real_check(ctx, row, plan_id)

    monkeypatch.setattr(ex, "_require_extendable", check)
    real = ex._materialize
    raced: list[str] = []

    def racing(ctx: Any, req: Any, target: Any, env: Any, **kw: Any) -> Any:
        out = real(ctx, req, target, env, **kw)
        if kw["plan_id"] == bg.plan_id and not raced:
            raced.append("sync")
            bypass["on"] = True
            try:
                assert prepare(lab, key="sync", extends=base.experiment_id, add_seeds=[2]).status == "ready"
            finally:
                bypass["on"] = False
        return out

    monkeypatch.setattr(ex, "_materialize", racing)
    drive_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", bg.plan_id)
    assert plan["status"] == "invalid"
    assert jload(plan["validation_json"])["blocking"] == ["background_prepare"]
    assert count(lab, "runs", "plan_id=?", (bg.plan_id,)) == 0
    op = lab.ctx.db.get("operations", "operation_id", bg.operation_id or "")
    assert op["state"] == "failed" and "nothing to add" in jload(op["error_json"])["message"]
    assert run_seeds(lab, base.experiment_id) == {1, 2}
    assert count(lab, "events", "entity_id=? AND event_type='plan.invalid'", (bg.plan_id,)) == 1
    # the experiment is not wedged: a later extension proceeds
    assert prepare(lab, key="later", extends=base.experiment_id, add_seeds=[3]).status == "ready"


def test_background_crash_in_final_write_ends_invalid_not_preparing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab)

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("disk full while writing runs")

    monkeypatch.setattr(ex, "_insert_runs", boom)
    drive_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", data.plan_id)
    assert plan["status"] == "invalid" and plan["plan_hash"] is None
    assert lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)["status"] == "invalid"
    op = lab.ctx.db.get("operations", "operation_id", data.operation_id or "")
    assert op["state"] == "failed" and "disk full" in jload(op["error_json"])["message"]
    assert count(lab, "runs") == 0 and count(lab, "execution_plans") == 0
    item = lab.ctx.db.one("SELECT state FROM outbox WHERE kind='prepare.snapshot'")
    assert item is not None and item["state"] == "done"


# ---------------------------------------------------------------------------
# Extensions that would change the identity of existing cells
# ---------------------------------------------------------------------------


def test_adding_seeds_to_a_seedless_experiment_is_refused(lab: Lab) -> None:
    spec = lab.spec(
        seeds=None, smoke=None, entrypoint={"argv": ["python", "train.py", "--method", "{method}"]}
    )
    base = prepare(lab, spec=spec)
    assert base.status == "ready" and all(r.seed is None for r in base.runs)
    with pytest.raises(InvalidArgument, match="no seeds"):
        prepare(lab, key="e1", extends=base.experiment_id, add_seeds=[1, 2])
    with pytest.raises(InvalidArgument, match="no seeds"):
        prepare(lab, key="e2", extends=base.experiment_id, add_matrix={"seed": [1]})
    assert "seeds" not in jload(
        lab.ctx.db.get("experiments", "experiment_id", base.experiment_id)["spec_json"]
    ) or not spec_seeds(lab, base.experiment_id)
    assert count(lab, "runs", "experiment_id=?", (base.experiment_id,)) == 2


# ---------------------------------------------------------------------------
# Non-finite numbers are caller errors, never internal errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "updates",
    [
        {"parameters": {"lr": float("nan")}},
        {"matrix": {"method": ["a"], "lr": [float("inf")]}},
    ],
)
def test_non_finite_numbers_in_spec_are_invalid_argument(lab: Lab, updates: dict[str, Any]) -> None:
    with pytest.raises(InvalidArgument, match="finite"):
        prepare(lab, spec=lab.spec(**updates))
    assert count(lab, "experiments") == 0


def test_non_finite_numbers_from_yaml_are_invalid_argument(lab: Lab, tmp_path: Path) -> None:
    doc = lab.spec(parameters={"lr": 0.1})
    doc["code"] = {"project_root": str(lab.project)}
    text = json.dumps(doc).replace("0.1", ".nan")
    path = tmp_path / "spec.yaml"
    path.write_text(text)  # JSON is YAML; `.nan` is YAML's NaN
    with pytest.raises(InvalidArgument, match="finite"):
        prepare(lab, spec_path=str(path))


def test_non_finite_seeds_in_extend_and_derive_are_invalid_argument(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    with pytest.raises(InvalidArgument):
        prepare(lab, key="e", extends=base.experiment_id, add_matrix={"seed": [float("inf")]})
    for bad in (float("inf"), float("nan")):
        with pytest.raises(InvalidArgument):
            prepare(
                lab,
                key=f"d{bad}",
                derive_from={"run_id": base.runs[0].run_id, "overrides": {"params": {"seed": bad}}},
            )
    with pytest.raises(InvalidArgument):
        prepare(
            lab,
            key="d-lr",
            derive_from={"run_id": base.runs[0].run_id, "overrides": {"params": {"lr": float("inf")}}},
        )


# ---------------------------------------------------------------------------
# Honest background receipt
# ---------------------------------------------------------------------------


def test_background_receipt_counts_the_smoke_run(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab)
    assert data.status == "preparing"
    assert [r.kind for r in data.runs][:1] == ["smoke"] and data.runs_total == 7
    assert data.runs[0].label == "smoke.method-baseline.seed-1"
    assert data.budget.smoke_gpu_hours == pytest.approx(0.25)  # 1 GPU x 15 min
    assert data.budget.requested_gpu_hours_total == pytest.approx(6.25)
    drive_outbox(lab)
    ready = ex.plan_get(lab.ctx, plan_id=data.plan_id)
    assert ready.status == "ready" and ready.runs_total == data.runs_total
    assert ready.budget is not None
    assert ready.budget["requested_gpu_hours_total"] == pytest.approx(data.budget.requested_gpu_hours_total)
    assert ready.budget["smoke_gpu_hours"] == pytest.approx(data.budget.smoke_gpu_hours)


# ---------------------------------------------------------------------------
# plan_script / plan_get paging
# ---------------------------------------------------------------------------


def unicode_plan(lab: Lab) -> ex.PrepareData:
    argv = ["python", "train.py", "--note", "héllo✓\U0001f600", "--method", "{method}", "--seed", "{seed}"]
    data = prepare(lab, spec=lab.spec(entrypoint={"argv": argv}))
    assert data.status == "ready", data.validation.blocking
    return data


@pytest.mark.parametrize("max_bytes", [1, 2, 3])
def test_plan_script_paging_always_progresses_over_multibyte_text(lab: Lab, max_bytes: int) -> None:
    data = unicode_plan(lab)
    raw = ex.plan_script(lab.ctx, plan_id=data.plan_id).script.encode()
    start = raw.index("h\u00e9llo".encode())
    end = raw.index(b"--method", start)
    # one big page up to the multi-byte region, then tiny pages across it
    head = ex.plan_script(lab.ctx, plan_id=data.plan_id, max_bytes=start)
    assert head.script.encode() == raw[:start] and head.next_cursor
    cursor: str | None = head.next_cursor
    collected = b""
    for _ in range(4 * (end - start)):
        page = ex.plan_script(lab.ctx, plan_id=data.plan_id, cursor=cursor, max_bytes=max_bytes)
        assert page.script, "an empty page with a cursor would loop forever"
        assert page.offset == start + len(collected)
        collected += page.script.encode()
        cursor = page.next_cursor
        if start + len(collected) >= end or cursor is None:
            break
    assert start + len(collected) >= end
    assert raw[start : start + len(collected)] == collected


def test_plan_script_cursor_that_lands_mid_character_or_is_forged(lab: Lab) -> None:
    data = unicode_plan(lab)
    full = ex.plan_script(lab.ctx, plan_id=data.plan_id)
    raw = full.script.encode()
    mid = raw.index("✓".encode()) + 1
    first = ex.plan_script(lab.ctx, plan_id=data.plan_id, max_bytes=100)
    assert first.next_cursor
    for bad in ("not-a-cursor!!", encode_cursor({"offset": "x"}), encode_cursor({"offset": -5})):
        with pytest.raises(InvalidArgument):
            ex.plan_script(lab.ctx, plan_id=data.plan_id, cursor=bad)
    # an offset inside a character is never decoded into garbage / an internal error
    forged = json.loads(
        __import__("base64").urlsafe_b64decode(first.next_cursor + "=" * (-len(first.next_cursor) % 4))
    )
    forged["offset"] = mid
    with pytest.raises((InvalidArgument, Stale)):
        ex.plan_script(lab.ctx, plan_id=data.plan_id, cursor=encode_cursor(forged))


def test_plan_script_cursor_is_bound_to_its_view_and_run(lab: Lab) -> None:
    data = prepare(lab)
    page = ex.plan_script(lab.ctx, plan_id=data.plan_id, max_bytes=1000)
    assert page.next_cursor
    with pytest.raises((InvalidArgument, Stale)):
        ex.plan_script(lab.ctx, plan_id=data.plan_id, view="generated", cursor=page.next_cursor)
    with pytest.raises((InvalidArgument, Stale)):
        ex.plan_script(lab.ctx, plan_id=data.plan_id, run_id=data.smoke_run_id, cursor=page.next_cursor)
    nxt = ex.plan_script(lab.ctx, plan_id=data.plan_id, max_bytes=1000, cursor=page.next_cursor)
    assert nxt.offset == 1000


def test_plan_get_cursor_errors_are_invalid_argument(lab: Lab) -> None:
    data = prepare(lab)
    for bad in (encode_cursor({"after": "x"}), "%%%"):
        with pytest.raises(InvalidArgument):
            ex.plan_get(lab.ctx, plan_id=data.plan_id, cursor=bad)


def test_plan_get_preview_cursor_is_not_reused_as_a_run_cursor(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab)
    page = ex.plan_get(lab.ctx, plan_id=data.plan_id, limit=2)
    assert page.planned_runs and page.next_cursor
    drive_outbox(lab)
    with pytest.raises(Stale):
        ex.plan_get(lab.ctx, plan_id=data.plan_id, limit=2, cursor=page.next_cursor)
    # a cursor of one plan never pages another plan
    other = prepare(lab, key="other")
    first = ex.plan_get(lab.ctx, plan_id=other.plan_id, limit=2)
    assert first.next_cursor
    with pytest.raises((InvalidArgument, Stale)):
        ex.plan_get(lab.ctx, plan_id=data.plan_id, limit=2, cursor=first.next_cursor)


# ---------------------------------------------------------------------------
# Unknown is never pass
# ---------------------------------------------------------------------------


class PartialFind(LocalTransport):
    """`find` that could not read part of site-packages (exit 1) but printed something."""

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        if argv[0] == "find":
            out = f"{argv[1]}/requests-2.0.dist-info/direct_url.json\n"
            return CommandResult(list(argv), 1, out, f"find: {argv[1]}/secret: Permission denied\n", 0.0)
        return super().run(argv, timeout=timeout, stdin=stdin)


def test_partial_site_packages_listing_is_unchecked_not_pass(lab: Lab) -> None:
    venv = make_venv(lab.scratch / "venvs" / "e", editable_dist={"requests": False})
    out = vd.check_env_guard_static(
        transport=PartialFind("lab", user="alice"),
        cluster=ResolvedCluster(profile=lab_profile(lab.root), overlay=lab_overlay()),
        runtime_venv=str(venv),
        gpus=1,
        gpu_types=["a100"],
        project={"mypkg"},
        now=lab.clock.now(),
        deadline=vd.Deadline(10.0),
    )
    bypass = {f.check: f for f in out}["snapshot_bypass"]
    assert bypass.status == "unchecked", bypass
    assert "Permission denied" in bypass.message


def test_gpu_arch_unknown_is_unchecked_not_pass(lab: Lab) -> None:
    venv = make_venv(lab.scratch / "venvs" / "t", torch_cuda="12.8")
    out = vd.check_env_guard_static(
        transport=LocalTransport("lab", user="alice"),
        cluster=ResolvedCluster(profile=lab_profile(lab.root), overlay=lab_overlay()),
        runtime_venv=str(venv),
        gpus=1,
        gpu_types=[],  # partition GPU types unknown and no resources.gpu_type
        project={"mypkg"},
        now=lab.clock.now(),
        deadline=vd.Deadline(10.0),
    )
    env = {f.check: f for f in out}["env_guard_static"]
    assert env.status == "unchecked"
    assert any(c["name"] == "arch" and c["status"] == "unchecked" for c in env.evidence["checks"])


def test_storage_checks_without_storage_roles_are_unchecked(lab: Lab) -> None:
    cluster = ResolvedCluster(
        profile=lab_profile(lab.root, storage={}, remote_root_template="/scratch/alice/hpclab"),
        overlay=lab_overlay(),
    )
    pc = plan_context(
        lab.spec(runtime={"python": "python3", "env": {"HF_HOME": "/home/alice/.cache"}}), cluster
    )
    assert vd.check_storage_home_avoidance(pc).status == "unchecked"
    assert vd.check_cache_locations(pc).status == "unchecked"


# ---------------------------------------------------------------------------
# §4 invariants
# ---------------------------------------------------------------------------


class _TxProbe:
    """Tracks, per thread, whether a ledger transaction is open."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.violations: list[str] = []

    def inside(self) -> bool:
        return bool(getattr(self.local, "depth", 0))

    def note(self, what: str) -> None:
        if self.inside():
            self.violations.append(what)


def _guard_io(monkeypatch: pytest.MonkeyPatch, lab: Lab) -> _TxProbe:
    probe = _TxProbe()
    real_tx = Database.tx

    @contextmanager
    def tx(self: Database, *a: Any, **k: Any) -> Iterator[Any]:
        probe.local.depth = getattr(probe.local, "depth", 0) + 1
        try:
            with real_tx(self, *a, **k) as t:
                yield t
        finally:
            probe.local.depth -= 1

    monkeypatch.setattr(Database, "tx", tx)
    for name in ("run", "run_script", "stat", "read_range", "write_file", "remote_user", "list_dir"):
        real = getattr(LocalTransport, name)

        def wrapped(self: Any, *a: Any, _real: Any = real, _name: str = name, **k: Any) -> Any:
            probe.note(f"transport.{_name}")
            return _real(self, *a, **k)

        monkeypatch.setattr(LocalTransport, name, wrapped)
    real_submit = lab.scheduler.submit

    def submit(*a: Any, **k: Any) -> Any:
        probe.note("scheduler.submit")
        return real_submit(*a, **k)

    monkeypatch.setattr(lab.scheduler, "submit", submit)
    return probe


def test_no_network_io_inside_a_transaction(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _guard_io(monkeypatch, lab)
    make_venv(lab.scratch / "venvs" / "x", torch_cuda="12.8")
    spec = lab.spec(runtime={"python": "python3", "venv": "{scratch}/venvs/x"})
    data = prepare(lab, spec=spec, validate_remote=True)
    assert data.status == "ready", data.validation.blocking
    assert lab.scheduler.calls, "the remote checks must actually have run"
    lab.ctx.settings.prepare_sync_max_bytes = 10
    bg = prepare(lab, key="bg", spec=spec, validate_remote=True)
    drive_outbox(lab)
    assert lab.ctx.db.get("plans", "plan_id", bg.plan_id)["status"] == "ready"
    assert len(lab.scheduler.calls) == 2
    assert probe.violations == []


def test_failed_final_write_stores_neither_ledger_rows_nor_idempotency(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 8: the idempotent response is stored in the same transaction as the mutation."""

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("crash between the run rows and the response")

    monkeypatch.setattr(ex, "_record_overrides", boom)
    with pytest.raises(RuntimeError):
        prepare(lab)
    for table in ("experiments", "plans", "runs", "execution_plans", "idempotency"):
        assert count(lab, table) == 0, table
    assert count(lab, "events", "entity_type != 'cluster'") == 0
    monkeypatch.undo()
    assert prepare(lab).status == "ready"


def test_every_state_change_has_an_event(lab: Lab) -> None:
    """Invariant 9, checked over a new experiment, an extension and a background plan."""
    base = prepare(lab, spec=lab.spec(seeds=[1]))
    prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    lab.ctx.settings.prepare_sync_max_bytes = 10
    bg = prepare(lab, key="bg")
    drive_outbox(lab)
    db = lab.ctx.db

    def ev(entity_id: str, event_type: str) -> int:
        return count(lab, "events", "entity_id=? AND event_type=?", (entity_id, event_type))

    for run in db.all("SELECT run_id FROM runs"):
        assert ev(run["run_id"], "run.created") == 1
    for plan in db.all("SELECT plan_id, status FROM plans"):
        assert ev(plan["plan_id"], "plan.created") == 1
        assert ev(plan["plan_id"], f"plan.{plan['status']}") == 1
    exp = db.get("experiments", "experiment_id", base.experiment_id)
    assert exp["version"] == 2 and ev(base.experiment_id, "experiment.extended") == 1
    assert ev(bg.experiment_id, "experiment.status_changed") == 1
    op = db.get("operations", "operation_id", bg.operation_id or "")
    updates = count(lab, "events", "entity_id=? AND event_type='operation.updated'", (op["operation_id"],))
    assert op["version"] == 1 + updates and jload(op["progress_json"]) == {}


def test_prepare_never_queues_a_job(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x")
    prepare(
        lab, spec=lab.spec(runtime={"python": "python3", "venv": "{scratch}/venvs/x"}), validate_remote=True
    )
    assert lab.scheduler.calls and all(c["test_only"] for c in lab.scheduler.calls)
    assert count(lab, "attempts") == 0 and count(lab, "outbox", "kind != 'prepare.snapshot'") == 0


def test_experiment_get_never_reports_a_stale_observation_as_running(lab: Lab) -> None:
    """A RUNNING attempt last observed days ago is unknown *now* ('?'), exactly as runs_list says."""
    from datetime import timedelta

    from hpclab.domain.clock import iso
    from hpclab.testing.factories import make_attempt

    data = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    fresh, old = (r.run_id or "" for r in data.runs)
    with lab.ctx.db.tx() as tx:
        for rid, seen in ((fresh, lab.clock.now()), (old, lab.clock.now() - timedelta(days=3))):
            ep = tx.one("SELECT current_exec_plan_hash AS h FROM runs WHERE run_id=?", (rid,))
            make_attempt(
                tx,
                rid,
                exec_plan_hash=ep["h"] if ep else None,
                phase="observing",
                scheduler_state="RUNNING",
                cluster_id="lab",
                last_observed_at=iso(seen),
            )
    got = ex.experiment_get(lab.ctx, experiment_id=data.experiment_id)
    assert got.run_counts.get("R") == 1 and got.run_counts.get("?") == 1, got.run_counts
    assert got.coverage.complete is False and any("stale" in n for n in got.coverage.notes)


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-c", "python train.py"],
        ["sh", "-c", "cd sub && python train.py"],
        ["python", "-c", "import runpy; runpy.run_path('train.py')"],
    ],
)
def test_shell_command_strings_are_not_taken_for_missing_files(lab: Lab, argv: list[str]) -> None:
    """A `-c` command string is not a file path: never a false *blocking* finding."""
    data = prepare(lab, spec=lab.spec(entrypoint={"argv": [*argv, "{method}", "{seed}"]}, smoke=None))
    f = {x.check: x for x in data.validation.findings}["entrypoint_in_snapshot"]
    assert f.status == "unchecked", f
    assert data.status == "ready", data.validation.blocking


def test_two_background_extensions_cannot_both_be_preparing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both requests pass the pre-check before either plan exists; the second must be refused
    in the transaction that would create it (two preparing extensions wait on each other and
    both end invalid)."""
    from hpclab.domain.errors import NotReady

    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    lab.ctx.settings.prepare_sync_max_bytes = 10
    monkeypatch.setattr(ex, "_require_extendable", lambda *a, **k: None)  # the pre-check raced
    first = prepare(lab, key="a", extends=base.experiment_id, add_seeds=[2])
    assert first.status == "preparing"
    with pytest.raises(NotReady):
        prepare(lab, key="b", extends=base.experiment_id, add_seeds=[3])
    assert count(lab, "plans", "status='preparing'") == 1
    assert count(lab, "idempotency", "idem_key='b'") == 0
    monkeypatch.undo()
    drive_outbox(lab)
    assert lab.ctx.db.get("plans", "plan_id", first.plan_id)["status"] == "ready"
