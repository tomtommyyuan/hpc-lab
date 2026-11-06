"""index_attempt: the integrity state machine and attempts.result_state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from artifacts_support import DEFAULT_OUTPUTS, Att, Lab, auth_error, conn_lost, metrics_jsonl

from hpclab.domain.errors import PreconditionFailed
from hpclab.services import artifacts
from hpclab.services.artifacts import handle_index, index_attempt, sweep_unindexed
from hpclab.services.outbox import OutboxItem


def item(att: Att, tries: int = 1) -> OutboxItem:
    return OutboxItem(1, "artifacts.index", None, att.attempt_id, {"attempt_id": att.attempt_id}, tries)


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_finalized_outputs_carry_compute_node_digests(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    doc = lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)

    assert res.result_state == "valid" and res.previous_result_state == "unchecked" and res.changed
    assert res.manifest == "present" and res.manifest_hash_state == "complete"
    by = {o.name: o for o in res.outputs}
    assert {o.integrity for o in res.outputs} == {"finalized"}
    assert by["metrics"].digest_source == "compute_node"
    entries = {f["name"]: f for f in doc["files"]}
    assert (
        lab.artifact(att, "metrics")["digest"]
        == entries["metrics"]["sha256"]
        == sha(att.path("outputs/metrics.jsonl"))
    )
    ckpt = lab.artifact(att, "checkpoints")
    assert ckpt["kind"] == "dir" and ckpt["digest"] == entries["checkpoints"]["sha256"]
    members = lab.ctx.db.all("SELECT * FROM artifacts WHERE name LIKE 'checkpoints/%' ORDER BY name")
    assert [m["name"] for m in members] == [
        "checkpoints/ckpt-000000020/hpclab-ckpt.json",
        "checkpoints/ckpt-000000020/model.bin",
        "checkpoints/ckpt-000000060/hpclab-ckpt.json",
        "checkpoints/ckpt-000000060/model.bin",
    ]
    assert {m["integrity"] for m in members} == {"finalized"} and all(m["required"] == 0 for m in members)
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "valid"
    assert len(lab.events("attempt.result_state_changed")) == 1
    assert len(lab.events("artifact.indexed")) == 3

    again = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert not again.changed and again.result_state == "valid"
    assert len(lab.events("artifact.indexed")) == 3  # no event without a state change


def test_missing_required_output_makes_the_result_incomplete(lab: Lab) -> None:
    att = lab.attempt()
    att.write("outputs/metrics.jsonl", metrics_jsonl())  # summary never written, no checkpoints
    doc = lab.manifest(att)
    assert doc["missing_required"] == ["summary"]
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    by = {o.name: o for o in res.outputs}
    assert by["summary"].integrity == "missing"
    assert by["metrics"].integrity == "finalized"
    assert by["checkpoints"].integrity == "finalized" and by["checkpoints"].members == 0  # empty dir
    assert res.result_state == "incomplete"
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "incomplete"


def test_optional_output_missing_keeps_the_result_valid(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    att.path("checkpoints").rmdir()
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "valid"
    assert {o.name: o.integrity for o in res.outputs}["checkpoints"] == "missing"


def test_file_changed_after_the_manifest_is_incomplete(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    with att.path("outputs/metrics.jsonl").open("a") as fh:
        fh.write('{"late": true}\n')
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    metrics = {o.name: o for o in res.outputs}["metrics"]
    assert metrics.integrity == "incomplete"
    assert "size_changed_after_manifest" in metrics.issues
    assert res.result_state == "incomplete"


def test_dir_member_added_after_manifest_is_incomplete_but_optional(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    att.write("checkpoints/ckpt-000000080/model.bin", b"late")
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    ck = {o.name: o for o in res.outputs}["checkpoints"]
    assert ck.integrity == "incomplete" and any(i.startswith("member_added: 1") for i in ck.issues)
    assert res.result_state == "valid"  # checkpoints are optional


def test_missing_digest_is_rehashed_on_the_login_node(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    quick = lab.manifest(att, mode="quick")  # the hashing pass never ran (job killed)
    assert quick["hash_state"] == "pending" and all(f["sha256"] is None for f in quick["files"])
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "valid"
    metrics = lab.artifact(att, "metrics")
    assert metrics["digest_source"] == "login_node" and metrics["digest"] == sha(
        att.path("outputs/metrics.jsonl")
    )
    ours = lab.artifact(att, "checkpoints")
    assert ours["digest_source"] == "login_node" and ours["integrity"] == "finalized"
    assert "sha256" in lab.transport.calls
    full = lab.manifest(att, mode="full")  # the wrapper's own tree digest must be identical
    assert {f["name"]: f for f in full["files"]}["checkpoints"]["sha256"] == ours["digest"]


def test_existing_digest_is_not_rehashed(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att, mode="quick")
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    lab.transport.calls.clear()
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert "sha256" not in lab.transport.calls and res.result_state == "valid"


def test_file_too_large_for_the_login_node_stays_unknown(tmp_path: Path) -> None:
    lab = Lab(tmp_path, max_login_hash_bytes=16)
    try:
        att = lab.attempt()
        lab.good_outputs(att, checkpoint=False)
        lab.manifest(att, mode="quick")
        res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
        metrics = {o.name: o for o in res.outputs}["metrics"]
        assert metrics.integrity == "unknown" and metrics.digest_state == "skipped_too_large"
        assert res.result_state == "unknown"
        assert "sha256" not in lab.transport.calls
    finally:
        lab.close()


def test_running_attempt_outputs_are_mutable(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    att.write("outputs/metrics.jsonl", metrics_jsonl(final=False))
    att.write("checkpoints/ckpt-000000020/model.bin", b"w")
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    by = {o.name: o for o in res.outputs}
    assert res.result_state == "pending" and res.manifest == "skipped"
    assert by["metrics"].integrity == "mutable" and by["metrics"].digest_state == "pending"
    assert by["summary"].integrity == "missing"
    assert by["checkpoints"].integrity == "mutable"
    assert lab.artifact(att, "checkpoints/ckpt-000000020/model.bin")["integrity"] == "mutable"


def test_requeue_turns_a_valid_result_back_to_pending(lab: Lab, indexed: Att) -> None:
    lab.set_attempt(indexed, phase="observing", scheduler_state="RUNNING", restart_count=1)
    res = index_attempt(lab.ctx, attempt_id=indexed.attempt_id)
    assert res.result_state == "pending"
    assert lab.artifact(indexed, "metrics")["integrity"] == "mutable"


def test_transport_down_is_unknown_never_missing(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    lab.transport.down = conn_lost()
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "unknown" and res.manifest == "unavailable"
    assert {o.integrity for o in res.outputs} == {"unknown"}
    assert res.coverage.complete is False and res.retry_after_s == 60.0
    assert handle_index(lab.ctx, item(att)).action == "retry"  # type: ignore[union-attr]

    lab.transport.down = None
    assert index_attempt(lab.ctx, attempt_id=att.attempt_id).result_state == "valid"
    lab.transport.down = conn_lost()
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "valid"  # an established result is not downgraded by an outage
    assert lab.artifact(att, "metrics")["integrity"] == "finalized"


def test_auth_required_is_retried_without_limit(lab: Lab, indexed: Att) -> None:
    lab.transport.down = auth_error()
    res = handle_index(lab.ctx, item(indexed, tries=500))
    assert res is not None and res.action == "retry" and res.delay_s == 120.0
    lab.transport.down = conn_lost()
    res = handle_index(lab.ctx, item(indexed, tries=500))
    assert res is not None and res.action == "done"  # bounded for ordinary outages


def test_manifest_of_another_attempt_is_not_trusted(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att, attempt_id="att_01J00000000000000000000000")
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.manifest == "invalid"
    assert res.result_state == "unknown"
    assert {o.integrity for o in res.outputs} == {"unknown"}


def test_missing_manifest_right_after_the_job_is_rechecked(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.manifest == "missing" and res.result_state == "unknown" and res.retry_after_s == 30.0
    lab.clock.advance(3600)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.retry_after_s is None  # outside the settle window it is believed


def test_symlink_escaping_the_attempt_dir_is_never_finalized(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    secret = lab.tmp / "secret.txt"
    secret.write_text("private\n")
    target = att.path("outputs/metrics.jsonl")
    target.unlink()
    target.symlink_to(secret)
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    metrics = {o.name: o for o in res.outputs}["metrics"]
    assert metrics.integrity == "unknown" and any("symlink_escape" in i for i in metrics.issues)
    assert res.result_state == "unknown"


def test_declared_file_that_is_a_directory_is_invalid(lab: Lab) -> None:
    att = lab.attempt()
    att.path("outputs/metrics.jsonl").mkdir(parents=True)
    att.write("outputs/metrics.json", "{}")
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "invalid"


def test_checkpoint_staging_dirs_are_not_member_rows(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    att.write("checkpoints/.staging-000000080-4242/model.bin", b"partial")
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert {o.name: o for o in res.outputs}["checkpoints"].integrity == "finalized"
    names = [r["name"] for r in lab.ctx.db.all("SELECT name FROM artifacts WHERE name LIKE 'checkpoints/%'")]
    assert names and not any(".staging" in n for n in names)


def test_vanished_member_is_marked_missing(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    att.write("checkpoints/ckpt-000000020/model.bin", b"w")
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    att.path("checkpoints/ckpt-000000020/model.bin").unlink()  # keep_last rotated it away
    att.write("checkpoints/ckpt-000000040/model.bin", b"w2")
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert lab.artifact(att, "checkpoints/ckpt-000000020/model.bin")["integrity"] == "missing"
    assert lab.artifact(att, "checkpoints/ckpt-000000040/model.bin")["integrity"] == "mutable"


def test_never_ran_attempt_is_refused_and_the_handler_finishes(lab: Lab) -> None:
    att = lab.attempt(phase="submission_rejected", scheduler_state="UNKNOWN", job_id=None)
    with pytest.raises(PreconditionFailed):
        index_attempt(lab.ctx, attempt_id=att.attempt_id)
    res = handle_index(lab.ctx, item(att))
    assert res is not None and res.action == "done"


def test_phase_change_during_indexing_discards_the_result(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    real = artifacts._observe

    def observe_then_requeue(*args: Any, **kwargs: Any) -> Any:
        obs = real(*args, **kwargs)
        lab.set_attempt(att, phase="observing", restart_count=1)
        return obs

    monkeypatch.setattr(artifacts, "_observe", observe_then_requeue)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.retry_after_s == 2.0 and not res.changed
    assert lab.ctx.db.all("SELECT * FROM artifacts") == []
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "unchecked"


def test_valid_result_refreshes_run_status_and_wakes_the_workflow(lab: Lab) -> None:
    att = lab.attempt()
    lab.set_attempt(att, workflow_id="wf_01J00000000000000000000000")
    with lab.ctx.db.tx() as tx:
        tx.execute("UPDATE runs SET status='completed_unverified' WHERE run_id=?", (att.run_id,))
    lab.good_outputs(att)
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert lab.row("runs", "run_id", att.run_id)["status"] == "succeeded"
    wake = lab.ctx.db.all("SELECT * FROM outbox WHERE kind='workflow.dispatch'")
    assert len(wake) == 1 and json.loads(wake[0]["payload_json"]) == {
        "workflow_id": "wf_01J00000000000000000000000"
    }


def test_spec_without_outputs_falls_back_to_the_manifest(lab: Lab) -> None:
    att = lab.attempt(spec_outputs=False)
    lab.good_outputs(att)
    lab.manifest(att, outputs=DEFAULT_OUTPUTS)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert sorted(o.name for o in res.outputs) == ["checkpoints", "metrics", "summary"]
    assert res.result_state == "valid"


def test_sweep_enqueues_terminal_attempts_that_were_never_indexed(lab: Lab) -> None:
    att = lab.attempt()
    lab.attempt(phase="observing", scheduler_state="RUNNING")
    assert sweep_unindexed(lab.ctx) == 0  # too recent: observe.finalize is expected to do it
    lab.clock.advance(120)
    assert sweep_unindexed(lab.ctx) == 1
    assert sweep_unindexed(lab.ctx) == 0  # already queued
    rows = lab.ctx.db.all("SELECT * FROM outbox WHERE kind='artifacts.index'")
    assert [r["target_id"] for r in rows] == [att.attempt_id]


def test_handlers_and_periodic_tasks_are_exported() -> None:
    assert set(artifacts.outbox_handlers()) == {"artifacts.index", "artifact.download"}
    tasks = artifacts.periodic_tasks()
    assert [t.name for t in tasks] == ["artifacts.sweep"]


def test_path_absent_from_a_stat_reply_is_unknown_not_missing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    real = lab.transport.stat

    def partial(paths: list[str], **kw: Any) -> Any:
        out = real(paths, **kw)
        return {p: st for p, st in out.items() if not p.endswith("metrics.json")}

    monkeypatch.setattr(lab.transport, "stat", partial)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    summary = {o.name: o for o in res.outputs}["summary"]
    assert summary.integrity == "unknown" and res.result_state == "unknown"


def test_symlinked_output_directory_is_not_verified(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    real_dir = att.path("real-ckpts")
    att.path("checkpoints").rename(real_dir)
    att.path("checkpoints").symlink_to(real_dir)
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    ck = {o.name: o for o in res.outputs}["checkpoints"]
    assert ck.integrity == "unknown" and any("symlinked directory" in i for i in ck.issues)
    assert res.result_state == "valid"  # optional output
