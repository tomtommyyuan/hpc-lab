"""Regressions found in review (each failed before its fix) plus the invariant guards
written with them (``..._not_swept_forever``, ``..._has_a_horizon``,
``..._keeps_a_login_hashed_directory_finalized``, ``test_no_remote_call_runs_inside_a_transaction``).

* a known digest mismatch must make *every* read stale, not only whole-file reads;
* a directory output that vanished (scratch purge) takes its member rows with it, and member
  transitions are audited by name;
* a long cluster outage must not leave a terminal attempt's result ``unknown`` forever
  (the smoke gate and batch collection wait on it);
* a mutable-artifact refresh must not overwrite a record finalized while the read ran;
* a directory is never finalized on an earlier tree digest its own size/mtime cannot vouch for;
* the offline (cache) read re-verifies what it labels verified;
* ``value_summary`` is bounded by ``max_bytes``;
* the vault never replaces a digest-verified metrics copy with bytes that no longer match,
  backups skip in-progress capture files, and every ended attempt gets captured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from artifacts_support import Att, Lab, accounting_record, conn_lost, keep_mtime, metrics_jsonl

from hpclab.domain.errors import NotFound, Stale
from hpclab.services import artifacts
from hpclab.services.artifacts import ByteRange, artifact_read, handle_index, index_attempt, sweep_unindexed
from hpclab.services.outbox import OutboxItem, OutboxRunner
from hpclab.services.vault import capture


def ref(lab: Lab, att: Att, name: str) -> str:
    return str(lab.artifact(att, name)["artifact_id"])


# ---------------------------------------------------------------------------
# artifact_read: a recorded mismatch is stale for partial reads too
# ---------------------------------------------------------------------------


def test_recorded_mismatch_refuses_partial_reads(lab: Lab, indexed: Att) -> None:
    summary = indexed.path("outputs/metrics.json")
    keep_mtime(summary, summary.read_bytes().replace(b"0.9", b"0.1"))  # same size + mtime
    with pytest.raises(Stale):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))  # whole read: detected
    assert lab.artifact(indexed, "summary")["digest_state"] == "mismatch"
    # a partial read cannot re-verify the digest, but the ledger already knows these bytes
    # are not the attested ones: returning them would be "silently reading new content"
    for kw in (
        {"view": "head", "max_bytes": 8},
        {"view": "tail", "max_bytes": 8},
        {"range": ByteRange(offset=0, length=8)},
    ):
        with pytest.raises(Stale) as err:
            artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"), **kw)
        assert err.value.details["digest_state"] == "mismatch"


def test_recorded_mismatch_on_a_large_jsonl_tail_is_stale(lab: Lab, indexed: Att) -> None:
    metrics = indexed.path("outputs/metrics.jsonl")
    tampered = metrics.read_bytes().replace(b'"acc": 0.9', b'"acc": 0.1')
    keep_mtime(metrics, tampered)
    with pytest.raises(Stale):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "metrics"), view="head", max_bytes=10**6)
    with pytest.raises(Stale):  # the tail window (< file size) is not verifiable on its own
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "metrics"), max_bytes=64)


# ---------------------------------------------------------------------------
# index: members of a vanished directory
# ---------------------------------------------------------------------------


def test_members_of_a_vanished_directory_become_missing(lab: Lab, indexed: Att) -> None:
    before = lab.ctx.db.all(
        "SELECT integrity FROM artifacts WHERE attempt_id=? AND name LIKE 'checkpoints/%'",
        (indexed.attempt_id,),
    )
    assert before and {r["integrity"] for r in before} == {"finalized"}
    for p in sorted(indexed.path("checkpoints").rglob("*"), reverse=True):  # scratch purge
        p.unlink() if p.is_file() else p.rmdir()
    indexed.path("checkpoints").rmdir()
    res = index_attempt(lab.ctx, attempt_id=indexed.attempt_id)
    assert {o.name: o.integrity for o in res.outputs}["checkpoints"] == "missing"
    after = lab.ctx.db.all(
        "SELECT name, integrity, size_bytes FROM artifacts WHERE attempt_id=? AND name LIKE 'checkpoints/%'",
        (indexed.attempt_id,),
    )
    assert {r["integrity"] for r in after} == {"missing"}, after
    assert all(r["size_bytes"] is None for r in after)
    with pytest.raises(NotFound):  # the member reference now says so instead of "finalized"
        artifact_read(
            lab.ctx, artifact_ref=f"{ref(lab, indexed, 'checkpoints')}#ckpt-000000060/hpclab-ckpt.json"
        )


# ---------------------------------------------------------------------------
# index: a long outage is retried until the result is established
# ---------------------------------------------------------------------------


def test_result_left_unknown_by_an_outage_is_swept_again(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    lab.transport.down = conn_lost()
    item = OutboxItem(1, "artifacts.index", None, att.attempt_id, {"attempt_id": att.attempt_id}, 30)
    res = handle_index(lab.ctx, item)
    assert res is not None and res.action == "done"  # the item's own retries are bounded
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "unknown"

    lab.clock.advance(120)
    assert sweep_unindexed(lab.ctx) == 1  # never observed: worth trying again
    assert sweep_unindexed(lab.ctx) == 0  # queued once
    lab.transport.down = None
    runner = OutboxRunner(lab.ctx)
    runner.register_all(artifacts.outbox_handlers())
    runner.drain()
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "valid"


def test_unknown_results_that_were_observed_are_not_swept_forever(tmp_path: Any) -> None:
    lab = Lab(tmp_path, max_login_hash_bytes=16)  # digests unknowable: a permanent "unknown"
    try:
        att = lab.attempt()
        lab.good_outputs(att, checkpoint=False)
        lab.manifest(att, mode="quick")
        assert index_attempt(lab.ctx, attempt_id=att.attempt_id).result_state == "unknown"
        lab.clock.advance(120)
        assert sweep_unindexed(lab.ctx) == 0
    finally:
        lab.close()


def test_outage_sweep_has_a_horizon(lab: Lab) -> None:
    att = lab.attempt()
    lab.transport.down = conn_lost()
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    lab.clock.advance(30 * 86400)
    assert sweep_unindexed(lab.ctx) == 0  # a cluster gone for a month is not polled forever


# ---------------------------------------------------------------------------
# artifact_read: refresh of a mutable record vs a concurrent index
# ---------------------------------------------------------------------------


def test_mutable_refresh_never_overwrites_a_concurrently_finalized_record(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    path = att.write("outputs/metrics.jsonl", metrics_jsonl(final=False))
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    art = lab.artifact(att, "metrics")
    assert art["integrity"] == "mutable"
    with path.open("a") as fh:  # the job keeps writing: the read will want to refresh the record
        fh.write(json.dumps({"schema": "hpclab.metrics/v1", "step": 40, "metrics": {"loss": 0.4}}) + "\n")
    real = lab.transport.stat

    def stat_while_the_job_ends(paths: list[str], **kw: Any) -> Any:
        out = real(paths, **kw)
        # meanwhile the job ended and the index finalized the record (other thread)
        with lab.ctx.db.tx() as tx:
            tx.execute(
                "UPDATE artifacts SET integrity='finalized', digest=?, digest_state='computed', "
                "digest_source='compute_node', size_bytes=?, version=version+1 WHERE artifact_id=?",
                ("sha256:" + "d" * 64, 12345, art["artifact_id"]),
            )
        return out

    monkeypatch.setattr(lab.transport, "stat", stat_while_the_job_ends)
    with pytest.raises(Stale):
        artifact_read(lab.ctx, artifact_ref=art["artifact_id"])
    row = lab.artifact(att, "metrics")
    assert row["integrity"] == "finalized" and row["size_bytes"] == 12345  # untouched
    assert not lab.events("artifact.observed")


# ---------------------------------------------------------------------------
# vault: the evidence copy of a metrics file is never replaced by other bytes
# ---------------------------------------------------------------------------


def _wrapper_files(att: Att) -> None:
    att.write("job.sbatch", "#!/bin/bash\necho hi\n")
    att.write("hpclab-plan.json", json.dumps({"attempt_id": att.attempt_id}))
    att.write("status.json", json.dumps({"schema": "hpclab.status/v1", "state": "exited"}))
    att.write("env.json", json.dumps({"schema": "hpclab.env/v1"}))
    att.write("logs/stdout.log", "ok\n")
    att.write("logs/stderr.log", "")


def test_recapture_keeps_the_verified_metrics_copy(lab: Lab, indexed: Att) -> None:
    _wrapper_files(indexed)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    first = capture(lab.ctx, attempt_id=indexed.attempt_id)
    kept = next(i for i in first.captured if i.kind == "metrics_file" and i.local_path.endswith(".jsonl"))
    recorded = lab.artifact(indexed, "metrics")["digest"]
    assert kept.digest == recorded
    original = Path(kept.local_path).read_bytes()

    # after the evaluation, the scratch copy is overwritten (a re-run into the same dir, a script)
    indexed.path("outputs/metrics.jsonl").write_text(metrics_jsonl(acc=0.1))
    again = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert Path(kept.local_path).read_bytes() == original  # the evidence the number came from
    row = lab.ctx.db.one(
        "SELECT digest FROM vault_items WHERE item_id=?",
        (kept.item_id,),
    )
    assert row is not None and row["digest"] == recorded
    assert not any(i.item_id == kept.item_id for i in again.captured)
    assert any("differ" in w for w in again.warnings)


# ---------------------------------------------------------------------------
# vault: backups hold only finished copies; every ended attempt gets captured
# ---------------------------------------------------------------------------


def test_backup_skips_in_progress_capture_files(lab: Lab, indexed: Att, tmp_path: Any) -> None:
    from hpclab.services.vault import backup, vault_root

    _wrapper_files(indexed)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    capture(lab.ctx, attempt_id=indexed.attempt_id)
    # a capture running concurrently (or one that crashed) leaves partial downloads here
    staging = vault_root(lab.paths) / indexed.attempt_id / ".incoming"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "metrics.1a2b3c4d").write_bytes(b"half a file")
    data = backup(lab.ctx, dest=str(tmp_path / "bk"))
    manifest = json.loads(Path(data.manifest_path).read_text())
    assert not any(".incoming" in f["path"] for f in manifest["files"])
    assert data.vault_files == len(manifest["files"]) - data.snapshot_manifests


def test_sweep_captures_every_ended_attempt_vault_status_counts(lab: Lab) -> None:
    from hpclab.services.vault import handle_capture, sweep_uncaptured, vault_status

    rejected = lab.attempt(phase="submission_rejected", scheduler_state="UNKNOWN", job_id=None)
    rejected.write("job.sbatch", "#!/bin/bash\n")
    abandoned = lab.attempt(phase="abandoned", scheduler_state="UNKNOWN", job_id="2002")
    abandoned.write("job.sbatch", "#!/bin/bash\n")
    assert vault_status(lab.ctx).not_captured == 2
    lab.clock.advance(3600)
    assert sweep_uncaptured(lab.ctx) == 2
    for row in lab.ctx.db.all("SELECT * FROM outbox WHERE kind='vault.capture' ORDER BY outbox_id"):
        handle_capture(
            lab.ctx,
            OutboxItem(
                row["outbox_id"], row["kind"], None, row["target_id"], json.loads(row["payload_json"]), 1
            ),
        )
    assert vault_status(lab.ctx).not_captured == 0


# ---------------------------------------------------------------------------
# artifact_read: the offline cache path re-verifies what it calls verified
# ---------------------------------------------------------------------------


def test_offline_read_never_serves_a_corrupted_cache_copy_as_verified(lab: Lab, indexed: Att) -> None:
    from hpclab.adapters.base import TransportError
    from hpclab.services.artifacts import download_to_cache

    cached = download_to_cache(lab.ctx, lab.artifact(indexed, "summary"))
    cached.chmod(0o644)
    cached.write_bytes(cached.read_bytes().replace(b"0.9", b"0.1"))  # local disk damage
    lab.transport.down = conn_lost()
    with pytest.raises(TransportError):  # no verified copy left: the outage is reported
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    assert not cached.exists()  # the damaged entry is dropped (re-downloaded later)


def test_member_transitions_are_audited_by_name(lab: Lab, indexed: Att) -> None:
    indexed.path("checkpoints/ckpt-000000020/model.bin").unlink()  # keep_last rotated it away
    index_attempt(lab.ctx, attempt_id=indexed.attempt_id)
    events = lab.events("artifact.members_indexed")
    last = json.loads(events[-1]["payload_json"])
    assert {"member": "ckpt-000000020/model.bin", "from": "finalized", "to": "missing"} in last["changes"]


def test_json_shape_summary_is_bounded_by_the_text_limit(lab: Lab) -> None:
    outputs = [{"name": "odd", "path": "{output_dir}/odd.json", "role": "other"}]
    att = lab.attempt(outputs=outputs)
    att.write("outputs/odd.json", json.dumps({("k" * 200_000) + str(i): i for i in range(10)}))
    lab.manifest(att, outputs=outputs)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    data = artifact_read(lab.ctx, artifact_ref=ref(lab, att, "odd"), view="json", max_bytes=4096)
    assert data.value is None and data.value_summary is not None and data.truncated
    assert len(data.model_dump_json()) < 64 * 1024
    assert data.value_summary["key_count"] == 10


# ---------------------------------------------------------------------------
# §4: network I/O never happens inside a ledger transaction
# ---------------------------------------------------------------------------


def test_no_remote_call_runs_inside_a_transaction(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    from hpclab.services.evaluations import evaluation_collect
    from hpclab.services.vault import backup

    calls: list[str] = []

    def guard(name: str) -> None:
        calls.append(name)
        assert not lab.ctx.db.conn.in_transaction, f"{name} called inside db.tx()"

    for name in ("stat", "read_range", "read_tail", "sha256", "list_dir", "download"):
        real = getattr(lab.transport, name)

        def wrapped(*a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
            guard(_name)
            return _real(*a, **kw)

        monkeypatch.setattr(lab.transport, name, wrapped)
    real_acct = lab.scheduler.query_accounting

    def acct(**kw: Any) -> Any:
        guard("sacct")
        return real_acct(**kw)

    monkeypatch.setattr(lab.scheduler, "query_accounting", acct)

    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att, mode="quick")  # forces the login-node re-hash path too
    _wrapper_files(att)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    artifact_read(lab.ctx, artifact_ref=ref(lab, att, "metrics"))
    artifact_read(lab.ctx, artifact_ref=f"{ref(lab, att, 'checkpoints')}#ckpt-000000060/model.bin")
    runner = OutboxRunner(lab.ctx)
    runner.register_all(artifacts.outbox_handlers())
    runner.drain()  # the artifact.download operation
    evaluation_collect(lab.ctx, idempotency_key="io", attempt_id=att.attempt_id)
    capture(lab.ctx, attempt_id=att.attempt_id)
    backup(lab.ctx, dest=str(lab.tmp / "bk"))
    assert {"stat", "read_range", "read_tail", "sha256", "list_dir", "download", "sacct"} <= set(calls)


def test_directory_is_not_finalized_on_an_old_tree_digest(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    from hpclab.adapters.base import TransportTimeout

    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att, mode="quick")  # the job was killed before its hashing pass
    index_attempt(lab.ctx, attempt_id=att.attempt_id)  # login node hashes the members
    first = lab.artifact(att, "checkpoints")
    assert first["integrity"] == "finalized" and first["digest_source"] == "login_node"
    # a member is rewritten in place with the same size: neither the member total nor the
    # directory's mtime changes
    member = att.path("checkpoints/ckpt-000000060/model.bin")
    member.write_bytes(b"WEIGHTS-60")
    lab.manifest(att, mode="quick")

    def no_hash(*a: Any, **kw: Any) -> Any:
        raise TransportTimeout("sha256 timed out")

    monkeypatch.setattr(lab.transport, "sha256", no_hash)  # this time nothing can be re-hashed
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    ck = lab.artifact(att, "checkpoints")
    assert ck["integrity"] != "finalized" or ck["digest"] != first["digest"], ck
    assert lab.artifact(att, "checkpoints/ckpt-000000060/model.bin")["integrity"] != "finalized"


def test_reindexing_keeps_a_login_hashed_directory_finalized(lab: Lab) -> None:
    att = lab.attempt(
        outputs=[
            {"name": "summary", "path": "{output_dir}/metrics.json", "role": "metrics"},
            {"name": "checkpoints", "path": "{checkpoint_dir}", "kind": "dir", "role": "checkpoint"},
        ]
    )
    lab.good_outputs(att)
    lab.manifest(att, mode="quick", outputs=[
        {"name": "summary", "path": "{output_dir}/metrics.json", "role": "metrics"},
        {"name": "checkpoints", "path": "{checkpoint_dir}", "kind": "dir", "role": "checkpoint"},
    ])  # fmt: skip
    first = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert first.result_state == "valid"
    digest = lab.artifact(att, "checkpoints")["digest"]
    lab.transport.calls.clear()
    again = index_attempt(lab.ctx, attempt_id=att.attempt_id)  # e.g. artifacts_list(refresh=true)
    assert again.result_state == "valid" and not again.changed
    ck = lab.artifact(att, "checkpoints")
    assert ck["integrity"] == "finalized" and ck["digest"] == digest
    assert "sha256" not in lab.transport.calls  # unchanged members are not re-hashed
