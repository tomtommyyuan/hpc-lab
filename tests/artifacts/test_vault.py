"""Vault capture / status / backup."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from artifacts_support import FINAL, Att, Lab, accounting_record, auth_error, conn_lost, metrics_jsonl

from hpclab.domain.errors import InvalidArgument, NotReady
from hpclab.domain.states import SchedulerState
from hpclab.services import vault
from hpclab.services.artifacts import index_attempt
from hpclab.services.outbox import OutboxItem
from hpclab.services.vault import backup, capture, handle_capture, sweep_uncaptured, vault_status


def wrapper_files(att: Att, *, stdout_bytes: int = 400 * 1024) -> None:
    att.write("job.sbatch", "#!/bin/bash\n#SBATCH --job-name=hl.run.abcdefgh\necho hi\n")
    att.write("hpclab-plan.json", json.dumps({"attempt_id": att.attempt_id, "token": "abcdefgh1234"}))
    att.write("status.json", json.dumps({"schema": "hpclab.status/v1", "state": "exited", "exit_code": 0}))
    att.write("env.json", json.dumps({"schema": "hpclab.env/v1", "host": "node1"}))
    att.write("logs/stdout.log", ("step line\n" * (stdout_bytes // 10 + 1))[:stdout_bytes])
    att.write("logs/stderr.log", "warning: small\n")


def item(att: Att, tries: int = 1) -> OutboxItem:
    return OutboxItem(1, "vault.capture", None, att.attempt_id, {"attempt_id": att.attempt_id}, tries)


def test_capture_keeps_bounded_verified_copies(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    kinds = sorted(i.kind for i in res.captured)
    assert kinds == sorted(
        [
            "script",
            "plan",
            "status",
            "env",
            "outputs_manifest",
            "log_head",
            "log_head",
            "log_tail",
            "metrics_file",
            "metrics_file",
            "sacct_raw",
        ]
    )
    assert res.missing == [] and res.unavailable == [] and res.retry_after_s is None and res.coverage.complete
    for it in res.captured:
        local = Path(it.local_path)
        assert local.is_file() and local.is_relative_to(lab.paths.home / "vault" / indexed.attempt_id)
        assert "sha256:" + hashlib.sha256(local.read_bytes()).hexdigest() == it.digest
        assert it.size_bytes == local.stat().st_size
    by = {(i.kind, Path(i.local_path).name): i for i in res.captured}
    head = by[("log_head", "stdout.head")]
    tail = by[("log_tail", "stdout.tail")]
    assert (
        head.size_bytes == 64 * 1024 and head.truncated and tail.size_bytes == 256 * 1024 and tail.truncated
    )
    assert not by[("log_head", "stderr.head")].truncated
    assert res.source_sizes[f"log_head:{indexed.path('logs/stdout.log')}"] == 400 * 1024
    sacct = json.loads(Path(by[("sacct_raw", "sacct.json")].local_path).read_text())
    assert sacct["records"][0]["job_id"] == "1001" and sacct["records"][0]["elapsed_s"] == 42
    metrics = by[("metrics_file", "metrics.jsonl")]
    assert metrics.digest == lab.artifact(indexed, "metrics")["digest"]

    again = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert again.captured == [] and again.unchanged == len(res.captured)
    assert len(lab.events("vault.captured")) == 2
    rows = lab.ctx.db.all("SELECT * FROM vault_items WHERE attempt_id=?", (indexed.attempt_id,))
    assert len(rows) == len(res.captured)


def test_changed_source_is_recaptured(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    capture(lab.ctx, attempt_id=indexed.attempt_id)
    indexed.write("logs/stdout.log", "requeued segment output\n")
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert [i.kind for i in res.captured] == ["log_head"]
    assert Path(res.captured[0].local_path).read_text() == "requeued segment output\n"


def test_accounting_lag_is_retried(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert "sacct_raw" in res.missing and res.retry_after_s == 300.0
    handled = handle_capture(lab.ctx, item(indexed))
    assert handled is not None and handled.action == "retry"
    handled = handle_capture(lab.ctx, item(indexed, tries=99))
    assert handled is not None and handled.action == "done"  # accounting may never show it


def test_unreachable_cluster_captures_nothing_and_retries(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    lab.transport.down = conn_lost()
    lab.scheduler.down = conn_lost()
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert res.captured == [] and res.unavailable and res.coverage.complete is False
    assert res.retry_after_s == 60.0
    assert lab.events("vault.capture_incomplete") and not lab.events("vault.captured")
    lab.transport.down = auth_error()
    handled = handle_capture(lab.ctx, item(indexed, tries=999))
    assert handled is not None and handled.action == "retry" and handled.delay_s == 120.0


def test_running_attempt_is_not_captured(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    with pytest.raises(NotReady):
        capture(lab.ctx, attempt_id=att.attempt_id)


def test_rejected_attempt_keeps_only_staged_files(lab: Lab) -> None:
    att = lab.attempt(phase="submission_rejected", scheduler_state="UNKNOWN", job_id=None)
    wrapper_files(att, stdout_bytes=10)
    res = capture(lab.ctx, attempt_id=att.attempt_id)
    assert sorted(i.kind for i in res.captured) == ["plan", "script"] and res.warnings


def test_status_counts_and_pages(lab: Lab) -> None:
    exp = lab.experiment()
    a = lab.ready(experiment=exp, seed=1)
    b = lab.ready(experiment=exp, seed=2)
    wrapper_files(a, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    capture(lab.ctx, attempt_id=a.attempt_id)
    wrapper_files(b, stdout_bytes=100)
    b.path("env.json").unlink()
    capture(lab.ctx, attempt_id=b.attempt_id)
    lab.attempt(experiment=exp, seed=3)  # ended, never captured
    other = lab.ready(seed=1)  # other experiment
    capture(lab.ctx, attempt_id=other.attempt_id)

    st = vault_status(lab.ctx, experiment_id=exp[0])
    assert st.attempts_in_scope == 3 and st.not_captured == 1
    assert st.fully_captured == 1 and st.partially_captured == 1
    by = {x.attempt_id: x for x in st.attempts}
    assert by[b.attempt_id].missing_kinds == ["env"] and by[a.attempt_id].missing_kinds == []
    assert st.total_items == by[a.attempt_id].items + by[b.attempt_id].items
    page = vault_status(lab.ctx, limit=2)
    assert len(page.attempts) == 2 and page.next_cursor
    rest = vault_status(lab.ctx, limit=2, cursor=page.next_cursor)
    assert len(rest.attempts) == 2 and rest.next_cursor is None
    local = lab.ctx.db.all(
        "SELECT local_path FROM vault_items WHERE attempt_id=? ORDER BY kind", (a.attempt_id,)
    )
    Path(local[0]["local_path"]).unlink()
    damaged = vault_status(lab.ctx, experiment_id=exp[0])
    assert damaged.coverage.complete is False
    assert {x.attempt_id: x for x in damaged.attempts}[a.attempt_id].missing_local_files == 1


def test_backup_is_a_consistent_verifiable_copy(lab: Lab, indexed: Att, tmp_path: Path) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    capture(lab.ctx, attempt_id=indexed.attempt_id)
    manifests = lab.paths.snapshots / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / ("ab" * 32 + ".json")).write_text('{"schema": "hpclab.snapshot/v1", "files": []}')

    data = backup(lab.ctx, dest=str(tmp_path / "backups"), idempotency_key="bk-1")
    assert data.integrity_check == "ok" and data.vault_files >= 10 and data.snapshot_manifests == 1
    conn = sqlite3.connect(data.db_path)
    try:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE attempt_id=?", (indexed.attempt_id,)
        ).fetchone()
    finally:
        conn.close()
    assert n == 7
    manifest = json.loads(Path(data.manifest_path).read_text())
    assert manifest["db"]["sha256"] == data.db_sha256
    for f in manifest["files"]:
        p = Path(data.backup_dir) / f["path"]
        assert "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
    assert (
        "sha256:" + hashlib.sha256(Path(data.manifest_path).read_bytes()).hexdigest() == data.manifest_digest
    )

    replay = backup(lab.ctx, dest=str(tmp_path / "backups"), idempotency_key="bk-1")
    assert replay == data and len(list((tmp_path / "backups").iterdir())) == 1
    second = backup(lab.ctx, dest=str(tmp_path / "backups"))
    assert second.backup_dir != data.backup_dir  # same second: a distinct directory
    assert len(lab.events("backup.created")) == 2


def test_backup_refuses_unsafe_destinations(lab: Lab) -> None:
    with pytest.raises(InvalidArgument):
        backup(lab.ctx, dest="relative/dir")
    with pytest.raises(InvalidArgument):
        backup(lab.ctx, dest=str(lab.paths.home / "backups"))


def test_sweep_enqueues_uncaptured_attempts_once(lab: Lab, indexed: Att) -> None:
    assert sweep_uncaptured(lab.ctx) == 0
    lab.clock.advance(3600)
    assert sweep_uncaptured(lab.ctx) == 1
    assert sweep_uncaptured(lab.ctx) == 0
    wrapper_files(indexed, stdout_bytes=10)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    with lab.ctx.db.tx() as tx:
        tx.execute("UPDATE outbox SET state='done' WHERE kind='vault.capture'")
    capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert sweep_uncaptured(lab.ctx) == 0  # captured: never again


def test_handlers_and_periodic_tasks_are_exported() -> None:
    assert set(vault.outbox_handlers()) == {"vault.capture"}
    assert [t.name for t in vault.periodic_tasks()] == ["vault.sweep"]


def test_cluster_without_accounting_is_reported_not_retried(lab: Lab, indexed: Att) -> None:
    from hpclab.adapters.base import TransportError
    from hpclab.domain.errors import ErrorCode

    wrapper_files(indexed, stdout_bytes=10)
    lab.scheduler.down = TransportError("sacct: accounting storage is disabled", code=ErrorCode.UNSUPPORTED)
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert res.retry_after_s is None and any("sacct_raw" in u for u in res.unavailable)
    assert "sacct_raw" not in {i.kind for i in res.captured}
    handled = handle_capture(lab.ctx, item(indexed))
    assert handled is not None and handled.action == "done"


# ---------------------------------------------------------------------------
# metrics files are confined to the attempt directory (symlinks are resolved)
# ---------------------------------------------------------------------------


def _link_summary(lab: Lab, att: Att, target: Path) -> Att:
    """The job made the declared metrics file ``summary`` (outputs/metrics.json) a symlink."""
    att.write("outputs/metrics.jsonl", metrics_jsonl())
    os.symlink(target, att.path("outputs/metrics.json"))
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    wrapper_files(att, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    lab.transport.calls.clear()
    return att


def _vault_dir(lab: Lab, att: Att) -> Path:
    return lab.paths.home / "vault" / att.attempt_id


def _metrics_captured(res: vault.VaultCaptureData) -> dict[str, Path]:
    return {
        Path(i.local_path).parent.name: Path(i.local_path) for i in res.captured if i.kind == "metrics_file"
    }


def test_metrics_symlink_to_an_outside_file_is_never_copied(lab: Lab, tmp_path: Path) -> None:
    secret = tmp_path / "cluster_home" / ".git-credentials"
    secret.parent.mkdir()
    secret.write_text("https://user:ghp_SUPERSECRETTOKEN@github.com\n")
    att = _link_summary(lab, lab.attempt(), secret)

    res = capture(lab.ctx, attempt_id=att.attempt_id)
    assert set(_metrics_captured(res)) == {"metrics"}
    assert lab.transport.calls.count("download") == 1  # only the confined metrics.jsonl
    refused = [u for u in res.unavailable if u.startswith("metrics_file:summary:")]
    assert len(refused) == 1 and "outside the attempt directory" in refused[0]
    assert res.retry_after_s is None  # a refusal, not an outage: retrying cannot change it
    for p in _vault_dir(lab, att).rglob("*"):
        assert p.is_dir() or b"SUPERSECRETTOKEN" not in p.read_bytes(), p
    assert not (_vault_dir(lab, att) / ".incoming").exists()
    handled = handle_capture(lab.ctx, item(att))
    assert handled is not None and handled.action == "done"


def test_metrics_symlink_to_an_outside_directory_is_never_mirrored(lab: Lab, tmp_path: Path) -> None:
    home = tmp_path / "cluster_home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY")
    (home / "big.bin").write_bytes(b"\0" * 1024 * 1024)
    att = _link_summary(lab, lab.attempt(), home)

    res = capture(lab.ctx, attempt_id=att.attempt_id)  # never IsADirectoryError
    assert set(_metrics_captured(res)) == {"metrics"}
    assert lab.transport.calls.count("download") == 1
    assert any(u.startswith("metrics_file:summary:") and "outside" in u for u in res.unavailable)
    root = _vault_dir(lab, att)
    assert not (root / ".incoming").exists()
    assert not list(root.rglob("id_ed25519")) and not list(root.rglob("big.bin"))
    assert lab.events("vault.captured")


def test_metrics_symlink_to_a_directory_inside_the_attempt_is_not_downloaded(lab: Lab) -> None:
    att = lab.attempt()
    att.write("checkpoints/ckpt-000000020/model.bin", b"weights-20")
    att = _link_summary(lab, att, att.path("checkpoints"))

    res = capture(lab.ctx, attempt_id=att.attempt_id)
    assert set(_metrics_captured(res)) == {"metrics"}
    assert lab.transport.calls.count("download") == 1
    assert "metrics_file:summary" in res.missing
    root = _vault_dir(lab, att)
    assert not (root / ".incoming").exists() and not list(root.rglob("model.bin"))


def test_metrics_symlink_inside_the_attempt_is_still_captured(lab: Lab) -> None:
    att = lab.attempt()
    real = att.write("outputs/final/metrics.json", json.dumps(dict(FINAL, split="test")))
    att = _link_summary(lab, att, real)

    res = capture(lab.ctx, attempt_id=att.attempt_id)
    got = _metrics_captured(res)
    assert set(got) == {"metrics", "summary"}
    assert got["summary"].read_bytes() == real.read_bytes()
    assert not [u for u in res.unavailable if u.startswith("metrics_file")]


def test_attempt_dir_behind_a_symlinked_prefix_is_still_captured(lab: Lab, tmp_path: Path) -> None:
    """Scratch roots are often symlinks (/scratch -> /mnt/...): containment compares realpaths."""
    link = tmp_path / "scr"
    os.symlink(lab.remote_root, link)
    att = lab.attempt()
    linked = dataclasses.replace(att, dir=link / "attempts" / att.attempt_id)
    lab.set_attempt(att, remote_dir=str(linked.dir))
    lab.good_outputs(linked)
    lab.manifest(linked)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    wrapper_files(linked, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")

    res = capture(lab.ctx, attempt_id=att.attempt_id)
    assert set(_metrics_captured(res)) == {"metrics", "summary"}
    assert res.unavailable == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_metrics_path_that_is_not_a_regular_file_is_not_downloaded(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    indexed.path("outputs/metrics.json").unlink()
    os.mkfifo(indexed.path("outputs/metrics.json"))  # rsync -a would recreate it; hashing it blocks
    lab.transport.calls.clear()

    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert set(_metrics_captured(res)) == {"metrics"}
    assert lab.transport.calls.count("download") == 1
    assert any(u.startswith("metrics_file:summary:") and "regular file" in u for u in res.unavailable)


def test_download_that_is_not_a_regular_file_is_discarded(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path changed between the stat and the copy (or a transport mirrored a directory)."""
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")

    def mirror(remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        (Path(local_path) / ".ssh").mkdir(parents=True)
        (Path(local_path) / ".ssh" / "id_ed25519").write_text("PRIVATE KEY")

    monkeypatch.setattr(lab.transport, "download", mirror)
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)  # never IsADirectoryError
    assert _metrics_captured(res) == {}
    assert sum(1 for u in res.unavailable if u.startswith("metrics_file:") and "regular file" in u) == 2
    root = _vault_dir(lab, indexed)
    assert not (root / ".incoming").exists() and not list(root.rglob("id_ed25519"))


def test_incoming_downloads_are_removed_when_capture_fails(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")

    def disk_full(path: Path, data: bytes | None, src: Path | None) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vault, "_write_atomic", disk_full)
    with pytest.raises(OSError):
        capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert not (_vault_dir(lab, indexed) / ".incoming").exists()


# ---------------------------------------------------------------------------
# backup permissions; accounting already stored is reused
# ---------------------------------------------------------------------------


def test_backup_is_private_whatever_the_umask(lab: Lab, indexed: Att, tmp_path: Path) -> None:
    """The ledger copy holds notify secrets: the target is 0700 and every copied file 0600."""
    wrapper_files(indexed, stdout_bytes=100)
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    capture(lab.ctx, attempt_id=indexed.attempt_id)
    manifests = lab.paths.snapshots / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / ("cd" * 32 + ".json")).write_text('{"schema": "hpclab.snapshot/v1", "files": []}')
    for p in (lab.paths.home / "vault").rglob("*"):
        os.chmod(p, 0o777 if p.is_dir() else 0o666)  # world-readable sources are not copied as such
    old = os.umask(0)
    try:
        data = backup(lab.ctx, dest=str(tmp_path / "backups"))
    finally:
        os.umask(old)
    target = Path(data.backup_dir)
    assert (target.stat().st_mode & 0o777) == 0o700
    seen = 0
    for p in target.rglob("*"):
        mode = p.stat().st_mode & 0o777
        assert mode == (0o700 if p.is_dir() else 0o600), (p, oct(mode))
        seen += 1
    assert seen > data.vault_files + 2  # ledger, manifest, vault files, snapshot manifest, dirs


def _store_final_accounting(lab: Lab, att: Att) -> None:
    rec = accounting_record("1001").model_copy(update={"observed_at": "2026-09-01T12:30:00.000Z"})
    lab.set_attempt(att, last_observation_json=rec.model_dump_json())


def test_capture_reuses_the_stored_accounting_record(lab: Lab, indexed: Att) -> None:
    """The observer already applied the final sacct record: the vault keeps it, sacct is not asked."""
    wrapper_files(indexed, stdout_bytes=100)
    _store_final_accounting(lab, indexed)
    lab.scheduler.down = AssertionError("sacct must not be queried again")
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    assert res.missing == [] and res.unavailable == []
    (sacct,) = [i for i in res.captured if i.kind == "sacct_raw"]
    doc = json.loads(Path(sacct.local_path).read_text())
    assert doc["records"][0]["job_id"] == "1001" and doc["records"][0]["elapsed_s"] == 42
    assert doc["observed_at"] == "2026-09-01T12:30:00.000Z" and "stored" in doc["note"]


def test_capture_asks_sacct_when_no_final_record_is_stored(lab: Lab, indexed: Att) -> None:
    wrapper_files(indexed, stdout_bytes=100)
    running = accounting_record("1001").model_copy(
        update={"state": SchedulerState.RUNNING, "elapsed_s": None}
    )
    lab.set_attempt(indexed, last_observation_json=running.model_dump_json())
    lab.scheduler.jobs["1001"] = accounting_record("1001")
    res = capture(lab.ctx, attempt_id=indexed.attempt_id)
    (sacct,) = [i for i in res.captured if i.kind == "sacct_raw"]
    doc = json.loads(Path(sacct.local_path).read_text())
    assert doc["records"][0]["state"] == "COMPLETED" and "stored" not in doc["note"]
