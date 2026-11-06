"""Notify channel targets (webhook / ntfy URLs, command lines) never enter the ledger.

``clusters.overlay_json`` records each channel as ``{channel_id, kind, events}``; the target
stays in the private 0600 overlay file (and in the registering process's memory) and is read
back only when a message is sent. The ledger is the file users back up and attach to bug
reports, so a webhook secret stored there would leak with every copy.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.profiles.loader import save_overlay
from hpclab.profiles.schema import ClusterProfile, NotifyChannel, UserOverlay
from hpclab.services import notify
from hpclab.services.context import ServiceContext, notify_channel_id
from hpclab.services.outbox import OutboxRunner
from hpclab.storage.db import jdump

SECRET = "SECRETXYZ"
WEBHOOK = f"https://hooks.slack.com/services/T000/B000/{SECRET}"
NTFY = f"https://ntfy.sh/hpclab-alice-{SECRET}"
RECORDER = r"""
import sys
with open(sys.argv[1], "a") as fh:
    fh.write(sys.stdin.read().replace("\n", " ") + "\n")
"""


def _profile(cluster_id: str = "c1") -> ClusterProfile:
    return ClusterProfile(cluster_id=cluster_id, display_name=cluster_id.upper(), transport="local")


def _overlay(*channels: NotifyChannel, cluster_id: str = "c1") -> UserOverlay:
    return UserOverlay(cluster_id=cluster_id, user="alice", lab_users=["bob"], notify=list(channels))


def _recorder(tmp_path: Path, inbox: Path) -> NotifyChannel:
    script = tmp_path / f"recorder-{SECRET}.py"
    script.write_text(RECORDER)
    return NotifyChannel(kind="command", target=f"{sys.executable} {script} {inbox}", events=["*"])


def _ledger_text(ctx: ServiceContext) -> str:
    return "\n".join(ctx.db.conn.iterdump())


def _drain(ctx: ServiceContext) -> int:
    runner = OutboxRunner(ctx)
    runner.register_all(notify.outbox_handlers())
    return runner.drain()


def _emit(ctx: ServiceContext, event: str = "attempt.uncertain", entity: str = "att_1") -> list[str]:
    with ctx.db.tx(source="worker") as tx:
        return notify.emit(tx, event, entity, "submission uncertain", "reconciling")


@pytest.fixture
def ctx(paths: Paths) -> Iterator[ServiceContext]:
    c = ServiceContext.create(paths, clock=FakeClock())
    yield c
    c.clusters.close()
    c.db.close()


def test_register_keeps_notify_targets_out_of_the_ledger(ctx: ServiceContext, tmp_path: Path) -> None:
    rec = _recorder(tmp_path, tmp_path / "inbox.jsonl")
    overlay = _overlay(
        NotifyChannel(kind="slack_webhook", target=WEBHOOK),
        NotifyChannel(kind="ntfy", target=NTFY, events=["*"]),
        rec,
        NotifyChannel(kind="log"),
    )
    ctx.clusters.register(_profile(), overlay)

    row = ctx.db.one("SELECT overlay_json FROM clusters WHERE cluster_id='c1'")
    assert row is not None and SECRET not in row["overlay_json"]
    assert SECRET not in _ledger_text(ctx)  # nowhere: not in events either
    stored = json.loads(row["overlay_json"])
    assert stored["user"] == "alice" and stored["lab_users"] == ["bob"]  # the rest is kept
    assert stored["notify"] == [
        {"channel_id": notify_channel_id(ch.kind, ch.target), "kind": ch.kind, "events": ch.events}
        for ch in overlay.notify
    ]
    # the registering process still resolves the full overlay (round trip is lossless)
    assert ctx.clusters.get("c1").overlay == overlay
    ctx.clusters.register(_profile(), ctx.clusters.get("c1").overlay)
    assert ctx.clusters.get("c1").overlay == overlay
    # notify sees channel ids and events only
    channels = notify.configured_channels(ctx.db)
    assert {c.channel_id for c in channels} == {
        notify_channel_id(ch.kind, ch.target) for ch in overlay.notify
    }
    assert all(c.target is None for c in channels)


def test_notify_channel_id_matches_notify_module() -> None:
    assert notify.channel_id("ntfy", NTFY) == notify_channel_id("ntfy", NTFY)
    assert notify_channel_id("log", None) == notify_channel_id("log", "")


def test_a_new_process_reads_targets_from_the_private_overlay_file(paths: Paths, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox.jsonl"
    webhook = NotifyChannel(kind="slack_webhook", target=WEBHOOK, events=["run.succeeded"])
    overlay = _overlay(_recorder(tmp_path, inbox), webhook)
    first = ServiceContext.create(paths, clock=FakeClock())
    save_overlay(overlay, paths)  # what cluster_register does before registering
    first.clusters.register(_profile(), overlay)
    first.db.close()

    # e.g. the controller restarted: nothing in memory, the target comes from the 0600 file
    ctx = ServiceContext.create(paths, clock=FakeClock())
    try:
        assert ctx.clusters.get("c1").overlay == overlay
        assert ctx.clusters.notify_channels("c1") == overlay.notify
        (nid,) = _emit(ctx)
        _drain(ctx)
        (line,) = inbox.read_text().splitlines()
        assert json.loads(line)["notification_ids"] == [nid]
        state = ctx.db.one("SELECT state FROM notifications WHERE notification_id=?", (nid,))
        assert state is not None and state["state"] == "sent"
        assert SECRET not in _ledger_text(ctx)
    finally:
        ctx.db.close()


def test_a_target_changed_in_the_file_without_re_registering_is_not_used(
    paths: Paths, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inbox, other = tmp_path / "inbox.jsonl", tmp_path / "other.jsonl"
    registered = _overlay(_recorder(tmp_path, inbox))
    first = ServiceContext.create(paths, clock=FakeClock())
    first.clusters.register(_profile(), registered)
    first.db.close()
    save_overlay(_overlay(_recorder(tmp_path, other)), paths)  # edited, never re-registered

    ctx = ServiceContext.create(paths, clock=FakeClock())
    try:
        with caplog.at_level(logging.WARNING, logger="hpclab.context"):
            assert ctx.clusters.get("c1").overlay.notify == []
            assert ctx.clusters.get("c1").overlay.notify == []
        warnings = [r for r in caplog.records if "not in the overlay file" in r.getMessage()]
        assert len(warnings) == 1  # once per file version, not on every lookup
        assert SECRET not in caplog.text
        _emit(ctx)  # the ledger still lists the registered channel ...
        _drain(ctx)
        assert not inbox.exists() and not other.exists()  # ... but no target is guessed
        row = ctx.db.one("SELECT state, last_error FROM notifications")
        assert row is not None and row["state"] == "queued"  # retried: the file may be fixed
        assert "re-register" in row["last_error"] and SECRET not in row["last_error"]
        result = notify.notify_test(ctx)
        (ch,) = result.channels
        assert not ch.ok and "re-register" in (ch.error or "") and SECRET not in (ch.error or "")
    finally:
        ctx.db.close()


def test_rows_written_before_the_fix_are_redacted_on_open(paths: Paths) -> None:
    overlay = _overlay(NotifyChannel(kind="slack_webhook", target=WEBHOOK), NotifyChannel(kind="log"))
    first = ServiceContext.create(paths, clock=FakeClock())
    with first.db.tx(source="api") as tx:  # the old format: the whole overlay, targets included
        tx.insert(
            "clusters",
            {
                "cluster_id": "c1",
                "profile_json": jdump(_profile().model_dump(mode="json", by_alias=True)),
                "overlay_json": jdump(overlay.model_dump(mode="json")),
                "profile_revision": "r1",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
    # a process that is still running with the old row can resolve it
    assert first.clusters.get("c1").overlay == overlay
    first.db.close()

    ctx = ServiceContext.create(paths, clock=FakeClock())
    try:
        row = ctx.db.one("SELECT overlay_json, profile_revision FROM clusters WHERE cluster_id='c1'")
        assert row is not None and SECRET not in row["overlay_json"] and row["profile_revision"] == "r1"
        assert SECRET not in _ledger_text(ctx)
        assert ctx.clusters.get("c1").overlay == overlay  # still usable in the redacting process
        assert ctx.clusters.redact_stored_notify_targets() == 0  # idempotent
    finally:
        ctx.db.close()


def test_backup_of_the_ledger_carries_no_notify_secret(ctx: ServiceContext, tmp_path: Path) -> None:
    from hpclab.services.vault import backup

    ctx.clusters.register(_profile(), _overlay(NotifyChannel(kind="slack_webhook", target=WEBHOOK)))
    backup(ctx, dest=str(tmp_path / "nas"))
    (db_file,) = (tmp_path / "nas").glob("hpclab-backup-*/hpclab.db")
    con = sqlite3.connect(db_file)
    try:
        dump = "\n".join(con.iterdump())
    finally:
        con.close()
    assert "hooks.slack.com" not in dump and SECRET not in dump
    for f in db_file.parent.rglob("*"):
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f


def test_cluster_register_keeps_the_user_edited_target_in_the_file_only(ctx: ServiceContext) -> None:
    from hpclab.profiles.loader import load_overlay
    from hpclab.services.clusters import cluster_register

    # the user adds a webhook to the private overlay file, then (re-)registers the cluster
    save_overlay(_overlay(NotifyChannel(kind="slack_webhook", target=WEBHOOK), cluster_id="sim"), ctx.paths)
    cluster_register(ctx, profile="sim", user="alice", account="lab")
    assert SECRET not in _ledger_text(ctx)
    on_disk = load_overlay("sim", ctx.paths)
    assert on_disk is not None and [c.target for c in on_disk.notify] == [WEBHOOK]
    assert [c.target for c in ctx.clusters.notify_channels("sim")] == [WEBHOOK]


def test_an_unreadable_overlay_file_does_not_break_lookups(
    paths: Paths, caplog: pytest.LogCaptureFixture
) -> None:
    first = ServiceContext.create(paths, clock=FakeClock())
    first.clusters.register(
        _profile(), _overlay(NotifyChannel(kind="slack_webhook", target=WEBHOOK), NotifyChannel(kind="log"))
    )
    first.db.close()
    overlay_file = paths.overlays / "c1.yaml"
    overlay_file.write_text(f"cluster_id: c1\nnotify: [{{kind: slack_webhook, target: {WEBHOOK}\n")  # broken
    overlay_file.chmod(0o600)

    ctx = ServiceContext.create(paths, clock=FakeClock())
    try:
        with caplog.at_level(logging.WARNING, logger="hpclab.context"):
            rc = ctx.clusters.get("c1")
        assert rc.overlay.user == "alice" and [c.kind for c in rc.overlay.notify] == ["log"]
        assert "cannot read notify targets" in caplog.text and SECRET not in caplog.text
    finally:
        ctx.db.close()
