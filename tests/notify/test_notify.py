"""Notifications: emit (in the caller's tx), batching window, immediate classes, dedupe,
payload hygiene, channels (command with JSON on stdin, log, ntfy/slack over HTTP), retries."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import InvalidArgument
from hpclab.profiles.schema import ClusterProfile, NotifyChannel, UserOverlay
from hpclab.services import notify
from hpclab.services.context import ServiceContext
from hpclab.services.notify import (
    MAX_TRIES,
    channel_id,
    emit,
    flush,
    handle_send,
    notify_test,
    outbox_handlers,
    periodic_tasks,
    sanitize_text,
)
from hpclab.services.outbox import OutboxItem, OutboxRunner

RECORDER = r"""
import json, sys
data = sys.stdin.read()
json.loads(data)  # the channel must receive valid JSON
with open(sys.argv[1], "a") as fh:
    fh.write(data.replace("\n", " ") + "\n")
"""
FAILER = "import sys; sys.stderr.write('webhook down /home/alice/secret-path\\n'); sys.exit(3)\n"


@dataclass
class NEnv:
    ctx: ServiceContext
    clock: FakeClock
    inbox: Path
    tmp: Path

    def channels(self, *chans: NotifyChannel) -> None:
        self.ctx.clusters.register(
            ClusterProfile(cluster_id="c1", display_name="C1", transport="local"),
            UserOverlay(cluster_id="c1", notify=list(chans)),
        )

    def recorder(self, events: list[str] | None = None, *, script: str = RECORDER) -> NotifyChannel:
        path = self.tmp / f"sink{abs(hash(script)) % 1000}.py"
        path.write_text(script)
        kw: dict[str, Any] = {"events": events} if events is not None else {}
        return NotifyChannel(kind="command", target=f"{sys.executable} {path} {self.inbox}", **kw)

    def emit(self, event: str, entity: str, title: str = "t", body: str = "b", **kw: Any) -> list[str]:
        with self.ctx.db.tx(source="worker") as tx:
            return emit(tx, event, entity, title, body, **kw)

    def drain(self) -> int:
        runner = OutboxRunner(self.ctx)
        runner.register_all(outbox_handlers())
        return runner.drain()

    def received(self) -> list[dict[str, Any]]:
        if not self.inbox.exists():
            return []
        return [json.loads(line) for line in self.inbox.read_text().splitlines() if line.strip()]

    def rows(self) -> list[dict[str, Any]]:
        return self.ctx.db.all("SELECT * FROM notifications ORDER BY created_at, notification_id")


@pytest.fixture
def nenv(paths: Paths, tmp_path: Path) -> Iterator[NEnv]:
    clock = FakeClock()
    ctx = ServiceContext.create(paths, clock=clock)
    env = NEnv(ctx=ctx, clock=clock, inbox=tmp_path / "inbox.jsonl", tmp=tmp_path)
    yield env
    ctx.db.close()


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------


def test_emit_without_channels_writes_nothing(nenv: NEnv) -> None:
    assert nenv.emit("run.failed", "run_1") == []
    nenv.channels()  # a cluster with no notify entries
    assert nenv.emit("run.failed", "run_1") == []
    assert nenv.rows() == [] and nenv.ctx.db.all("SELECT * FROM outbox") == []


def test_emit_validates_input(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    for bad in (("", "run_1"), ("run.failed", "")):
        with pytest.raises(InvalidArgument):
            nenv.emit(*bad)
    with pytest.raises(InvalidArgument):
        nenv.emit("run.failed", "run_1", urgency="now")  # type: ignore[arg-type]


def test_channel_event_filters_and_aliases(nenv: NEnv) -> None:
    nenv.channels(
        nenv.recorder(events=["run.failed"]),
        NotifyChannel(kind="log", events=["*"]),
        NotifyChannel(kind="desktop", events=["workflow.*", "submission_uncertain"]),
    )
    assert len(nenv.emit("run.failed", "run_1")) == 2  # command + log
    assert len(nenv.emit("run.succeeded", "run_2")) == 1  # log only
    assert len(nenv.emit("workflow.finished", "wf_1")) == 2  # log + desktop (prefix pattern)
    ids = nenv.emit("attempt.uncertain", "att_1")  # alias match for the desktop channel
    assert len(ids) == 2
    rows = {r["notification_id"]: r for r in nenv.rows()}
    assert {rows[i]["state"] for i in ids} == {"queued"}  # immediate class


def test_same_channel_twice_across_clusters_is_one_channel(nenv: NEnv) -> None:
    rec = nenv.recorder()
    nenv.channels(rec)
    nenv.ctx.clusters.register(
        ClusterProfile(cluster_id="c2", display_name="C2", transport="local"),
        UserOverlay(cluster_id="c2", notify=[rec]),
    )
    (cid,) = {c.channel_id for c in notify.configured_channels(nenv.ctx.db)}
    assert cid == channel_id("command", rec.target) and len(nenv.emit("run.failed", "r")) == 1


# ---------------------------------------------------------------------------
# batching, immediate, dedupe
# ---------------------------------------------------------------------------


def test_sweep_batches_into_one_message_and_uncertain_goes_out_at_once(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder(events=["*"]))
    for i in range(20):  # 20 terminal events over 10 minutes
        nenv.emit("run.failed" if i % 4 == 0 else "run.succeeded", f"run_{i}", f"run {i} finished", "state X")
        nenv.clock.advance(30)
    nenv.emit("attempt.uncertain", "att_9", "submission uncertain", "reconciling; never resubmitted")
    assert flush(nenv.ctx) == 0  # the batch window (30 min) has not elapsed
    nenv.drain()
    first = nenv.received()
    assert len(first) == 1 and first[0]["urgency"] == "immediate" and first[0]["count"] == 1
    assert first[0]["items"][0]["event_type"] == "attempt.uncertain"
    nenv.clock.advance(25 * 60)  # oldest pending is now > 30 min old
    assert flush(nenv.ctx) == 20
    nenv.drain()
    batch = nenv.received()[1:]
    assert len(batch) == 1
    msg = batch[0]
    assert msg["urgency"] == "batch" and msg["count"] == 20 and len(msg["notification_ids"]) == 20
    assert msg["title"] == "hpclab: 20 updates" and msg["body"] == "run.failed x5, run.succeeded x15"
    assert {r["state"] for r in nenv.rows()} == {"sent"}
    assert flush(nenv.ctx) == 0 and nenv.drain() == 0


def test_dedupe_while_unsent_then_notify_again(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    assert len(nenv.emit("run.failed", "run_1")) == 1
    assert nenv.emit("run.failed", "run_1") == []  # same (event, entity) still pending
    assert len(nenv.emit("run.failed", "run_2")) == 1
    flush(nenv.ctx, force=True)
    nenv.drain()
    assert len(nenv.emit("run.failed", "run_1")) == 1  # the first one was sent: news again
    assert len(nenv.emit("budget.threshold", "exp_1", dedupe_key="exp_1:80%")) == 1
    nenv.drain()
    assert nenv.emit("budget.threshold", "exp_1", dedupe_key="exp_1:80%")  # sent -> allowed again
    events = {
        e["event_type"]
        for e in nenv.ctx.db.all("SELECT event_type FROM events WHERE entity_type='notification'")
    }
    assert {"notification.created", "notification.batched", "notification.sent"} <= events


def test_explicit_immediate_urgency(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    nenv.emit("run.failed", "run_1", urgency="immediate")
    nenv.drain()
    assert [m["urgency"] for m in nenv.received()] == ["immediate"]


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------


def test_payload_carries_no_paths_metrics_logs_or_secrets(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    nenv.emit(
        "run.failed",
        "run_1",
        "run method-mixup.seed-1 failed in /hai/scratch/alice/hpclab/attempts/att_1",
        'acc=0.912 loss: 0.53\nTraceback (most recent call last):\n  File "train.py", line 3\n'
        "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789 see https://wandb.ai/x/y?key=1 outputs/metrics.json 3/5 cells",
        urgency="immediate",
    )
    nenv.drain()
    (msg,) = nenv.received()
    raw = json.dumps(msg)

    def keys(obj: Any) -> set[str]:
        if isinstance(obj, dict):
            return set(obj) | {k for v in obj.values() for k in keys(v)}
        if isinstance(obj, list):
            return {k for v in obj for k in keys(v)}
        return set()

    assert not {"path", "log", "paths", "logs", "stdout", "stderr"} & keys(msg)
    for leaked in ("/hai/scratch", "0.912", "0.53", "ghp_", "wandb.ai", "metrics.json", "\\n"):
        assert leaked not in raw, leaked
    assert "method-mixup.seed-1" in msg["title"] and "3/5 cells" in msg["body"]
    assert "acc=<value>" in msg["body"] and "token=<redacted>" in msg["body"]


def test_sanitize_text_is_bounded_and_single_line() -> None:
    out = sanitize_text("a\nb\tc " + "word " * 100, 50)
    assert "\n" not in out and "\t" not in out and len(out) == 50 and out.endswith("…")
    assert sanitize_text("x" * 40, 100) == "<redacted>"  # long opaque tokens (keys, digests)
    assert sanitize_text("exit 137, 2 GPUs, seed 3", 100) == "exit 137, 2 GPUs, seed 3"


# ---------------------------------------------------------------------------
# delivery failures and retries
# ---------------------------------------------------------------------------


def test_failing_channel_retries_then_fails_without_leaking_the_target(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder(script=FAILER))
    (nid,) = nenv.emit("attempt.uncertain", "att_1")
    item = OutboxItem(outbox_id=1, kind="notify.send", operation_id=None, target_id=nid,
                      payload={"notification_ids": [nid], "urgency": "immediate"}, tries=1)  # fmt: skip
    delays = []
    for _ in range(MAX_TRIES - 1):
        res = handle_send(nenv.ctx, item)
        assert res.action == "retry"
        delays.append(res.delay_s)
    assert delays == sorted(delays) and delays[0] > 0  # backoff grows
    row = nenv.ctx.db.one("SELECT * FROM notifications WHERE notification_id=?", (nid,))
    assert row["state"] == "queued" and row["tries"] == MAX_TRIES - 1
    assert "exited 3" in row["last_error"] and "secret-path" not in row["last_error"]
    assert str(nenv.tmp) not in row["last_error"]  # no command path either
    assert handle_send(nenv.ctx, item).action == "done"
    row = nenv.ctx.db.one("SELECT * FROM notifications WHERE notification_id=?", (nid,))
    assert row["state"] == "failed" and row["tries"] == MAX_TRIES
    assert handle_send(nenv.ctx, item).action == "done"  # nothing left: idempotent re-entry


def test_removed_channel_marks_queued_rows_failed(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    (nid,) = nenv.emit("attempt.uncertain", "att_1")
    nenv.channels()  # the user removed the channel before delivery
    nenv.drain()
    row = nenv.ctx.db.one("SELECT * FROM notifications WHERE notification_id=?", (nid,))
    assert row["state"] == "failed" and "no longer configured" in row["last_error"]


def test_missing_command_is_a_delivery_failure(nenv: NEnv) -> None:
    nenv.channels(NotifyChannel(kind="command", target="/nonexistent/hpclab-hook --flag"))
    nenv.emit("attempt.uncertain", "att_1")
    nenv.drain()
    row = nenv.rows()[0]
    assert row["tries"] == 1 and row["state"] == "queued" and "not found: hpclab-hook" in row["last_error"]


# ---------------------------------------------------------------------------
# HTTP channels + log + notify_test
# ---------------------------------------------------------------------------


class _Sink(BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []
    status = 200

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Sink.seen.append({"path": self.path, "headers": dict(self.headers), "body": body.decode()})
        self.send_response(_Sink.status)
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        return None


@pytest.fixture
def http_sink() -> Iterator[str]:
    _Sink.seen = []
    _Sink.status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_ntfy_and_slack_webhook_posts(nenv: NEnv, http_sink: str) -> None:
    nenv.channels(
        NotifyChannel(kind="ntfy", target=f"{http_sink}/hpclab-alice-SECRET"),
        NotifyChannel(kind="slack_webhook", target=f"{http_sink}/services/T0/B0/SECRET"),
    )
    nenv.emit("attempt.uncertain", "att_1", "submission uncertain", "reconciling")
    nenv.drain()
    by_path = {s["path"]: s for s in _Sink.seen}
    ntfy = by_path["/hpclab-alice-SECRET"]
    assert ntfy["headers"]["Title"] == "submission uncertain" and ntfy["headers"]["Priority"] == "high"
    assert ntfy["body"] == "reconciling"
    slack = json.loads(by_path["/services/T0/B0/SECRET"]["body"])
    assert slack == {"text": "*submission uncertain*\nreconciling"}
    assert {r["state"] for r in nenv.rows()} == {"sent"}
    assert all("SECRET" not in r["channel"] for r in nenv.rows())


def test_http_error_is_retried_and_never_shows_the_url(nenv: NEnv, http_sink: str) -> None:
    _Sink.status = 500
    nenv.channels(NotifyChannel(kind="slack_webhook", target=f"{http_sink}/services/T0/B0/SECRET"))
    nenv.emit("attempt.uncertain", "att_1")
    nenv.drain()
    row = nenv.rows()[0]
    assert row["state"] == "queued" and "HTTP 500 from 127.0.0.1" in row["last_error"]
    assert "SECRET" not in row["last_error"]


def test_log_channel_and_notify_test(nenv: NEnv, http_sink: str) -> None:
    nenv.channels(
        nenv.recorder(),
        NotifyChannel(kind="log"),
        NotifyChannel(kind="ntfy", target="http://127.0.0.1:9/topic-SECRET"),  # nothing listens on :9
        NotifyChannel(kind="ntfy", target="file:///etc/passwd"),
    )
    data = notify_test(nenv.ctx, cluster_id="c1")
    results = {(r.kind, r.target): r for r in data.channels}
    assert data.sent == 2 and data.failed == 2
    assert results[("command", "python" + sys.executable.rsplit("python", 1)[-1])].ok
    assert results[("log", "log")].ok
    unreachable = results[("ntfy", "http://127.0.0.1/…")]
    assert not unreachable.ok and "SECRET" not in (unreachable.error or "")
    bad_scheme = next(r for r in data.channels if r.target.startswith("file"))
    assert not bad_scheme.ok and "http(s)" in (bad_scheme.error or "")
    (msg,) = nenv.received()
    assert msg["title"] == "hpclab test notification" and msg["test"] is True
    log_lines = (nenv.ctx.paths.home / "logs" / "notifications.jsonl").read_text().splitlines()
    assert json.loads(log_lines[-1])["title"] == "hpclab test notification"
    assert nenv.ctx.db.one("SELECT COUNT(*) AS n FROM events WHERE event_type='notify.tested'")["n"] == 1
    assert nenv.rows() == []  # a test sends directly; it records no notifications


def test_notify_test_without_channels(nenv: NEnv) -> None:
    data = notify_test(nenv.ctx)
    assert data.channels == [] and "no notification channels configured" in (data.note or "")


def test_controller_wiring() -> None:
    assert set(outbox_handlers()) == {"notify.send"}
    (task,) = periodic_tasks()
    assert task.name == "notify.flush" and task.interval_s == 60.0


def test_rows_of_a_dead_send_item_are_requeued_not_lost(nenv: NEnv) -> None:
    """At-least-once: if the outbox item carrying queued rows died (handler crashed 3 times),
    the periodic flush hands them to a new item - otherwise they stay 'queued' forever and the
    dedupe key silently swallows every later notification of the same event."""
    nenv.channels(nenv.recorder(events=["auth.required"]))
    (nid,) = nenv.emit("auth_required", "c1", "log in to c1", "polling paused")
    with nenv.ctx.db.tx() as tx:  # the send item crashed three times -> dead
        tx.execute("UPDATE outbox SET state='dead' WHERE kind='notify.send'")
    assert nenv.drain() == 0 and nenv.received() == []
    assert nenv.emit("auth_required", "c1") == []  # still deduped against the stuck row
    nenv.clock.advance(30 * 60)
    flush(nenv.ctx)
    assert nenv.drain() == 0  # not yet: a slow but live item must not be doubled
    nenv.clock.advance(31 * 60)
    flush(nenv.ctx)
    nenv.drain()
    (msg,) = nenv.received()
    assert msg["notification_ids"] == [nid] and msg["urgency"] == "immediate"
    assert nenv.rows()[0]["state"] == "sent"
    assert len(nenv.emit("auth_required", "c1")) == 1  # news again once delivered


def test_queued_rows_with_a_live_item_are_not_requeued(nenv: NEnv) -> None:
    nenv.channels(nenv.recorder())
    nenv.emit("attempt.uncertain", "att_1")
    nenv.clock.advance(3 * 3600)  # e.g. the controller was down; the item is still pending
    flush(nenv.ctx)
    assert nenv.ctx.db.one("SELECT COUNT(*) AS n FROM outbox WHERE kind='notify.send'")["n"] == 1
    nenv.drain()
    assert len(nenv.received()) == 1
