"""Bulk snapshot uploads on slow or flaky links.

A large snapshot on a slow link must be able to finish: the upload deadline grows with the
snapshot's size, a retry after an interrupted upload resumes into the same staging dir
(rsync skips the files already there) instead of starting from zero, and concurrent
publishes of one digest from the same machine are serialized instead of uploading the same
bytes twice.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from snapshot_support import GitRepo

from hpclab.adapters.base import CommandFailed, CommandResult, ConnectionLost, TransportTimeout
from hpclab.adapters.local import LocalTransport
from hpclab.domain.models import CodeOptions
from hpclab.services import snapshot as snap
from hpclab.services.snapshot import (
    MARKER_NAME,
    REMOTE_MANIFEST_NAME,
    SnapshotResult,
    SnapshotStore,
    capture_snapshot,
    digest_hex,
    publish_snapshot,
    transfer_timeout_s,
)


@pytest.fixture
def captured(repo: GitRepo, store: SnapshotStore) -> SnapshotResult:
    return capture_snapshot(repo.root, CodeOptions(), store)


@pytest.fixture
def transport() -> LocalTransport:
    return LocalTransport("local")


def staging_entries(remote_root: Path) -> list[str]:
    staging = remote_root / ".staging"
    return sorted(os.listdir(staging)) if staging.exists() else []


class UploadSpy:
    """Wraps ``upload_dir``: records (remote_dir, timeout, files already there) per call."""

    def __init__(self, transport: LocalTransport, fail: list[BaseException | None] | None = None) -> None:
        self.real = transport.upload_dir
        self.fail = list(fail or [])
        self.calls: list[tuple[str, float, list[str]]] = []

    def __call__(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        present = sorted(
            os.path.relpath(os.path.join(d, f), remote_dir) for d, _, fs in os.walk(remote_dir) for f in fs
        )
        self.calls.append((remote_dir, timeout, present))
        exc = self.fail.pop(0) if self.fail else None
        if exc is not None:
            partial = Path(remote_dir)  # a transfer that got part of the way
            partial.mkdir(parents=True, exist_ok=True)
            (partial / "train.py").write_text("print('train')\n")
            raise exc
        self.real(local_dir, remote_dir, timeout=timeout)


# ---------------------------------------------------------------------------
# deadline
# ---------------------------------------------------------------------------


def test_transfer_timeout_grows_with_size_and_file_count() -> None:
    small = transfer_timeout_s(10_000, 3)
    assert small >= 600.0
    one_gib = transfer_timeout_s(1024**3, 1)
    # 1 GiB must not need more than ~1 Mbit/s to finish inside one attempt
    assert one_gib >= 1024**3 / (128 * 1024)
    assert transfer_timeout_s(256 * 1024**2, 50_000) > transfer_timeout_s(256 * 1024**2, 1)


def test_publish_scales_the_upload_timeout_with_the_snapshot_size(
    repo: GitRepo,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo.write("weights.txt", "x" * 300_000)
    repo.commit("big text file")
    res = capture_snapshot(repo.root, CodeOptions(), store)
    monkeypatch.setattr(snap, "TRANSFER_MIN_RATE", 1024)  # a very slow link: 300 KB -> ~5 min
    spy = UploadSpy(transport)
    monkeypatch.setattr(transport, "upload_dir", spy)

    publish_snapshot(transport, store, res.ref, str(remote_root))

    [(_, timeout, _)] = spy.calls
    assert timeout == pytest.approx(transfer_timeout_s(res.ref.total_bytes, res.ref.file_count))
    assert timeout > 600.0 + 290.0


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        TransportTimeout("rsync timed out after 600s"),
        ConnectionLost("connection dropped mid-transfer"),
    ],
)
def test_interrupted_upload_is_resumed_into_the_same_staging_dir(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    spy = UploadSpy(transport, fail=[failure])
    monkeypatch.setattr(transport, "upload_dir", spy)

    with pytest.raises(type(failure)) as err:
        publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert err.value.retryable is True
    assert len(staging_entries(remote_root)) == 1  # kept for the retry
    assert not (remote_root / "snapshots" / digest_hex(captured.ref.digest)).exists()

    final = publish_snapshot(transport, store, captured.ref, str(remote_root))

    (first_dir, _, _), (second_dir, _, second_present) = spy.calls
    assert second_dir == first_dir  # same staging dir: rsync only sends what is missing
    assert "train.py" in second_present  # the bytes from the first attempt are still there
    assert (Path(final) / MARKER_NAME).exists()
    assert (Path(final) / "train.py").read_text() == "print('train')\n"
    assert staging_entries(remote_root) == []


def test_staging_dir_is_stable_per_store_and_distinct_across_stores(
    repo: GitRepo, captured: SnapshotResult, store: SnapshotStore, tmp_path: Path
) -> None:
    other = SnapshotStore(tmp_path / "other-machine")
    capture_snapshot(repo.root, CodeOptions(), other)
    root = "/remote/hpclab"
    hex_ = digest_hex(captured.ref.digest)
    mine = snap.staging_dir(root, hex_, store)
    assert mine == snap.staging_dir(root, hex_, SnapshotStore(store.root))  # stable across restarts
    assert mine.startswith(f"{root}/.staging/{hex_}.")
    assert snap.staging_dir(root, hex_, other) != mine  # never shared with another machine


def test_leftover_sealed_staging_dir_is_reused_after_a_crash(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the seal step leaves a read-only staging dir (with a marker) behind; the
    next publish must be able to reuse it instead of failing on permissions forever."""
    real_write = transport.write_file

    def crash_on_marker(path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0) -> None:
        real_write(path, content, mode=mode, timeout=timeout)
        if path.endswith("/" + MARKER_NAME):
            raise ConnectionLost("connection dropped after the marker was written")

    monkeypatch.setattr(transport, "write_file", crash_on_marker)
    monkeypatch.setattr(snap, "_remove_tree_best_effort", lambda transport, path: None)  # link is down
    with pytest.raises(ConnectionLost):
        publish_snapshot(transport, store, captured.ref, str(remote_root))
    [leftover] = staging_entries(remote_root)
    assert (remote_root / ".staging" / leftover / MARKER_NAME).exists()
    monkeypatch.undo()

    final = publish_snapshot(transport, store, captured.ref, str(remote_root))

    assert (Path(final) / MARKER_NAME).exists()
    assert (Path(final) / REMOTE_MANIFEST_NAME).exists()
    assert staging_entries(remote_root) == []


def test_non_transport_upload_failure_still_removes_staging(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = CommandFailed(
        "rsync failed during upload_dir (exit 11)",
        CommandResult(["rsync"], 11, "", "No space left on device", 0.1),
        retryable=False,
    )
    monkeypatch.setattr(transport, "upload_dir", UploadSpy(transport, fail=[quota]))
    with pytest.raises(CommandFailed):
        publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert staging_entries(remote_root) == []


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_concurrent_publishes_of_one_digest_upload_once(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_upload = transport.upload_dir
    active = 0
    peak = 0
    uploads = 0
    guard = threading.Lock()
    started = threading.Event()

    def slow_upload(local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        nonlocal active, peak, uploads
        with guard:
            active += 1
            uploads += 1
            peak = max(peak, active)
        started.set()
        time.sleep(0.5)
        try:
            real_upload(local_dir, remote_dir, timeout=timeout)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(transport, "upload_dir", slow_upload)
    results: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(publish_snapshot(transport, store, captured.ref, str(remote_root)))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    first = threading.Thread(target=worker)
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=worker)
    second.start()
    first.join(10)
    second.join(10)

    assert errors == []
    assert len(results) == 2 and results[0] == results[1]
    assert uploads == 1 and peak == 1
    assert staging_entries(remote_root) == []


def test_lock_wait_is_bounded_and_retryable(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hex_ = digest_hex(captured.ref.digest)
    with snap._publish_lock(store, hex_, wait_s=1.0):
        monkeypatch.setattr(snap, "_publish_lock_wait_s", lambda ref: 0.3)
        with pytest.raises(snap.Unavailable) as err:
            publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert err.value.retryable is True
    assert "being published" in err.value.message
    # once the holder is gone, the publish goes through
    assert (Path(publish_snapshot(transport, store, captured.ref, str(remote_root))) / MARKER_NAME).exists()
