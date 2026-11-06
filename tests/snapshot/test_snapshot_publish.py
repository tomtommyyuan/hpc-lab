"""publish_snapshot through LocalTransport into a temporary "remote" filesystem."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from snapshot_support import GitRepo

from hpclab.adapters.base import ConnectionLost
from hpclab.adapters.local import LocalTransport
from hpclab.domain.errors import InvalidArgument, PreconditionFailed
from hpclab.domain.models import CodeOptions, CodeSnapshotRef
from hpclab.services.snapshot import (
    MARKER_NAME,
    REMOTE_MANIFEST_NAME,
    SnapshotResult,
    SnapshotStore,
    SnapshotVerificationError,
    capture_snapshot,
    digest_hex,
    marker_bytes,
    publish_snapshot,
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


def test_publish_creates_a_complete_read_only_snapshot(
    captured: SnapshotResult, store: SnapshotStore, transport: LocalTransport, remote_root: Path
) -> None:
    final = publish_snapshot(transport, store, captured.ref, str(remote_root))

    hex_ = digest_hex(captured.ref.digest)
    assert final == f"{remote_root}/snapshots/{hex_}"
    root = Path(final)
    assert (root / "train.py").read_text() == "print('train')\n"
    assert (root / "src" / "model.py").exists()
    assert os.access(root / "run.sh", os.X_OK)
    assert (root / MARKER_NAME).read_bytes() == marker_bytes(captured.ref)
    manifest = (root / REMOTE_MANIFEST_NAME).read_text()
    assert captured.ref.digest in manifest
    # read-only: files, subdirectories and the top-level dir; directories stay traversable
    for path in (root, root / "src", root / "train.py", root / MARKER_NAME):
        assert not os.access(path, os.W_OK), path
    assert os.access(root / "src", os.X_OK)
    assert staging_entries(remote_root) == []


def test_republish_is_idempotent_and_does_not_upload(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = publish_snapshot(transport, store, captured.ref, str(remote_root))

    def no_upload(*args: object, **kwargs: object) -> None:
        raise AssertionError("a complete snapshot must not be uploaded again")

    monkeypatch.setattr(transport, "upload_dir", no_upload)
    assert publish_snapshot(transport, store, captured.ref, str(remote_root)) == first


def test_republish_from_another_machine_needs_no_local_store(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    tmp_path: Path,
) -> None:
    first = publish_snapshot(transport, store, captured.ref, str(remote_root))
    empty_store = SnapshotStore(tmp_path / "other-machine-store")
    assert publish_snapshot(transport, empty_store, captured.ref, str(remote_root)) == first


def test_checksum_mismatch_fails_removes_staging_and_writes_no_marker(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sha = transport.sha256

    def corrupt(paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        out = real_sha(paths, timeout=timeout)
        for p in out:
            if p.endswith("/train.py"):
                out[p] = "sha256:" + "0" * 64
        return out

    monkeypatch.setattr(transport, "sha256", corrupt)
    with pytest.raises(SnapshotVerificationError) as err:
        publish_snapshot(transport, store, captured.ref, str(remote_root))

    assert err.value.retryable is True
    assert [m["path"] for m in err.value.details["mismatched"]] == ["train.py"]
    final = remote_root / "snapshots" / digest_hex(captured.ref.digest)
    assert not final.exists()
    assert staging_entries(remote_root) == []


def test_missing_and_extra_uploaded_files_fail_verification(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_upload = transport.upload_dir

    def sloppy_upload(local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        real_upload(local_dir, remote_dir, timeout=timeout)
        Path(remote_dir, "stray.txt").write_text("not in the manifest")

    monkeypatch.setattr(transport, "upload_dir", sloppy_upload)
    with pytest.raises(SnapshotVerificationError) as err:
        publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert err.value.details["remote_file_count"] == captured.ref.file_count + 1
    assert staging_entries(remote_root) == []


def test_transport_failure_during_upload_keeps_staging_for_the_retry(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_upload = transport.upload_dir

    def half_upload(local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        real_upload(local_dir, remote_dir, timeout=timeout)
        raise ConnectionLost("connection dropped mid-transfer")

    monkeypatch.setattr(transport, "upload_dir", half_upload)
    with pytest.raises(ConnectionLost):
        publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert len(staging_entries(remote_root)) == 1  # the retry resumes into it (no marker yet)
    assert not (remote_root / "snapshots" / digest_hex(captured.ref.digest)).exists()

    monkeypatch.undo()
    final = publish_snapshot(transport, store, captured.ref, str(remote_root))
    assert (Path(final) / MARKER_NAME).exists()
    assert staging_entries(remote_root) == []


def test_concurrent_publisher_winning_the_race_is_tolerated(
    repo: GitRepo,
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    other = LocalTransport("local")
    other_store = SnapshotStore(tmp_path / "other-machine-store")  # a rival on another machine
    assert capture_snapshot(repo.root, CodeOptions(), other_store).ref == captured.ref
    real_upload = transport.upload_dir

    def upload_while_someone_else_publishes(
        local_dir: str, remote_dir: str, *, timeout: float = 600.0
    ) -> None:
        publish_snapshot(other, other_store, captured.ref, str(remote_root))  # the rival finishes first
        real_upload(local_dir, remote_dir, timeout=timeout)

    monkeypatch.setattr(transport, "upload_dir", upload_while_someone_else_publishes)
    final = publish_snapshot(transport, store, captured.ref, str(remote_root))

    assert (Path(final) / MARKER_NAME).exists()
    assert staging_entries(remote_root) == []


def test_incomplete_final_dir_is_quarantined_and_republished(
    captured: SnapshotResult, store: SnapshotStore, transport: LocalTransport, remote_root: Path
) -> None:
    final = remote_root / "snapshots" / digest_hex(captured.ref.digest)
    final.mkdir(parents=True)
    (final / "train.py").write_text("half-written garbage")  # no marker: never completed

    path = publish_snapshot(transport, store, captured.ref, str(remote_root))

    assert (Path(path) / "train.py").read_text() == "print('train')\n"
    assert (Path(path) / MARKER_NAME).exists()
    assert staging_entries(remote_root) == []


def test_marker_for_different_content_is_refused(
    captured: SnapshotResult, store: SnapshotStore, transport: LocalTransport, remote_root: Path
) -> None:
    final = remote_root / "snapshots" / digest_hex(captured.ref.digest)
    final.mkdir(parents=True)
    (final / MARKER_NAME).write_text('{"digest": "something else"}\n')
    with pytest.raises(PreconditionFailed, match="different content"):
        publish_snapshot(transport, store, captured.ref, str(remote_root))


def test_ref_that_does_not_match_the_manifest_is_rejected(
    captured: SnapshotResult, store: SnapshotStore, transport: LocalTransport, remote_root: Path
) -> None:
    bad = CodeSnapshotRef.model_validate({**captured.ref.model_dump(), "file_count": 999})
    with pytest.raises(InvalidArgument):
        publish_snapshot(transport, store, bad, str(remote_root))


@pytest.mark.parametrize("root", ["relative/path", "/has space/x", "/a/../b", "/$USER/x", ""])
def test_unsafe_remote_roots_are_rejected(
    captured: SnapshotResult, store: SnapshotStore, transport: LocalTransport, root: str
) -> None:
    with pytest.raises(InvalidArgument):
        publish_snapshot(transport, store, captured.ref, root)


def test_published_snapshot_supports_hardlinked_work_dirs(
    captured: SnapshotResult,
    store: SnapshotStore,
    transport: LocalTransport,
    remote_root: Path,
    tmp_path: Path,
) -> None:
    """The wrapper builds work dirs with `cp -al` from the read-only snapshot."""
    import subprocess

    final = publish_snapshot(transport, store, captured.ref, str(remote_root))
    work = remote_root / "attempts" / "att_1" / "work"
    work.parent.mkdir(parents=True)
    subprocess.run(["cp", "-al", final, str(work)], check=True)
    assert os.stat(work / "train.py").st_ino == os.stat(Path(final) / "train.py").st_ino
