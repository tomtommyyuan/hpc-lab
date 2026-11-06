"""Regression tests from the adversarial review: secret leaks, provenance lies, escapes."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from snapshot_support import GitRepo

from hpclab.adapters.local import LocalTransport
from hpclab.domain.errors import HpcLabError, InvalidArgument, PreconditionFailed
from hpclab.domain.models import CodeOptions
from hpclab.services.snapshot import (
    PathMatcher,
    SnapshotStore,
    capture_snapshot,
    digest_hex,
    publish_snapshot,
    scan_project,
)


def captured_paths(store: SnapshotStore, digest: str) -> set[str]:
    return {e["path"] for e in store.manifest(digest)["files"]}


# ---------------------------------------------------------------------------
# secrets that the name rules cannot see
# ---------------------------------------------------------------------------


def test_service_account_json_with_a_private_key_is_never_uploaded(
    repo: GitRepo, store: SnapshotStore, blob_exists: Callable[[bytes], bool]
) -> None:
    """GCP keys embed the PEM block inside a JSON string, so it never starts a line."""
    key = {
        "type": "service_account",
        "project_id": "lab-123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----\n",
        "client_email": "trainer@lab-123.iam.gserviceaccount.com",
    }
    content = json.dumps(key, indent=2).encode()
    repo.write("configs/lab-123-4f2a.json", content)
    repo.commit("add gcs access")

    res = capture_snapshot(repo.root, CodeOptions(), store)

    assert "configs/lab-123-4f2a.json" not in captured_paths(store, res.ref.digest)
    assert "configs/lab-123-4f2a.json" in res.excluded
    assert any(w.startswith("SECRET:") and "lab-123-4f2a.json" in w for w in res.warnings)
    assert not blob_exists(content)


def test_json_mentioning_private_key_without_a_key_block_is_kept(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("schema.json", '{"private_key": "path to your key file"}\n')
    repo.commit("schema")
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert "schema.json" in captured_paths(store, res.ref.digest)


@pytest.mark.parametrize("rel", ["kaggle.json", ".kaggle/kaggle.json"])
def test_kaggle_api_tokens_are_secrets(repo: GitRepo, store: SnapshotStore, rel: str) -> None:
    repo.write(rel, '{"username": "u", "key": "0123456789abcdef"}\n')
    repo.commit("kaggle")
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert rel not in captured_paths(store, res.ref.digest)
    assert rel in res.excluded


def test_symlink_into_dot_git_is_never_captured(
    repo: GitRepo, store: SnapshotStore, blob_exists: Callable[[bytes], bool]
) -> None:
    """.git/config can hold a token-bearing remote URL; a tracked link must not smuggle it out."""
    repo.git("remote", "add", "origin", "https://alice:ghp_SECRETTOKEN@github.com/lab/proj.git")
    os.symlink(".git/config", repo.root / "gitcfg")
    repo.commit("link")

    res = capture_snapshot(repo.root, CodeOptions(), store)

    assert "gitcfg" not in captured_paths(store, res.ref.digest)
    assert "gitcfg" in res.excluded
    assert not blob_exists((repo.root / ".git" / "config").read_bytes())
    manifest = store.manifest(res.ref.digest)
    assert "ghp_SECRETTOKEN" not in json.dumps(manifest)  # remote_url is sanitized too


# ---------------------------------------------------------------------------
# provenance: git flags that hide edits from `git status`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_edits_hidden_by_index_flags_still_make_the_snapshot_dirty(
    repo: GitRepo, store: SnapshotStore, flag: str
) -> None:
    clean = capture_snapshot(repo.root, CodeOptions(), store).ref
    repo.git("update-index", flag, "train.py")
    repo.write("train.py", "print('local hack, invisible to git status')\n")

    res = capture_snapshot(repo.root, CodeOptions(), store)

    assert res.ref.git_dirty is True
    assert res.ref.dirty_patch_digest is not None
    assert res.ref.dirty_patch_digest != clean.dirty_patch_digest
    assert any("train.py" in w and flag.strip("-") in w for w in res.warnings)
    with pytest.raises(PreconditionFailed) as err:
        capture_snapshot(repo.root, CodeOptions(allow_dirty=False), store)
    assert "train.py" in err.value.details["dirty_files"]


def test_hidden_edit_changes_the_dirty_patch_digest(repo: GitRepo, store: SnapshotStore) -> None:
    repo.git("update-index", "--assume-unchanged", "train.py")
    repo.write("train.py", "v1\n")
    first = capture_snapshot(repo.root, CodeOptions(), store).ref.dirty_patch_digest
    repo.write("train.py", "v2\n")
    second = capture_snapshot(repo.root, CodeOptions(), store).ref.dirty_patch_digest
    assert first and second and first != second


def test_flagged_but_unchanged_file_keeps_the_tree_clean(repo: GitRepo, store: SnapshotStore) -> None:
    repo.git("update-index", "--assume-unchanged", "train.py")
    res = capture_snapshot(repo.root, CodeOptions(allow_dirty=False), store)
    assert res.ref.git_dirty is False and res.ref.dirty_patch_digest is None


def test_files_outside_a_sparse_checkout_are_reported_not_silently_dropped(
    repo: GitRepo, store: SnapshotStore
) -> None:
    repo.git("update-index", "--skip-worktree", "src/model.py")
    (repo.root / "src" / "model.py").unlink()
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert "src/model.py" not in captured_paths(store, res.ref.digest)
    assert any("not checked out" in w for w in res.warnings)


def test_dirty_files_are_project_relative_in_a_monorepo(repo: GitRepo) -> None:
    repo.write("projA/train.py", "a\n")
    repo.commit("mono")
    repo.write("projA/train.py", "edited\n")
    repo.write("projB/other.py", "not our project\n")
    with pytest.raises(PreconditionFailed) as err:
        scan_project(repo.root / "projA", CodeOptions(allow_dirty=False))
    assert err.value.details["dirty_files"] == ["train.py"]


# ---------------------------------------------------------------------------
# escapes and reserved names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", [":(top)projB", ":/projB", ":!x", ":(glob)**"])
def test_pathspec_magic_cannot_reach_outside_the_project(repo: GitRepo, entry: str) -> None:
    repo.write("projA/a.py", "a\n")
    repo.commit("mono")
    repo.write("projB/private_notes.txt", "outside projA\n")
    with pytest.raises(InvalidArgument):
        scan_project(repo.root / "projA", CodeOptions(include_untracked=[entry]))


def test_scan_never_lists_paths_outside_the_project_root(repo: GitRepo) -> None:
    repo.write("projA/a.py", "a\n")
    repo.commit("mono")
    scan = scan_project(repo.root / "projA", CodeOptions())
    assert all(not f.path.startswith("../") and ".." not in f.path.split("/") for f in scan.files)


def test_reserved_snapshot_marker_names_are_not_captured(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write(".hpclab-complete", "mine\n")
    repo.write(".hpclab-manifest.json", "{}\n")
    repo.write("sub/.hpclab-complete", "fine: only the root names are reserved\n")
    repo.commit("reserved")
    res = capture_snapshot(repo.root, CodeOptions(), store)
    files = captured_paths(store, res.ref.digest)
    assert ".hpclab-complete" not in files and ".hpclab-manifest.json" not in files
    assert "sub/.hpclab-complete" in files
    assert any(".hpclab-complete" in w and "reserved" in w for w in res.warnings)


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/logs", "logs/a.txt", True),
        ("**/logs", "x/y/logs/a.txt", True),
        ("**/logs/", "logs/a.txt", True),
        ("**/*.csv", "t.csv", True),
        ("**/cache/tmp", "cache/tmp/a", True),
        ("**/cache/tmp", "a/cache/tmp/b", True),
        ("**/logs", "catalogs/a.txt", False),
    ],
)
def test_leading_double_star_matches_at_any_depth_including_the_root(
    pattern: str, path: str, expected: bool
) -> None:
    assert PathMatcher([pattern]).matches(path) is expected


# ---------------------------------------------------------------------------
# publish integrity
# ---------------------------------------------------------------------------


def test_publish_refuses_a_local_manifest_that_does_not_hash_to_the_ref(
    repo: GitRepo, store: SnapshotStore, remote_root: Path
) -> None:
    res = capture_snapshot(repo.root, CodeOptions(), store)
    path = store.manifest_path(res.ref.digest)
    manifest = json.loads(path.read_text())
    manifest["files"][0]["mode"] = 0o755 if manifest["files"][0]["mode"] == 0o644 else 0o644
    path.chmod(0o644)
    path.write_text(json.dumps(manifest))

    with pytest.raises(HpcLabError, match="does not match"):
        publish_snapshot(LocalTransport("local"), store, res.ref, str(remote_root))
    assert not (remote_root / "snapshots" / digest_hex(res.ref.digest)).exists()


def test_publish_rejects_an_invalid_argument_ref_before_touching_the_remote(
    store: SnapshotStore, remote_root: Path
) -> None:
    from hpclab.domain.models import CodeSnapshotRef

    bogus = CodeSnapshotRef(digest="sha256:../../etc", file_count=0, total_bytes=0)
    with pytest.raises(InvalidArgument):
        publish_snapshot(LocalTransport("local"), store, bogus, str(remote_root))
