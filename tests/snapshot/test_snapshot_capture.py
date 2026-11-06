"""scan_project / capture_snapshot: what goes in, what never goes in, identity, limits."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from snapshot_support import GitRepo

from hpclab.domain.errors import InvalidArgument, LimitExceeded, NotFound, PreconditionFailed
from hpclab.domain.models import CodeOptions
from hpclab.services import snapshot as snap
from hpclab.services.snapshot import (
    MODE_EXEC,
    MODE_PLAIN,
    PathMatcher,
    SnapshotStore,
    capture_snapshot,
    digest_hex,
    is_secret_path,
    scan_project,
)


def paths_of(store: SnapshotStore, digest: str) -> dict[str, dict]:
    return {e["path"]: e for e in store.manifest(digest)["files"]}


# ---------------------------------------------------------------------------
# git-aware capture
# ---------------------------------------------------------------------------


def test_clean_repo_captures_tracked_files_with_git_facts(repo: GitRepo, store: SnapshotStore) -> None:
    res = capture_snapshot(repo.root, CodeOptions(), store)

    files = paths_of(store, res.ref.digest)
    assert sorted(files) == [".gitignore", "run.sh", "src/model.py", "train.py"]
    assert res.ref.git_commit == repo.git("rev-parse", "HEAD")
    assert res.ref.git_branch == "main"
    assert res.ref.git_dirty is False
    assert res.ref.dirty_patch_digest is None
    assert res.ref.file_count == 4
    assert res.ref.total_bytes == sum(f["size"] for f in files.values())
    assert res.ref.project_root == str(repo.root.resolve())
    assert files["run.sh"]["mode"] == MODE_EXEC
    assert files["train.py"]["mode"] == MODE_PLAIN
    assert files["train.py"]["sha256"] == "sha256:" + hashlib.sha256(b"print('train')\n").hexdigest()
    manifest = store.manifest(res.ref.digest)
    assert manifest["schema"] == "hpclab.snapshot/v1"
    assert manifest["git"]["commit"] == res.ref.git_commit
    assert not any(".git/" in p or p == ".git" for p in files)


def test_modified_tracked_file_is_captured_from_the_working_tree(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("train.py", "print('edited, not committed')\n")
    res = capture_snapshot(repo.root, CodeOptions(), store)

    entry = paths_of(store, res.ref.digest)["train.py"]
    assert entry["sha256"] == "sha256:" + hashlib.sha256(b"print('edited, not committed')\n").hexdigest()
    assert res.ref.git_dirty is True
    assert res.ref.dirty_patch_digest is not None and res.ref.dirty_patch_digest.startswith("sha256:")


def test_dirty_patch_digest_tracks_the_diff(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("train.py", "v1\n")
    first = capture_snapshot(repo.root, CodeOptions(), store).ref.dirty_patch_digest
    repo.write("train.py", "v2\n")
    second = capture_snapshot(repo.root, CodeOptions(), store).ref.dirty_patch_digest
    assert first and second and first != second


def test_allow_dirty_false_refuses_a_dirty_tree(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("src/model.py", "changed\n")
    with pytest.raises(PreconditionFailed) as err:
        capture_snapshot(repo.root, CodeOptions(allow_dirty=False), store)
    assert err.value.details["dirty_files"] == ["src/model.py"]


def test_deleted_tracked_file_is_absent_from_the_snapshot(repo: GitRepo, store: SnapshotStore) -> None:
    (repo.root / "src" / "model.py").unlink()
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert "src/model.py" not in paths_of(store, res.ref.digest)
    assert res.ref.git_dirty is True


def test_untracked_files_are_left_out_with_a_loud_warning(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("new_module.py", "x = 1\n")
    repo.write("ignored.log", "noise\n")  # gitignored: not worth a warning
    res = capture_snapshot(repo.root, CodeOptions(), store)

    assert "new_module.py" not in paths_of(store, res.ref.digest)
    warning = next(w for w in res.warnings if "untracked" in w)
    assert "new_module.py" in warning and "ignored.log" not in warning


def test_include_untracked_explicit_file_and_pathspec(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("configs/local.yaml", "lr: 0.1\n")
    repo.write("configs/other.yaml", "lr: 0.2\n")
    repo.write("notes/todo.txt", "later\n")
    repo.write("ignored.log", "explicitly wanted\n")  # gitignored but explicitly listed -> included
    opts = CodeOptions(include_untracked=["configs/*.yaml", "ignored.log"])
    res = capture_snapshot(repo.root, opts, store)

    files = paths_of(store, res.ref.digest)
    assert {"configs/local.yaml", "configs/other.yaml", "ignored.log"} <= set(files)
    assert "notes/todo.txt" not in files
    assert res.ref.untracked_included == ["configs/local.yaml", "configs/other.yaml", "ignored.log"]
    assert any("notes/" in w for w in res.warnings)
    assert not any("configs/" in w and "untracked" in w for w in res.warnings)


def test_include_untracked_that_matches_nothing_warns(repo: GitRepo, store: SnapshotStore) -> None:
    res = capture_snapshot(repo.root, CodeOptions(include_untracked=["missing.yaml"]), store)
    assert any("missing.yaml" in w and "matched no" in w for w in res.warnings)


@pytest.mark.parametrize("entry", ["../outside.py", "/etc/passwd", "a/../../b"])
def test_include_untracked_rejects_paths_outside_the_project(
    repo: GitRepo, store: SnapshotStore, entry: str
) -> None:
    with pytest.raises(InvalidArgument):
        scan_project(repo.root, CodeOptions(include_untracked=[entry]))


# ---------------------------------------------------------------------------
# never uploaded: secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        ".env",
        ".env.production",
        "certs/server.pem",
        "deploy/id_rsa",
        "cfg/credentials.json",
        ".aws/config",
        "keys/api.key",
    ],
)
def test_tracked_secret_files_are_excluded_loudly(
    repo: GitRepo, store: SnapshotStore, blob_exists: Callable[[bytes], bool], rel: str
) -> None:
    content = f"SECRET-{rel}\n".encode()
    repo.write(rel, content)
    repo.commit("oops, committed a secret")
    res = capture_snapshot(repo.root, CodeOptions(), store)

    assert rel not in paths_of(store, res.ref.digest)
    assert rel in res.excluded
    assert any(w.startswith("SECRET:") and rel in w for w in res.warnings)
    assert not blob_exists(content)


def test_explicitly_included_secret_is_still_excluded(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write(".env.local", "TOKEN=abc\n")
    res = capture_snapshot(repo.root, CodeOptions(include_untracked=[".env.local"]), store)
    assert ".env.local" not in paths_of(store, res.ref.digest)
    assert res.ref.untracked_included == []


def test_private_key_content_is_excluded_even_with_an_innocent_name(
    repo: GitRepo, store: SnapshotStore, blob_exists: Callable[[bytes], bool]
) -> None:
    content = b"notes\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n"
    repo.write("notes.txt", content)
    repo.write("doc.md", "Keys look like `-----BEGIN RSA PRIVATE KEY-----` (quoted, not at line start)\n")
    repo.commit("add notes")
    res = capture_snapshot(repo.root, CodeOptions(), store)

    files = paths_of(store, res.ref.digest)
    assert "notes.txt" not in files
    assert "doc.md" in files
    assert any("notes.txt" in w and "private key" in w for w in res.warnings)
    assert not blob_exists(content)


def test_is_secret_path_is_case_insensitive_and_component_aware() -> None:
    assert is_secret_path("A/.ENV")
    assert is_secret_path("home/.ssh/config")
    assert is_secret_path("x/ID_RSA.pub")
    assert not is_secret_path("src/environment.py")
    assert not is_secret_path("src/keyboard.py")


# ---------------------------------------------------------------------------
# default excludes, user excludes, symlinks
# ---------------------------------------------------------------------------


def test_default_excludes_apply_to_tracked_caches_and_large_weights(
    repo: GitRepo, store: SnapshotStore
) -> None:
    repo.write("src/__pycache__/model.cpython-312.pyc", b"\x00bytecode")
    repo.write("weights/big.pt", b"\x00" * (snap.LARGE_WEIGHT_BYTES + 1))
    repo.write("weights/tiny.pt", b"\x00" * 16)
    repo.write("data/splits/train.txt", "1\n2\n")  # tracked data is kept: code may depend on it
    repo.commit("add stuff")
    res = capture_snapshot(repo.root, CodeOptions(), store)

    files = paths_of(store, res.ref.digest)
    assert "src/__pycache__/model.cpython-312.pyc" not in files
    assert "weights/big.pt" not in files
    assert "weights/tiny.pt" in files
    assert "data/splits/train.txt" in files
    reasons = {e.path: e.reason for e in res.excluded_detail}
    assert reasons["weights/big.pt"] == "large_weights"
    assert reasons["src/__pycache__/model.cpython-312.pyc"] == "cache"


def test_user_exclude_patterns(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("logs/run1.txt", "x")
    repo.write("results/table.csv", "a,b")
    repo.write("scripts/tmp.sh", "echo")
    repo.write("other/scripts/tmp.sh", "echo")
    repo.commit("more")
    opts = CodeOptions(exclude=["logs/", "*.csv", "/scripts/tmp.sh"])
    res = capture_snapshot(repo.root, opts, store)

    files = paths_of(store, res.ref.digest)
    assert "logs/run1.txt" not in files
    assert "results/table.csv" not in files
    assert "scripts/tmp.sh" not in files
    assert "other/scripts/tmp.sh" in files  # anchored pattern only matches at the root


def test_symlinks_inside_outside_and_to_directories(
    repo: GitRepo, store: SnapshotStore, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("not yours\n")
    os.symlink(outside, repo.root / "escape.txt")
    os.symlink("train.py", repo.root / "alias.py")
    os.symlink("src", repo.root / "srclink")
    repo.commit("links")
    res = capture_snapshot(repo.root, CodeOptions(), store)

    files = paths_of(store, res.ref.digest)
    assert "escape.txt" not in files
    assert any("escape.txt" in w and "outside" in w for w in res.warnings)
    assert files["alias.py"]["sha256"] == files["train.py"]["sha256"]  # stored with the target's content
    assert "srclink" not in files
    assert any("srclink" in w and "directories" in w for w in res.warnings)


def test_subdirectory_project_in_a_monorepo(repo: GitRepo, store: SnapshotStore) -> None:
    repo.write("projA/train.py", "a\n")
    repo.write("projB/train.py", "b\n")
    repo.commit("mono")
    res = capture_snapshot(repo.root / "projA", CodeOptions(), store)
    assert sorted(paths_of(store, res.ref.digest)) == ["train.py"]
    assert res.ref.git_commit == repo.git("rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# identity / dedupe
# ---------------------------------------------------------------------------


def test_identical_trees_dedupe_regardless_of_commit(repo: GitRepo, store: SnapshotStore) -> None:
    first = capture_snapshot(repo.root, CodeOptions(), store)
    repo.commit("empty commit, same tree")
    second = capture_snapshot(repo.root, CodeOptions(), store)

    assert first.ref.digest == second.ref.digest
    assert first.ref.git_commit != second.ref.git_commit
    assert first.reused is False and second.reused is True


def test_executable_bit_is_part_of_the_identity(repo: GitRepo, store: SnapshotStore) -> None:
    before = capture_snapshot(repo.root, CodeOptions(), store).ref.digest
    (repo.root / "train.py").chmod(0o755)
    after = capture_snapshot(repo.root, CodeOptions(), store).ref.digest
    assert before != after


def test_digest_is_the_hash_of_schema_and_sorted_files(repo: GitRepo, store: SnapshotStore) -> None:
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert res.ref.digest == snap.snapshot_digest(store.manifest(res.ref.digest)["files"])


def test_file_changing_during_capture_is_reported(
    repo: GitRepo, store: SnapshotStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter(range(1000))
    monkeypatch.setattr(snap, "_stat_key", lambda path: (1, next(ticks)))
    with pytest.raises(PreconditionFailed, match="changed while"):
        capture_snapshot(repo.root, CodeOptions(), store)


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------


def test_max_files_limit_lists_offenders(repo: GitRepo) -> None:
    with pytest.raises(LimitExceeded) as err:
        scan_project(repo.root, CodeOptions(max_files=2))
    details = err.value.details
    assert details["file_count"] == 4 and details["max_files"] == 2
    assert details["largest_files"] and "hint" in details


def test_max_bytes_limit_names_the_biggest_file_and_directory(repo: GitRepo) -> None:
    repo.write("assets/huge.txt", "x" * 50_000)
    repo.commit("big")
    with pytest.raises(LimitExceeded) as err:
        scan_project(repo.root, CodeOptions(max_bytes=10_000))
    details = err.value.details
    assert details["largest_files"][0] == {"path": "assets/huge.txt", "size": 50_000}
    assert details["largest_top_level"][0]["path"] == "assets/"


# ---------------------------------------------------------------------------
# non-git projects
# ---------------------------------------------------------------------------


def test_plain_directory_is_walked_with_default_excludes(tmp_path: Path, store: SnapshotStore) -> None:
    root = tmp_path / "plain"
    (root / "pkg" / "data").mkdir(parents=True)
    (root / "pkg" / "data" / "__init__.py").write_text("")  # a package named data is code
    (root / "pkg" / "data" / "loader.py").write_text("load = 1\n")
    (root / "data").mkdir()
    (root / "data" / "train.bin").write_bytes(b"\x00" * 10)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_ed25519").write_text("k")
    (root / "main.py").write_text("print(1)\n")
    res = capture_snapshot(root, CodeOptions(), store)

    files = paths_of(store, res.ref.digest)
    assert sorted(files) == ["main.py", "pkg/data/__init__.py", "pkg/data/loader.py"]
    assert res.ref.git_commit is None and res.ref.git_dirty is False
    assert any("not a git repository" in w for w in res.warnings)
    assert any(w.startswith("SECRET:") and ".ssh" in w for w in res.warnings)
    assert {"data/", ".venv/", ".ssh/"} <= set(res.excluded)


def test_project_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument):
        scan_project(tmp_path / "nope", CodeOptions())


def test_scan_is_deterministic(repo: GitRepo) -> None:
    a = scan_project(repo.root, CodeOptions())
    b = scan_project(repo.root, CodeOptions())
    assert [f.path for f in a.files] == [f.path for f in b.files] == sorted(f.path for f in a.files)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_tree_dir_materializes_hardlinked_read_only_files(repo: GitRepo, store: SnapshotStore) -> None:
    res = capture_snapshot(repo.root, CodeOptions(), store)
    tree = store.tree_dir(res.ref.digest)

    assert (tree / "src" / "model.py").read_text() == "class Model: ...\n"
    train = tree / "train.py"
    blob = store.object_path(digest_hex(paths_of(store, res.ref.digest)["train.py"]["sha256"]))
    assert os.stat(train).st_ino == os.stat(blob).st_ino
    assert not os.access(train, os.W_OK)
    assert os.access(tree / "run.sh", os.X_OK)
    assert not os.access(tree / "train.py", os.X_OK)
    assert store.tree_dir(res.ref.digest) == tree  # cached


def test_tree_dir_reports_missing_blobs(repo: GitRepo, store: SnapshotStore) -> None:
    res = capture_snapshot(repo.root, CodeOptions(), store)
    entry = paths_of(store, res.ref.digest)["train.py"]
    blob = store.object_path(digest_hex(entry["sha256"]))
    blob.chmod(0o644)
    blob.unlink()
    with pytest.raises(NotFound, match="missing"):
        store.tree_dir(res.ref.digest)


@pytest.mark.parametrize("bad", ["../../etc", "sha256:xyz", "sha256:" + "A" * 64, ""])
def test_digest_hex_rejects_non_digests(bad: str) -> None:
    with pytest.raises(InvalidArgument):
        digest_hex(bad)


def test_unknown_manifest_is_not_found(store: SnapshotStore) -> None:
    assert store.has("sha256:" + "0" * 64) is False
    with pytest.raises(NotFound):
        store.manifest("sha256:" + "0" * 64)


# ---------------------------------------------------------------------------
# matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "is_dir", "expected"),
    [
        ("logs/", "logs/a.txt", False, True),
        ("logs/", "src/logs/a.txt", False, True),
        ("logs/", "logs", False, False),  # a *file* named logs is not a directory match
        ("*.csv", "a/b/c.csv", False, True),
        ("/build", "build/x.o", False, True),
        ("/build", "src/build/x.o", False, False),
        ("src/gen", "src/gen/a.py", False, True),
        ("src/*.py", "src/a.py", False, True),
        ("tmp", "a/tmp", True, True),
        ("# comment", "comment", False, False),
    ],
)
def test_path_matcher(pattern: str, path: str, is_dir: bool, expected: bool) -> None:
    assert PathMatcher([pattern]).matches(path, is_dir=is_dir) is expected
