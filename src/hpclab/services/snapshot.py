"""Content-addressed code snapshots (build spec §2.4, system design §6.1).

A snapshot freezes *exactly* the code a Run executes, so every reported number can be traced
back to it and a later edit of the working tree can never change a queued job.

Pipeline::

    scan_project()      cheap pre-scan: which files, how many bytes, git facts, warnings
    capture_snapshot()  hash every file into the local blob store + write the manifest
    publish_snapshot()  upload to <remote_root>/snapshots/<hex> (staging -> verify -> seal
                        -> marker -> atomic rename); idempotent and race tolerant

What goes in (git projects): tracked files with their *working-tree* content, plus untracked
files explicitly listed in ``CodeOptions.include_untracked`` (git pathspecs). Never ``.git``.
Non-git projects are walked with the same exclusion rules (and a warning).

The project root itself may not be the home directory, one of its ancestors, or a credential
store such as ``~/.ssh``, ``~/.config/gh`` or ``~/.cache/huggingface`` (nor lie inside one):
the snapshot lands on shared cluster storage and the job can print any file in it, so no
frontend (an agent's inline spec included) can turn a snapshot into a copy of the user's logins.

Exclusion rules, strongest first:

1. likely secrets (``.env*``, ``*.pem``, ``*.key``, ``id_rsa*``, ``.netrc``, ``*.p12``,
   ``credentials*``, ``*credentials.json``, ``kaggle.json``, ``.npmrc``, ``*.tfvars``,
   ``.aws/``, ``.ssh/``, ``.config/gh/``, ``.config/gcloud/``, ``.cache/huggingface/``,
   ``.docker/config.json``, ``.kube/config`` ... and files whose first 64 KiB hold a
   ``-----BEGIN ... PRIVATE KEY-----`` block at a line start or as a JSON ``"private_key"``
   value, or a token in a well-known format: GitHub ``gh?_``, Hugging Face ``hf_``, Slack
   ``xox?-``, ``sk-`` API keys, AWS keys, Google ``ya29.``, OAuth ``refresh_token`` /
   ``client_secret`` JSON values, docker/npm registry auth, kubeconfig client keys) are
   excluded *even when tracked or explicitly included* and reported loudly - they are never
   uploaded. Name and content rules are a safety net, not a guarantee;
2. virtualenvs and caches (``.venv/``, ``venv/``, ``__pycache__/``, ``node_modules/`` ...);
3. user ``CodeOptions.exclude`` patterns (gitignore-like: ``name``, ``dir/``, ``/anchored``);
4. data/output directories (``data/``, ``outputs/``, ``wandb/``, ``runs/``, ``checkpoints/``
   ...) unless the file is tracked, explicitly included, or the directory is a Python package
   (has ``__init__.py``) - ``src/pkg/data/`` modules are code, not data;
5. large weight/array files (``*.pt``, ``*.ckpt``, ``*.safetensors`` ... over 5 MB) unless
   explicitly included - they belong in ``inputs``;
6. symlinks that resolve outside the project or into ``.git/`` (warned). Symlinks to files
   inside the project are stored as regular files with the target's content.

Dirty tracking: ``git status`` plus files marked ``--assume-unchanged``/``--skip-worktree``
whose content differs from the index (``git status`` and ``git diff`` never look at those,
so a "local config hack" would otherwise be recorded as a clean commit).

Identity: the snapshot digest is ``digest({"schema", "files"})`` where each file entry is
``{path, sha256, size, mode}`` sorted by path and ``mode`` keeps only the executable bit
(0o755 / 0o644). Timestamps and git metadata are *not* part of the identity, so identical
trees captured from different commits dedupe to one snapshot.

Local store layout (``SnapshotStore(root)``, root = ``Paths.snapshots``)::

    objects/<aa>/<hex>       read-only blobs (0444); ``<hex>.x`` = executable variant (0555)
    manifests/<hex>.json     manifest (first writer wins; identity = the files list)
    trees/<hex>/             materialized tree (hardlinks to blobs when possible), on demand
    tmp/                     scratch for atomic writes
    locks/publish-<hex>.lock serializes local publishes of one digest (flock)
    publisher-id             random seed naming this store's remote staging dirs

Remote contract (``publish_snapshot``): ``<remote_root>/snapshots/<hex>/`` exists only when
complete - it is renamed into place atomically *after* every file's sha256 was verified on the
cluster, the tree was made read-only and ``.hpclab-complete`` was written (LAST). Job wrappers
refuse to run a snapshot without that marker. Read-only permissions are an operational guard
against accidental edits, not a security boundary against the same Unix user. Uploads go to
``.staging/<hex>.<publisher>`` with a deadline that grows with the snapshot's size; an upload
interrupted by the link keeps that dir, so the retry resumes instead of starting from zero.
"""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hpclab.adapters.base import CommandFailed, CommandResult, Transport, TransportError
from hpclab.domain.clock import Clock, SystemClock, iso
from hpclab.domain.errors import (
    ErrorCode,
    HpcLabError,
    InvalidArgument,
    LimitExceeded,
    NotFound,
    PreconditionFailed,
    Unavailable,
)
from hpclab.domain.hashing import canonical_json
from hpclab.domain.hashing import digest as canonical_digest
from hpclab.domain.models import CodeOptions, CodeSnapshotRef

log = logging.getLogger("hpclab.snapshot")

SNAPSHOT_SCHEMA = "hpclab.snapshot/v1"
MARKER_SCHEMA = "hpclab.snapshot-marker/v1"
MARKER_NAME = ".hpclab-complete"
REMOTE_MANIFEST_NAME = ".hpclab-manifest.json"

MODE_EXEC = 0o755
MODE_PLAIN = 0o644

LARGE_WEIGHT_BYTES = 5 * 1024 * 1024
LARGE_FILE_WARN_BYTES = 20 * 1024 * 1024
_HASH_CHUNK = 1 << 20
_SNIFF_BYTES = 64 * 1024
_MAX_LISTED = 20  # bounded lists inside warnings / error details
_GIT_TIMEOUT_S = 120.0

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REMOTE_RE = re.compile(r"^/[A-Za-z0-9_./@+,=:~-]*$")
_PUBLISHER_SEED_RE = re.compile(r"^[0-9a-f]{32}$")
# Content that marks a file as a credential whatever its name (checked on the first
# ``_SNIFF_BYTES``). Every pattern is anchored on a token *format*, not on a word such as
# "token" (``tokenizer.py`` is code), and skips placeholders like ``${TOKEN}`` / ``<key>``.
# Patterns start with a literal and check the preceding character with a lookbehind placed
# *after* it, so the regex engine can skip ahead with a fast literal search (a leading
# lookbehind would make every pattern ~20x slower on a large tree).
_SECRET_CONTENT_PATTERNS: tuple[tuple[str, bytes], ...] = (
    # a PEM private key block starting a line, or one embedded in a JSON string the way cloud
    # service-account key files store it ({"private_key": "-----BEGIN PRIVATE KEY-----\n...")
    ("a private key block", rb"-----BEGIN (?<![^\n]-----BEGIN )[A-Z0-9 ]*PRIVATE KEY-----"),
    ("a private key block", rb'"private_key"\s*:\s*"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----'),
    ("a GitHub token", rb"gh[pousr]_(?<![A-Za-z0-9_]gh[pousr]_)[A-Za-z0-9]{36,}"),
    ("a GitHub token", rb"github_pat_(?<![A-Za-z0-9_]github_pat_)[A-Za-z0-9_]{50,}"),
    (
        "a Hugging Face token",
        rb"hf_(?<![A-Za-z0-9_]hf_)(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])[A-Za-z0-9]{30,}(?![A-Za-z0-9_])",
    ),
    ("a Slack token", rb"xox[abposr]-(?<![A-Za-z0-9_]xox[abposr]-)[A-Za-z0-9-]{10,}"),
    (
        "an API secret key",
        rb"sk-(?<![A-Za-z0-9_-]sk-)(?:ant-|proj-|svcacct-|admin-)?"
        rb"(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}",
    ),
    ("an API secret key", rb"k_live_(?<![A-Za-z0-9_][rs]k_live_)(?<=[rs]k_live_)[A-Za-z0-9]{20,}"),
    ("an AWS access key", rb"AKIA(?<![A-Z0-9]AKIA)(?![A-Z0-9]{9}EXAMPLE)[A-Z0-9]{16}(?![A-Z0-9])"),
    ("an AWS access key", rb"ASIA(?<![A-Z0-9]ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ("an AWS secret key", rb"aws_secret_access_key\s*[=:]\s*[\"']?[A-Za-z0-9/+]{30,}"),
    ("an AWS secret key", rb"AWS_SECRET_ACCESS_KEY\s*[=:]\s*[\"']?[A-Za-z0-9/+]{30,}"),
    ("a Google OAuth token", rb"ya29\.(?<![A-Za-z0-9_]ya29\.)[A-Za-z0-9_-]{20,}"),
    (
        "an OAuth token or client secret",
        rb'"(?i:refresh_?token|access_?token|client_?secret)"\s*:\s*"[^"\s$<{%][^"\s]{15,}"',
    ),
    ("a registry credential", rb'"auth"\s*:\s*"[A-Za-z0-9+/]{16,}={0,2}"'),
    ("a registry credential", rb"_authToken\s*=\s*[\"']?[^\s\"'$<{][^\s\"']{7,}"),
    ("a registry credential", rb"npmAuthToken\s*:\s*[\"']?[^\s\"'$<{][^\s\"']{7,}"),
    ("a Kubernetes client key", rb"client-key-data:[ \t]*[A-Za-z0-9+/=]{20,}"),
    ("a Kubernetes token", rb"token:[ \t]*[\"']?eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),
)
_SECRET_CONTENT_RES = tuple((what, re.compile(pat)) for what, pat in _SECRET_CONTENT_PATTERNS)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_URL_CREDENTIALS_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@]+@")

# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

_SECRET_NAME_PATTERNS = (
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    "_netrc",
    ".pgpass",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "credentials*",
    ".git-credentials",
    ".pypirc",
    ".htpasswd",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.toml",
    "kaggle.json",
    "*credentials.json",  # gcloud application_default_credentials.json, Claude .credentials.json
    "access_tokens.db",  # gcloud
    "stored_tokens",  # huggingface
    ".npmrc",
    ".dockercfg",
    ".vault-token",
    ".s3cfg",
    ".boto",
    ".databrickscfg",
    "rclone.conf",
    "*.tfvars",
    "*.tfstate",
    "*.tfstate.backup",
)
_SECRET_DIRS = frozenset(
    {
        ".aws",
        ".ssh",
        ".gnupg",
        ".kaggle",
        ".azure",
        ".huggingface",
        ".password-store",
        ".terraform.d",
        "legacy_credentials",  # gcloud
    }
)
# Credential stores recognised by a run of directory names (``x/.config/gh/hosts.yml``) and
# files recognised by their parent (``.kube/config``): the last name alone is too generic.
_SECRET_DIR_RUNS: tuple[tuple[str, ...], ...] = (
    (".config", "gh"),
    (".config", "gcloud"),
    (".config", "hub"),
    (".config", "rclone"),
    (".cache", "huggingface"),
    (".local", "share", "keyrings"),
)
_SECRET_FILE_SUFFIXES: tuple[tuple[str, ...], ...] = ((".docker", "config.json"), (".kube", "config"))
# Credential stores under the home directory. A project root that is one of them, lies inside
# one, or is an ancestor of one ($HOME itself, "/", ~/.config, ~/.cache ...) is refused, so no
# frontend can point a snapshot - which lands on shared cluster storage - at the user's logins.
_HOME_CREDENTIAL_PATHS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".docker",
    ".kube",
    ".azure",
    ".kaggle",
    ".password-store",
    ".terraform.d",
    ".huggingface",
    ".config/gh",
    ".config/gcloud",
    ".config/hub",
    ".config/rclone",
    ".cache/huggingface",
    ".local/share/keyrings",
    "Library/Keychains",
    ".claude/.credentials.json",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
)
# Root-level names publish_snapshot writes into the remote tree (a project file with one of
# these names would be overwritten there and deleted from the job's work dir).
_RESERVED_ROOT_NAMES = frozenset({MARKER_NAME, REMOTE_MANIFEST_NAME})
_CACHE_DIRS = frozenset(
    {
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".ipynb_checkpoints",
        ".hg",
        ".svn",
    }
)
_CACHE_FILE_PATTERNS = (".DS_Store", "*.pyc", "*.pyo")
_DATA_DIRS = frozenset(
    {"data", "outputs", "wandb", "runs", "checkpoints", "mlruns", "lightning_logs", "multirun"}
)
_DATA_FILE_PATTERNS = ("slurm-*.out",)
_WEIGHT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".h5",
        ".hdf5",
        ".onnx",
        ".gguf",
        ".npy",
        ".npz",
        ".pkl",
        ".pickle",
        ".parquet",
        ".arrow",
        ".tfrecord",
    }
)

ExcludeReason = Literal[
    "secret",
    "cache",
    "user_exclude",
    "data_dir",
    "large_weights",
    "symlink_outside",
    "symlink_to_dir",
    "broken_symlink",
    "special_file",
    "unsupported_name",
    "submodule_not_checked_out",
    "git_internal",
    "outside_project",
]


@dataclass(frozen=True)
class _Pattern:
    glob: str
    dir_only: bool
    anchored: bool
    floating: bool = False  # "**/a/b": the anchored glob may start at any depth


class PathMatcher:
    """A small, deterministic subset of gitignore matching.

    * ``name`` matches any path component (file or directory) at any depth;
    * ``name/`` matches directories only;
    * a pattern containing ``/`` (or starting with ``/``) is anchored at the project root and
      matches the path itself or any of its parent directories;
    * a leading ``**/`` matches at any depth *including the root* (``**/logs`` == ``logs``);
    * ``*``/``?``/``[]`` are fnmatch globs (``*`` also crosses ``/`` in anchored patterns).
    Negation (``!``) is not supported. ``casefold=True`` matches case-insensitively.
    """

    def __init__(self, patterns: Iterable[str], *, casefold: bool = False) -> None:
        self._casefold = casefold
        compiled: list[_Pattern] = []
        for raw in patterns:
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            if casefold:
                text = text.lower()
            dir_only = text.endswith("/")
            body = text.strip("/")
            floating = False
            while body.startswith("**/"):
                body, floating = body[3:], True
            if not body:
                continue
            if floating:
                compiled.append(_Pattern(glob=body, dir_only=dir_only, anchored="/" in body, floating=True))
            else:
                compiled.append(
                    _Pattern(glob=body, dir_only=dir_only, anchored=text.startswith("/") or "/" in body)
                )
        self._patterns = tuple(compiled)

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def matches(self, rel: str, *, is_dir: bool = False) -> bool:
        if not self._patterns:
            return False
        if self._casefold:
            rel = rel.lower()
        parts = rel.split("/")
        dir_parts = parts if is_dir else parts[:-1]
        for pat in self._patterns:
            if pat.anchored:
                upto = len(dir_parts) if pat.dir_only else len(parts)
                globs = (pat.glob, "*/" + pat.glob) if pat.floating else (pat.glob,)
                if any(
                    fnmatch.fnmatchcase("/".join(parts[:i]), g) for i in range(1, upto + 1) for g in globs
                ):
                    return True
            else:
                candidates = dir_parts if pat.dir_only else parts
                if any(fnmatch.fnmatchcase(c, pat.glob) for c in candidates):
                    return True
        return False


_SECRET_NAMES = PathMatcher(_SECRET_NAME_PATTERNS, casefold=True)
_CACHE_FILES = PathMatcher(_CACHE_FILE_PATTERNS)
_DATA_FILES = PathMatcher(_DATA_FILE_PATTERNS)


def _has_run(parts: Sequence[str], run: tuple[str, ...]) -> bool:
    n = len(run)
    return any(tuple(parts[i : i + n]) == run for i in range(len(parts) - n + 1))


def _is_secret_dir(parts: Sequence[str]) -> bool:
    """True if a directory (given as lowercased path components) is or lies in a credential store."""
    return any(p in _SECRET_DIRS for p in parts) or any(_has_run(parts, run) for run in _SECRET_DIR_RUNS)


def is_secret_path(rel: str) -> bool:
    """True if a relative path looks like a credential (name-based check)."""
    parts = rel.split("/")
    lowered = [p.lower() for p in parts]
    if _is_secret_dir(lowered[:-1]):
        return True
    if any(len(lowered) >= len(sfx) and tuple(lowered[-len(sfx) :]) == sfx for sfx in _SECRET_FILE_SUFFIXES):
        return True
    return _SECRET_NAMES.matches(parts[-1])


def sniff_secret(head: bytes) -> str | None:
    """What credential ``head`` (a file's first bytes) contains, e.g. "a GitHub token", or None."""
    for what, pattern in _SECRET_CONTENT_RES:
        if pattern.search(head):
            return what
    return None


# ---------------------------------------------------------------------------
# Scan results
# ---------------------------------------------------------------------------

FileOrigin = Literal["tracked", "untracked", "walk"]


@dataclass(frozen=True)
class ScannedFile:
    path: str  # posix path relative to the project root (the name inside the snapshot)
    source: Path  # file whose bytes are captured (the resolved target for in-project symlinks)
    size: int
    executable: bool
    origin: FileOrigin


@dataclass(frozen=True)
class ExcludedPath:
    path: str  # directories carry a trailing "/"
    reason: ExcludeReason


@dataclass(frozen=True)
class GitInfo:
    toplevel: str
    commit: str | None
    branch: str | None
    dirty: bool
    dirty_patch_digest: str | None
    remote_url: str | None
    dirty_files: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "dirty_patch_digest": self.dirty_patch_digest,
            "remote_url": self.remote_url,
        }


@dataclass(frozen=True)
class ProjectScan:
    project_root: Path
    files: tuple[ScannedFile, ...]
    total_bytes: int
    git: GitInfo | None
    warnings: tuple[str, ...]
    excluded: tuple[ExcludedPath, ...]
    untracked_included: tuple[str, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class SnapshotResult:
    ref: CodeSnapshotRef
    warnings: list[str]
    excluded: list[str]
    excluded_detail: list[ExcludedPath] = field(default_factory=list)
    reused: bool = False  # the identical tree was already in the local store


class SnapshotVerificationError(HpcLabError):
    """Uploaded bytes did not match the manifest. Retryable: a re-upload usually fixes it."""

    code = ErrorCode.PRECONDITION_FAILED

    def __init__(self, message: str, **kw: Any) -> None:
        kw.setdefault("retryable", True)
        super().__init__(message, **kw)


class _FileChanged(Exception):
    """A file changed between stat/hash/copy; the capture retries once, then fails."""


# ---------------------------------------------------------------------------
# Local blob store
# ---------------------------------------------------------------------------


def digest_hex(digest: str) -> str:
    """Hex part of ``sha256:<hex>`` (or a bare hex). Rejects anything else: digests become
    path components, so this is also the path-traversal guard."""
    value = digest.split(":", 1)[1] if digest.startswith("sha256:") else digest
    if not _HEX_RE.match(value):
        raise InvalidArgument(f"not a sha256 digest: {digest!r}")
    return value


def _safe_manifest_path(path: str) -> str:
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or any(p in ("", ".", "..") for p in parts)
        or _CONTROL_RE.search(path)
    ):
        raise HpcLabError(f"snapshot manifest contains an unsafe path {path!r}", details={"path": path})
    return path


class SnapshotStore:
    """Content-addressed local store; safe for concurrent writers (atomic renames)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- layout --------------------------------------------------------------
    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def trees_dir(self) -> Path:
        return self.root / "trees"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def object_path(self, sha_hex: str, *, executable: bool = False) -> Path:
        name = sha_hex + (".x" if executable else "")
        return self.objects_dir / sha_hex[:2] / name

    def manifest_path(self, digest: str) -> Path:
        return self.manifests_dir / f"{digest_hex(digest)}.json"

    @property
    def locks_dir(self) -> Path:
        return self.root / "locks"

    def publisher_id(self) -> str:
        """Stable id of this store on this host (12 hex): names its remote staging dirs.

        Derived from a random seed kept in the store plus the host name, so it survives
        controller restarts (a retried upload resumes into the same staging dir) but is never
        shared with another machine - not even one holding a copy of this store.
        """
        seed = self._publisher_seed()
        return hashlib.sha256(f"{seed}\0{socket.gethostname()}".encode()).hexdigest()[:12]

    def _publisher_seed(self) -> str:
        path = self.root / "publisher-id"
        for _ in range(3):
            try:
                value: str | None = path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                value = None
            except (OSError, UnicodeDecodeError):
                value = ""
            if value and _PUBLISHER_SEED_RE.match(value):
                return value
            self._ensure_dirs()
            fd, tmp = tempfile.mkstemp(prefix=".publisher-", dir=self.tmp_dir)
            try:
                with os.fdopen(fd, "w", encoding="ascii") as fh:
                    fh.write(secrets.token_hex(16) + "\n")
                if value is None:
                    try:
                        os.link(tmp, path)  # first writer wins
                    except FileExistsError:
                        pass
                    except OSError:  # filesystem without hardlinks
                        if not path.exists():
                            os.replace(tmp, path)
                else:  # unreadable or corrupt: replace it
                    os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        raise HpcLabError(f"could not create the snapshot store's publisher id at {path}")

    # -- queries -------------------------------------------------------------
    def has(self, digest: str) -> bool:
        return self.manifest_path(digest).is_file()

    def manifest(self, digest: str) -> dict[str, Any]:
        path = self.manifest_path(digest)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NotFound(
                f"snapshot {digest} is not in the local store", details={"digest": digest}
            ) from exc
        if not isinstance(data, dict) or data.get("schema") != SNAPSHOT_SCHEMA:
            raise HpcLabError(f"snapshot manifest {path} is not {SNAPSHOT_SCHEMA}")
        return data

    # -- writes --------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        for d in (self.objects_dir, self.manifests_dir, self.trees_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def ensure_blob(self, source: Path, sha_hex: str, size: int) -> bool:
        """Store ``source`` as blob ``sha_hex`` unless present. Returns True if written.

        The copy is re-hashed while writing; if the bytes differ from ``sha_hex`` the source
        changed after it was hashed and ``_FileChanged`` is raised (nothing is stored).
        """
        dest = self.object_path(sha_hex)
        if dest.is_file():
            return False
        self._ensure_dirs()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".blob-", dir=self.tmp_dir)
        try:
            h = hashlib.sha256()
            written = 0
            with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
                for chunk in iter(lambda: src.read(_HASH_CHUNK), b""):
                    h.update(chunk)
                    out.write(chunk)
                    written += len(chunk)
            if h.hexdigest() != sha_hex or written != size:
                raise _FileChanged(str(source))
            os.chmod(tmp, 0o444)
            os.replace(tmp, dest)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return True

    def _exec_blob(self, sha_hex: str) -> Path:
        """Executable variant of a blob (a single inode cannot carry two modes)."""
        dest = self.object_path(sha_hex, executable=True)
        if dest.is_file():
            return dest
        fd, tmp = tempfile.mkstemp(prefix=".blobx-", dir=self.tmp_dir)
        os.close(fd)
        try:
            shutil.copyfile(self.object_path(sha_hex), tmp)
            os.chmod(tmp, 0o555)  # noqa: S103 - read+exec only; blobs are never writable
            os.replace(tmp, dest)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return dest

    def save_manifest(self, manifest: dict[str, Any]) -> bool:
        """Write the manifest for ``manifest['digest']``; first writer wins. True if created."""
        self._ensure_dirs()
        final = self.manifest_path(manifest["digest"])
        if final.is_file():
            return False
        fd, tmp = tempfile.mkstemp(prefix=".manifest-", dir=self.tmp_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, sort_keys=True, indent=1, ensure_ascii=False)
                fh.write("\n")
            os.chmod(tmp, 0o444)
            try:
                os.link(tmp, final)
            except FileExistsError:
                return False
            except OSError:  # filesystem without hardlinks: fall back to a rename
                if final.exists():
                    return False
                os.replace(tmp, final)
            return True
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def tree_dir(self, digest: str) -> Path:
        """Materialized snapshot tree (built from blobs on first use; hardlinks when possible).

        Directories are 0755 and files keep blob modes (0444 / 0555). The final directory
        only ever appears complete (built under ``trees/.tmp-*`` and renamed).
        """
        sha_hex = digest_hex(digest)
        final = self.trees_dir / sha_hex
        if final.is_dir():
            return final
        manifest = self.manifest(digest)
        self._ensure_dirs()
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{sha_hex[:12]}-", dir=self.trees_dir))
        try:
            for entry in manifest["files"]:
                self._materialize_file(tmp, entry)
            os.chmod(tmp, 0o755)  # noqa: S103 - mkdtemp creates 0700; trees are plain code dirs
            try:
                os.rename(tmp, final)
            except OSError:
                if not final.is_dir():
                    raise
                shutil.rmtree(tmp, ignore_errors=True)  # a concurrent builder won
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return final

    def _materialize_file(self, tree: Path, entry: dict[str, Any]) -> None:
        rel = _safe_manifest_path(str(entry["path"]))
        sha_hex = digest_hex(str(entry["sha256"]))
        blob = self.object_path(sha_hex)
        try:
            blob_size = blob.stat().st_size
        except FileNotFoundError as exc:
            raise NotFound(
                f"blob {sha_hex[:12]} of {rel!r} is missing from the local snapshot store; capture again",
                details={"path": rel, "sha256": entry["sha256"]},
            ) from exc
        if blob_size != int(entry["size"]):
            raise HpcLabError(f"blob {sha_hex[:12]} has size {blob_size}, manifest says {entry['size']}")
        if int(entry["mode"]) & 0o111:
            blob = self._exec_blob(sha_hex)
        dest = tree / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(blob, dest)
        except OSError:
            shutil.copyfile(blob, dest)
            os.chmod(dest, 0o555 if int(entry["mode"]) & 0o111 else 0o444)


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


class _GitMissing(Exception):
    pass


def _git(
    root: Path, *args: str, timeout: float = _GIT_TIMEOUT_S, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    try:
        return subprocess.run(
            ["git", "-C", str(root), "-c", "core.quotepath=off", *args],
            input=stdin,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _GitMissing() from exc
    except subprocess.TimeoutExpired as exc:
        raise Unavailable(
            f"git {args[0]} timed out after {timeout:.0f}s in {root}", details={"project_root": str(root)}
        ) from exc


def _git_ok(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    proc = _git(root, *args, stdin=stdin)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise PreconditionFailed(
            f"git {' '.join(args[:2])} failed in {root}: {msg}",
            details={"project_root": str(root), "stderr": msg[-2000:]},
        )
    return proc.stdout


def _split_z(data: bytes) -> list[bytes]:
    return [p for p in data.split(b"\0") if p]


def _git_toplevel(root: Path, warnings: list[str]) -> str | None:
    """The repository top level, or None for a non-git project (git missing -> warning)."""
    try:
        proc = _git(root, "rev-parse", "--show-toplevel")
    except _GitMissing:
        warnings.append("git is not installed: the project is snapshotted as a plain directory tree")
        return None
    if proc.returncode == 0:
        return proc.stdout.decode("utf-8", "replace").strip()
    msg = proc.stderr.decode("utf-8", "replace").strip()
    if "not a git repository" in msg.lower():
        return None
    raise PreconditionFailed(
        f"git refused to read {root}: {msg}",
        details={
            "project_root": str(root),
            "stderr": msg[-2000:],
            "hint": "for 'dubious ownership' run: git config --global --add safe.directory <path>",
        },
    )


def _sanitize_remote_url(url: str | None) -> str | None:
    if not url:
        return None
    return _URL_CREDENTIALS_RE.sub(lambda m: m.group("scheme"), url.strip())


def _parse_status_paths(data: bytes) -> list[str]:
    """Paths from ``git status --porcelain=v1 -z`` (rename entries carry an extra source path)."""
    out: list[str] = []
    items = data.split(b"\0")
    i = 0
    while i < len(items):
        item = items[i]
        i += 1
        if len(item) < 4:
            continue
        code = item[:2].decode("ascii", "replace")
        out.append(item[3:].decode("utf-8", "replace"))
        if "R" in code or "C" in code:
            i += 1  # skip the rename/copy source
    return out


_FLAG_NAMES = {"assume": "assume-unchanged", "skip": "skip-worktree"}


def _index_flagged_files(root: Path) -> list[tuple[str, str, str]]:
    """(path, flag, index blob id) of regular files marked assume-unchanged / skip-worktree.

    ``git status`` and ``git diff`` never look at such files, so their working-tree edits are
    invisible to both. ``ls-files -v`` tags them: lowercase = assume-unchanged, ``S`` =
    skip-worktree (``s`` = both).
    """
    out: list[tuple[str, str, str]] = []
    for item in _split_z(_git_ok(root, "ls-files", "-v", "-s", "-z")):
        meta, sep, raw_path = item.partition(b"\t")
        fields = meta.split()
        if not sep or len(fields) != 4 or fields[1] not in (b"100644", b"100755"):
            continue
        tag = fields[0].decode("ascii", "replace")
        if tag in ("S", "s"):
            flag = _FLAG_NAMES["skip"]
        elif tag.islower():
            flag = _FLAG_NAMES["assume"]
        else:
            continue
        out.append((raw_path.decode("utf-8", "replace"), flag, fields[2].decode("ascii")))
    return out


def _hidden_edits(root: Path, warnings: list[str]) -> dict[str, str]:
    """Flagged files whose working-tree content differs from the index: {path: worktree blob id}.

    Blob ids come from ``git hash-object --stdin-paths``, which applies the same clean filters
    (git-lfs, eol conversion) as ``git add``, so an untouched LFS file is not a false positive.
    Flagged files that are not checked out (sparse checkout) are only reported.
    """
    flagged = _index_flagged_files(root)
    present = [(p, f, oid) for p, f, oid in flagged if (root / p).is_file() and "\n" not in p]
    absent = len(flagged) - len(present)
    if absent:
        warnings.append(
            f"{absent} tracked file(s) marked skip-worktree/assume-unchanged are not checked out "
            "(sparse checkout?) and are not in the snapshot"
        )
    if not present:
        return {}
    listing = "".join(p + "\n" for p, _, _ in present).encode("utf-8")
    blob_ids = _git_ok(root, "hash-object", "--stdin-paths", stdin=listing).decode("ascii").split()
    if len(blob_ids) != len(present):
        raise PreconditionFailed(
            "git hash-object returned an unexpected number of ids", details={"project_root": str(root)}
        )
    hidden = {p: oid for (p, _, index_oid), oid in zip(present, blob_ids, strict=True) if oid != index_oid}
    if hidden:
        flags = {p: f for p, f, _ in present}
        listed = ", ".join(f"{p} ({flags[p]})" for p in sorted(hidden)[:10])
        warnings.append(
            f"{len(hidden)} file(s) are edited but hidden from `git status` by index flags: {listed}; "
            "their working-tree content is captured and the snapshot is marked dirty"
        )
    return hidden


def _git_info(root: Path, toplevel: str, warnings: list[str]) -> GitInfo:
    head = _git(root, "rev-parse", "--verify", "-q", "HEAD")
    commit = head.stdout.decode().strip() if head.returncode == 0 else None
    br = _git(root, "symbolic-ref", "-q", "--short", "HEAD")
    branch = br.stdout.decode("utf-8", "replace").strip() if br.returncode == 0 else None
    status = _git_ok(root, "status", "--porcelain=v1", "-z", "--untracked-files=no", "--", ".")
    # porcelain v1 paths are relative to the repository root; report them project-relative
    prefix = Path(os.path.relpath(root, toplevel)).as_posix()
    status_paths = [
        p[len(prefix) + 1 :] if prefix != "." and p.startswith(prefix + "/") else p
        for p in _parse_status_paths(status)
    ]
    hidden = _hidden_edits(root, warnings)
    dirty_files = sorted(set(status_paths) | set(hidden))
    dirty = bool(dirty_files)
    patch_digest = None
    if dirty and commit:
        patch = _git_ok(
            root,
            "diff",
            "--binary",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "HEAD",
            "--",
            ".",
        )
        h = hashlib.sha256(patch)
        for path in sorted(hidden):  # `git diff` does not show edits hidden by index flags
            h.update(f"\0hidden-edit\0{path}\0{hidden[path]}".encode())
        patch_digest = "sha256:" + h.hexdigest()
    remote = _git(root, "config", "--get", "remote.origin.url")
    remote_url = remote.stdout.decode("utf-8", "replace") if remote.returncode == 0 else None
    return GitInfo(
        toplevel=toplevel,
        commit=commit,
        branch=branch,
        dirty=dirty,
        dirty_patch_digest=patch_digest,
        remote_url=_sanitize_remote_url(remote_url),
        dirty_files=tuple(sorted(dirty_files)),
    )


def _decode_git_path(raw: bytes, warnings: list[str]) -> str | None:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        warnings.append(f"skipped a file with a non-UTF-8 name: {raw.decode('utf-8', 'replace')!r}")
        return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    rel: str
    origin: FileOrigin
    explicit: bool = False


class _Scanner:
    """Collects candidates, applies the exclusion rules and accumulates warnings."""

    def __init__(self, root: Path, options: CodeOptions) -> None:
        self.root = root
        self.options = options
        self.user_excludes = PathMatcher(options.exclude)
        self.warnings: list[str] = []
        self.excluded: list[ExcludedPath] = []
        self.files: list[ScannedFile] = []
        self._package_cache: dict[str, bool] = {}

    # -- rules ---------------------------------------------------------------
    def _is_package_dir(self, rel_dir: str) -> bool:
        cached = self._package_cache.get(rel_dir)
        if cached is None:
            cached = (self.root / rel_dir / "__init__.py").is_file()
            self._package_cache[rel_dir] = cached
        return cached

    def _in_data_dir(self, parts: Sequence[str]) -> bool:
        for i, part in enumerate(parts):
            if part in _DATA_DIRS and not self._is_package_dir("/".join(parts[: i + 1])):
                return True
        return False

    def dir_exclusion(self, rel_dir: str) -> ExcludeReason | None:
        """Reason to prune a whole directory during a plain walk (never for git listings)."""
        parts = rel_dir.split("/")
        name = parts[-1]
        if _is_secret_dir([p.lower() for p in parts]):
            return "secret"
        if name in _CACHE_DIRS or name == ".git":
            return "cache"
        if self.user_excludes.matches(rel_dir, is_dir=True):
            return "user_exclude"
        if self._in_data_dir(parts):
            return "data_dir"
        return None

    def path_exclusion(self, cand: _Candidate) -> ExcludeReason | None:
        parts = cand.rel.split("/")
        if is_secret_path(cand.rel):
            return "secret"
        if any(p in _CACHE_DIRS for p in parts[:-1]) or _CACHE_FILES.matches(parts[-1]):
            return "cache"
        if self.user_excludes.matches(cand.rel):
            return "user_exclude"
        in_data = self._in_data_dir(parts[:-1]) or _DATA_FILES.matches(parts[-1])
        if in_data and cand.origin != "tracked" and not cand.explicit:
            return "data_dir"
        return None

    # -- recording -----------------------------------------------------------
    def exclude(self, rel: str, reason: ExcludeReason, warning: str | None = None) -> None:
        self.excluded.append(ExcludedPath(rel, reason))
        if reason == "secret":
            msg = (
                f"SECRET: excluded likely credential file {rel!r} from the snapshot (never uploaded); "
                "if it is tracked by git, consider removing it from the repository"
            )
            self.warnings.append(msg)
            log.warning("%s", msg)
        elif warning:
            self.warnings.append(warning)

    def add(self, cand: _Candidate) -> None:
        rel = cand.rel
        parts = rel.split("/")
        if ".git" in parts:
            return
        if not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in parts):
            # git never lists such paths for plain pathspecs; refuse them defensively
            self.exclude(rel, "outside_project", f"skipped {rel!r}: not a path inside the project root")
            return
        if rel in _RESERVED_ROOT_NAMES:
            self.exclude(
                rel,
                "unsupported_name",
                f"skipped {rel!r}: the name is reserved for hpclab's snapshot marker files",
            )
            return
        if _CONTROL_RE.search(rel):
            self.exclude(
                rel,
                "unsupported_name",
                f"skipped {rel!r}: control characters in file names are not supported",
            )
            return
        reason = self.path_exclusion(cand)
        if reason is not None:
            self.exclude(rel, reason)
            return
        full = self.root / rel
        try:
            st = os.lstat(full)
        except FileNotFoundError:
            return  # tracked but deleted in the working tree: the snapshot mirrors the tree
        source = full
        if stat.S_ISLNK(st.st_mode):
            resolved = self._resolve_symlink(rel, full)
            if resolved is None:
                return
            source, st = resolved
        elif stat.S_ISDIR(st.st_mode):
            self.exclude(
                rel + "/", "submodule_not_checked_out", f"skipped {rel!r}: git submodule is not checked out"
            )
            return
        elif not stat.S_ISREG(st.st_mode):
            self.exclude(rel, "special_file", f"skipped {rel!r}: not a regular file")
            return
        if (
            not cand.explicit
            and st.st_size > LARGE_WEIGHT_BYTES
            and os.path.splitext(rel)[1].lower() in _WEIGHT_SUFFIXES
        ):
            self.exclude(rel, "large_weights")
            return
        self.files.append(
            ScannedFile(
                path=rel,
                source=source,
                size=st.st_size,
                executable=bool(st.st_mode & stat.S_IXUSR),
                origin=cand.origin,
            )
        )

    def _resolve_symlink(self, rel: str, full: Path) -> tuple[Path, os.stat_result] | None:
        target = Path(os.path.realpath(full))
        root_real = str(self.root)
        if str(target) != root_real and not str(target).startswith(root_real + os.sep):
            self.exclude(
                rel, "symlink_outside", f"excluded symlink {rel!r}: it points outside the project ({target})"
            )
            return None
        target_rel = target.relative_to(self.root).as_posix()
        if ".git" in target_rel.split("/"):
            self.exclude(
                rel,
                "git_internal",
                f"excluded symlink {rel!r}: it points into .git/ (git internals can hold credentials)",
            )
            return None
        if is_secret_path(target_rel):
            self.exclude(rel, "secret")
            return None
        try:
            tst = os.stat(target)
        except OSError:
            self.exclude(rel, "broken_symlink", f"excluded broken symlink {rel!r}")
            return None
        if stat.S_ISDIR(tst.st_mode):
            self.exclude(
                rel,
                "symlink_to_dir",
                f"excluded symlink {rel!r}: symlinks to directories are not captured (the target is "
                "captured under its real path if it is part of the project)",
            )
            return None
        if not stat.S_ISREG(tst.st_mode):
            self.exclude(rel, "special_file", f"skipped {rel!r}: symlink to a non-regular file")
            return None
        return target, tst


def _home_dirs() -> list[Path]:
    """``$HOME`` and the account's passwd home (they differ when HOME was overridden)."""
    raw: list[str] = []
    if os.environ.get("HOME"):
        raw.append(os.environ["HOME"])
    try:
        import pwd

        raw.append(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError, OSError):
        pass
    homes: list[Path] = []
    for value in raw:
        try:
            home = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if home not in homes:
            homes.append(home)
    return homes


def _credential_store_conflict(root: Path) -> tuple[str, Path] | None:
    """(relation, store) when ``root`` is, lies inside, or contains a home credential store.

    Directories are compared by path *and* by identity (device + inode): ``resolve()`` keeps
    the caller's spelling, so ``~/.SSH`` on a case-insensitive filesystem (macOS's default),
    ``/System/Volumes/Data/Users/me`` (a macOS firmlink) or a bind mount would otherwise name
    a credential store - or the whole home - without matching it as a string.
    """
    keys: dict[Path, tuple[int, int] | None] = {}

    def key(path: Path) -> tuple[int, int] | None:
        if path not in keys:
            try:
                st = os.stat(path)
            except (OSError, ValueError):
                keys[path] = None
            else:
                keys[path] = (st.st_dev, st.st_ino)
        return keys[path]

    root_key = key(root)
    root_chain = {k for k in map(key, root.parents) if k is not None}
    for home in _home_dirs():
        for rel in _HOME_CREDENTIAL_PATHS:
            store = home / rel
            try:
                targets = {store, store.resolve()}  # ~/.config may be a symlink into dotfiles/
            except (OSError, RuntimeError):
                targets = {store}
            for target in sorted(targets):
                target_key = key(target)
                if root == target or (root_key is not None and target_key == root_key):
                    return "is", target
                if target in root.parents or (target_key is not None and target_key in root_chain):
                    return "is inside", target
                if root in target.parents or (
                    root_key is not None and any(key(p) == root_key for p in target.parents)
                ):
                    return "contains", target
    return None


def _resolve_project_root(project_root: Path) -> Path:
    root = Path(project_root).expanduser()
    if not root.is_dir():
        raise InvalidArgument(f"project root {str(project_root)!r} is not a directory")
    root = root.resolve()
    conflict = _credential_store_conflict(root)
    if conflict is not None:
        relation, store = conflict
        raise InvalidArgument(
            f"refusing to snapshot {root}: it {relation} the credential store {store}. A code snapshot "
            "is uploaded to shared cluster storage, so its root must be the project directory itself, "
            "never the home directory, one of its ancestors or a login/token store",
            details={
                "project_root": str(root),
                "credential_store": str(store),
                "hint": "set code.project_root to the project's own directory (e.g. ~/code/my-project)",
            },
        )
    return root


def _validate_include(entry: str) -> str:
    """A relative path or glob inside the project. Pathspec magic (``:(top)``, ``:/``, ``:!``)
    is refused: ``:(top)``/``:/`` resolve against the repository root and would reach outside
    a project that is a subdirectory of a monorepo (write ``./:name`` for a literal colon)."""
    text = entry.strip()
    if not text or text.startswith(("/", ":")) or ".." in text.split("/") or "\0" in text:
        raise InvalidArgument(
            f"code.include_untracked entry {entry!r} must be a relative path or glob inside the project "
            "(no absolute paths, '..' or git pathspec magic)"
        )
    return text


def _collect_git(scanner: _Scanner, toplevel: str) -> tuple[GitInfo, list[str]]:
    root = scanner.root
    info = _git_info(root, toplevel, scanner.warnings)
    tracked_raw = _git_ok(root, "ls-files", "-z", "--cached", "--recurse-submodules")
    tracked: set[str] = set()
    for raw in _split_z(tracked_raw):
        rel = _decode_git_path(raw, scanner.warnings)
        if rel is not None and rel not in tracked:
            tracked.add(rel)
            scanner.add(_Candidate(rel, "tracked"))

    included: list[str] = []
    for entry in scanner.options.include_untracked:
        spec = _validate_include(entry)
        listing = _git_ok(root, "ls-files", "-z", "--others", "--", spec)
        decoded = (_decode_git_path(raw, scanner.warnings) for raw in _split_z(listing))
        matched = [rel for rel in decoded if rel is not None]
        new = [rel for rel in matched if rel not in tracked and rel not in included]
        if not matched and spec not in tracked:
            scanner.warnings.append(f"code.include_untracked entry {entry!r} matched no untracked files")
        for rel in sorted(new):
            before = len(scanner.files)
            scanner.add(_Candidate(rel, "untracked", explicit=True))
            if len(scanner.files) > before:
                included.append(rel)
            tracked.add(rel)  # never add twice across overlapping pathspecs

    _warn_untracked_left_out(scanner, set(included))
    return info, included


def _warn_untracked_left_out(scanner: _Scanner, included: set[str]) -> None:
    """Loudly list untracked, non-ignored files that will NOT be in the snapshot.

    A new module that was never ``git add``-ed is the classic "ImportError after two days in
    the queue"; ``--directory`` keeps the listing bounded for large untracked trees.
    """
    raw = _git_ok(
        scanner.root,
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--directory",
        "--no-empty-directory",
    )
    left: list[str] = []
    for item in _split_z(raw):
        rel = item.decode("utf-8", "replace")
        is_dir = rel.endswith("/")
        clean = rel.rstrip("/")
        if clean in included or (is_dir and any(i.startswith(rel) for i in included)):
            continue
        parts = clean.split("/")
        if is_dir and (scanner.dir_exclusion(clean) is not None or any(p in _CACHE_DIRS for p in parts)):
            continue
        if not is_dir and scanner.path_exclusion(_Candidate(clean, "untracked")) is not None:
            continue
        left.append(rel)
    if left:
        shown = ", ".join(left[:10]) + (f", ... (+{len(left) - 10} more)" if len(left) > 10 else "")
        scanner.warnings.append(
            f"{len(left)} untracked path(s) are NOT in the snapshot (git add them or list them in "
            f"code.include_untracked): {shown}"
        )


def _collect_walk(scanner: _Scanner) -> None:
    """Plain directory walk for non-git projects, pruning excluded directories early."""
    root = scanner.root
    limit = scanner.options.max_files + 1
    scanner.warnings.append(
        f"{root} is not a git repository: snapshotting the directory tree with default excludes "
        "(no commit/dirty tracking; consider `git init`)"
    )
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else Path(rel_dir).as_posix()
        keep: list[str] = []
        for name in sorted(dirnames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if os.path.islink(os.path.join(dirpath, name)):
                scanner.add(_Candidate(rel, "walk"))  # recorded as symlink_to_dir / outside
                continue
            reason = scanner.dir_exclusion(rel)
            if reason is not None:
                scanner.exclude(rel + "/", reason)
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError:
                scanner.warnings.append(f"skipped a file with a non-UTF-8 name under {rel_dir or '.'!r}")
                continue
            scanner.add(_Candidate(rel, "walk"))
            if len(scanner.files) >= limit:
                scanner.warnings.append(f"directory walk stopped after {limit} files (limit exceeded)")
                return


def _largest_offenders(files: Sequence[ScannedFile]) -> dict[str, Any]:
    by_size = sorted(files, key=lambda f: (-f.size, f.path))[:15]
    dirs: dict[str, list[int]] = {}
    for f in files:
        top = f.path.split("/", 1)[0] + ("/" if "/" in f.path else "")
        acc = dirs.setdefault(top, [0, 0])
        acc[0] += f.size
        acc[1] += 1
    top_dirs = sorted(dirs.items(), key=lambda kv: (-kv[1][0], kv[0]))[:10]
    return {
        "largest_files": [{"path": f.path, "size": f.size} for f in by_size],
        "largest_top_level": [{"path": k, "bytes": v[0], "files": v[1]} for k, v in top_dirs],
    }


def _enforce_limits(files: Sequence[ScannedFile], total: int, options: CodeOptions) -> None:
    over_files = len(files) > options.max_files
    over_bytes = total > options.max_bytes
    if not (over_files or over_bytes):
        return
    what = []
    if over_files:
        what.append(f"{len(files)} files > max_files {options.max_files}")
    if over_bytes:
        what.append(f"{total} bytes > max_bytes {options.max_bytes}")
    raise LimitExceeded(
        "code snapshot too large: " + "; ".join(what),
        details={
            "file_count": len(files),
            "max_files": options.max_files,
            "total_bytes": total,
            "max_bytes": options.max_bytes,
            **_largest_offenders(files),
            "hint": "add code.exclude patterns (e.g. 'logs/', '*.csv'), move data/weights to inputs, "
            "or raise code.max_bytes / code.max_files",
        },
    )


def scan_project(project_root: Path, options: CodeOptions) -> ProjectScan:
    """Cheap pre-scan: which files a snapshot would contain, their total size, git facts.

    ``project_root`` wins over ``options.project_root`` (the caller resolves the latter).
    Raises ``LimitExceeded`` (with the largest offenders) when ``max_files``/``max_bytes`` are
    exceeded, ``PreconditionFailed`` when the tree is dirty and ``allow_dirty`` is False or git
    refuses the repository, ``InvalidArgument`` for a bad root or include entry.
    Deterministic: files are sorted by path.
    """
    root = _resolve_project_root(project_root)
    scanner = _Scanner(root, options)
    toplevel = _git_toplevel(root, scanner.warnings)
    git: GitInfo | None = None
    included: list[str] = []
    if toplevel is None:
        _collect_walk(scanner)
    else:
        git, included = _collect_git(scanner, toplevel)
        if git.dirty and not options.allow_dirty:
            raise PreconditionFailed(
                "the working tree has uncommitted changes and code.allow_dirty is false",
                details={
                    "dirty_files": list(git.dirty_files[:_MAX_LISTED]),
                    "dirty_count": len(git.dirty_files),
                },
            )
    files = sorted(scanner.files, key=lambda f: f.path)
    total = sum(f.size for f in files)
    _enforce_limits(files, total, options)
    big = [f for f in files if f.size > LARGE_FILE_WARN_BYTES]
    if big:
        listed = ", ".join(f"{f.path} ({f.size // (1024 * 1024)} MiB)" for f in big[:10])
        scanner.warnings.append(f"large files are included in the code snapshot: {listed}")
    if not files:
        scanner.warnings.append("the code snapshot is empty")
    return ProjectScan(
        project_root=root,
        files=tuple(files),
        total_bytes=total,
        git=git,
        warnings=tuple(scanner.warnings),
        excluded=tuple(sorted(scanner.excluded, key=lambda e: e.path)),
        untracked_included=tuple(sorted(included)),
    )


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> tuple[str, int, bytes]:
    """(sha256 hex, size, first ``_SNIFF_BYTES`` bytes) in one read."""
    h = hashlib.sha256()
    size = 0
    head = b""
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            if len(head) < _SNIFF_BYTES:
                head += chunk[: _SNIFF_BYTES - len(head)]
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size, head


def _stat_key(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _capture_one(store: SnapshotStore, f: ScannedFile) -> tuple[str, int] | str:
    """Hash + store one file -> (sha256 hex, size); or, when its content holds a credential
    (a private key, an access token ...), what it holds - such a file is never stored.

    Retries once if the file changes underneath us, then fails loudly.
    """
    for attempt in range(2):
        try:
            before = _stat_key(f.source)
            sha_hex, size, head = _hash_file(f.source)
            if _stat_key(f.source) != before or size != before[0]:
                raise _FileChanged(f.path)
            secret = sniff_secret(head)
            if secret is not None:
                return secret
            store.ensure_blob(f.source, sha_hex, size)
            return sha_hex, size
        except _FileChanged:
            if attempt == 1:
                raise PreconditionFailed(
                    f"{f.path!r} changed while the snapshot was being captured; retry when writes have stopped",
                    details={"path": f.path},
                ) from None
        except FileNotFoundError as exc:
            raise PreconditionFailed(
                f"{f.path!r} disappeared while the snapshot was being captured", details={"path": f.path}
            ) from exc
    raise AssertionError("unreachable")


def snapshot_digest(files: Sequence[dict[str, Any]]) -> str:
    """Identity of a tree: paths + contents + executable bits only."""
    return canonical_digest({"schema": SNAPSHOT_SCHEMA, "files": list(files)})


def capture_snapshot(
    project_root: Path, options: CodeOptions, store: SnapshotStore, *, clock: Clock | None = None
) -> SnapshotResult:
    """Scan, hash and store the project; returns the ``CodeSnapshotRef`` for a RunSpec.

    Every file is read once for hashing and (if the blob is new) once more while copying;
    the copy is verified against the first hash, so a file edited mid-capture is either
    captured consistently or reported (``PreconditionFailed``) - never torn.
    """
    scan = scan_project(project_root, options)
    warnings = list(scan.warnings)
    excluded = list(scan.excluded)
    entries: list[dict[str, Any]] = []
    untracked = set(scan.untracked_included)
    for f in scan.files:
        captured = _capture_one(store, f)
        if isinstance(captured, str):
            excluded.append(ExcludedPath(f.path, "secret"))
            untracked.discard(f.path)
            msg = f"SECRET: excluded {f.path!r}: its content contains {captured} (never uploaded)"
            warnings.append(msg)
            log.warning("%s", msg)
            continue
        sha_hex, size = captured
        entries.append(
            {
                "path": f.path,
                "sha256": "sha256:" + sha_hex,
                "size": size,
                "mode": MODE_EXEC if f.executable else MODE_PLAIN,
            }
        )
    digest = snapshot_digest(entries)
    total = sum(int(e["size"]) for e in entries)
    now = (clock or SystemClock()).now()
    manifest: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "digest": digest,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": total,
        "git": scan.git.to_manifest() if scan.git else None,
        "untracked_included": sorted(untracked),
        "project_root": str(scan.project_root),
        "created_at": iso(now),
    }
    created = store.save_manifest(manifest)
    git = scan.git
    ref = CodeSnapshotRef(
        digest=digest,
        file_count=len(entries),
        total_bytes=total,
        git_commit=git.commit if git else None,
        git_branch=git.branch if git else None,
        git_dirty=git.dirty if git else False,
        dirty_patch_digest=git.dirty_patch_digest if git else None,
        untracked_included=sorted(untracked),
        project_root=str(scan.project_root),
    )
    excluded_sorted = sorted(excluded, key=lambda e: e.path)
    log.info(
        "snapshot %s: %d files, %d bytes (%s)",
        digest[:19],
        len(entries),
        total,
        "new" if created else "reused",
    )
    return SnapshotResult(
        ref=ref,
        warnings=warnings,
        excluded=[e.path for e in excluded_sorted],
        excluded_detail=excluded_sorted,
        reused=not created,
    )


# ---------------------------------------------------------------------------
# Publish (remote)
# ---------------------------------------------------------------------------

# Fixed, hpclab-authored snippets (run via Transport.run_script; all data arrives as args).
_RENAME_SCRIPT = r"""set -u
src=$1
dst=$2
mkdir -p "${dst%/*}" || exit 3
if [ -e "$dst" ]; then exit 10; fi
if command -v python3 >/dev/null 2>&1; then
python3 - "$src" "$dst" <<'HPCLAB_EOF'
import errno, os, sys
try:
    os.rename(sys.argv[1], sys.argv[2])
except OSError as exc:
    if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
        sys.exit(10)
    sys.stderr.write("rename failed: %s\n" % exc)
    sys.exit(4)
HPCLAB_EOF
exit $?
fi
if mv -T "$src" "$dst" 2>/dev/null; then exit 0; fi
if [ -e "$dst" ]; then exit 10; fi
echo "no atomic directory rename available on this host (need python3 or GNU mv -T)" >&2
exit 5
"""

# Staging trees and quarantined trees contain uploaded *copies*, never hardlinks into other
# snapshots, so making everything writable before deleting is safe here.
_REMOVE_TREE_SCRIPT = r"""set -u
d=$1
[ -e "$d" ] || exit 0
chmod -R u+w "$d" 2>/dev/null
rm -rf "$d"
"""

_SEAL_SCRIPT = r"""set -u
d=$1
chmod -R a-w "$d" && chmod u+w "$d"
"""

_FREEZE_TOP_SCRIPT = r"""set -u
chmod a-w "$1"
"""

_COUNT_SCRIPT = r"""set -u
find "$1" \( -type f -o -type l \) -print | wc -l
"""

_QUARANTINE_SCRIPT = r"""set -u
src=$1
dst=$2
if [ -e "$src/.hpclab-complete" ]; then exit 20; fi
mkdir -p "${dst%/*}" || exit 3
chmod u+w "$src" 2>/dev/null
mv "$src" "$dst"
"""

# A staging dir left behind by an interrupted publish is reused (see ``staging_dir``). If that
# publish got past the seal step, make the tree writable again and drop the files only a
# complete, verified upload may carry, so the upload refreshes it and everything is verified
# again. Staging dirs hold uploaded copies (never hardlinks), so ``chmod`` touches nothing else.
_PREPARE_STAGING_SCRIPT = r"""set -u
d=$1
shift
[ -e "$d" ] || exit 0
chmod -R u+w "$d" 2>/dev/null
for name in "$@"; do rm -f "$d/$name"; done
exit 0
"""

_VERIFY_BATCH_FILES = 256
_VERIFY_BATCH_BYTES = 256 * 1024 * 1024

# Bulk transfer deadline (``transfer_timeout_s``): a floor, plus the time the bytes take on a
# slow but working link (1 Mbit/s: home VPN, conference Wi-Fi), plus a per-file cost. A dead
# link is caught much earlier by ssh keep-alives; a slow one must be able to finish.
TRANSFER_BASE_TIMEOUT_S = 600.0
TRANSFER_MIN_RATE = 128 * 1024  # bytes per second
TRANSFER_PER_FILE_S = 0.02
_PUBLISH_LOCK_POLL_S = 0.2


def transfer_timeout_s(total_bytes: int, file_count: int = 1) -> float:
    """Deadline for one bulk transfer (upload of a snapshot, download of an artifact)."""
    return (
        TRANSFER_BASE_TIMEOUT_S
        + max(0, total_bytes) / TRANSFER_MIN_RATE
        + max(0, file_count) * TRANSFER_PER_FILE_S
    )


def staging_dir(root: str, sha_hex: str, store: SnapshotStore) -> str:
    """Where this machine uploads snapshot ``sha_hex`` before the atomic rename.

    Deterministic per (digest, store, host): a retry after an interrupted upload reuses it and
    rsync only sends the files that are not there yet, instead of starting from zero. Another
    machine uses its own dir, so its cleanup or seal step can never touch this upload.
    """
    return f"{root}/.staging/{sha_hex}.{store.publisher_id()}"


def _publish_lock_wait_s(ref: CodeSnapshotRef) -> float:
    """How long to wait for another local publisher of the same digest: its whole upload
    plus verification and the remote steps."""
    verify_s = 60.0 + ref.total_bytes / (40 * 1024 * 1024)
    return transfer_timeout_s(ref.total_bytes, ref.file_count) + verify_s + 600.0


@contextmanager
def _publish_lock(store: SnapshotStore, sha_hex: str, *, wait_s: float) -> Iterator[bool]:
    """Serialize publishes of one digest from one store (threads and processes alike): they
    share a staging dir, and the second would only upload the same bytes again. Yields True
    when it had to wait (the other publisher has probably finished the job).

    Raises retryable ``Unavailable`` after ``wait_s``.
    """
    store.locks_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(store.locks_dir / f"publish-{sha_hex}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        waited = False
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise Unavailable(
                        f"snapshot {sha_hex[:12]} is still being published by another worker on this "
                        "machine; retrying later",
                        details={"digest": "sha256:" + sha_hex, "waited_s": round(wait_s, 1)},
                    ) from None
                waited = True
                time.sleep(_PUBLISH_LOCK_POLL_S)
        try:
            yield waited
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _validate_remote_root(remote_root: str) -> str:
    root = remote_root.rstrip("/")
    if not root or not _SAFE_REMOTE_RE.match(root) or "/../" in f"{root}/" or "/./" in f"{root}/":
        raise InvalidArgument(
            f"remote_root {remote_root!r} must be an absolute path of [A-Za-z0-9_./@+,=:~-] characters",
            details={"remote_root": remote_root},
        )
    return root


def marker_bytes(ref: CodeSnapshotRef) -> bytes:
    """Deterministic marker content derived from the ref alone: every publisher writes
    byte-identical markers, so the marker's sha256 proves both completion and which snapshot
    a directory holds (and checking it needs no local store)."""
    payload = {
        "schema": MARKER_SCHEMA,
        "digest": ref.digest,
        "file_count": ref.file_count,
        "total_bytes": ref.total_bytes,
    }
    return (canonical_json(payload) + "\n").encode("utf-8")


def _script(
    transport: Transport, script: str, args: list[str], *, what: str, timeout: float = 120.0
) -> CommandResult:
    result = transport.run_script(script, args=args, timeout=timeout)
    if result.exit_code != 0:
        raise CommandFailed(
            f"{what} failed on {transport.cluster_id} (exit {result.exit_code}): {result.stderr.strip()[-500:]}",
            result,
            details={"step": what, "exit_code": result.exit_code, "stderr": result.stderr[-2000:]},
        )
    return result


def _remove_tree_best_effort(transport: Transport, path: str) -> None:
    try:
        res = transport.run_script(_REMOVE_TREE_SCRIPT, args=[path], timeout=300.0)
        if res.exit_code != 0:
            log.warning("could not remove %s: %s", path, res.stderr.strip()[-300:])
    except Exception as exc:  # noqa: BLE001 - cleanup must never mask the original failure
        log.warning("could not remove %s: %s", path, exc)


FinalState = Literal["complete", "absent", "incomplete"]


def _final_state(transport: Transport, final: str, expected_marker: bytes) -> FinalState:
    marker = f"{final}/{MARKER_NAME}"
    st = transport.stat([final, marker])
    if not st[final].exists:
        return "absent"
    if not st[marker].exists:
        return "incomplete"
    found = transport.sha256([marker]).get(marker)
    expected = "sha256:" + hashlib.sha256(expected_marker).hexdigest()
    if found != expected:
        raise PreconditionFailed(
            f"{final} carries a completion marker for different content; refusing to reuse it",
            details={"path": final, "expected_marker_sha256": expected, "found": found},
        )
    return "complete"


def _quarantine(transport: Transport, root: str, sha_hex: str, final: str) -> None:
    """Move an unmarked (never completed) final dir aside so a fresh publish can proceed."""
    dest = f"{root}/.staging/{sha_hex}.quarantine.{secrets.token_hex(4)}"
    res = transport.run_script(_QUARANTINE_SCRIPT, args=[final, dest], timeout=120.0)
    if res.exit_code == 20:
        return  # completed by a concurrent publisher in the meantime
    if res.exit_code != 0:
        raise CommandFailed(
            f"could not move incomplete snapshot {final} aside: {res.stderr.strip()[-300:]}", res
        )
    log.warning("moved incomplete snapshot dir %s to %s", final, dest)
    _remove_tree_best_effort(transport, dest)


def _verify_batches(
    entries: Sequence[dict[str, Any]], base: str
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    batch: list[tuple[str, dict[str, Any]]] = []
    batch_bytes = 0
    for e in entries:
        size = int(e["size"])
        if batch and (len(batch) >= _VERIFY_BATCH_FILES or batch_bytes + size > _VERIFY_BATCH_BYTES):
            yield batch
            batch, batch_bytes = [], 0
        batch.append((f"{base}/{e['path']}", e))
        batch_bytes += size
    if batch:
        yield batch


def _verify_remote_tree(transport: Transport, base: str, entries: Sequence[dict[str, Any]]) -> None:
    """Every manifest file must exist remotely with the manifest sha256, and nothing else."""
    mismatched: list[dict[str, Any]] = []
    missing: list[str] = []
    for batch in _verify_batches(entries, base):
        batch_bytes = sum(int(e["size"]) for _, e in batch)
        timeout = 60.0 + batch_bytes / (40 * 1024 * 1024)
        found = transport.sha256([p for p, _ in batch], timeout=timeout)
        for path, e in batch:
            got = found.get(path)
            if got is None:
                missing.append(e["path"])
            elif got != e["sha256"]:
                mismatched.append({"path": e["path"], "expected": e["sha256"], "found": got})
    count_out = _script(transport, _COUNT_SCRIPT, [base], what="count uploaded files").stdout.strip()
    try:
        remote_count: int | None = int(count_out.split()[-1])
    except (ValueError, IndexError):
        remote_count = None
    extra = remote_count is not None and remote_count != len(entries)
    if mismatched or missing or extra:
        raise SnapshotVerificationError(
            f"uploaded snapshot failed verification: {len(mismatched)} mismatched, {len(missing)} missing"
            + (f", {remote_count} files present but {len(entries)} expected" if extra else ""),
            details={
                "mismatched": mismatched[:_MAX_LISTED],
                "missing": missing[:_MAX_LISTED],
                "remote_file_count": remote_count,
                "expected_file_count": len(entries),
            },
        )


def _upload_and_rename(
    transport: Transport,
    store: SnapshotStore,
    ref: CodeSnapshotRef,
    manifest: dict[str, Any],
    *,
    root: str,
    final: str,
    marker: bytes,
) -> None:
    """Steps 2-4 of ``publish_snapshot`` (the caller holds the per-digest lock)."""
    sha_hex = digest_hex(ref.digest)
    tree = store.tree_dir(ref.digest)
    staging = staging_dir(root, sha_hex, store)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    _script(
        transport,
        _PREPARE_STAGING_SCRIPT,
        [staging, MARKER_NAME, REMOTE_MANIFEST_NAME],
        what="prepare snapshot staging dir",
    )
    renamed = False
    resumable = False
    try:
        try:
            transport.upload_dir(
                str(tree), staging, timeout=transfer_timeout_s(ref.total_bytes, ref.file_count)
            )
        except TransportError as exc:
            # The link failed mid-transfer: keep what arrived so the retry resumes. A remote
            # rsync error (quota, permissions) is not resumable and is cleaned up.
            resumable = not isinstance(exc, CommandFailed)
            if resumable:
                log.info("upload of snapshot %s interrupted; %s is kept for the retry", sha_hex[:12], staging)
            raise
        _verify_remote_tree(transport, staging, manifest["files"])
        transport.write_file(f"{staging}/{REMOTE_MANIFEST_NAME}", manifest_bytes, mode=0o444)
        _script(transport, _SEAL_SCRIPT, [staging], what="make snapshot read-only")
        transport.write_file(f"{staging}/{MARKER_NAME}", marker, mode=0o444)
        res = transport.run_script(_RENAME_SCRIPT, args=[staging, final], timeout=120.0)
        if res.exit_code == 0:
            renamed = True
        elif res.exit_code == 10:
            if _final_state(transport, final, marker) != "complete":
                raise PreconditionFailed(
                    f"{final} appeared during publish but is not complete", details={"path": final}
                )
            log.info("snapshot %s was published concurrently; using the existing copy", sha_hex[:12])
        else:
            raise CommandFailed(f"atomic rename to {final} failed: {res.stderr.strip()[-300:]}", res)
    finally:
        if not renamed and not resumable:
            _remove_tree_best_effort(transport, staging)


def publish_snapshot(
    transport: Transport, store: SnapshotStore, ref: CodeSnapshotRef, remote_root: str
) -> str:
    """Publish a captured snapshot to ``<remote_root>/snapshots/<hex>`` and return that path.

    Idempotent and safe under concurrent publishers:

    1. a final dir with a valid ``.hpclab-complete`` marker -> return immediately;
    2. otherwise (holding a per-digest lock in the local store, so this machine uploads a
       digest once at a time) upload to ``.staging/<hex>.<publisher>`` (``staging_dir``) with
       a deadline that grows with the snapshot's size, verify every file's sha256 on the
       cluster (bounded batches) and the file count, write ``.hpclab-manifest.json``, make the
       tree read-only (dirs stay traversable), write the marker LAST, then rename the staging
       dir into place atomically (``rename(2)``: python3 or GNU ``mv -T``);
    3. if another publisher won the rename race, its (complete) result is used;
    4. on failure the error propagates (``SnapshotVerificationError`` for corrupt uploads,
       ``TransportError`` subclasses for transport problems, ``CommandFailed`` when a remote
       step fails). An upload interrupted by the link (timeout, dropped connection, expired
       login) keeps its staging dir, so the retry resumes where it stopped; after any other
       failure the staging dir is removed. A staging dir is never trusted: whatever is in it
       is refreshed by the next upload and verified again.
    Only ``run_script``/``upload_dir``/``write_file``/``sha256``/``stat`` are used, so this works
    over SSH, locally and against the simulator.
    """
    root = _validate_remote_root(remote_root)
    sha_hex = digest_hex(ref.digest)
    final = f"{root}/snapshots/{sha_hex}"
    marker = marker_bytes(ref)

    state = _final_state(transport, final, marker)
    if state == "complete":
        return final
    manifest = store.manifest(ref.digest)
    if (manifest.get("digest"), manifest.get("file_count"), manifest.get("total_bytes")) != (
        ref.digest,
        ref.file_count,
        ref.total_bytes,
    ):
        raise InvalidArgument(
            "CodeSnapshotRef does not match the stored manifest",
            details={"digest": ref.digest, "manifest_file_count": manifest.get("file_count")},
        )
    if snapshot_digest(manifest["files"]) != ref.digest:
        # The remote marker is trusted forever: never publish content under a name it lacks.
        raise HpcLabError(
            f"local manifest for {ref.digest} does not match its digest (corrupt local store?); "
            "capture the snapshot again",
            details={"digest": ref.digest, "manifest_path": str(store.manifest_path(ref.digest))},
        )
    with _publish_lock(store, sha_hex, wait_s=_publish_lock_wait_s(ref)) as waited:
        if waited:  # the other local publisher has most likely finished the job
            state = _final_state(transport, final, marker)
            if state == "complete":
                return final
        if state == "incomplete":
            _quarantine(transport, root, sha_hex, final)
        _upload_and_rename(transport, store, ref, manifest, root=root, final=final, marker=marker)
    try:
        _script(transport, _FREEZE_TOP_SCRIPT, [final], what="freeze snapshot dir")
    except Exception as exc:  # noqa: BLE001 - content is complete and verified; this is cosmetic
        log.warning("could not make %s read-only: %s", final, exc)
    log.info("published snapshot %s to %s:%s", sha_hex[:12], transport.cluster_id, final)
    return final
