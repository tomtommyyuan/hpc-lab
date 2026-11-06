"""Artifacts: the declared outputs of an Attempt, indexed with an honest integrity state.

Only *declared* outputs (``RunSpec.outputs``) are indexed; hpclab is not a remote file
browser. At job end the wrapper writes ``outputs.json`` on the compute node (size, mtime and
sha256 hashed *there*, §6.4). Indexing combines that manifest with a fresh login-node
``stat`` (and, for directories, a bounded listing):

========== ================================================================================
integrity  meaning
========== ================================================================================
mutable    the attempt is still active (or abandoned): the file may still change
finalized  attempt terminal + exists + digest known + size/mtime (dir: member set and
           sizes) equal to the manifest
incomplete changed after the manifest was written, appeared after it, or a directory that
           could only be partially listed
missing    does not exist
unknown    could not be established: no manifest, transport down, digest unknown (e.g. a
           file too large to hash), symlink escaping the attempt dir
========== ================================================================================

``attempts.result_state`` is derived from the *required* outputs only: ``valid`` iff all are
finalized, ``invalid`` when one fails validation (kind mismatch, digest mismatch),
``incomplete`` when one is missing/incomplete, ``unknown`` otherwise; ``pending`` while the
attempt is active. The login node re-hashes a file only when the manifest lacks its digest
and it is at most ``policy.max_login_hash_bytes`` (bounded per index run).

Reads (``artifact_read``) are confined to the attempt directory (lexical check, then realpath
check: a symlink escaping it is ``unauthorized``) and never silently return new content: a
size/mtime/digest different from the indexed record is ``stale``, and so is every read (even a
partial one) of an output already found not to match its digest (for mutable outputs of a
running attempt the record is refreshed and the reply says so). Binary content is not
returned inline; it gets a ``download`` operation (outbox ``artifact.download``) into the
content-addressed cache ``paths.cache/sha256/<xx>/<hex>``, verified against the digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import posixpath
import re
import secrets
import shutil
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.adapters.base import AuthRequiredError, StatInfo, Transport, TransportError
from hpclab.api.envelope import (
    Coverage,
    SourceStatusValue,
    clamp_bytes,
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from hpclab.config import Paths
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import (
    HpcLabError,
    InvalidArgument,
    LimitExceeded,
    NotFound,
    PreconditionFailed,
    Stale,
    Unauthorized,
    Unsupported,
)
from hpclab.domain.ids import kind_of, new_id
from hpclab.domain.models import OutputDecl
from hpclab.profiles.schema import SitePolicy
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.services.snapshot import transfer_timeout_s
from hpclab.storage.db import Row, Tx, jload

log = logging.getLogger("hpclab.artifacts")

ARTIFACT_URI = "hpclab://artifacts/"
OPERATION_URI = "hpclab://operations/"
OUTPUTS_SCHEMA = "hpclab.outputs/v1"

MANIFEST_MAX_BYTES = 16 * 1024 * 1024  # outputs.json with 10k dir members is ~2 MB
MAX_MEMBER_ROWS = 1000  # member rows per declared directory (the dir row covers the rest)
MEMBER_CHANGES_IN_EVENT = 100  # member transitions listed in one artifact.members_indexed event
MAX_DIR_LISTING = 10_000  # members compared per directory (= the wrapper's MAX_MEMBERS)
MAX_REHASH_FILES = 256  # login-node sha256 per index run
DOWNLOAD_MAX_BYTES = 1024**3
JSON_VIEW_MAX_BYTES = 4 * 1024 * 1024
MAX_JSONL_RECORDS = 1000
MEMBERS_IN_READ = 50
MTIME_TOLERANCE_S = 1e-5  # st_mtime is compared exactly (µs ISO round trip), not "roughly"
SETTLE_WINDOW_S = 300.0  # NFS attribute caches: re-check mismatches seen right after the job ended
MAX_INDEX_RETRIES = 30
UNKNOWN_SWEEP_HORIZON_S = 14 * 86400.0  # re-sweep never-observed outputs of attempts this recent
BINARY_SNIFF_BYTES = 8192

Integrity = Literal["mutable", "incomplete", "finalized", "missing", "unknown"]
ReadView = Literal["head", "tail", "text", "json", "jsonl_tail"]
ListKind = Literal["file", "dir", "metrics", "checkpoint", "log", "prediction", "figure", "other"]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTIVE_RESULT = "pending"


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class ArtifactData(BaseModel):
    """One indexed output (or a member file of a declared output directory)."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_ref: str
    attempt_id: str
    run_id: str
    name: str  # declared output name, or "<output>/<relpath>" for a directory member
    output: str  # the declared output this row belongs to
    member_path: str | None = None  # relpath inside the declared directory (members only)
    role: str
    kind: str  # file | dir
    format: str | None = None
    remote_path: str
    size_bytes: int | None = None
    mtime: str | None = None  # ISO-8601 UTC with microseconds (exact st_mtime)
    digest: str | None = None
    digest_state: str  # pending | computed | skipped_too_large | mismatch | unknown
    digest_source: str | None = None  # compute_node | login_node | controller
    integrity: str  # mutable | incomplete | finalized | missing | unknown
    required: bool
    cached: bool = False
    observed_at: str
    version: int


class IndexedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    name: str
    kind: str
    role: str
    required: bool
    integrity: str
    digest_state: str
    digest_source: str | None = None
    size_bytes: int | None = None
    members: int | None = None
    issues: list[str] = Field(default_factory=list)


class IndexResult(BaseModel):
    """Outcome of one ``index_attempt`` run (also the ``artifacts.index`` handler's work)."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    phase: str
    result_state: str
    previous_result_state: str
    changed: bool
    manifest: str  # present | missing | invalid | unreadable | too_large | unavailable | skipped
    manifest_hash_state: str | None = None
    outputs: list[IndexedOutput] = Field(default_factory=list)
    member_rows: int = 0
    issues: list[str] = Field(default_factory=list)
    retry_after_s: float | None = None  # background hint: worth indexing again later
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


class ArtifactsListData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    phase: str
    scheduler_state: str
    result_state: str
    artifacts: list[ArtifactData]
    total: int
    last_indexed_at: str | None = None
    next_cursor: str | None = None
    coverage: Coverage
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


class ByteRange(BaseModel):
    """A byte window for ``artifact_read(view="text")``; ``length`` is capped by ``max_bytes``."""

    model_config = ConfigDict(extra="forbid")

    offset: int = Field(default=0, ge=0)
    length: int | None = Field(default=None, ge=1)


class DownloadRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str | None = None
    operation_ref: str | None = None
    state: str  # queued | running | blocked | succeeded | failed | cached
    poll_after_ms: int | None = None
    cached: bool = False


class ArtifactReadData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: ArtifactData
    view: str  # head | tail | text | json | jsonl_tail | members | metadata
    media_type: str
    text: str | None = None
    value: Any = None  # parsed document for view=json (when it fits max_bytes)
    value_summary: dict[str, Any] | None = None  # shape of a JSON document too large to return
    records: list[Any] | None = None  # view=jsonl_tail
    parse_errors: int = 0
    offset: int | None = None
    bytes_read: int = 0
    next_offset: int | None = None
    size: int | None = None
    eof: bool = False
    truncated: bool = False
    binary: bool = False
    digest_verified: bool = False
    source: Literal["remote", "cache", "ledger"] = "remote"
    refreshed: bool = False  # a mutable artifact's record was refreshed before reading
    members: list[ArtifactData] | None = None
    members_total: int | None = None
    download: DownloadRef | None = None
    note: str | None = None
    coverage: Coverage
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _within(path: str, root: str) -> bool:
    p = posixpath.normpath(path)
    r = posixpath.normpath(root)
    return p == r or p.startswith(r.rstrip("/") + "/")


def _iso_us(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _epoch(iso_text: str | None) -> float | None:
    dt = parse_iso(iso_text) if iso_text else None
    return dt.timestamp() if dt is not None else None


def _mtime_matches(recorded: float | None, current: float | None, *, coarse: bool = False) -> bool:
    if recorded is None or current is None:
        return False
    if coarse or float(recorded).is_integer():
        # degraded manifests (no python on the node) record whole seconds
        return int(recorded) == int(current)
    return abs(recorded - current) <= MTIME_TOLERANCE_S


def _norm_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not v.startswith("sha256:"):
        v = "sha256:" + v
    return v if _DIGEST_RE.match(v) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _timeout(ctx: ServiceContext) -> float:
    return max(5.0, min(30.0, float(ctx.settings.request_deadline_s)))


def _dot_dir(relpath: str) -> bool:
    """True when a directory component starts with '.' (checkpoint staging/replaced dirs)."""
    parts = relpath.split("/")[:-1]
    return any(p.startswith(".") for p in parts)


def artifact_ref(artifact_id: str) -> str:
    return f"{ARTIFACT_URI}{artifact_id}"


def cache_path_for(paths: Paths, digest: str) -> Path:
    """Content-addressed cache location of a ``sha256:<hex>`` digest."""
    hex_ = digest.split(":", 1)[1]
    return paths.cache / "sha256" / hex_[:2] / hex_


def _attempt_row(ctx: ServiceContext, attempt_id: str) -> Row:
    if kind_of(attempt_id) != "attempt":
        raise InvalidArgument(f"{attempt_id!r} is not an attempt id (att_...)")
    return ctx.db.get("attempts", "attempt_id", attempt_id)


def _attempt_mode(attempt: Row) -> str:
    phase = attempt["phase"]
    if phase == "terminal":
        return "terminal"
    if phase in ("submission_rejected", "cancelled_before_submit"):
        return "never_ran"
    if phase == "abandoned":
        return "abandoned"
    return "active"


def _policy(ctx: ServiceContext, cluster_id: str) -> SitePolicy:
    return ctx.clusters.get(cluster_id).profile.policy


# ---------------------------------------------------------------------------
# Declared outputs and the attempt layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptPaths:
    dir: str
    output_dir: str
    checkpoint_dir: str
    work: str
    outputs_manifest: str
    script: str
    token_file: str
    status_json: str
    env_json: str
    stdout: str
    stderr: str


def attempt_paths(attempt: Row) -> AttemptPaths:
    """Remote paths of an attempt (§6.1), derived from ``attempts.remote_dir``."""
    d = str(attempt["remote_dir"]).rstrip("/")
    try:
        from hpclab.services.jobscript import attempt_layout

        suffix = f"/attempts/{attempt['attempt_id']}"
        if d.endswith(suffix):
            lay = attempt_layout(d[: -len(suffix)], attempt["attempt_id"])
            if lay.dir == d:
                return AttemptPaths(
                    dir=lay.dir,
                    output_dir=lay.output_dir,
                    checkpoint_dir=lay.checkpoint_dir,
                    work=lay.work,
                    outputs_manifest=lay.outputs_manifest,
                    script=lay.script,
                    token_file=lay.token_file,
                    status_json=lay.status_json,
                    env_json=lay.env_json,
                    stdout=lay.stdout,
                    stderr=lay.stderr,
                )
    except HpcLabError:  # unusual characters: fall back to the documented layout
        pass
    return AttemptPaths(
        dir=d,
        output_dir=f"{d}/outputs",
        checkpoint_dir=f"{d}/checkpoints",
        work=f"{d}/work",
        outputs_manifest=f"{d}/outputs.json",
        script=f"{d}/job.sbatch",
        token_file=f"{d}/hpclab-plan.json",
        status_json=f"{d}/status.json",
        env_json=f"{d}/env.json",
        stdout=f"{d}/logs/stdout.log",
        stderr=f"{d}/logs/stderr.log",
    )


@dataclass(frozen=True)
class DeclaredOutput:
    name: str
    role: str
    kind: str
    format: str | None
    required: bool
    template: str
    path: str | None  # resolved absolute remote path; None when it cannot be resolved here
    note: str | None = None


_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def _resolve_template(template: str, attempt: Row, lay: AttemptPaths) -> tuple[str | None, str | None]:
    static = {
        "output_dir": lay.output_dir,
        "checkpoint_dir": lay.checkpoint_dir,
        "attempt_dir": lay.dir,
        "code_dir": lay.work,
        "run_id": str(attempt["run_id"]),
        "attempt_id": str(attempt["attempt_id"]),
    }
    if attempt.get("job_id"):
        static["job_id"] = str(attempt["job_id"])
    unresolved: list[str] = []

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in static:
            return static[key]
        unresolved.append(key)
        return m.group(0)

    path = _PLACEHOLDER_RE.sub(sub, template)
    if unresolved:
        return None, f"placeholder {{{unresolved[0]}}} is only known on the compute node"
    if not path.startswith("/"):
        path = f"{lay.work}/{path}"
    path = posixpath.normpath(path)
    if not _within(path, lay.dir):
        return None, "declared path resolves outside the attempt directory"
    return path, None


def declared_outputs(
    attempt: Row, run: Row, manifest_entries: dict[str, dict[str, Any]] | None = None
) -> list[DeclaredOutput]:
    """Declared outputs of the attempt's Run with resolved remote paths.

    ``RunSpec.outputs`` is authoritative. Only when the stored spec has no ``outputs`` key at
    all (hand-made ledgers) are the manifest's entries used as the declaration.
    """
    lay = attempt_paths(attempt)
    spec = jload(run.get("spec_json"), {}) or {}
    raw = spec.get("outputs") if isinstance(spec, dict) else None
    decls: list[DeclaredOutput] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                d = OutputDecl.model_validate(item)
            except ValueError:
                log.warning("run %s has an invalid output declaration: %r", run.get("run_id"), item)
                continue
            path, note = _resolve_template(d.path, attempt, lay)
            decls.append(DeclaredOutput(d.name, d.role, d.kind, d.format, d.required, d.path, path, note))
        return decls
    for name, entry in sorted((manifest_entries or {}).items()):
        p = entry.get("path")
        path = posixpath.normpath(p) if isinstance(p, str) and p.startswith("/") else None
        note = None
        if path is None or not _within(path, lay.dir):
            path, note = None, "manifest path is outside the attempt directory"
        decls.append(
            DeclaredOutput(
                name=name,
                role=str(entry.get("role") or "other"),
                kind="dir" if entry.get("kind") == "dir" else "file",
                format=None,
                required=bool(entry.get("required", True)),
                template=str(p),
                path=path,
                note=note,
            )
        )
    return decls


# ---------------------------------------------------------------------------
# Observation (I/O, outside any transaction)
# ---------------------------------------------------------------------------


@dataclass
class _Manifest:
    state: str  # present | missing | invalid | unreadable | too_large | unavailable | skipped
    hash_state: str | None = None
    degraded: bool = False
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class _Listing:
    files: dict[str, StatInfo]
    truncated: bool


@dataclass
class _Member:
    relpath: str
    size: int | None
    mtime: float | None
    digest: str | None
    digest_state: str
    digest_source: str | None
    integrity: str


@dataclass
class _Assessment:
    integrity: str
    exists: bool
    size: int | None = None
    mtime: float | None = None
    digest: str | None = None
    digest_state: str = "unknown"
    digest_source: str | None = None
    issues: list[str] = field(default_factory=list)
    invalid: bool = False
    members: list[_Member] | None = None
    member_total: int | None = None


def _read_manifest(transport: Transport, path: str, attempt_id: str, timeout: float) -> _Manifest:
    try:
        rr = transport.read_range(path, 0, MANIFEST_MAX_BYTES + 1, timeout=timeout)
    except FileNotFoundError:
        return _Manifest("missing")
    except TransportError:
        raise
    except OSError as exc:
        return _Manifest("unreadable", error=str(exc))
    if rr.size > MANIFEST_MAX_BYTES or len(rr.data) > MANIFEST_MAX_BYTES:
        return _Manifest("too_large", error=f"{rr.size} bytes > {MANIFEST_MAX_BYTES}")
    try:
        doc = json.loads(rr.data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _Manifest("invalid", error="outputs.json is not valid JSON")
    if not isinstance(doc, dict) or doc.get("schema") != OUTPUTS_SCHEMA:
        return _Manifest("invalid", error=f"outputs.json is not {OUTPUTS_SCHEMA}")
    if doc.get("attempt_id") != attempt_id:
        return _Manifest("invalid", error=f"outputs.json belongs to attempt {doc.get('attempt_id')!r}")
    files = doc.get("files")
    if not isinstance(files, list):
        return _Manifest("invalid", error="outputs.json has no files list")
    entries: dict[str, dict[str, Any]] = {}
    for f in files:
        if isinstance(f, dict) and isinstance(f.get("name"), str):
            entries.setdefault(f["name"], f)
    hs = doc.get("hash_state")
    return _Manifest(
        "present",
        hash_state=hs if isinstance(hs, str) else None,
        degraded=bool(doc.get("degraded")),
        entries=entries,
    )


def _listing(transport: Transport, path: str, timeout: float) -> _Listing:
    items = transport.list_dir(path, max_entries=MAX_DIR_LISTING + 1, timeout=timeout)
    files: dict[str, StatInfo] = {}
    root = path.rstrip("/")
    for st in items[:MAX_DIR_LISTING]:
        p = st.path
        rel = p[len(root) + 1 :] if p.startswith(root + "/") else posixpath.relpath(p, root)
        files[rel] = st
    return _Listing(files=files, truncated=len(items) > MAX_DIR_LISTING)


def _entry_matches(st: StatInfo, entry: dict[str, Any], coarse: bool) -> list[str]:
    """Why a terminal file does not match its manifest entry (empty list = it matches)."""
    if entry.get("exists") is not True:
        return ["appeared_after_manifest"]
    issues = []
    if _as_int(entry.get("size")) != st.size:
        issues.append("size_changed_after_manifest")
    if not _mtime_matches(_as_float(entry.get("mtime")), st.mtime, coarse=coarse):
        issues.append("mtime_changed_after_manifest")
    return issues


def _assess_file(
    *,
    active: bool,
    st: StatInfo | None,
    entry: dict[str, Any] | None,
    manifest: _Manifest,
    root_real: str,
    login_digest: str | None,
    max_login: int,
) -> _Assessment:
    if st is None:
        return _Assessment("unknown", exists=False, issues=["not reported by the stat call"])
    if not st.exists:
        return _Assessment("missing", exists=False)
    a = _Assessment("unknown", exists=True, size=st.size, mtime=st.mtime)
    if st.realpath and not _within(st.realpath, root_real):
        a.issues.append("symlink_escape: resolves outside the attempt directory")
        return a
    if st.kind == "dir":
        a.size = None
        a.issues.append("kind_mismatch: declared as a file but is a directory")
        a.invalid = True
        return a
    if active:
        a.integrity, a.digest_state = "mutable", "pending"
        return a
    if entry is None:
        a.issues.append(
            "not_in_manifest" if manifest.state == "present" else f"no_manifest ({manifest.state})"
        )
        return a
    if entry.get("error"):
        a.issues.append(f"manifest_error: {entry['error']}")
        a.invalid = True
        return a
    why = _entry_matches(st, entry, manifest.degraded)
    if why:
        a.integrity = "incomplete"
        a.issues.extend(why)
        return a
    sha = _norm_digest(entry.get("sha256"))
    if sha is not None:
        a.digest, a.digest_state, a.digest_source, a.integrity = sha, "computed", "compute_node", "finalized"
        return a
    if login_digest is not None:
        a.digest, a.digest_state, a.digest_source, a.integrity = (
            login_digest,
            "computed",
            "login_node",
            "finalized",
        )
        return a
    skipped = entry.get("skipped")
    if skipped == "too_large" or (st.size or 0) > max_login:
        a.digest_state = "skipped_too_large"
        a.issues.append("digest_unknown: too large to hash on the login node")
    else:
        a.digest_state = "pending"
        a.issues.append(f"digest_unknown: {skipped or 'not hashed by the wrapper'}")
    return a


def _tree_digest(members: list[tuple[str, str]]) -> str:
    """Same formula as the wrapper's helper (``describe_output``)."""
    tree = json.dumps([[rel, sha] for rel, sha in members], separators=(",", ":"))
    return "sha256:" + hashlib.sha256(tree.encode("utf-8")).hexdigest()


def _manifest_members(entry: dict[str, Any]) -> list[tuple[str, int | None, str | None, str | None]] | None:
    raw = entry.get("members")
    if not isinstance(raw, list):
        return None
    out: list[tuple[str, int | None, str | None, str | None]] = []
    for m in raw:
        if isinstance(m, dict) and isinstance(m.get("relpath"), str):
            skipped = m.get("skipped")
            out.append(
                (
                    m["relpath"],
                    _as_int(m.get("size")),
                    _norm_digest(m.get("sha256")),
                    skipped if isinstance(skipped, str) else None,
                )
            )
    return out


def _assess_dir(
    *,
    active: bool,
    st: StatInfo | None,
    entry: dict[str, Any] | None,
    manifest: _Manifest,
    root_real: str,
    listing: _Listing | None,
    login_digests: dict[str, str | None],
    dir_path: str,
    prior: dict[str, Row] | None = None,
) -> _Assessment:
    """``prior``: this directory's member rows from earlier index runs, by relpath. A member's
    earlier digest is reused only while *that member's* size and mtime are unchanged (the
    directory's own size/mtime would not notice an in-place rewrite)."""
    if st is None:
        return _Assessment("unknown", exists=False, issues=["not reported by the stat call"])
    if not st.exists:
        return _Assessment("missing", exists=False)
    a = _Assessment("unknown", exists=True, mtime=st.mtime)
    if st.realpath and not _within(st.realpath, root_real):
        a.issues.append("symlink_escape: resolves outside the attempt directory")
        return a
    if st.kind == "symlink":
        a.issues.append(
            "symlinked directory output: its members are not verified (declare the real directory)"
        )
        return a
    if st.kind != "dir":
        a.size = st.size
        a.issues.append("kind_mismatch: declared as a directory but is not one")
        a.invalid = True
        return a
    if listing is None:
        a.issues.append("listing_unavailable")
        if active:
            a.integrity, a.digest_state = "mutable", "pending"
        return a
    files = listing.files
    a.size = sum(s.size or 0 for s in files.values() if s.kind == "file")
    escapes = sorted(rp for rp, s in files.items() if s.realpath and not _within(s.realpath, root_real))
    if active:
        a.integrity, a.digest_state = "mutable", "pending"
        a.members = [
            _Member(rp, s.size, s.mtime, None, "pending", None, "unknown" if rp in escapes else "mutable")
            for rp, s in sorted(files.items())
        ]
        a.member_total = len(files)
        if listing.truncated:
            a.issues.append(f"partially_listed: more than {MAX_DIR_LISTING} files")
        return a
    if entry is None:
        a.issues.append(
            "not_in_manifest" if manifest.state == "present" else f"no_manifest ({manifest.state})"
        )
        a.members = [
            _Member(rp, s.size, s.mtime, None, "unknown", None, "unknown") for rp, s in sorted(files.items())
        ]
        a.member_total = len(files)
        return a
    if entry.get("exists") is not True:
        a.integrity = "incomplete"
        a.issues.append("appeared_after_manifest")
        return a
    if entry.get("error"):
        a.issues.append(f"manifest_error: {entry['error']}")
        a.invalid = True
        return a
    mm = _manifest_members(entry)
    if mm is None:
        a.issues.append("manifest_has_no_member_list")
        return a
    by_rel = {rel: (size, sha, skipped) for rel, size, sha, skipped in mm}
    missing = [rel for rel in by_rel if rel not in files]
    added = [rel for rel in files if rel not in by_rel]
    changed = [
        rel
        for rel, (size, _sha, _sk) in by_rel.items()
        if rel in files and files[rel].kind == "file" and files[rel].size != size
    ]
    members: list[_Member] = []
    known: list[tuple[str, str]] = []
    used_login = False
    for rel, size, sha, _skipped in mm:
        s = files.get(rel)
        if s is None:
            members.append(_Member(rel, size, None, sha, "computed" if sha else "unknown", None, "missing"))
            continue
        if rel in escapes:
            members.append(_Member(rel, s.size, s.mtime, None, "unknown", None, "unknown"))
            continue
        if rel in changed:
            members.append(
                _Member(rel, s.size, s.mtime, sha, "computed" if sha else "unknown", None, "incomplete")
            )
            continue
        digest, source = sha, "compute_node" if sha else None
        if digest is None:
            login = login_digests.get(f"{dir_path}/{rel}")
            if login is not None:
                digest, source, used_login = login, "login_node", True
            else:
                p = (prior or {}).get(rel)
                if p is not None and p["digest"] and p["digest_state"] == "computed" and _same_bytes(p, s):
                    digest, source = p["digest"], p["digest_source"]
                    used_login = used_login or source == "login_node"
        if digest is None:
            members.append(_Member(rel, s.size, s.mtime, None, "pending", None, "unknown"))
            continue
        known.append((rel, digest))
        members.append(_Member(rel, s.size, s.mtime, digest, "computed", source, "finalized"))
    a.members = members
    a.member_total = len(mm)
    partial = bool(entry.get("truncated")) or listing.truncated
    if escapes:
        a.issues.append(f"symlink_escape: {len(escapes)} member(s) resolve outside the attempt directory")
        return a
    if missing or added or changed or partial:
        a.integrity = "incomplete"
        if partial:
            a.issues.append("partially_listed")
        for label, items in (
            ("member_missing", missing),
            ("member_added", added),
            ("member_changed", changed),
        ):
            if items:
                a.issues.append(
                    f"{label}: {len(items)} ({', '.join(items[:3])}{', ...' if len(items) > 3 else ''})"
                )
        return a
    if len(known) != len(mm):
        a.digest_state = "pending"
        a.issues.append(f"digest_unknown: {len(mm) - len(known)} member(s) not hashed")
        return a
    tree = _norm_digest(entry.get("sha256"))
    if tree is not None and not used_login:
        a.digest, a.digest_source = tree, "compute_node"
    else:
        a.digest, a.digest_source = _tree_digest(known), "login_node" if used_login else "compute_node"
    a.digest_state, a.integrity = "computed", "finalized"
    return a


@dataclass
class _Observation:
    transport_ok: bool
    error: HpcLabError | None = None
    root_real: str = ""
    stats: dict[str, StatInfo] = field(default_factory=dict)
    manifest: _Manifest = field(default_factory=lambda: _Manifest("skipped"))
    listings: dict[str, _Listing | None] = field(default_factory=dict)
    login_digests: dict[str, str | None] = field(default_factory=dict)


def _existing_rows(ctx: ServiceContext, attempt_id: str) -> dict[str, Row]:
    return {r["name"]: r for r in ctx.db.all("SELECT * FROM artifacts WHERE attempt_id=?", (attempt_id,))}


def _rehash_candidates(
    decls: list[DeclaredOutput],
    obs: _Observation,
    existing: dict[str, Row],
    max_login: int,
) -> list[str]:
    """Paths the login node should hash: the manifest lacks the digest, the file still matches
    the manifest, it is small enough, and no earlier index already hashed the same bytes."""
    out: list[str] = []
    budget = max_login
    m = obs.manifest
    for d in decls:
        entry = m.entries.get(d.name)
        if d.path is None or entry is None or entry.get("exists") is not True:
            continue
        st = obs.stats.get(d.path)
        if st is None or not st.exists:
            continue
        if d.kind == "file" and st.kind != "dir":
            if _norm_digest(entry.get("sha256")) is not None or _entry_matches(st, entry, m.degraded):
                continue
            row = existing.get(d.name)
            if row is not None and row["digest"] and _same_bytes(row, st):
                continue
            size = st.size or 0
            if size <= max_login and size <= budget:
                out.append(d.path)
                budget -= size
        elif d.kind == "dir":
            listing = obs.listings.get(d.name)
            mm = _manifest_members(entry)
            if listing is None or mm is None:
                continue
            for rel, msize, sha, _sk in mm:
                s = listing.files.get(rel)
                if sha is not None or s is None or s.kind != "file" or s.size != msize:
                    continue
                row = existing.get(f"{d.name}/{rel}")
                if row is not None and row["digest"] and _same_bytes(row, s):
                    continue
                n = s.size or 0
                if n <= max_login and n <= budget:
                    out.append(f"{d.path}/{rel}")
                    budget -= n
        if len(out) >= MAX_REHASH_FILES:
            break
    return out[:MAX_REHASH_FILES]


def _same_bytes(row: Row, st: StatInfo) -> bool:
    return row["size_bytes"] == st.size and _mtime_matches(_epoch(row["mtime"]), st.mtime)


def _observe(
    ctx: ServiceContext,
    attempt: Row,
    decls: list[DeclaredOutput],
    mode: str,
    lay: AttemptPaths,
    policy: SitePolicy,
    existing: dict[str, Row],
    *,
    background: bool,
) -> _Observation:
    timeout = _timeout(ctx)
    try:
        transport = ctx.clusters.transport(attempt["cluster_id"])
        obs = _Observation(transport_ok=True)
        if mode == "terminal":
            obs.manifest = _read_manifest(transport, lay.outputs_manifest, attempt["attempt_id"], timeout)
        paths = [lay.dir] + [d.path for d in decls if d.path]
        obs.stats = transport.stat(paths, timeout=timeout)
        root = obs.stats.get(lay.dir)
        obs.root_real = (
            root.realpath if root is not None and root.exists and root.realpath else None
        ) or lay.dir
        for d in decls:
            st = obs.stats.get(d.path) if d.path else None
            if d.kind == "dir" and d.path and st is not None and st.exists and st.kind == "dir":
                try:
                    obs.listings[d.name] = _listing(transport, d.path, timeout)
                except (FileNotFoundError, NotADirectoryError):
                    obs.listings[d.name] = _Listing(files={}, truncated=False)
                except OSError:
                    obs.listings[d.name] = None
                except TransportError as exc:
                    if isinstance(exc, AuthRequiredError):
                        raise
                    obs.listings[d.name] = None  # e.g. list_dir unsupported without remote python
        if mode == "terminal" and obs.manifest.state == "present":
            candidates = _rehash_candidates(decls, obs, existing, policy.max_login_hash_bytes)
            if candidates:
                try:
                    obs.login_digests = transport.sha256(candidates, timeout=300.0 if background else timeout)
                except TransportError as exc:
                    if isinstance(exc, AuthRequiredError):
                        raise
                    log.info("login-node re-hash failed for %s: %s", attempt["attempt_id"], exc.message)
        return obs
    except TransportError as exc:
        return _Observation(transport_ok=False, error=exc)


# ---------------------------------------------------------------------------
# index_attempt
# ---------------------------------------------------------------------------


def _row_values(
    *,
    role: str,
    kind: str,
    fmt: str | None,
    remote_path: str,
    a: _Assessment | _Member,
    required: bool,
) -> dict[str, Any]:
    exists = a.integrity != "missing"
    return {
        "role": role,
        "kind": kind,
        "format": fmt,
        "remote_path": remote_path,
        "size_bytes": a.size if exists else None,
        "mtime": _iso_us(a.mtime) if exists else None,
        "digest": a.digest,
        "digest_state": a.digest_state,
        "digest_source": a.digest_source,
        "integrity": a.integrity,
        "required": 1 if required else 0,
    }


def _merge_known(values: dict[str, Any], row: Row | None) -> dict[str, Any]:
    """Keep facts an earlier index established that this run could not re-establish:
    a digest for identical size/mtime, and a discovered digest mismatch."""
    if row is None:
        return values
    same = (
        values["size_bytes"] is not None
        and row["size_bytes"] == values["size_bytes"]
        and _mtime_matches(_epoch(row["mtime"]), _epoch(values["mtime"]))
    )
    if not same:
        return values
    if row["digest_state"] == "mismatch" and (values["digest"] in (None, row["digest"])):
        return {**values, "digest": row["digest"], "digest_state": "mismatch", "integrity": "unknown"}
    if values["kind"] != "file":
        # a directory's size total and mtime survive an in-place same-size rewrite of a member,
        # so they do not vouch for an earlier tree digest
        return values
    if values["digest"] is None and row["digest"] and values["integrity"] in ("unknown", "finalized"):
        merged = {
            **values,
            "digest": row["digest"],
            "digest_state": "computed",
            "digest_source": row["digest_source"],
        }
        if values["integrity"] == "unknown" and values["digest_state"] in ("pending", "skipped_too_large"):
            merged["integrity"] = "finalized"
        return merged
    return values


def _upsert(
    tx: Tx, attempt: Row, name: str, values: dict[str, Any], existing: Row | None, observed_at: str
) -> tuple[str, bool]:
    """Insert or update one artifact row. Returns (artifact_id, changed)."""
    if existing is None:
        art_id = new_id("artifact")
        tx.insert(
            "artifacts",
            {
                "artifact_id": art_id,
                "attempt_id": attempt["attempt_id"],
                "run_id": attempt["run_id"],
                "name": name,
                **values,
                "cache_path": None,
                "observed_at": observed_at,
                "version": 1,
            },
        )
        return art_id, True
    changed = {k: v for k, v in values.items() if existing[k] != v}
    if changed:
        if "digest" in changed:
            changed["cache_path"] = None  # the cache is keyed by the old digest
        changed["observed_at"] = observed_at
        tx.update("artifacts", "artifact_id", existing["artifact_id"], changed)
        return existing["artifact_id"], True
    tx.update(
        "artifacts", "artifact_id", existing["artifact_id"], {"observed_at": observed_at}, bump_version=False
    )
    return existing["artifact_id"], False


def _derive_result_state(mode: str, required: list[_Assessment]) -> tuple[str, list[str]]:
    if mode == "active":
        return _ACTIVE_RESULT, []
    if mode == "abandoned":
        return "unknown", ["attempt was abandoned: a job may still be writing its outputs"]
    reasons: list[str] = []
    if any(a.invalid or a.digest_state == "mismatch" for a in required):
        return "invalid", reasons
    if any(a.integrity in ("missing", "incomplete") for a in required):
        return "incomplete", reasons
    if any(a.integrity != "finalized" for a in required):
        return "unknown", reasons
    return "valid", reasons


def index_attempt(ctx: ServiceContext, *, attempt_id: str, background: bool = False) -> IndexResult:
    """Index the declared outputs of an attempt and set ``attempts.result_state``.

    Remote I/O (manifest read, stat, bounded dir listing, bounded login-node re-hash) happens
    before the single write transaction. If the attempt's phase changed meanwhile the result
    is discarded (``retry_after_s`` set). Transport failures never become "missing": the
    outputs are ``unknown`` and a previously established result is kept.
    """
    attempt = _attempt_row(ctx, attempt_id)
    run = ctx.db.get("runs", "run_id", attempt["run_id"])
    mode = _attempt_mode(attempt)
    if mode == "never_ran":
        raise PreconditionFailed(
            f"attempt {attempt_id} never ran ({attempt['phase']}): there are no outputs to index",
            details={"attempt_id": attempt_id, "phase": attempt["phase"]},
        )
    policy = _policy(ctx, attempt["cluster_id"])
    lay = attempt_paths(attempt)
    existing = _existing_rows(ctx, attempt_id)
    decls = declared_outputs(attempt, run)
    spec = jload(run.get("spec_json"), {}) or {}
    needs_manifest_decls = not (isinstance(spec, dict) and isinstance(spec.get("outputs"), list))
    obs = _observe(ctx, attempt, decls, mode, lay, policy, existing, background=background)
    if needs_manifest_decls and obs.manifest.state == "present":
        decls = declared_outputs(attempt, run, obs.manifest.entries)
        obs = _observe(ctx, attempt, decls, mode, lay, policy, existing, background=background)
    observed_at = iso(ctx.clock.now()) or ""
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    warnings: list[str] = []
    issues: list[str] = []
    active = mode != "terminal"
    retry: float | None = None

    assessed: list[tuple[DeclaredOutput, _Assessment]] = []
    if obs.transport_ok:
        cov.add("stat", "ok", observed_at)
        if mode == "terminal":
            m = obs.manifest
            m_status: SourceStatusValue = "ok" if m.state == "present" else "partial"
            cov.add("outputs.json", m_status, observed_at, m.error or m.state)
            if m.state != "present":
                issues.append(f"outputs manifest {m.state}" + (f": {m.error}" if m.error else ""))
            elif m.hash_state == "pending":
                issues.append("the wrapper's hashing pass did not finish (hash_state=pending)")
        for d in decls:
            if d.path is None:
                a = _Assessment("unknown", exists=False, issues=[d.note or "path unresolvable"])
            elif d.kind == "dir":
                a = _assess_dir(
                    active=active,
                    st=obs.stats.get(d.path),
                    entry=obs.manifest.entries.get(d.name),
                    manifest=obs.manifest,
                    root_real=obs.root_real,
                    listing=obs.listings.get(d.name),
                    login_digests=obs.login_digests,
                    dir_path=d.path,
                    prior={
                        n[len(d.name) + 1 :]: r for n, r in existing.items() if n.startswith(f"{d.name}/")
                    },
                )
            else:
                a = _assess_file(
                    active=active,
                    st=obs.stats.get(d.path),
                    entry=obs.manifest.entries.get(d.name),
                    manifest=obs.manifest,
                    root_real=obs.root_real,
                    login_digest=obs.login_digests.get(d.path),
                    max_login=policy.max_login_hash_bytes,
                )
            assessed.append((d, a))
    else:
        err = obs.error
        c_status: SourceStatusValue = "unauthorized" if isinstance(err, AuthRequiredError) else "unavailable"
        cov.add("cluster", c_status, observed_at, err.message if err else None)
        issues.append(f"cluster unreachable: {err.message if err else 'unknown error'}")
        retry = 120.0 if isinstance(err, AuthRequiredError) else 60.0

    with ctx.db.tx(source="worker" if background else "api") as tx:
        current = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
        assert current is not None
        prev_result = str(current["result_state"])
        if current["phase"] != attempt["phase"] or current["restart_count"] != attempt["restart_count"]:
            return IndexResult(
                attempt_id=attempt_id,
                phase=current["phase"],
                result_state=prev_result,
                previous_result_state=prev_result,
                changed=False,
                manifest=obs.manifest.state,
                retry_after_s=2.0,
                coverage=cov,
                observed_at=observed_at,
                warnings=["the attempt changed while its outputs were being indexed; index again"],
            )
        rows = {r["name"]: r for r in tx.all("SELECT * FROM artifacts WHERE attempt_id=?", (attempt_id,))}
        outputs: list[IndexedOutput] = []
        member_rows = 0
        changed_any = False
        required: list[_Assessment] = []
        if not obs.transport_ok:
            for d in decls:
                row = rows.get(d.name)
                if row is None:
                    a = _Assessment("unknown", exists=False, issues=["cluster unreachable"])
                    vals = _row_values(
                        role=d.role,
                        kind=d.kind,
                        fmt=d.format,
                        remote_path=d.path or d.template,
                        a=a,
                        required=d.required,
                    )
                    art_id, _ = _upsert(tx, current, d.name, vals, None, observed_at)
                    tx.event(
                        "artifact",
                        art_id,
                        "artifact.indexed",
                        {"name": d.name, "integrity": "unknown", "reason": "cluster unreachable"},
                    )
                    changed_any = True
                    outputs.append(
                        IndexedOutput(
                            artifact_id=art_id,
                            name=d.name,
                            kind=d.kind,
                            role=d.role,
                            required=d.required,
                            integrity="unknown",
                            digest_state="unknown",
                            issues=a.issues,
                        )
                    )
                else:
                    outputs.append(_indexed_from_row(row, ["not re-checked: cluster unreachable"]))
            if mode == "terminal" and prev_result not in ("unchecked", _ACTIVE_RESULT):
                new_result = prev_result
            else:
                new_result = _ACTIVE_RESULT if mode == "active" else "unknown"
        else:
            for d, a in assessed:
                vals = _row_values(
                    role=d.role,
                    kind=d.kind,
                    fmt=d.format,
                    remote_path=d.path or d.template,
                    a=a,
                    required=d.required,
                )
                vals = _merge_known(vals, rows.get(d.name))
                if vals["digest_state"] == "mismatch":
                    a.invalid = True
                    a.issues.append("digest_mismatch: content differs from the recorded digest")
                a.integrity = vals["integrity"]
                a.digest_state = vals["digest_state"]
                prev = rows.get(d.name)
                art_id, changed = _upsert(tx, current, d.name, vals, prev, observed_at)
                if changed:
                    changed_any = True
                    tx.event(
                        "artifact",
                        art_id,
                        "artifact.indexed",
                        {
                            "name": d.name,
                            "integrity": vals["integrity"],
                            "previous_integrity": prev["integrity"] if prev else None,
                            "digest": vals["digest"],
                            "digest_state": vals["digest_state"],
                            "digest_source": vals["digest_source"],
                            "size_bytes": vals["size_bytes"],
                            "issues": a.issues[:10],
                        },
                        observed_at=observed_at,
                    )
                n_members = 0
                seen = a
                if d.kind == "dir" and a.members is None and vals["integrity"] == "missing":
                    seen = replace(a, members=[])  # the directory is gone, and so are its members
                if d.kind == "dir" and seen.members is not None and d.path:
                    n_members, m_changed = _upsert_members(tx, current, d, seen, rows, observed_at, art_id)
                    member_rows += n_members
                    changed_any = changed_any or m_changed
                if d.required:
                    required.append(a)
                outputs.append(
                    IndexedOutput(
                        artifact_id=art_id,
                        name=d.name,
                        kind=d.kind,
                        role=d.role,
                        required=d.required,
                        integrity=vals["integrity"],
                        digest_state=vals["digest_state"],
                        digest_source=vals["digest_source"],
                        size_bytes=vals["size_bytes"],
                        members=a.member_total,
                        issues=a.issues[:20],
                    )
                )
            new_result, reasons = _derive_result_state(mode, required)
            warnings.extend(reasons)
            if mode == "terminal" and not decls:
                warnings.append("the run declares no outputs: nothing to validate")
            elif mode == "terminal" and not required:
                warnings.append("no output is declared required: result_state is valid by definition")
            retry = _settle_retry(current, assessed, obs.manifest, ctx)
        if new_result != prev_result:
            changed_any = True
            _set_result_state(tx, current, new_result, prev_result, observed_at, reason="artifacts indexed")
        if changed_any:
            tx.event(
                "attempt",
                attempt_id,
                "attempt.artifacts_indexed",
                {
                    "result_state": new_result,
                    "outputs": {o.name: o.integrity for o in outputs},
                    "member_rows": member_rows,
                    "manifest": obs.manifest.state,
                },
                observed_at=observed_at,
            )
    return IndexResult(
        attempt_id=attempt_id,
        phase=attempt["phase"],
        result_state=new_result,
        previous_result_state=prev_result,
        changed=changed_any,
        manifest=obs.manifest.state if obs.transport_ok else "unavailable",
        manifest_hash_state=obs.manifest.hash_state,
        outputs=outputs,
        member_rows=member_rows,
        issues=issues,
        retry_after_s=retry,
        coverage=cov,
        observed_at=observed_at,
        warnings=warnings,
    )


def _indexed_from_row(row: Row, issues: list[str]) -> IndexedOutput:
    return IndexedOutput(
        artifact_id=row["artifact_id"],
        name=row["name"],
        kind=row["kind"],
        role=row["role"],
        required=bool(row["required"]),
        integrity=row["integrity"],
        digest_state=row["digest_state"],
        digest_source=row["digest_source"],
        size_bytes=row["size_bytes"],
        issues=issues,
    )


def _settle_retry(
    attempt: Row, assessed: list[tuple[DeclaredOutput, _Assessment]], manifest: _Manifest, ctx: ServiceContext
) -> float | None:
    """Right after a job ends a login node may still see cached attributes (NFS): mismatches
    and a missing manifest are re-checked a few times before they are believed."""
    if attempt["phase"] != "terminal":
        return None
    ended = parse_iso(attempt.get("terminal_at") or attempt.get("end_time"))
    if ended is None or (ctx.clock.now() - ended) > timedelta(seconds=SETTLE_WINDOW_S):
        return None
    suspicious = manifest.state == "missing" or any(
        a.integrity in ("incomplete", "missing") for d, a in assessed if d.required
    )
    return 30.0 if suspicious else None


def _set_result_state(tx: Tx, attempt: Row, new: str, prev: str, observed_at: str, *, reason: str) -> None:
    version = tx.update("attempts", "attempt_id", attempt["attempt_id"], {"result_state": new})
    tx.event(
        "attempt",
        attempt["attempt_id"],
        "attempt.result_state_changed",
        {"from": prev, "to": new, "reason": reason},
        entity_version=version,
        observed_at=observed_at,
    )
    _refresh_run_status(tx, attempt, new)
    if new == "valid" and attempt.get("scheduler_state") == "COMPLETED":
        run = tx.one("SELECT spec_json FROM runs WHERE run_id=?", (attempt["run_id"],))
        if run is not None and (jload(run["spec_json"], {}) or {}).get("evaluation"):
            # the declared default evaluator runs automatically (handler: services/evaluations.py)
            tx.enqueue(
                "evaluation.auto",
                target_id=attempt["attempt_id"],
                payload={"attempt_id": attempt["attempt_id"]},
                dedupe_key=f"evaluation.auto:{attempt['attempt_id']}",
            )
    wf = attempt.get("workflow_id")
    if wf:
        # the smoke gate / dispatch loop waits on result_state; wake it (handler: services/workflows.py)
        tx.enqueue(
            "workflow.dispatch",
            target_id=wf,
            payload={"workflow_id": wf},
            dedupe_key=f"workflow.dispatch:{wf}",
        )


def _refresh_run_status(tx: Tx, attempt: Row, result_state: str) -> None:
    """RunStatus distinguishes SUCCEEDED (COMPLETED + valid) from COMPLETED_UNVERIFIED."""
    if attempt.get("scheduler_state") != "COMPLETED" or attempt.get("phase") != "terminal":
        return
    run = tx.one("SELECT run_id, status, latest_attempt_id FROM runs WHERE run_id=?", (attempt["run_id"],))
    if run is None or run["latest_attempt_id"] != attempt["attempt_id"]:
        return
    if run["status"] not in ("succeeded", "completed_unverified"):
        return
    status = "succeeded" if result_state == "valid" else "completed_unverified"
    if status != run["status"]:
        v = tx.update("runs", "run_id", run["run_id"], {"status": status})
        tx.event(
            "run",
            run["run_id"],
            "run.status_changed",
            {"from": run["status"], "to": status, "reason": f"result_state={result_state}"},
            entity_version=v,
        )


def _upsert_members(
    tx: Tx,
    attempt: Row,
    d: DeclaredOutput,
    a: _Assessment,
    rows: dict[str, Row],
    observed_at: str,
    parent_id: str,
) -> tuple[int, bool]:
    assert a.members is not None and d.path is not None
    prefix = f"{d.name}/"
    existing = {n: r for n, r in rows.items() if n.startswith(prefix)}
    members = [m for m in a.members if not (d.role == "checkpoint" and _dot_dir(m.relpath))]
    room = max(0, MAX_MEMBER_ROWS - len(existing))
    changed_n = 0
    changes: list[dict[str, Any]] = []  # audit trail of member rows (bounded in the event)
    kept: set[str] = set()
    for m in sorted(members, key=lambda x: x.relpath):
        name = prefix + m.relpath
        prev = existing.get(name)
        if prev is None:
            if room <= 0:
                continue
            room -= 1
        kept.add(name)
        vals = _row_values(
            role=d.role, kind="file", fmt=None, remote_path=f"{d.path}/{m.relpath}", a=m, required=False
        )
        vals = _merge_known(vals, prev)
        _, changed = _upsert(tx, attempt, name, vals, prev, observed_at)
        if changed:
            changed_n += 1
            changes.append(
                {"member": m.relpath, "from": prev["integrity"] if prev else None, "to": vals["integrity"]}
            )
    for name, prev in existing.items():
        if name not in kept and prev["integrity"] != "missing":
            tx.update(
                "artifacts",
                "artifact_id",
                prev["artifact_id"],
                {"integrity": "missing", "size_bytes": None, "mtime": None, "observed_at": observed_at},
            )
            changed_n += 1
            changes.append({"member": name[len(prefix) :], "from": prev["integrity"], "to": "missing"})
    if changed_n:
        tx.event(
            "artifact",
            parent_id,
            "artifact.members_indexed",
            {
                "output": d.name,
                "rows_changed": changed_n,
                "members": a.member_total,
                "rows": len(kept),
                "changes": changes[:MEMBER_CHANGES_IN_EVENT],
                "changes_truncated": len(changes) > MEMBER_CHANGES_IN_EVENT,
            },
            observed_at=observed_at,
        )
    return len(kept), bool(changed_n)


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def artifact_from_row(row: Row) -> ArtifactData:
    name = str(row["name"])
    output, sep, member = name.partition("/")
    cache = row.get("cache_path")
    return ArtifactData(
        artifact_id=row["artifact_id"],
        artifact_ref=artifact_ref(row["artifact_id"]),
        attempt_id=row["attempt_id"],
        run_id=row["run_id"],
        name=name,
        output=output,
        member_path=member if sep else None,
        role=row["role"],
        kind=row["kind"],
        format=row["format"],
        remote_path=row["remote_path"],
        size_bytes=row["size_bytes"],
        mtime=row["mtime"],
        digest=row["digest"],
        digest_state=row["digest_state"],
        digest_source=row["digest_source"],
        integrity=row["integrity"],
        required=bool(row["required"]),
        cached=bool(cache) and Path(str(cache)).is_file(),
        observed_at=row["observed_at"],
        version=int(row["version"]),
    )


def _clean_relpath(rel: str) -> str:
    rel = rel.strip()
    if not rel or rel.startswith("/") or "\0" in rel:
        raise InvalidArgument(f"invalid member path {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise InvalidArgument(f"member path {rel!r} must be a plain relative path (no '.', '..' or '//')")
    return rel


def resolve_artifact(ctx: ServiceContext, ref: str) -> Row:
    """Artifact row for ``art_...``, ``hpclab://artifacts/art_...`` or ``...#<member relpath>``."""
    text = (ref or "").strip()
    if text.startswith(ARTIFACT_URI):
        text = text[len(ARTIFACT_URI) :]
    art_id, sep, member = text.partition("#")
    if kind_of(art_id) != "artifact":
        raise InvalidArgument(f"{ref!r} is not an artifact reference (art_... or {ARTIFACT_URI}<id>)")
    row = ctx.db.get("artifacts", "artifact_id", art_id)
    if not sep:
        return row
    rel = _clean_relpath(member)
    if row["kind"] != "dir":
        raise InvalidArgument(f"artifact {art_id} is a file; '#<member>' only applies to directory outputs")
    mrow = ctx.db.one(
        "SELECT * FROM artifacts WHERE attempt_id=? AND name=?", (row["attempt_id"], f"{row['name']}/{rel}")
    )
    if mrow is None:
        raise NotFound(
            f"member {rel!r} of {row['name']} is not indexed",
            details={
                "artifact_id": art_id,
                "member": rel,
                "hint": "list members with artifacts_list(include_members=true); at most "
                f"{MAX_MEMBER_ROWS} members per directory are indexed",
            },
        )
    return mrow


def artifact_get(ctx: ServiceContext, *, artifact_id: str) -> ArtifactData:
    """Metadata of one artifact (resource ``hpclab://artifacts/{artifact_id}``)."""
    return artifact_from_row(resolve_artifact(ctx, artifact_id))


def artifacts_list(
    ctx: ServiceContext,
    *,
    attempt_id: str,
    kind: ListKind | None = None,
    include_members: bool = False,
    refresh: bool = False,
    cursor: str | None = None,
    limit: int | None = None,
) -> ArtifactsListData:
    """Declared outputs of an attempt with integrity, digest state and size.

    ``kind`` filters by ``file``/``dir`` or by role (``metrics``, ``checkpoint``, ...).
    ``include_members`` adds indexed member files of declared directories. ``refresh``
    re-indexes first (bounded remote stat/listing) - useful while the attempt runs.
    """
    lim = clamp_limit(limit)
    attempt = _attempt_row(ctx, attempt_id)
    observed_at = iso(ctx.clock.now()) or ""
    warnings: list[str] = []
    if refresh and _attempt_mode(attempt) != "never_ran":
        res = index_attempt(ctx, attempt_id=attempt_id)
        cov = res.coverage.model_copy(deep=True)
        warnings.extend(res.warnings)
        warnings.extend(res.issues)
        attempt = _attempt_row(ctx, attempt_id)
    else:
        cov = Coverage()
        cov.add("ledger", "ok", observed_at)
    where = ["attempt_id = ?"]
    params: list[Any] = [attempt_id]
    if not include_members:
        where.append("instr(name, '/') = 0")
    if kind in ("file", "dir"):
        where.append("kind = ?")
        params.append(kind)
    elif kind is not None:
        where.append("role = ?")
        params.append(kind)
    base = " AND ".join(where)
    total_row = ctx.db.one(f"SELECT COUNT(*) AS n FROM artifacts WHERE {base}", params)
    total = int(total_row["n"]) if total_row else 0
    after = decode_cursor(cursor).get("after")
    page_sql = f"SELECT * FROM artifacts WHERE {base}"
    page_params = list(params)
    if isinstance(after, str):
        page_sql += " AND name > ?"
        page_params.append(after)
    rows = ctx.db.all(page_sql + " ORDER BY name LIMIT ?", [*page_params, lim + 1])
    next_cursor = encode_cursor({"after": rows[lim - 1]["name"]}) if len(rows) > lim else None
    rows = rows[:lim]
    stats = ctx.db.one(
        "SELECT COUNT(*) AS n, MAX(observed_at) AS last FROM artifacts WHERE attempt_id=?", (attempt_id,)
    )
    indexed = int(stats["n"]) if stats else 0
    mode = _attempt_mode(attempt)
    if indexed == 0:
        if mode == "never_ran":
            cov.notes.append(f"attempt never ran ({attempt['phase']}): it has no outputs")
        else:
            cov.complete = False
            cov.missing.append("artifacts_index")
            if mode == "terminal":
                cov.notes.append(
                    "outputs are indexed shortly after the job ends; retry soon or pass refresh=true"
                )
            else:
                cov.notes.append(
                    "attempt is still active; pass refresh=true to list the outputs written so far"
                )
    elif mode != "terminal":
        cov.notes.append(
            f"attempt is {attempt['phase']}: outputs are mutable; entries reflect the index at {stats['last'] if stats else '?'}"
        )
    return ArtifactsListData(
        attempt_id=attempt_id,
        run_id=attempt["run_id"],
        phase=attempt["phase"],
        scheduler_state=attempt["scheduler_state"],
        result_state=attempt["result_state"],
        artifacts=[artifact_from_row(r) for r in rows],
        total=total,
        last_indexed_at=stats["last"] if stats else None,
        next_cursor=next_cursor,
        coverage=cov,
        observed_at=observed_at,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# artifact_read
# ---------------------------------------------------------------------------


def _media_type(path: str, fmt: str | None) -> str:
    ext = posixpath.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return "application/x-ndjson"
    if ext == ".json":
        return "application/json"
    if ext in (".log", ".out", ".err", ".txt"):
        return "text/plain"
    if ext in (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz", ".npy", ".pkl"):
        return "application/octet-stream"
    guessed, _ = mimetypes.guess_type(path)
    return guessed or ("application/x-ndjson" if fmt == "hpclab.metrics/v1" else "application/octet-stream")


def _default_view(path: str, fmt: str | None) -> str:
    ext = posixpath.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return "jsonl_tail"
    if ext == ".json":
        return "json"
    if fmt == "hpclab.metrics/v1":
        return "jsonl_tail"
    return "head"


def _looks_binary(data: bytes) -> bool:
    sample = data[:BINARY_SNIFF_BYTES]
    if b"\x00" in sample:
        return True
    text = sample.decode("utf-8", errors="replace")
    bad = text.count("�")
    return bad > 8 and bad > len(text) // 20


def _utf8_trim_end(b: bytes) -> bytes:
    """Drop an incomplete UTF-8 sequence at the end (at most 3 bytes)."""
    n = len(b)
    for i in (1, 2, 3):
        if i > n:
            break
        c = b[n - i]
        if c & 0xC0 == 0x80:
            continue  # continuation byte: look further back
        if c & 0x80 == 0:
            return b
        need = 2 if c & 0xE0 == 0xC0 else 3 if c & 0xF0 == 0xE0 else 4 if c & 0xF8 == 0xF0 else 1
        return b if i >= need else b[: n - i]
    return b


def _utf8_trim_start(b: bytes) -> bytes:
    i = 0
    while i < min(3, len(b)) and b[i] & 0xC0 == 0x80:
        i += 1
    return b[i:]


def _changed(row: Row, st: StatInfo) -> bool:
    return row["size_bytes"] != st.size or not _mtime_matches(_epoch(row["mtime"]), st.mtime)


def _stale(row: Row, st: StatInfo | None, why: str) -> HpcLabError:
    return Stale(
        f"artifact {row['name']} changed since it was indexed ({why}); refusing to read the new content",
        details={
            "artifact_id": row["artifact_id"],
            "recorded": {"size_bytes": row["size_bytes"], "mtime": row["mtime"], "digest": row["digest"]},
            "current": {"size_bytes": st.size if st else None, "mtime": _iso_us(st.mtime) if st else None},
            "hint": f"re-index with artifacts_list(attempt_id={row['attempt_id']!r}, refresh=true) and read the new reference",
        },
    )


def _record_mismatch(ctx: ServiceContext, row: Row, actual: str, *, source: str) -> None:
    """Content with the recorded size/mtime hashes differently: the attested bytes are gone."""
    with ctx.db.tx(source="api") as tx:
        cur = tx.one("SELECT * FROM artifacts WHERE artifact_id=?", (row["artifact_id"],))
        if cur is None:
            return
        if cur["digest_state"] != "mismatch":
            v = tx.update(
                "artifacts",
                "artifact_id",
                cur["artifact_id"],
                {"digest_state": "mismatch", "integrity": "unknown"},
            )
            tx.event(
                "artifact",
                cur["artifact_id"],
                "artifact.digest_mismatch",
                {"recorded": cur["digest"], "actual": actual, "detected_by": source},
                entity_version=v,
            )
        required = bool(cur["required"])
        if "/" in cur["name"]:
            parent = tx.one(
                "SELECT required FROM artifacts WHERE attempt_id=? AND name=?",
                (cur["attempt_id"], cur["name"].split("/", 1)[0]),
            )
            required = required or bool(parent and parent["required"])
        attempt = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (cur["attempt_id"],))
        if (
            required
            and attempt is not None
            and attempt["result_state"] != "invalid"
            and attempt["phase"] == "terminal"
        ):
            _set_result_state(
                tx,
                attempt,
                "invalid",
                attempt["result_state"],
                tx.now,
                reason=f"digest mismatch in {cur['name']}",
            )


def _refresh_mutable(ctx: ServiceContext, row: Row, st: StatInfo, observed_at: str) -> Row:
    with ctx.db.tx(source="api") as tx:
        cur = tx.one("SELECT integrity FROM artifacts WHERE artifact_id=?", (row["artifact_id"],))
        if cur is None or cur["integrity"] != "mutable":
            # the job ended and the index recorded the final state while this read ran; that
            # record must not be overwritten with this read's (older) observation
            raise Stale(
                f"artifact {row['name']} was re-indexed while it was being read; read it again",
                details={
                    "artifact_id": row["artifact_id"],
                    "integrity": cur["integrity"] if cur else None,
                    "hint": "read the artifact again (its record is now final)",
                },
            )
        v = tx.update(
            "artifacts",
            "artifact_id",
            row["artifact_id"],
            {"size_bytes": st.size, "mtime": _iso_us(st.mtime), "observed_at": observed_at},
        )
        tx.event(
            "artifact",
            row["artifact_id"],
            "artifact.observed",
            {"size_bytes": st.size, "integrity": "mutable"},
            entity_version=v,
        )
        new = tx.one("SELECT * FROM artifacts WHERE artifact_id=?", (row["artifact_id"],))
    assert new is not None
    return new


def _cache_file(ctx: ServiceContext, row: Row) -> Path | None:
    digest = _norm_digest(row["digest"])
    if digest is None or row["digest_state"] == "mismatch":
        return None
    p = cache_path_for(ctx.paths, digest)
    return p if p.is_file() else None


def _verified_cache_file(ctx: ServiceContext, row: Row) -> Path | None:
    """The cache entry, re-hashed now: it is served as ``digest_verified`` while the cluster is
    unreachable, so local damage since the download must not pass. A bad entry is dropped."""
    cached = _cache_file(ctx, row)
    if cached is None:
        return None
    try:
        if _sha256_file(cached) == _norm_digest(row["digest"]):
            return cached
        log.warning("cache entry %s does not match its digest; removing it", cached)
        cached.unlink()
    except OSError:
        log.warning("cache entry %s is unreadable", cached, exc_info=True)
    return None


def ensure_download(ctx: ServiceContext, row: Row) -> tuple[DownloadRef | None, str | None]:
    """Reuse or create the ``download`` operation for an artifact (outbox ``artifact.download``)."""
    if row["kind"] != "file":
        return None, "directories are not downloaded as a whole; read or download member files"
    digest = _norm_digest(row["digest"])
    if digest is None or row["digest_state"] == "mismatch":
        return (
            None,
            "no verified digest is recorded, so the download could not be verified; index the attempt first",
        )
    if (row["size_bytes"] or 0) > DOWNLOAD_MAX_BYTES:
        return (
            None,
            f"artifact is larger than the {DOWNLOAD_MAX_BYTES} byte download limit; copy it with rsync yourself",
        )
    if _cache_file(ctx, row) is not None:
        return DownloadRef(state="cached", cached=True), None
    with ctx.db.tx(source="api") as tx:
        op = tx.one(
            "SELECT * FROM operations WHERE kind='download' AND target_type='artifact' AND target_id=? "
            "AND state IN ('queued','running','blocked') ORDER BY created_at DESC LIMIT 1",
            (row["artifact_id"],),
        )
        if op is None:
            op_id = tx.create_operation(
                "download", target_type="artifact", target_id=row["artifact_id"], stage="queued"
            )
            tx.enqueue(
                "artifact.download",
                operation_id=op_id,
                target_id=row["artifact_id"],
                payload={"artifact_id": row["artifact_id"], "digest": digest},
            )
            op = tx.one("SELECT * FROM operations WHERE operation_id=?", (op_id,))
        assert op is not None
    return (
        DownloadRef(
            operation_id=op["operation_id"],
            operation_ref=f"{OPERATION_URI}{op['operation_id']}",
            state=op["state"],
            poll_after_ms=int(op["poll_after_ms"]),
        ),
        None,
    )


def artifact_read(
    ctx: ServiceContext,
    *,
    artifact_ref: str,
    range: ByteRange | None = None,
    view: ReadView | None = None,
    max_bytes: int | None = None,
) -> ArtifactReadData:
    """Bounded, integrity-checked view of an artifact.

    Views: ``head``/``tail``/``text`` (``range`` = byte window), ``json`` (parsed document),
    ``jsonl_tail`` (last records). The path must stay inside the attempt directory (symlink
    escape -> ``unauthorized``). Content that changed since indexing -> ``stale``. Binary
    content is not returned; a download operation is queued instead.
    """
    window = range
    limit = clamp_bytes(max_bytes)
    if window is not None and view not in (None, "text", "head"):
        raise InvalidArgument("range can only be used with view='text' (or 'head')")
    row = resolve_artifact(ctx, artifact_ref)
    attempt = _attempt_row(ctx, row["attempt_id"])
    observed_at = iso(ctx.clock.now()) or ""
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    media = _media_type(row["remote_path"], row["format"])
    if row["kind"] == "dir":
        members = ctx.db.all(
            "SELECT * FROM artifacts WHERE attempt_id=? AND name LIKE ? ESCAPE '\\' ORDER BY name LIMIT ?",
            (row["attempt_id"], _like_prefix(row["name"]), MEMBERS_IN_READ),
        )
        n = ctx.db.one(
            "SELECT COUNT(*) AS n FROM artifacts WHERE attempt_id=? AND name LIKE ? ESCAPE '\\'",
            (row["attempt_id"], _like_prefix(row["name"])),
        )
        return ArtifactReadData(
            artifact=artifact_from_row(row),
            view="members",
            media_type="inode/directory",
            members=[artifact_from_row(m) for m in members],
            members_total=int(n["n"]) if n else 0,
            source="ledger",
            note=f"directory output: read a member with {artifact_ref_for_member(row, '<relpath>')}",
            coverage=cov,
            observed_at=observed_at,
        )
    lay = attempt_paths(attempt)
    path = str(row["remote_path"])
    if not _within(path, lay.dir):
        raise Unauthorized(
            f"artifact {row['artifact_id']} points outside its attempt directory",
            details={"artifact_id": row["artifact_id"]},
        )
    if row["digest_state"] == "mismatch":
        # the bytes on the cluster are known not to be the attested ones; a partial read could
        # not re-verify them, so no view of them is returned
        raise Stale(
            f"artifact {row['name']} no longer matches its recorded digest; refusing to read it",
            details={
                "artifact_id": row["artifact_id"],
                "digest_state": "mismatch",
                "recorded": {"size_bytes": row["size_bytes"], "mtime": row["mtime"], "digest": row["digest"]},
                "hint": "the attested content is gone from the cluster; recover or re-run the attempt",
            },
        )
    transport = ctx.clusters.transport(attempt["cluster_id"])
    timeout = _timeout(ctx)
    try:
        stats = transport.stat([path, lay.dir], timeout=timeout)
    except TransportError as exc:
        cached = _verified_cache_file(ctx, row)
        if cached is None:
            raise
        cov.add("cluster", "unavailable", observed_at, exc.message)
        return _read_local(ctx, row, cached, view, window, limit, media, cov, observed_at)
    cov.add("stat", "ok", observed_at)
    st = stats.get(path)
    root = stats.get(lay.dir)
    if st is None or not st.exists:
        if row["integrity"] == "missing":
            raise NotFound(
                f"artifact {row['name']} does not exist on the cluster",
                details={"artifact_id": row["artifact_id"]},
            )
        raise _stale(row, None, "it no longer exists")
    root_real = (root.realpath if root is not None and root.exists and root.realpath else None) or lay.dir
    if st.realpath and not _within(st.realpath, root_real):
        raise Unauthorized(
            f"artifact {row['name']} resolves outside the attempt directory (symlink escape); refusing to read it",
            details={"artifact_id": row["artifact_id"], "allowed_root": lay.dir},
        )
    if st.kind == "dir":
        raise _stale(row, st, "it is a directory now")
    mutable = row["integrity"] == "mutable" and _attempt_mode(attempt) != "terminal"
    refreshed = False
    if _changed(row, st):
        if not mutable:
            raise _stale(row, st, "size or mtime differ")
        row = _refresh_mutable(ctx, row, st, observed_at)
        refreshed = True
    chosen = "text" if window is not None else (view or _default_view(path, row["format"]))
    out = _read_remote(transport, row, st, chosen, window, limit, timeout)
    if not mutable and (out.rr_size != st.size or not _mtime_matches(out.rr_mtime, st.mtime)):
        raise _stale(row, st, "it changed while being read")
    warnings: list[str] = []
    digest_verified = False
    if out.whole and row["digest"] and not mutable:
        actual = "sha256:" + hashlib.sha256(out.raw).hexdigest()
        if actual != _norm_digest(row["digest"]):
            _record_mismatch(ctx, row, actual, source="artifact_read")
            raise Stale(
                f"artifact {row['name']} content does not match its recorded digest; refusing to return it",
                details={"artifact_id": row["artifact_id"], "recorded": row["digest"], "actual": actual},
            )
        digest_verified = True
    data = _finish_read(ctx, row, out, chosen, limit, media, cov, observed_at)
    data.digest_verified = digest_verified
    data.refreshed = refreshed
    if mutable:
        data.note = (
            data.note + " " if data.note else ""
        ) + "attempt is still active: this output may still change."
    data.warnings.extend(warnings)
    return data


def artifact_ref_for_member(row: Row, rel: str) -> str:
    return f"{ARTIFACT_URI}{row['artifact_id']}#{rel}"


def _like_prefix(name: str) -> str:
    return name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"


@dataclass
class _Read:
    raw: bytes
    offset: int
    rr_size: int
    rr_mtime: float | None
    eof: bool
    whole: bool
    fallback_note: str | None = None
    view: str = "head"


def _read_remote(
    transport: Transport,
    row: Row,
    st: StatInfo,
    view: str,
    window: ByteRange | None,
    limit: int,
    timeout: float,
) -> _Read:
    path = str(row["remote_path"])
    size = st.size or 0
    note = None
    if view == "json":
        if size <= JSON_VIEW_MAX_BYTES:
            rr = transport.read_range(path, 0, JSON_VIEW_MAX_BYTES + 1, timeout=timeout)
            return _Read(
                rr.data, rr.offset, rr.size, rr.mtime, rr.eof, rr.offset == 0 and rr.eof, view="json"
            )
        view, note = (
            "head",
            f"view=json needs the whole document (at most {JSON_VIEW_MAX_BYTES} bytes); showing the head",
        )
    if view in ("tail", "jsonl_tail"):
        rr = transport.read_tail(path, limit, timeout=timeout)
        return _Read(rr.data, rr.offset, rr.size, rr.mtime, rr.eof, rr.offset == 0 and rr.eof, note, view)
    offset = window.offset if window is not None else 0
    length = min(window.length or limit, limit) if window is not None else limit
    rr = transport.read_range(path, offset, length, timeout=timeout)
    return _Read(rr.data, rr.offset, rr.size, rr.mtime, rr.eof, rr.offset == 0 and rr.eof, note, view)


def _read_local(
    ctx: ServiceContext,
    row: Row,
    cached: Path,
    view: str | None,
    window: ByteRange | None,
    limit: int,
    media: str,
    cov: Coverage,
    observed_at: str,
) -> ArtifactReadData:
    """Serve a read from the verified content-addressed cache while the cluster is unreachable."""
    raw_all = cached.read_bytes() if cached.stat().st_size <= JSON_VIEW_MAX_BYTES else None
    size = cached.stat().st_size
    chosen = "text" if window is not None else (view or _default_view(str(row["remote_path"]), row["format"]))
    with open(cached, "rb") as fh:
        if chosen in ("tail", "jsonl_tail"):
            start = max(0, size - limit)
            fh.seek(start)
            raw = fh.read(limit)
            off = start
        elif chosen == "json" and raw_all is not None:
            raw, off = raw_all, 0
        else:
            if chosen == "json":
                chosen = "head"
            off = window.offset if window is not None else 0
            fh.seek(min(off, size))
            raw = fh.read(min(window.length or limit, limit) if window is not None else limit)
    out = _Read(raw, off, size, None, off + len(raw) >= size, off == 0 and len(raw) >= size, view=chosen)
    data = _finish_read(ctx, row, out, chosen, limit, media, cov, observed_at)
    data.source = "cache"
    data.digest_verified = True  # the cache only holds files verified against the digest
    data.note = (
        data.note + " " if data.note else ""
    ) + "served from the local verified cache (cluster unreachable)."
    return data


def _finish_read(
    ctx: ServiceContext,
    row: Row,
    out: _Read,
    chosen: str,
    limit: int,
    media: str,
    cov: Coverage,
    observed_at: str,
) -> ArtifactReadData:
    art = artifact_from_row(row)
    note = out.fallback_note
    base: dict[str, Any] = {
        "artifact": art,
        "media_type": media,
        "offset": out.offset,
        "size": out.rr_size,
        "coverage": cov,
        "observed_at": observed_at,
    }
    view = out.view if out.view != chosen and out.fallback_note else chosen
    if _looks_binary(out.raw):
        dl, why = ensure_download(ctx, row)
        return ArtifactReadData(
            **base,
            view="metadata",
            binary=True,
            eof=out.eof,
            download=dl,
            note=why
            or "binary content is not returned inline; a verified download to the local cache was queued",
        )
    if view == "json":
        try:
            value = json.loads(out.raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            text_raw = _utf8_trim_end(out.raw[:limit])
            return ArtifactReadData(
                **base,
                note=f"not valid JSON ({exc.__class__.__name__}); showing the head",
                view="head",
                text=text_raw.decode("utf-8", errors="replace"),
                bytes_read=len(text_raw),
                next_offset=len(text_raw) if len(text_raw) < out.rr_size else None,
                eof=len(text_raw) >= out.rr_size,
                truncated=len(text_raw) < out.rr_size,
            )
        compact = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
        if len(compact.encode("utf-8")) <= limit:
            return ArtifactReadData(
                **base, note=note, view="json", value=value, bytes_read=len(out.raw), eof=True
            )
        return ArtifactReadData(
            **base,
            note=f"document larger than max_bytes={limit}: only its shape is returned; use view='text' with a range",
            view="json",
            value_summary=_json_summary(value, budget=max(256, limit // 2)),
            bytes_read=len(out.raw),
            eof=True,
            truncated=True,
        )
    if view == "jsonl_tail":
        raw = out.raw
        partial_first = out.offset > 0
        lines = raw.split(b"\n")
        if partial_first and lines:
            lines = lines[1:]
        records: list[Any] = []
        errors = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s.decode("utf-8")))
            except (UnicodeDecodeError, ValueError):
                errors += 1
        dropped = max(0, len(records) - MAX_JSONL_RECORDS)
        return ArtifactReadData(
            **base,
            note=note,
            view="jsonl_tail",
            records=records[-MAX_JSONL_RECORDS:],
            parse_errors=errors,
            bytes_read=len(raw),
            eof=True,
            truncated=partial_first or dropped > 0,
        )
    if view == "tail":
        raw = _utf8_trim_start(out.raw) if out.offset > 0 else out.raw
        return ArtifactReadData(
            **base,
            note=note,
            view="tail",
            text=raw.decode("utf-8", errors="replace"),
            bytes_read=len(raw),
            eof=True,
            truncated=out.offset > 0,
        )
    raw = out.raw if out.eof else _utf8_trim_end(out.raw)
    nxt = out.offset + len(raw)
    return ArtifactReadData(
        **base,
        note=note,
        view=view,
        text=raw.decode("utf-8", errors="replace"),
        bytes_read=len(raw),
        next_offset=None if out.eof and nxt >= out.rr_size else nxt,
        eof=out.eof and nxt >= out.rr_size,
        truncated=not (out.eof and nxt >= out.rr_size) or out.offset > 0,
    )


def _json_summary(value: Any, budget: int = 8192) -> dict[str, Any]:
    """Shape of a document too large to return, itself bounded: at most 100 keys, each cut to
    128 characters, listed until ``budget`` characters are used."""
    if isinstance(value, dict):
        keys: list[str] = []
        used = 0
        cut = False
        for k in list(value)[:100]:
            shown = k if len(k) <= 128 else k[:128] + "..."
            cut = cut or shown is not k
            used += len(shown) + 4
            if used > budget:
                break
            keys.append(shown)
        return {
            "type": "object",
            "keys": keys,
            "key_count": len(value),
            "keys_truncated": cut or len(keys) < len(value),
        }
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return {"type": type(value).__name__}


# ---------------------------------------------------------------------------
# Verified download into the content-addressed cache
# ---------------------------------------------------------------------------


def download_to_cache(
    ctx: ServiceContext,
    row: Row,
    *,
    max_bytes: int = DOWNLOAD_MAX_BYTES,
    timeout: float | None = None,
) -> Path:
    """Copy a file artifact into ``paths.cache/sha256/<xx>/<hex>`` and verify its digest.

    Preconditions: a recorded ``sha256`` digest, size known and <= ``max_bytes``, the remote
    file still inside the attempt dir with the recorded size/mtime (else ``stale``). The
    downloaded bytes must hash to the digest; otherwise nothing is cached, the mismatch is
    recorded on the artifact, and ``stale`` is raised. The default deadline grows with the
    file's size (``snapshot.transfer_timeout_s``), so a large file on a slow link can finish.
    """
    if row["kind"] != "file":
        raise Unsupported(f"artifact {row['name']} is a directory; download its member files")
    digest = _norm_digest(row["digest"])
    if digest is None or row["digest_state"] == "mismatch":
        raise PreconditionFailed(
            f"artifact {row['name']} has no verified digest; it cannot be downloaded verifiably",
            details={"artifact_id": row["artifact_id"], "digest_state": row["digest_state"]},
        )
    size = row["size_bytes"]
    if size is None:
        raise PreconditionFailed(
            f"artifact {row['name']} has no recorded size", details={"artifact_id": row["artifact_id"]}
        )
    if size > max_bytes:
        raise LimitExceeded(
            f"artifact {row['name']} is {size} bytes; the limit is {max_bytes}",
            details={"artifact_id": row["artifact_id"], "size_bytes": size, "limit": max_bytes},
        )
    final = cache_path_for(ctx.paths, digest)
    if final.is_file():
        if final.stat().st_size == size and _sha256_file(final) == digest:
            _record_cache(ctx, row, final)
            return final
        final.unlink()
    attempt = _attempt_row(ctx, row["attempt_id"])
    lay = attempt_paths(attempt)
    path = str(row["remote_path"])
    if not _within(path, lay.dir):
        raise Unauthorized(f"artifact {row['artifact_id']} points outside its attempt directory")
    transport = ctx.clusters.transport(attempt["cluster_id"])
    stats = transport.stat([path, lay.dir], timeout=_timeout(ctx))
    st = stats.get(path)
    root = stats.get(lay.dir)
    if st is None or not st.exists:
        raise _stale(row, None, "it no longer exists")
    root_real = (root.realpath if root is not None and root.exists and root.realpath else None) or lay.dir
    if st.realpath and not _within(st.realpath, root_real):
        raise Unauthorized(
            f"artifact {row['name']} resolves outside the attempt directory (symlink escape)",
            details={"artifact_id": row["artifact_id"]},
        )
    if st.kind == "dir" or _changed(row, st):
        raise _stale(row, st, "size or mtime differ")
    final.parent.mkdir(parents=True, exist_ok=True)
    # A private dir per try: rsync writes into a temp file *next to* its target, and when it
    # is killed at the deadline nobody else would ever remove that file.
    work = final.parent / f".partial-{secrets.token_hex(6)}"
    work.mkdir(mode=0o700)
    tmp = work / "download"
    try:
        try:
            transport.download(path, str(tmp), timeout=timeout or transfer_timeout_s(size))
        except FileNotFoundError as exc:
            raise _stale(row, None, "it no longer exists") from exc
        actual = _sha256_file(tmp)
        if actual != digest:
            after = transport.stat([path], timeout=_timeout(ctx)).get(path)
            if after is not None and after.exists and not _changed(row, after):
                _record_mismatch(ctx, row, actual, source="download")
                raise Stale(
                    f"artifact {row['name']} content does not match its recorded digest",
                    details={"artifact_id": row["artifact_id"], "recorded": digest, "actual": actual},
                )
            raise _stale(row, after, "it changed during the download")
        os.chmod(tmp, 0o444)
        os.replace(tmp, final)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _record_cache(ctx, row, final)
    return final


def _record_cache(ctx: ServiceContext, row: Row, final: Path) -> None:
    if row.get("cache_path") == str(final):
        return
    with ctx.db.tx(source="worker") as tx:
        cur = tx.one("SELECT cache_path, digest FROM artifacts WHERE artifact_id=?", (row["artifact_id"],))
        if cur is None or cur["cache_path"] == str(final) or cur["digest"] != row["digest"]:
            return
        v = tx.update("artifacts", "artifact_id", row["artifact_id"], {"cache_path": str(final)})
        tx.event(
            "artifact", row["artifact_id"], "artifact.cached", {"digest": row["digest"]}, entity_version=v
        )


def read_verified_bytes(
    ctx: ServiceContext, row: Row, *, max_bytes: int, timeout: float | None = None
) -> bytes:
    """Bytes of a file artifact, fetched through the verified cache (used by evaluations)."""
    path = download_to_cache(
        ctx, row, max_bytes=max_bytes, timeout=timeout or max(30.0, ctx.settings.request_deadline_s)
    )
    data = path.read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != _norm_digest(row["digest"]):
        raise Stale(
            "cached artifact bytes changed on the local disk", details={"artifact_id": row["artifact_id"]}
        )
    return data


# ---------------------------------------------------------------------------
# Outbox handlers and periodic sweep
# ---------------------------------------------------------------------------


def handle_index(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    attempt_id = str(item.payload.get("attempt_id") or item.target_id or "")
    try:
        res = index_attempt(ctx, attempt_id=attempt_id, background=True)
    except NotFound as exc:
        return HandlerResult.done(exc.message)
    except PreconditionFailed as exc:
        return HandlerResult.done(exc.message)
    if res.retry_after_s is not None:
        waiting_for_auth = any(s.status == "unauthorized" for s in res.coverage.sources)
        if waiting_for_auth or item.tries < MAX_INDEX_RETRIES:
            return HandlerResult.retry(res.retry_after_s, "; ".join(res.issues)[:500] or None)
    return HandlerResult.done(f"result_state={res.result_state}")


def _set_op(ctx: ServiceContext, op_id: str | None, **kw: Any) -> None:
    if not op_id:
        return
    with ctx.db.tx(source="worker") as tx:
        op = tx.one("SELECT state FROM operations WHERE operation_id=?", (op_id,))
        if op is None or op["state"] in ("succeeded", "failed"):
            return
        tx.set_operation(op_id, **kw)


def handle_download(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    art_id = str(item.payload.get("artifact_id") or item.target_id or "")
    op_id = item.operation_id
    try:
        row = ctx.db.get("artifacts", "artifact_id", art_id)
    except NotFound as exc:
        _set_op(ctx, op_id, state="failed", error=exc.to_dict())
        return HandlerResult.done("artifact not found")
    _set_op(ctx, op_id, state="running", stage="downloading")
    try:
        path = download_to_cache(ctx, row)
    except AuthRequiredError as exc:
        cluster = ctx.db.one("SELECT cluster_id FROM attempts WHERE attempt_id=?", (row["attempt_id"],))
        err = exc.to_dict()
        err["details"] = {
            **err.get("details", {}),
            "remediation": f"hpclab auth {cluster['cluster_id'] if cluster else ''}".strip(),
        }
        _set_op(ctx, op_id, state="blocked", stage="waiting_for_auth", error=err)
        return HandlerResult.retry(120.0, "auth_required")
    except TransportError:
        raise  # retried with backoff by the runner; the operation stays running
    except HpcLabError as exc:
        _set_op(ctx, op_id, state="failed", error=exc.to_dict())
        return HandlerResult.done(exc.message)
    _set_op(
        ctx,
        op_id,
        state="succeeded",
        stage="cached",
        result={
            "artifact_id": art_id,
            "digest": row["digest"],
            "size_bytes": row["size_bytes"],
            "cache_path": str(path),
        },
    )
    return HandlerResult.done()


def sweep_unindexed(ctx: ServiceContext, *, limit: int = 100) -> int:
    """Enqueue ``artifacts.index`` for terminal attempts whose outputs were never checked.

    That is: never indexed (e.g. a finalize step that ran before this module was deployed),
    or left ``unknown`` because the cluster could not be reached at all (no output was ever
    observed) after the index item's own retries ran out. Without the latter a long outage
    right after a smoke run would stall its workflow's gate forever. ``unknown`` results that
    *were* observed (no manifest, too large to hash) are final and not swept again.
    Ledger-only; returns how many were enqueued.
    """
    now = ctx.clock.now()
    cutoff = iso(now - timedelta(seconds=60))
    horizon = iso(now - timedelta(seconds=UNKNOWN_SWEEP_HORIZON_S))
    rows = ctx.db.all(
        "SELECT a.attempt_id FROM attempts a WHERE a.phase='terminal' "
        "AND COALESCE(a.terminal_at, a.updated_at) <= ? AND ("
        " a.result_state IN ('unchecked','pending')"
        " OR (a.result_state='unknown' AND COALESCE(a.terminal_at, a.updated_at) >= ?"
        "     AND NOT EXISTS (SELECT 1 FROM artifacts r WHERE r.attempt_id=a.attempt_id AND instr(r.name,'/')=0"
        "     AND (r.integrity != 'unknown' OR r.size_bytes IS NOT NULL OR r.mtime IS NOT NULL)))"
        ") AND NOT EXISTS (SELECT 1 FROM outbox o "
        "WHERE o.kind='artifacts.index' AND o.target_id=a.attempt_id AND o.state IN ('pending','leased')) "
        "ORDER BY a.attempt_id LIMIT ?",
        (cutoff, horizon, limit),
    )
    if not rows:
        return 0
    with ctx.db.tx(source="worker") as tx:
        for r in rows:
            tx.enqueue(
                "artifacts.index",
                target_id=r["attempt_id"],
                payload={"attempt_id": r["attempt_id"]},
                dedupe_key=f"artifacts.index:{r['attempt_id']}",
            )
    return len(rows)


def outbox_handlers() -> dict[str, Handler]:
    return {"artifacts.index": handle_index, "artifact.download": handle_download}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    def run(ctx: ServiceContext) -> None:
        sweep_unindexed(ctx)

    return [PeriodicTask(name="artifacts.sweep", interval_s=300.0, fn=run, jitter_s=30.0, run_at_start=False)]
