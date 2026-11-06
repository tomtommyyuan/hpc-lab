"""Local provenance vault and ledger backup.

Scratch filesystems get purged and ``sacct`` is not an archive (slurmdbd purges, MinJobAge),
so when an attempt ends hpclab keeps small, durable local copies of the evidence a reported
number depends on, under ``<HPCLAB_HOME>/vault/<attempt_id>/``:

=================  ==============================================================
kind               content
=================  ==============================================================
sacct_raw          the scheduler's accounting record for the job (parsed record
                   JSON; the adapter does not expose the raw sacct line)
script             ``job.sbatch`` (the exact wrapper that ran)
plan               ``hpclab-plan.json`` (run spec, execution plan, token)
status / env       wrapper ``status.json`` / ``env.json``
outputs_manifest   ``outputs.json`` (compute-node digests)
log_head/log_tail  first 64 KiB / last 256 KiB of stdout and stderr
metrics_file       declared ``role: metrics`` files up to 50 MB each
=================  ==============================================================

Each item is a ``vault_items`` row with the sha256 and size of the *stored* bytes and a
``truncated`` flag (logs: the full source size is in the capture event). Re-capturing only
rewrites items whose bytes changed. ``backup`` copies the ledger with SQLite's online backup
API (consistent while the controller keeps writing) plus the vault and the snapshot
manifests, and writes a ``BACKUP.json`` with the digest of every copied file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import secrets
import shutil
import sqlite3
import stat as stat_mod
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hpclab.adapters.base import AuthRequiredError, StatInfo, TransportError
from hpclab.api.envelope import Coverage, SourceStatusValue, clamp_limit, decode_cursor, encode_cursor
from hpclab.config import Paths
from hpclab.domain.clock import iso
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument, NotFound, NotReady
from hpclab.domain.hashing import digest
from hpclab.domain.ids import kind_of, ulid
from hpclab.domain.models import JobObservation
from hpclab.services.artifacts import attempt_paths, declared_outputs
from hpclab.services.context import ServiceContext
from hpclab.services.observe import stored_accounting
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.storage.db import Row

log = logging.getLogger("hpclab.vault")

BACKUP_SCHEMA = "hpclab.backup/v1"
SMALL_FILE_MAX = 16 * 1024 * 1024
LOG_HEAD_BYTES = 64 * 1024
LOG_TAIL_BYTES = 256 * 1024
METRICS_FILE_MAX = 50 * 1000 * 1000
MAX_METRICS_FILES = 32
MAX_CAPTURE_RETRIES = 12
EXPECTED_KINDS = ("sacct_raw", "script", "plan", "status", "env", "outputs_manifest", "log_head")
_NO_JOB_KINDS = ("script", "plan")  # what exists for attempts whose job never ran
_ACTIVE_PHASES = (
    "staging",
    "submission_in_flight",
    "submission_uncertain",
    "duplicate_detected",
    "job_identified",
    "observing",
)


def vault_root(paths: Paths) -> Path:
    return paths.home / "vault"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class VaultItemData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    attempt_id: str
    kind: str
    source_path: str | None = None
    local_path: str
    digest: str | None = None
    size_bytes: int | None = None  # bytes stored locally
    truncated: bool = False
    captured_at: str
    present: bool = True  # the local file exists


class VaultCaptureData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    captured: list[VaultItemData] = Field(default_factory=list)  # new or changed items
    unchanged: int = 0
    source_sizes: dict[str, int] = Field(default_factory=dict)  # full size of each captured source
    missing: list[str] = Field(default_factory=list)  # sources that do not exist (yet)
    unavailable: list[str] = Field(default_factory=list)  # sources that could not be read
    retry_after_s: float | None = None
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


class VaultAttemptStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    phase: str
    items: int
    kinds: list[str]
    missing_kinds: list[str]
    stored_bytes: int
    truncated_items: int
    missing_local_files: int
    last_captured_at: str | None = None


class VaultStatusData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault_dir: str
    experiment_id: str | None = None
    attempts_in_scope: int
    fully_captured: int
    partially_captured: int
    not_captured: int
    total_items: int
    total_bytes: int
    attempts: list[VaultAttemptStatus]
    next_cursor: str | None = None
    coverage: Coverage
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


class BackupData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backup_dir: str
    db_path: str
    db_bytes: int
    db_sha256: str
    integrity_check: str
    vault_files: int
    vault_bytes: int
    snapshot_manifests: int
    manifest_path: str
    manifest_digest: str
    created_at: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


@dataclass
class _Captured:
    kind: str
    source_path: str
    rel_local: str
    data: bytes | None = None
    tmp_file: Path | None = None
    source_size: int | None = None
    truncated: bool = False
    expected_digest: str | None = None  # metrics files: the digest the artifact index attests


@dataclass
class _Reader:
    """Remote reads that degrade per source: missing files are recorded, transport failures
    mark the source unavailable, and an auth failure stops all further remote reads."""

    timeout: float
    items: list[_Captured] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    errors: list[TransportError] = field(default_factory=list)
    stopped: bool = False

    def fail(self, label: str, exc: TransportError) -> None:
        self.unavailable.append(f"{label}: {exc.message}")
        self.errors.append(exc)
        if isinstance(exc, AuthRequiredError):
            self.stopped = True


def _safe_name(name: str) -> str:
    base = posixpath.basename(name)
    return base if base not in ("", ".", "..") else "file"


def _read_sources(ctx: ServiceContext, attempt: Row, reader: _Reader, tmp_dir: Path) -> None:
    lay = attempt_paths(attempt)
    try:
        transport = ctx.clusters.transport(attempt["cluster_id"])
    except TransportError as exc:
        reader.fail("cluster", exc)
        return
    small = [
        ("script", lay.script, "job.sbatch"),
        ("plan", lay.token_file, "hpclab-plan.json"),
        ("status", lay.status_json, "status.json"),
        ("env", lay.env_json, "env.json"),
        ("outputs_manifest", lay.outputs_manifest, "outputs.json"),
    ]
    for kind, path, local in small:
        if reader.stopped:
            reader.unavailable.append(kind)
            continue
        try:
            rr = transport.read_range(path, 0, SMALL_FILE_MAX, timeout=reader.timeout)
        except FileNotFoundError:
            reader.missing.append(kind)
            continue
        except TransportError as exc:
            reader.fail(kind, exc)
            continue
        except OSError as exc:
            reader.unavailable.append(f"{kind}: {exc}")
            continue
        reader.items.append(
            _Captured(kind, path, local, data=rr.data, source_size=rr.size, truncated=rr.size > len(rr.data))
        )
    for stream, path in (("stdout", lay.stdout), ("stderr", lay.stderr)):
        if reader.stopped:
            reader.unavailable.append(f"log:{stream}")
            continue
        try:
            head = transport.read_range(path, 0, LOG_HEAD_BYTES, timeout=reader.timeout)
            reader.items.append(
                _Captured(
                    "log_head",
                    path,
                    f"logs/{stream}.head",
                    data=head.data,
                    source_size=head.size,
                    truncated=head.size > len(head.data),
                )
            )
            if head.size > LOG_HEAD_BYTES:
                tail = transport.read_tail(path, LOG_TAIL_BYTES, timeout=reader.timeout)
                reader.items.append(
                    _Captured(
                        "log_tail",
                        path,
                        f"logs/{stream}.tail",
                        data=tail.data,
                        source_size=tail.size,
                        truncated=tail.offset > 0,
                    )
                )
        except FileNotFoundError:
            reader.missing.append(f"log:{stream}")
        except TransportError as exc:
            reader.fail(f"log:{stream}", exc)
        except OSError as exc:
            reader.unavailable.append(f"log:{stream}: {exc}")
    _read_metrics(ctx, attempt, transport, reader, tmp_dir)


def _metrics_sources(ctx: ServiceContext, attempt: Row) -> list[tuple[str, str, str | None]]:
    """(output name, remote path, recorded digest) of declared metrics files."""
    rows = ctx.db.all(
        "SELECT name, remote_path, digest FROM artifacts WHERE attempt_id=? AND role='metrics' AND kind='file' "
        "AND instr(name,'/')=0 ORDER BY name",
        (attempt["attempt_id"],),
    )
    if rows:
        return [(r["name"], r["remote_path"], r["digest"]) for r in rows]
    run = ctx.db.get("runs", "run_id", attempt["run_id"])
    return [
        (d.name, d.path, None)
        for d in declared_outputs(attempt, run)
        if d.role == "metrics" and d.kind == "file" and d.path
    ]


def _inside(path: str, root: str) -> bool:
    p = posixpath.normpath(path)
    r = posixpath.normpath(root)
    return p == r or p.startswith(r.rstrip("/") + "/")


def _resolve_metrics_source(
    transport: Any, path: str, attempt_dir: str, timeout: float
) -> tuple[StatInfo | None, str]:
    """(stat of ``path``, what it resolves to): ``file``, ``dir``, ``other``, ``missing``,
    ``outside`` (not within the attempt directory once symlinks are followed) or
    ``unresolved`` (a symlink whose target could not be resolved).

    The confinement ``artifact_read`` and ``download_to_cache`` apply: the transports follow
    symlinks when they copy (rsync -L; a symlink to a directory is mirrored whole), so copying
    anything that resolves outside the attempt directory's realpath would pull arbitrary cluster
    files (credentials, a whole home directory) into the vault and every backup of it.
    """
    if not _inside(path, attempt_dir):
        return None, "outside"
    stats = transport.stat([path, attempt_dir], timeout=timeout)
    st, root = stats.get(path), stats.get(attempt_dir)
    if st is None or not st.exists:
        return st, "missing"
    root_real = (root.realpath if root is not None and root.exists and root.realpath else None) or attempt_dir
    if st.realpath and not _inside(st.realpath, root_real):
        return st, "outside"
    if st.kind != "symlink":
        return st, st.kind
    if not st.realpath:
        return st, "unresolved"
    target = transport.stat([st.realpath], timeout=timeout).get(st.realpath)
    if target is None or not target.exists:
        return st, "missing"
    return st, target.kind if target.kind in ("file", "dir") else "other"


def _is_regular_file(path: Path) -> bool:
    try:
        return stat_mod.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _discard(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _read_metrics(ctx: ServiceContext, attempt: Row, transport: Any, reader: _Reader, tmp_dir: Path) -> None:
    sources = _metrics_sources(ctx, attempt)[:MAX_METRICS_FILES]
    attempt_dir = attempt_paths(attempt).dir
    for name, path, recorded in sources:
        label = f"metrics_file:{name}"
        if reader.stopped:
            reader.unavailable.append(label)
            continue
        try:
            st, resolved = _resolve_metrics_source(transport, path, attempt_dir, reader.timeout)
            if resolved in ("missing", "dir"):
                reader.missing.append(label)
                continue
            if resolved != "file":
                why = {
                    "outside": "resolves outside the attempt directory (symlink escape)",
                    "unresolved": "is a symlink whose target could not be resolved",
                }.get(resolved, "is not a regular file")
                log.warning("vault: not copying %s: it %s", path, why)
                reader.unavailable.append(f"{label}: {path} {why}; not copied")
                continue
            size = st.size if st is not None else None
            if (size or 0) > METRICS_FILE_MAX:
                reader.unavailable.append(
                    f"{label}: {size} bytes is over the {METRICS_FILE_MAX} byte vault limit"
                )
                continue
            tmp = tmp_dir / f"{_safe_name(name)}.{secrets.token_hex(4)}"
            transport.download(path, str(tmp), timeout=max(60.0, reader.timeout))
            if not _is_regular_file(tmp):
                # the path changed between the stat and the copy: never keep what came instead
                _discard(tmp)
                reader.unavailable.append(f"{label}: the downloaded copy is not a regular file; discarded")
                continue
        except FileNotFoundError:
            reader.missing.append(label)
            continue
        except TransportError as exc:
            reader.fail(label, exc)
            continue
        except OSError as exc:
            reader.unavailable.append(f"{label}: {exc}")
            continue
        item = _Captured(
            "metrics_file",
            path,
            f"metrics/{_safe_name(name)}/{_safe_name(path)}",
            tmp_file=tmp,
            source_size=tmp.stat().st_size,
            expected_digest=recorded,
        )
        reader.items.append(item)
        if recorded:
            actual = _sha256_path(tmp)
            if actual != recorded:
                log.warning("vault copy of %s differs from its indexed digest", path)
                reader.notes.append(
                    f"{label}: the copied bytes ({actual}) differ from the indexed digest {recorded}"
                )


def _read_accounting(ctx: ServiceContext, attempt: Row, reader: _Reader) -> None:
    job_id = attempt.get("job_id")
    if not job_id:
        return
    stored = stored_accounting(ctx.db, attempt)
    if stored is not None and stored.job_id == str(job_id):
        # the observer / finalize already applied (or cached) the final record: keep that one
        # instead of spending a sacct call per attempt against the cluster's rate limit
        _keep_accounting(
            reader,
            str(job_id),
            source=stored.source,
            observed_at=stored.observed_at,
            complete=True,
            records=[stored],
            note=(
                "the final accounting record hpclab stored when the attempt ended (parsed sacct -X; "
                "no new sacct query); the raw sacct line is not exposed by the adapter"
            ),
        )
        return
    if reader.stopped:
        reader.unavailable.append("sacct_raw")
        return
    try:
        result = ctx.clusters.scheduler(attempt["cluster_id"]).query_accounting(job_ids=[str(job_id)])
    except TransportError as exc:
        if exc.code == ErrorCode.UNSUPPORTED:  # no accounting on this cluster: retrying cannot help
            reader.unavailable.append(f"sacct_raw: {exc.message}")
        else:
            reader.fail("sacct_raw", exc)
        return
    except HpcLabError as exc:
        reader.unavailable.append(f"sacct_raw: {exc.message}")
        return
    jobs = [j for j in result.jobs if j.job_id == str(job_id)]
    if not jobs:
        reader.missing.append("sacct_raw")
        return
    _keep_accounting(
        reader,
        str(job_id),
        source=result.source,
        observed_at=result.observed_at,
        complete=result.complete,
        records=jobs,
        note="parsed accounting record (sacct -X); the raw sacct line is not exposed by the adapter",
    )


def _keep_accounting(
    reader: _Reader,
    job_id: str,
    *,
    source: str,
    observed_at: str,
    complete: bool,
    records: list[JobObservation],
    note: str,
) -> None:
    doc = {
        "schema": "hpclab.sacct-record/v1",
        "source": source,
        "observed_at": observed_at,
        "complete": complete,
        "job_id": job_id,
        "records": [j.model_dump(mode="json") for j in records],
        "note": note,
    }
    data = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8")
    reader.items.append(
        _Captured("sacct_raw", f"sacct:{job_id}", "sacct.json", data=data, source_size=len(data))
    )


def _stable_sacct(data: bytes) -> Any:
    """Accounting records without per-query timestamps (they change on every query)."""
    try:
        doc = json.loads(data)
    except ValueError:
        return None
    records = doc.get("records") if isinstance(doc, dict) else None
    if not isinstance(records, list):
        return None
    return [{k: v for k, v in r.items() if k != "observed_at"} for r in records if isinstance(r, dict)]


def _same_content(item: _Captured, prev: Row, new_digest: str) -> bool:
    if prev["digest"] == new_digest:
        return True
    if item.kind != "sacct_raw" or item.data is None:
        return False
    try:
        old = Path(str(prev["local_path"])).read_bytes()
    except OSError:
        return False
    stable = _stable_sacct(item.data)
    return stable is not None and stable == _stable_sacct(old)


def _keeps_attested_copy(item: _Captured, prev: Row, new_digest: str, reader: _Reader) -> bool:
    """A vault copy that still matches the digest the index attested is the evidence an
    evaluation was computed from: it is never replaced by bytes that no longer match (a
    scratch file overwritten by a re-run or a script after the job ended)."""
    expected = item.expected_digest
    if expected is None or new_digest == expected or prev["digest"] != expected:
        return False
    try:
        if _sha256_path(Path(str(prev["local_path"]))) != expected:
            return False  # the local copy itself is damaged: take the new bytes
    except OSError:
        return False
    reader.notes.append(
        f"metrics_file:{item.source_path}: kept the earlier vault copy that matches the indexed digest {expected}"
    )
    return True


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_atomic(path: Path, data: bytes | None, src: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{secrets.token_hex(6)}"
    try:
        if src is not None:
            shutil.move(str(src), tmp)
        else:
            tmp.write_bytes(data or b"")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def capture(ctx: ServiceContext, *, attempt_id: str) -> VaultCaptureData:
    """Copy the attempt's evidence into the local vault (outbox ``vault.capture``).

    Remote reads are bounded and happen before the ledger transaction; sources that are
    missing or unreachable are reported (and retried by the handler), never faked.
    """
    if kind_of(attempt_id) != "attempt":
        raise InvalidArgument(f"{attempt_id!r} is not an attempt id (att_...)")
    attempt = ctx.db.get("attempts", "attempt_id", attempt_id)
    if attempt["phase"] in _ACTIVE_PHASES:
        raise NotReady(
            f"attempt {attempt_id} is {attempt['phase']}; the vault captures an attempt after it ends",
            details={"attempt_id": attempt_id, "phase": attempt["phase"]},
        )
    observed_at = iso(ctx.clock.now()) or ""
    root = vault_root(ctx.paths) / attempt_id
    root.mkdir(parents=True, exist_ok=True)
    reader = _Reader(timeout=max(10.0, min(60.0, ctx.settings.request_deadline_s * 3)))
    incoming = root / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        ran = attempt["phase"] in ("terminal", "abandoned") or bool(attempt.get("job_id"))
        if ran:
            _read_sources(ctx, attempt, reader, incoming)
            _read_accounting(ctx, attempt, reader)
        else:
            # nothing ran: keep what was staged for the audit trail (script, plan)
            _read_sources_subset(ctx, attempt, reader)
        existing = {
            (r["kind"], r["source_path"]): r
            for r in ctx.db.all("SELECT * FROM vault_items WHERE attempt_id=?", (attempt_id,))
        }
        captured: list[VaultItemData] = []
        unchanged = 0
        rows_to_write: list[dict[str, Any]] = []
        source_sizes: dict[str, int] = {}
        for item in reader.items:
            if item.tmp_file is not None:
                payload_digest, stored = _sha256_path(item.tmp_file), item.tmp_file.stat().st_size
            else:
                blob = item.data or b""
                payload_digest, stored = _sha256_bytes(blob), len(blob)
            if item.source_size is not None:
                source_sizes[f"{item.kind}:{item.source_path}"] = item.source_size
            local = root / item.rel_local
            prev = existing.get((item.kind, item.source_path))
            if (
                prev is not None
                and Path(prev["local_path"]).is_file()
                and (
                    _same_content(item, prev, payload_digest)
                    or _keeps_attested_copy(item, prev, payload_digest, reader)
                )
            ):
                unchanged += 1
                if item.tmp_file is not None and item.tmp_file.exists():
                    item.tmp_file.unlink()
                continue
            _write_atomic(local, item.data, item.tmp_file)
            rows_to_write.append(
                {
                    "item_id": prev["item_id"] if prev is not None else f"vlt_{ulid()}",
                    "attempt_id": attempt_id,
                    "kind": item.kind,
                    "source_path": item.source_path,
                    "local_path": str(local),
                    "digest": payload_digest,
                    "size_bytes": stored,
                    "truncated": 1 if item.truncated else 0,
                    "captured_at": observed_at,
                }
            )
    finally:
        # also after a failure: nothing downloaded for this capture outlives it
        shutil.rmtree(incoming, ignore_errors=True)
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    status: SourceStatusValue = "ok"
    if reader.errors:
        status = (
            "unauthorized" if any(isinstance(e, AuthRequiredError) for e in reader.errors) else "unavailable"
        )
    elif reader.unavailable:
        status = "partial"
    cov.add("cluster", status, observed_at, "; ".join(reader.unavailable)[:500] or None)
    if reader.missing:
        cov.notes.append("missing on the cluster: " + ", ".join(reader.missing))
    retry: float | None = None
    if reader.errors:
        retry = 120.0 if status == "unauthorized" else 60.0
    elif "sacct_raw" in reader.missing and ran:
        retry = 300.0  # accounting lags behind the end of the job
    with ctx.db.tx(source="worker") as tx:
        for row in rows_to_write:
            tx.execute(
                "INSERT INTO vault_items (item_id, attempt_id, kind, source_path, local_path, digest, size_bytes, "
                "truncated, captured_at) VALUES (:item_id, :attempt_id, :kind, :source_path, :local_path, :digest, "
                ":size_bytes, :truncated, :captured_at) ON CONFLICT(attempt_id, kind, source_path) DO UPDATE SET "
                "local_path=excluded.local_path, digest=excluded.digest, size_bytes=excluded.size_bytes, "
                "truncated=excluded.truncated, captured_at=excluded.captured_at",
                row,
            )
            captured.append(_item_model(row))
        event = "vault.captured" if not reader.errors else "vault.capture_incomplete"
        tx.event(
            "attempt",
            attempt_id,
            event,
            {
                "items": [
                    {"kind": r["kind"], "source": r["source_path"], "digest": r["digest"]}
                    for r in rows_to_write
                ],
                "unchanged": unchanged,
                "source_sizes": source_sizes,
                "missing": reader.missing,
                "unavailable": reader.unavailable[:20],
            },
            observed_at=observed_at,
        )
    warnings = list(reader.notes)
    if not ran:
        warnings.append(f"attempt {attempt['phase']}: no job ran, only staged files were captured")
    return VaultCaptureData(
        attempt_id=attempt_id,
        captured=captured,
        unchanged=unchanged,
        source_sizes=source_sizes,
        missing=reader.missing,
        unavailable=reader.unavailable,
        retry_after_s=retry,
        coverage=cov,
        observed_at=observed_at,
        warnings=warnings,
    )


def _read_sources_subset(ctx: ServiceContext, attempt: Row, reader: _Reader) -> None:
    lay = attempt_paths(attempt)
    try:
        transport = ctx.clusters.transport(attempt["cluster_id"])
    except TransportError as exc:
        reader.fail("cluster", exc)
        return
    for kind, path, local in (
        ("script", lay.script, "job.sbatch"),
        ("plan", lay.token_file, "hpclab-plan.json"),
    ):
        if reader.stopped:
            break
        try:
            rr = transport.read_range(path, 0, SMALL_FILE_MAX, timeout=reader.timeout)
        except FileNotFoundError:
            reader.missing.append(kind)
            continue
        except TransportError as exc:
            reader.fail(kind, exc)
            continue
        except OSError as exc:
            reader.unavailable.append(f"{kind}: {exc}")
            continue
        reader.items.append(
            _Captured(kind, path, local, data=rr.data, source_size=rr.size, truncated=rr.size > len(rr.data))
        )


def _item_model(row: Row) -> VaultItemData:
    return VaultItemData(
        item_id=row["item_id"],
        attempt_id=row["attempt_id"],
        kind=row["kind"],
        source_path=row["source_path"],
        local_path=row["local_path"],
        digest=row["digest"],
        size_bytes=row["size_bytes"],
        truncated=bool(row["truncated"]),
        captured_at=row["captured_at"],
        present=Path(str(row["local_path"])).is_file(),
    )


# ---------------------------------------------------------------------------
# vault_status
# ---------------------------------------------------------------------------


def vault_status(
    ctx: ServiceContext,
    *,
    experiment_id: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> VaultStatusData:
    """What the local vault holds for ended attempts (optionally of one experiment)."""
    lim = clamp_limit(limit)
    observed_at = iso(ctx.clock.now()) or ""
    if experiment_id is not None:
        if kind_of(experiment_id) != "experiment":
            raise InvalidArgument(f"{experiment_id!r} is not an experiment id (exp_...)")
        ctx.db.get("experiments", "experiment_id", experiment_id)
    marks = ",".join("?" for _ in _ACTIVE_PHASES)
    scope = f"a.phase NOT IN ({marks}) AND a.phase != 'cancelled_before_submit'"
    params: list[Any] = list(_ACTIVE_PHASES)
    if experiment_id is not None:
        scope += " AND r.experiment_id = ?"
        params.append(experiment_id)
    base = f"FROM attempts a JOIN runs r ON r.run_id = a.run_id WHERE {scope}"
    agg = ctx.db.one(
        f"SELECT COUNT(*) AS n, SUM(CASE WHEN EXISTS (SELECT 1 FROM vault_items v WHERE v.attempt_id=a.attempt_id) "
        f"THEN 1 ELSE 0 END) AS captured {base}",
        params,
    )
    totals = ctx.db.one(
        f"SELECT COUNT(*) AS items, COALESCE(SUM(v.size_bytes), 0) AS bytes FROM vault_items v WHERE v.attempt_id IN "
        f"(SELECT a.attempt_id {base})",
        params,
    )
    after = decode_cursor(cursor).get("after")
    page_sql = f"SELECT a.attempt_id, a.run_id, a.phase, a.job_id {base}"
    page_params = list(params)
    if isinstance(after, str):
        page_sql += " AND a.attempt_id > ?"
        page_params.append(after)
    rows = ctx.db.all(page_sql + " ORDER BY a.attempt_id LIMIT ?", [*page_params, lim + 1])
    next_cursor = encode_cursor({"after": rows[lim - 1]["attempt_id"]}) if len(rows) > lim else None
    rows = rows[:lim]
    items_by_attempt: dict[str, list[Row]] = {}
    if rows:
        ids = [r["attempt_id"] for r in rows]
        q = ",".join("?" for _ in ids)
        for it in ctx.db.all(f"SELECT * FROM vault_items WHERE attempt_id IN ({q})", ids):
            items_by_attempt.setdefault(it["attempt_id"], []).append(it)
    attempts: list[VaultAttemptStatus] = []
    fully = partially = 0
    missing_files = 0
    for r in rows:
        its = items_by_attempt.get(r["attempt_id"], [])
        kinds = sorted({i["kind"] for i in its})
        expected = (
            EXPECTED_KINDS if (r["phase"] in ("terminal", "abandoned") or r["job_id"]) else _NO_JOB_KINDS
        )
        missing_kinds = [k for k in expected if k not in kinds]
        absent = sum(1 for i in its if not Path(str(i["local_path"])).is_file())
        missing_files += absent
        if its and not missing_kinds and not absent:
            fully += 1
        elif its:
            partially += 1
        attempts.append(
            VaultAttemptStatus(
                attempt_id=r["attempt_id"],
                run_id=r["run_id"],
                phase=r["phase"],
                items=len(its),
                kinds=kinds,
                missing_kinds=missing_kinds,
                stored_bytes=sum(int(i["size_bytes"] or 0) for i in its),
                truncated_items=sum(1 for i in its if i["truncated"]),
                missing_local_files=absent,
                last_captured_at=max((i["captured_at"] for i in its), default=None),
            )
        )
    n = int(agg["n"]) if agg else 0
    captured = int(agg["captured"] or 0) if agg else 0
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    cov.add("vault", "ok" if not missing_files else "partial", observed_at)
    if missing_files:
        cov.notes.append(f"{missing_files} vault file(s) on this page are missing from disk")
    return VaultStatusData(
        vault_dir=str(vault_root(ctx.paths)),
        experiment_id=experiment_id,
        attempts_in_scope=n,
        fully_captured=fully,
        partially_captured=partially,
        not_captured=n - captured,
        total_items=int(totals["items"]) if totals else 0,
        total_bytes=int(totals["bytes"]) if totals else 0,
        attempts=attempts,
        next_cursor=next_cursor,
        coverage=cov,
        observed_at=observed_at,
        warnings=["fully/partially_captured count this page only; not_captured counts the whole scope"]
        if next_cursor
        else [],
    )


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------


_BACKUP_DIR_MODE = 0o700
_BACKUP_FILE_MODE = 0o600


def _private_dirs(path: Path, base: Path) -> None:
    """Create ``path`` (inside the backup ``base``) and its missing parents as 0700; the modes
    are set explicitly, so the umask cannot widen them."""
    missing: list[Path] = []
    p = path
    while p != base and base in p.parents and not p.is_dir():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        d.mkdir(exist_ok=True)
        os.chmod(d, _BACKUP_DIR_MODE)


def _copy_tree(src: Path, dst: Path, files: list[dict[str, Any]], base: Path) -> tuple[int, int]:
    count = total = 0
    if not src.is_dir():
        return 0, 0
    for dirpath, dirnames, filenames in os.walk(src):
        # dot-dirs hold in-progress work (a capture's ``.incoming`` downloads), never evidence
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".tmp-") or name.startswith(".partial-"):
                continue
            s = Path(dirpath) / name
            if s.is_symlink() or not s.is_file():
                continue
            d = dst / s.relative_to(src)
            _private_dirs(d.parent, base)
            shutil.copy2(s, d)
            os.chmod(d, _BACKUP_FILE_MODE)  # copy2 carries the source's mode bits over
            size = d.stat().st_size
            files.append({"path": str(d.relative_to(base)), "sha256": _sha256_path(d), "bytes": size})
            count += 1
            total += size
    return count, total


def backup(
    ctx: ServiceContext,
    *,
    dest: str,
    idempotency_key: str | None = None,
    include_vault: bool = True,
) -> BackupData:
    """Back up the ledger (SQLite online backup API), the vault and snapshot manifests into
    ``<dest>/hpclab-backup-<UTC stamp>/``. ``dest`` must be absolute and outside HPCLAB_HOME."""
    raw = Path(dest).expanduser()
    if not raw.is_absolute():
        raise InvalidArgument(
            "dest must be an absolute path (the controller's working directory is not yours)"
        )
    target_root = raw.resolve()
    home = ctx.paths.home.resolve()
    if target_root == home or home in target_root.parents:
        raise InvalidArgument("dest must be outside HPCLAB_HOME (a backup inside it would copy itself)")
    if ctx.db.path == ":memory:":
        raise InvalidArgument("an in-memory ledger cannot be backed up")
    request = {"dest": str(target_root), "include_vault": include_vault}
    payload_hash = digest(request)
    if idempotency_key:
        with ctx.db.tx(source="api") as tx:
            stored = tx.idem_lookup(ctx.principal, "backup", idempotency_key, payload_hash)
        if stored is not None:
            return BackupData.model_validate(stored)
    now = ctx.clock.now()
    created_at = iso(now) or ""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    target = target_root / f"hpclab-backup-{stamp}"
    n = 2
    while target.exists():
        target = target_root / f"hpclab-backup-{stamp}-{n}"
        n += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    # the ledger copy holds notify secrets: the backup is private to its owner (0700 / 0600,
    # set explicitly so a permissive umask cannot widen it)
    target.mkdir(mode=_BACKUP_DIR_MODE)
    os.chmod(target, _BACKUP_DIR_MODE)
    db_file = target / "hpclab.db"
    os.close(os.open(db_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _BACKUP_FILE_MODE))
    os.chmod(db_file, _BACKUP_FILE_MODE)
    src = sqlite3.connect(ctx.db.path, timeout=30.0)
    try:
        dst = sqlite3.connect(str(db_file))
        try:
            src.backup(dst)
            check = dst.execute("PRAGMA integrity_check").fetchone()
            integrity = str(check[0]) if check else "unknown"
        finally:
            dst.close()
    finally:
        src.close()
    os.chmod(db_file, _BACKUP_FILE_MODE)
    if integrity != "ok":
        raise HpcLabError(
            f"the backed-up ledger failed PRAGMA integrity_check: {integrity}",
            details={"backup_dir": str(target)},
        )
    files: list[dict[str, Any]] = []
    vault_files = vault_bytes = 0
    if include_vault:
        vault_files, vault_bytes = _copy_tree(vault_root(ctx.paths), target / "vault", files, target)
    manifests, _ = _copy_tree(
        ctx.paths.snapshots / "manifests", target / "snapshots" / "manifests", files, target
    )
    db_sha = _sha256_path(db_file)
    manifest = {
        "schema": BACKUP_SCHEMA,
        "created_at": created_at,
        "source_home": str(home),
        "db": {
            "path": "hpclab.db",
            "sha256": db_sha,
            "bytes": db_file.stat().st_size,
            "integrity_check": integrity,
        },
        "files": files,
        "counts": {"vault_files": vault_files, "vault_bytes": vault_bytes, "snapshot_manifests": manifests},
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    manifest_path = target / "BACKUP.json"
    fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _BACKUP_FILE_MODE)
    with os.fdopen(fd, "wb") as fh:
        fh.write(manifest_bytes)
    os.chmod(manifest_path, _BACKUP_FILE_MODE)
    data = BackupData(
        backup_dir=str(target),
        db_path=str(db_file),
        db_bytes=db_file.stat().st_size,
        db_sha256=db_sha,
        integrity_check=integrity,
        vault_files=vault_files,
        vault_bytes=vault_bytes,
        snapshot_manifests=manifests,
        manifest_path=str(manifest_path),
        manifest_digest=_sha256_bytes(manifest_bytes),
        created_at=created_at,
        warnings=[] if include_vault else ["vault not included"],
    )
    with ctx.db.tx(source="api") as tx:
        if idempotency_key:
            again = tx.idem_lookup(ctx.principal, "backup", idempotency_key, payload_hash)
            if again is not None:
                return BackupData.model_validate(again)
        tx.event(
            "backup",
            target.name,
            "backup.created",
            {
                "backup_dir": str(target),
                "db_sha256": db_sha,
                "vault_files": vault_files,
                "manifest_digest": data.manifest_digest,
            },
        )
        if idempotency_key:
            tx.idem_store(
                ctx.principal, "backup", idempotency_key, payload_hash, data.model_dump(mode="json")
            )
    return data


# ---------------------------------------------------------------------------
# Outbox handler and periodic sweep
# ---------------------------------------------------------------------------


def handle_capture(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    attempt_id = str(item.payload.get("attempt_id") or item.target_id or "")
    try:
        res = capture(ctx, attempt_id=attempt_id)
    except (NotFound, InvalidArgument) as exc:
        return HandlerResult.done(exc.message)
    except NotReady as exc:
        return (
            HandlerResult.retry(300.0, exc.message)
            if item.tries < MAX_CAPTURE_RETRIES
            else HandlerResult.done(exc.message)
        )
    if res.retry_after_s is not None:
        waiting_for_auth = any(s.status == "unauthorized" for s in res.coverage.sources)
        if waiting_for_auth or item.tries < MAX_CAPTURE_RETRIES:
            return HandlerResult.retry(
                res.retry_after_s, "; ".join(res.unavailable + res.missing)[:500] or None
            )
    return HandlerResult.done(f"{len(res.captured)} item(s) captured")


def sweep_uncaptured(ctx: ServiceContext, *, limit: int = 50) -> int:
    """Enqueue ``vault.capture`` for ended attempts that were never captured: terminal ones
    (normally enqueued by observe.finalize) and the abandoned / rejected ones no other step
    captures - the same scope ``vault_status`` counts, so its ``not_captured`` drains to 0."""
    cutoff = iso(ctx.clock.now() - timedelta(seconds=600))
    rows = ctx.db.all(
        "SELECT a.attempt_id FROM attempts a WHERE a.phase IN ('terminal','abandoned','submission_rejected') "
        "AND COALESCE(a.terminal_at, a.updated_at) <= ? "
        "AND NOT EXISTS (SELECT 1 FROM events e WHERE e.entity_type='attempt' AND e.entity_id=a.attempt_id "
        "AND e.event_type='vault.captured') AND NOT EXISTS (SELECT 1 FROM outbox o WHERE o.kind='vault.capture' "
        "AND o.target_id=a.attempt_id AND o.state IN ('pending','leased')) ORDER BY a.attempt_id LIMIT ?",
        (cutoff, limit),
    )
    if not rows:
        return 0
    with ctx.db.tx(source="worker") as tx:
        for r in rows:
            tx.enqueue(
                "vault.capture",
                target_id=r["attempt_id"],
                payload={"attempt_id": r["attempt_id"]},
                dedupe_key=f"vault.capture:{r['attempt_id']}",
            )
    return len(rows)


def outbox_handlers() -> dict[str, Handler]:
    return {"vault.capture": handle_capture}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    def run(ctx: ServiceContext) -> None:
        sweep_uncaptured(ctx)

    return [PeriodicTask(name="vault.sweep", interval_s=600.0, fn=run, jitter_s=60.0, run_at_start=False)]
