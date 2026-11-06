"""Checkpoint validation for ``hpclab.checkpoint/v1`` (design §6.5) and bare checkpoint dirs.

Metadata only: from a manifest and the file stats the caller gathered (login-node ``stat`` plus
optional digests) this decides whether a checkpoint is *complete and intact* (``status``) and
whether it can *continue this Run exactly* (``continuation``). It never loads tensors: that needs a
compute probe Run, which is why ``needs_load_probe`` stays True until the caller reports a passed
probe. A ``.pt`` file existing is not proof of anything (it may be truncated mid-write).

Issue severities: ``error`` = integrity failure (status ``invalid``); ``warning`` = a limit on how
the checkpoint may be used (continuation downgraded) or something unverified; ``info`` = FYI.

Continuation levels:

* ``exact``        – same experiment (same run spec hash, or identical semantic config), resumable
  training state, same world size: resuming continues this Run. The run spec hash does not
  cover resources, so when ``world_size`` is a declared semantic key the resuming world size
  must be *known* (``expected_world_size`` or ``expected_config["world_size"]``) and equal.
* ``weights_only`` – usable to initialise weights, but not a faithful continuation (provenance
  unconfirmed, no optimizer state, or a different world size).
* ``incompatible`` – declared semantic config differs (e.g. ``lr``): resuming would silently mix
  two experiments; only an explicitly derived Run may start from it.
* ``unknown``      – no usable manifest.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.adapters.base import StatInfo
from hpclab.domain.hashing import canonical_json, digest

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MANIFEST_NAME",
    "CheckpointCheck",
    "CheckpointIssue",
    "best_resume_candidate",
    "check_manifest_text",
    "classify_checkpoint_dir",
    "validate_checkpoint_manifest",
]

CHECKPOINT_SCHEMA = "hpclab.checkpoint/v1"
MANIFEST_NAME = "hpclab-ckpt.json"
MAX_ISSUES = 50

Status = Literal["valid", "invalid", "unverified"]
Continuation = Literal["exact", "weights_only", "incompatible", "unknown"]
Severity = Literal["error", "warning", "info"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONT_ORDER: dict[str, int] = {"exact": 0, "weights_only": 1, "incompatible": 2, "unknown": 3}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CheckpointIssue(_Frozen):
    code: str
    severity: Severity
    message: str


class CheckpointCheck(_Frozen):
    """Outcome of a checkpoint check. See the module docstring for the meaning of each level."""

    status: Status
    continuation: Continuation
    step: int | None = None
    issues: list[CheckpointIssue] = Field(default_factory=list)
    needs_load_probe: bool = True
    files_checked: int = 0
    digests_verified: int = 0
    epoch: float | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    # semantic key -> {"checkpoint": value, "expected": value}
    config_diffs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # sha256 of the canonical manifest: the identity to record as RecoverySource.checkpoint_digest
    manifest_digest: str | None = None

    @property
    def errors(self) -> list[CheckpointIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def issue_codes(self) -> list[str]:
        return [i.code for i in self.issues]

    @property
    def metadata_ok_for_resume(self) -> bool:
        """Intact and an exact continuation, loadability not yet considered."""
        return self.status == "valid" and self.continuation == "exact"

    @property
    def ready_for_resume(self) -> bool:
        """Intact, exact, and a compute probe has confirmed it loads."""
        return self.metadata_ok_for_resume and not self.needs_load_probe


@dataclass
class _Collector:
    issues: list[CheckpointIssue]

    def add(self, code: str, severity: Severity, message: str) -> None:
        if len(self.issues) < MAX_ISSUES:
            self.issues.append(CheckpointIssue(code=code, severity=severity, message=message))

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def _norm_digest(value: Any) -> str | None:
    """Lower-case hex of a ``sha256:<hex>`` or bare-hex digest; None if absent or malformed."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    v = v.removeprefix("sha256:")
    return v if _HEX64.match(v) else None


def _safe_relpath(path: str) -> bool:
    if not path or path.startswith(("/", "\\")) or "\\" in path or "\x00" in path:
        return False
    norm = posixpath.normpath(path)
    return norm != "." and not norm.startswith("../") and norm != ".."


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _same(a: Any, b: Any) -> bool:
    """Config equality: numbers by value (1e-9 relative), bools only equal bools, rest canonically."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError, RecursionError):
        pass
    try:
        return bool(a == b)
    except RecursionError:
        return False  # cannot tell: treat as different (never a false "exact")


def _preview(value: Any, limit: int = 120) -> str:
    try:
        return repr(value)[:limit]
    except RecursionError:
        return f"<{type(value).__name__} nested too deeply>"


def _stat_view(stat: Any) -> tuple[bool, str | None, int | None, str | None]:
    """(exists, kind, size, hex digest) from a stat dict or ``StatInfo``."""
    if isinstance(stat, StatInfo):
        return stat.exists, stat.kind, stat.size, None
    if not isinstance(stat, Mapping):
        return False, None, None, None
    size = _as_int(stat.get("size"))
    exists = stat.get("exists")
    exists_b = bool(exists) if exists is not None else size is not None
    kind = stat.get("kind")
    return (
        exists_b,
        kind if isinstance(kind, str) else None,
        size,
        _norm_digest(stat.get("sha256") or stat.get("digest")),
    )


@dataclass
class _FileTally:
    checked: int = 0
    digests_verified: int = 0
    digests_unverified: int = 0


def _check_files(files: Any, file_stats: Mapping[str, Any], out: _Collector) -> _FileTally:
    tally = _FileTally()
    if not isinstance(files, list) or not files:
        out.add("no_files", "error", "The manifest lists no files.")
        return tally
    # callers may key stats as "./model.pt"; the manifest says "model.pt"
    stats_by_norm = {posixpath.normpath(k): v for k, v in file_stats.items() if isinstance(k, str) and k}
    seen: set[str] = set()
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path:
            out.add("bad_file_entry", "error", f"Malformed file entry: {_preview(entry)}")
            continue
        if not _safe_relpath(path):
            out.add(
                "unsafe_path", "error", f"File path {path!r} is absolute or escapes the checkpoint directory."
            )
            continue
        norm = posixpath.normpath(path)
        if norm in seen:
            out.add("duplicate_path", "error", f"File {path!r} is listed twice.")
            continue
        seen.add(norm)
        _check_one_file(path, entry, file_stats.get(path, stats_by_norm.get(norm)), out, tally)
    return tally


def _check_one_file(path: str, entry: dict[str, Any], stat: Any, out: _Collector, tally: _FileTally) -> None:
    want_size = _as_int(entry.get("size"))
    want_digest = _norm_digest(entry.get("sha256"))
    if entry.get("sha256") is not None and want_digest is None:
        out.add("bad_digest_format", "warning", f"{path}: manifest digest is not 'sha256:<64 hex>'.")
    tally.checked += 1
    exists, kind, size, have_digest = _stat_view(stat)
    if stat is None or not exists:
        out.add("file_missing", "error", f"{path} is listed in the manifest but does not exist.")
        return
    if kind is not None and kind not in ("file", "symlink"):
        out.add("not_a_file", "error", f"{path} is a {kind}, not a file.")
        return
    if want_size is None:
        out.add("size_unknown", "warning", f"{path}: the manifest has no size.")
    elif size is None:
        out.add("size_unchecked", "warning", f"{path}: the current size could not be read.")
    elif size != want_size:
        out.add(
            "size_mismatch",
            "error",
            f"{path} is {size} bytes but the manifest says {want_size} (truncated or overwritten).",
        )
        return
    if want_digest and have_digest:
        if want_digest != have_digest:
            out.add("digest_mismatch", "error", f"{path}: content digest differs from the manifest.")
        else:
            tally.digests_verified += 1
    elif want_digest:
        tally.digests_unverified += 1


def _check_header(manifest: dict[str, Any], out: _Collector) -> bool:
    """Schema/complete/step checks; returns whether the schema is one we understand."""
    schema = manifest.get("schema")
    schema_ok = schema == CHECKPOINT_SCHEMA
    if schema is None:
        out.add("schema_missing", "error", f"The manifest has no 'schema' (expected {CHECKPOINT_SCHEMA}).")
    elif not schema_ok:
        out.add(
            "unsupported_schema",
            "error",
            f"Unsupported manifest schema {schema!r} (expected {CHECKPOINT_SCHEMA}).",
        )
    if manifest.get("complete") is not True:
        out.add(
            "incomplete",
            "error",
            "The manifest is not marked complete: the writer did not finish this checkpoint.",
        )
    step = _as_int(manifest.get("step"))
    if step is None:
        out.add("step_invalid", "error", "The manifest has no integer 'step'.")
    elif step <= 0:
        out.add(
            "step_not_positive", "error", f"step={step}: a checkpoint must be taken after at least one step."
        )
    return schema_ok


@dataclass
class _Compat:
    continuation: Continuation
    diffs: dict[str, dict[str, Any]]


def _semantic_diffs(
    semantic_keys: list[str],
    config: dict[str, Any],
    expected: dict[str, Any],
    out: _Collector,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    diffs: dict[str, dict[str, Any]] = {}
    unverifiable: list[str] = []
    for key in semantic_keys:
        if key == "world_size":
            continue  # compared separately (may come from the plan's resources)
        if key not in config:
            out.add(
                "semantic_key_missing",
                "warning",
                f"Semantic key {key!r} is declared but absent from the checkpoint config.",
            )
            unverifiable.append(key)
        elif key not in expected:
            unverifiable.append(key)
        elif not _same(config[key], expected[key]):
            diffs[key] = {"checkpoint": config[key], "expected": expected[key]}
    return diffs, unverifiable


def _compatibility(
    manifest: dict[str, Any],
    *,
    expected_run_spec_hash: str | None,
    expected_config: dict[str, Any] | None,
    expected_world_size: int | None,
    expected_run_id: str | None,
    out: _Collector,
) -> _Compat:
    config = manifest.get("config")
    if config is None:
        config = {}
    elif not isinstance(config, dict):
        out.add(
            "bad_config",
            "warning",
            "The manifest 'config' is not an object; config compatibility is unknown.",
        )
        config = {}
    raw_keys = manifest.get("semantic_keys")
    semantic_keys = [k for k in raw_keys if isinstance(k, str)] if isinstance(raw_keys, list) else []
    if raw_keys is not None and not isinstance(raw_keys, list):
        out.add("bad_semantic_keys", "warning", "The manifest 'semantic_keys' is not a list.")

    diffs, unverifiable = (
        _semantic_diffs(semantic_keys, config, expected_config or {}, out)
        if expected_config is not None
        else ({}, [])
    )
    ws = _world_size_check(manifest, config, semantic_keys, expected_config, expected_world_size, diffs, out)
    if ws == "unverified":
        unverifiable.append("world_size")

    ckpt_hash = manifest.get("run_spec_hash")
    if not isinstance(ckpt_hash, str) or not ckpt_hash:
        out.add("no_run_spec_hash", "warning", "The manifest does not record which run spec produced it.")
        ckpt_hash = None
    run_id = manifest.get("run_id")
    if expected_run_id and isinstance(run_id, str) and run_id != expected_run_id:
        out.add("foreign_run", "warning", f"Written by run {run_id}, not {expected_run_id}.")

    if diffs:
        keys = ", ".join(sorted(diffs))
        out.add(
            "semantic_config_mismatch",
            "warning",
            f"Semantic config differs ({keys}): resuming would mix two experiments; only a derived Run may start from it.",
        )
        return _Compat("incompatible", diffs)

    same_hash = expected_run_spec_hash is not None and ckpt_hash == expected_run_spec_hash
    comparable = expected_config is not None and bool(semantic_keys) and not unverifiable
    level: Continuation = "exact"
    if not same_hash and comparable:
        out.add(
            "same_semantic_config",
            "info",
            "Different run spec hash, but every declared semantic key matches.",
        )
    elif not same_hash:
        reason = (
            "no expected run spec hash was given"
            if expected_run_spec_hash is None
            else "the run spec hash differs"
        )
        if unverifiable:
            reason += f" and semantic keys {', '.join(sorted(unverifiable))} could not be compared"
        out.add(
            "provenance_unconfirmed", "warning", f"Cannot confirm this is the same experiment ({reason})."
        )
        level = "weights_only"
    resumable = manifest.get("resumable")
    if resumable is not True:
        code = "not_resumable" if resumable is False else "resumable_unknown"
        out.add(
            code,
            "warning",
            "The manifest does not declare resumable training state (optimizer/scheduler/RNG).",
        )
        level = _cap(level, "weights_only")
    if ws in ("changed", "unverified"):
        # the run spec hash does not cover resources, so even the same hash cannot vouch for it
        level = _cap(level, "weights_only")
    return _Compat(level, diffs)


def _world_size_check(
    manifest: dict[str, Any],
    config: dict[str, Any],
    semantic_keys: list[str],
    expected_config: dict[str, Any] | None,
    expected_world_size: int | None,
    diffs: dict[str, dict[str, Any]],
    out: _Collector,
) -> Literal["same", "changed", "semantic_diff", "unverified", "unknown"]:
    """Compare the checkpoint's world size with the resuming Run's.

    Declared semantic (``world_size`` in ``semantic_keys``): a difference is a config diff
    (``semantic_diff`` -> incompatible), and an unknown side is ``unverified`` (never exact).
    Otherwise a difference is ``changed`` (not an exact continuation) and unknown is ``unknown``
    (no finding: the writer did not say it matters).
    """
    semantic = "world_size" in semantic_keys
    raw = config.get("world_size", manifest.get("world_size"))
    ckpt_ws = _as_int(raw)
    exp_ws = (
        expected_world_size
        if expected_world_size is not None
        else _as_int((expected_config or {}).get("world_size"))
    )
    if semantic and ckpt_ws is None:
        if raw is None:
            out.add(
                "semantic_key_missing",
                "warning",
                "Semantic key 'world_size' is declared but absent from the checkpoint config.",
            )
        else:
            out.add(
                "world_size_unverified",
                "warning",
                f"Checkpoint world_size {_preview(raw)} is not an integer.",
            )
        return "unverified"
    if semantic and exp_ws is None:
        out.add(
            "world_size_unverified",
            "warning",
            "world_size is a declared semantic key but the resuming Run's world size is unknown: pass "
            "expected_world_size (total GPUs/ranks of the new plan) to allow an exact resume.",
        )
        return "unverified"
    if ckpt_ws is None or exp_ws is None:
        return "unknown"
    if ckpt_ws == exp_ws:
        return "same"
    if semantic:
        diffs["world_size"] = {"checkpoint": ckpt_ws, "expected": exp_ws}
        return "semantic_diff"
    out.add(
        "world_size_changed",
        "warning",
        f"World size changes from {ckpt_ws} to {exp_ws}: sharded optimizer state, data-sampler position and per-rank "
        "RNG cannot be continued exactly, and the global batch changes unless per-GPU batch or accumulation is adjusted.",
    )
    return "changed"


def _cap(level: Continuation, at_most: Continuation) -> Continuation:
    return level if _CONT_ORDER[level] >= _CONT_ORDER[at_most] else at_most


def _manifest_digest(manifest: dict[str, Any]) -> str | None:
    try:
        return digest(manifest)
    except (TypeError, ValueError, RecursionError):
        return None  # NaN/Infinity or absurd nesting: not canonical JSON, so no stable identity


def validate_checkpoint_manifest(
    manifest: dict[str, Any],
    file_stats: dict[str, dict[str, Any]],
    *,
    expected_run_spec_hash: str | None,
    expected_config: dict[str, Any] | None = None,
    expected_world_size: int | None = None,
    expected_run_id: str | None = None,
    probe_passed: bool = False,
) -> CheckpointCheck:
    """Validate an ``hpclab-ckpt.json`` manifest against observed file stats.

    ``file_stats`` maps each path *relative to the checkpoint dir* to ``{"exists", "size",
    "sha256"?, "kind"?}`` (``StatInfo`` values are accepted too); a manifest file without an
    entry counts as missing. ``expected_config`` is the config of the Run that wants to resume
    (usually its params), ``expected_world_size`` the new plan's total ranks (nodes x tasks, i.e.
    usually total GPUs) if not in the config; without it a checkpoint that declares
    ``world_size`` semantic is at most ``weights_only``. ``needs_load_probe`` is False only
    when ``probe_passed`` or the checkpoint is invalid anyway. Never raises on odd manifests.
    """
    out = _Collector([])
    if not isinstance(manifest, dict):
        out.add("bad_manifest", "error", "The manifest is not a JSON object.")
        return CheckpointCheck(
            status="invalid", continuation="unknown", issues=out.issues, needs_load_probe=False
        )
    schema_ok = _check_header(manifest, out)
    tally = _check_files(manifest.get("files"), file_stats, out)
    listed = manifest.get("files")
    manifest_paths = {
        posixpath.normpath(f["path"])
        for f in (listed if isinstance(listed, list) else [])
        if isinstance(f, dict) and isinstance(f.get("path"), str)
    }
    extra = sorted(
        p
        for p in file_stats
        if posixpath.normpath(p) not in manifest_paths and posixpath.basename(p) != MANIFEST_NAME
    )
    if extra:
        out.add("extra_files", "info", f"{len(extra)} file(s) not in the manifest, e.g. {extra[0]!r}.")
    compat = _compatibility(
        manifest,
        expected_run_spec_hash=expected_run_spec_hash,
        expected_config=expected_config,
        expected_world_size=expected_world_size,
        expected_run_id=expected_run_id,
        out=out,
    )
    if tally.digests_unverified and not out.has_errors:
        out.add(
            "digest_not_verified",
            "warning",
            f"{tally.digests_unverified} file(s) matched by size only; content digests were not computed "
            "(large files are hashed on a compute node, e.g. by the probe run).",
        )
    status: Status = "invalid" if out.has_errors else "valid"
    epoch = manifest.get("epoch")
    return CheckpointCheck(
        status=status,
        continuation=compat.continuation if schema_ok else "unknown",
        step=_as_int(manifest.get("step")),
        issues=out.issues,
        needs_load_probe=status != "invalid" and not probe_passed,
        files_checked=tally.checked,
        digests_verified=tally.digests_verified,
        epoch=float(epoch) if isinstance(epoch, int | float) and not isinstance(epoch, bool) else None,
        run_id=manifest.get("run_id") if isinstance(manifest.get("run_id"), str) else None,
        attempt_id=manifest.get("attempt_id") if isinstance(manifest.get("attempt_id"), str) else None,
        config_diffs=compat.diffs,
        manifest_digest=_manifest_digest(manifest),
    )


def check_manifest_text(
    raw: str | bytes, file_stats: dict[str, dict[str, Any]], **kwargs: Any
) -> CheckpointCheck:
    """Parse manifest text then validate; unreadable/truncated JSON is an ``invalid`` checkpoint."""
    try:
        manifest = json.loads(raw)
    except (ValueError, RecursionError) as exc:  # UnicodeDecodeError is a ValueError
        issue = CheckpointIssue(
            code="manifest_unreadable",
            severity="error",
            message=f"{MANIFEST_NAME} is not valid JSON ({type(exc).__name__}); it was not written atomically.",
        )
        return CheckpointCheck(
            status="invalid", continuation="unknown", issues=[issue], needs_load_probe=False
        )
    return validate_checkpoint_manifest(manifest, file_stats, **kwargs)


# ---------------------------------------------------------------------------
# Directories without a manifest
# ---------------------------------------------------------------------------

_WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".pkl",
    ".npz",
    ".npy",
    ".h5",
    ".msgpack",
    ".distcp",
)
_PARTIAL_RE = re.compile(r"(?:\.tmp|\.partial|\.incomplete|\.part|~)$|^\.?tmp", re.IGNORECASE)
_STATE_RE = re.compile(r"optim|scheduler|rng_state|trainer_state|zero_pp_rank|model_states", re.IGNORECASE)
_STEP_RES = (
    re.compile(r"(?:global_step|step|iter|iteration|checkpoint|ckpt)[-_=]?(\d+)", re.IGNORECASE),
    re.compile(r"_(\d{4,})\.[a-z]+$", re.IGNORECASE),
)


@dataclass(frozen=True)
class _Entry:
    name: str  # path relative to the directory
    size: int | None
    kind: str | None


def _entries(listing: Sequence[StatInfo | Mapping[str, Any]], dir_path: str | None) -> list[_Entry]:
    out: list[_Entry] = []
    prefix = dir_path.rstrip("/") + "/" if dir_path else None
    for item in listing:
        kind: str | None
        if isinstance(item, StatInfo):
            path, size, kind, exists = item.path, item.size, item.kind, item.exists
        elif isinstance(item, Mapping):
            path = str(item.get("path") or item.get("name") or "")
            raw_kind = item.get("kind")
            size, kind = _as_int(item.get("size")), raw_kind if isinstance(raw_kind, str) else None
            exists = item.get("exists", True) is not False
        else:
            continue
        if not path or not exists or kind in ("dir", "missing"):
            continue
        if prefix and path.startswith(prefix):
            path = path[len(prefix) :]
        out.append(_Entry(path, size, kind))
    return out


def _infer_step(names: list[str]) -> int | None:
    best: int | None = None
    for name in names:
        for pattern in _STEP_RES:
            for m in pattern.finditer(name):
                value = int(m.group(1))
                best = value if best is None else max(best, value)
    return best


def classify_checkpoint_dir(
    listing: Sequence[StatInfo | Mapping[str, Any]], *, dir_path: str | None = None
) -> CheckpointCheck:
    """Classify a checkpoint directory that has no validated manifest.

    Never ``valid``: without ``hpclab-ckpt.json`` file existence proves nothing. The result is
    ``unverified`` (a compute probe could still establish loadability, not provenance), or
    ``invalid`` when something is certainly wrong (no files, 0-byte weight files).
    """
    out = _Collector([])
    entries = _entries(listing, dir_path)
    if any(posixpath.basename(e.name) == MANIFEST_NAME for e in entries):
        out.add(
            "manifest_present",
            "info",
            f"{MANIFEST_NAME} exists: read it and validate it with validate_checkpoint_manifest().",
        )
        return CheckpointCheck(
            status="unverified", continuation="unknown", issues=out.issues, needs_load_probe=True
        )
    if not entries:
        out.add("no_files", "error", "The checkpoint directory is empty.")
        return CheckpointCheck(
            status="invalid", continuation="unknown", issues=out.issues, needs_load_probe=False
        )
    out.add(
        "no_manifest",
        "warning",
        f"No {MANIFEST_NAME}: an existing .pt/.safetensors file is not proof that the checkpoint is complete or "
        "loadable (it may have been cut off mid-write). Probe it on a compute node, or write checkpoints with "
        "hpclab_io.write_checkpoint.",
    )
    weights = [e for e in entries if e.name.lower().endswith(_WEIGHT_SUFFIXES)]
    if not weights:
        out.add("no_weight_files", "warning", "No recognisable weight files (.pt, .safetensors, .ckpt, ...).")
    empty = [e.name for e in weights if e.size == 0]
    if empty:
        out.add("empty_file", "error", f"{len(empty)} weight file(s) are 0 bytes, e.g. {empty[0]!r}.")
    partial = [e.name for e in entries if _PARTIAL_RE.search(posixpath.basename(e.name))]
    if partial:
        out.add(
            "partial_write_suspected", "warning", f"Temporary/partial files present, e.g. {partial[0]!r}."
        )
    if any(_STATE_RE.search(e.name) for e in entries):
        out.add("training_state_present", "info", "Optimizer/scheduler/RNG state files are present.")
    names = [e.name for e in weights] + ([posixpath.basename(dir_path.rstrip("/"))] if dir_path else [])
    step = _infer_step(names)
    if step is not None:
        out.add(
            "step_from_name", "info", f"Step {step} inferred from file/directory names, not from a manifest."
        )
    status: Status = "invalid" if out.has_errors else "unverified"
    return CheckpointCheck(
        status=status,
        continuation="unknown",
        step=step,
        issues=out.issues,
        needs_load_probe=status != "invalid",
        files_checked=len(weights),
    )


def best_resume_candidate(checks: Mapping[str, CheckpointCheck], *, require_probe: bool = True) -> str | None:
    """Key of the highest-step checkpoint usable for exact resume (probe-confirmed if required)."""
    usable = [
        (c.step or 0, key)
        for key, c in checks.items()
        if (c.ready_for_resume if require_probe else c.metadata_ok_for_resume)
    ]
    return max(usable)[1] if usable else None
