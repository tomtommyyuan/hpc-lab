"""hpclab_io — the few conventions a training script needs to be first-class under hpclab.

Copy this single file into your project (stdlib only, Python 3.8+). It implements:

* ``MetricsWriter``: append ``hpclab.metrics/v1`` JSON lines (train progress, eval results,
  the ``final`` record a paper table is built from).
* ``write_checkpoint`` / ``latest_checkpoint`` / ``read_checkpoint``: ``hpclab.checkpoint/v1``
  directories whose manifest ``hpclab-ckpt.json`` is written last, so a half-written
  checkpoint (job killed, node died, disk full) is never mistaken for a good one.
* ``hpclab_env``: the ``HPCLAB_*`` variables the job wrapper exports, parsed.
* ``GracefulStop``: turn SIGUSR1/SIGTERM (pre-timeout signal, preemption, scancel) into a flag
  your loop polls so it can checkpoint and exit cleanly.

Nothing here talks to hpclab; these are file formats. The same script runs unchanged on a
laptop, under the simulator and on a real cluster.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import time
from collections.abc import Iterable, Mapping
from typing import Any

METRICS_SCHEMA = "hpclab.metrics/v1"
CHECKPOINT_SCHEMA = "hpclab.checkpoint/v1"
MANIFEST_NAME = "hpclab-ckpt.json"
CHECKPOINT_PREFIX = "ckpt-"

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MANIFEST_NAME",
    "METRICS_SCHEMA",
    "CheckpointError",
    "GracefulStop",
    "MetricsWriter",
    "atomic_write_json",
    "hpclab_env",
    "latest_checkpoint",
    "read_checkpoint",
    "write_checkpoint",
]


class CheckpointError(Exception):
    """A checkpoint is missing, incomplete, corrupted or incompatible."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_KEYS = {
    "run_id": "HPCLAB_RUN_ID",
    "attempt_id": "HPCLAB_ATTEMPT_ID",
    "token": "HPCLAB_TOKEN",
    "output_dir": "HPCLAB_OUTPUT_DIR",
    "checkpoint_dir": "HPCLAB_CHECKPOINT_DIR",
    "resume_from": "HPCLAB_RESUME_FROM",
    "code_dir": "HPCLAB_CODE_DIR",
    "attempt_dir": "HPCLAB_ATTEMPT_DIR",
    "metrics_path": "HPCLAB_METRICS_PATH",
    "job_id": "SLURM_JOB_ID",
    "restart_count": "SLURM_RESTART_COUNT",
}


def hpclab_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The hpclab job context. Empty variables become ``None``.

    Keys: run_id, attempt_id, token, output_dir, checkpoint_dir, resume_from, code_dir,
    attempt_dir, metrics_path, job_id, restart_count, seed (int or None), params (dict parsed
    from HPCLAB_PARAMS_JSON, {} if absent/invalid), run_spec_hash (from the attempt's
    hpclab-plan.json when readable, else None), managed (True when running under hpclab).
    """
    env = os.environ if environ is None else environ
    out: dict[str, Any] = {key: (env.get(var) or None) for key, var in _ENV_KEYS.items()}
    out["seed"] = _int_or_none(env.get("HPCLAB_SEED"))
    try:
        params = json.loads(env.get("HPCLAB_PARAMS_JSON") or "{}")
    except ValueError:
        params = {}
    out["params"] = params if isinstance(params, dict) else {}
    out["run_spec_hash"] = _run_spec_hash(out["attempt_dir"])
    out["managed"] = out["run_id"] is not None
    return out


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text) if text else None
    except ValueError:
        return None


def _run_spec_hash(attempt_dir: str | None) -> str | None:
    if not attempt_dir:
        return None
    try:
        with open(os.path.join(attempt_dir, "hpclab-plan.json"), encoding="utf-8") as fh:
            plan = json.load(fh)
        value = plan["execution_plan"]["run_spec_hash"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricsWriter:
    """Append-only writer for ``hpclab.metrics/v1`` JSONL.

    One line per call, flushed immediately so ``tail``/progress estimation sees it while the
    job runs. Non-finite values (NaN/inf) are not valid JSON: they are written as ``null``
    and listed in ``"nonfinite": {"loss": "nan"}`` so a diverged run stays machine-readable.

    By default an existing file is appended to (a *resumed* run continues its log). Pass
    ``overwrite=True`` for a run that starts from scratch: otherwise records of an earlier
    run in the same directory, including its ``final`` record, would be attributed to this one.
    """

    def __init__(self, path: str, overwrite: bool = False) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        mode = "w" if overwrite else "a"
        self._fh = open(path, mode, encoding="utf-8")  # noqa: SIM115 - long-lived handle, closed in close()

    def log(
        self,
        step: int,
        split: str,
        metrics: Mapping[str, float],
        total_steps: int | None = None,
        final: bool = False,
        epoch: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Write one record and return it.

        ``extra`` keys (e.g. dataset, seed, units, higher_is_better) are copied verbatim; the
        final evaluation record should carry them so results can be compared safely.
        """
        clean, nonfinite = _split_nonfinite(metrics)
        record: dict[str, Any] = {"schema": METRICS_SCHEMA, "step": int(step)}
        if total_steps is not None:
            record["total_steps"] = int(total_steps)
        if epoch is not None:
            record["epoch"] = int(epoch)
        record["split"] = split
        if final:
            record["final"] = True
        record["time"] = time.time()
        record["metrics"] = clean
        if nonfinite:
            record["nonfinite"] = nonfinite
        record.update(extra)
        self._fh.write(json.dumps(record, sort_keys=False, allow_nan=False) + "\n")
        self._fh.flush()
        if final:
            os.fsync(self._fh.fileno())
        return record

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _split_nonfinite(metrics: Mapping[str, float]) -> tuple[dict[str, float | None], dict[str, str]]:
    clean: dict[str, float | None] = {}
    nonfinite: dict[str, str] = {}
    for name, value in metrics.items():
        number = float(value)
        if math.isfinite(number):
            clean[name] = number
        else:
            clean[name] = None
            nonfinite[name] = "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")
    return clean, nonfinite


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON via temp file + fsync + rename: readers see the old or the new file, never half."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        _remove_quietly(tmp)
        raise


def _remove_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def write_checkpoint(
    root: str,
    files: Mapping[str, bytes],
    step: int,
    config: Mapping[str, Any],
    semantic_keys: Iterable[str],
    epoch: int | None = None,
    framework: Mapping[str, str] | None = None,
    resumable: bool = True,
    keep_last: int | None = None,
) -> str:
    """Write ``root/ckpt-<step>/`` atomically and return its path.

    Order of operations: files into a hidden staging directory, fsync each, write the manifest
    (with sha256 and size of every file) last, then rename the staging directory into place.
    An existing checkpoint of the same step is moved aside (hidden) first and deleted only
    after the new one is in place, so a crash never leaves a half-deleted "complete" one.
    ``semantic_keys`` names the ``config`` entries that must match for a resume to be valid
    (e.g. lr, batch size, method); hpclab checks them before offering a checkpoint to resume.
    ``keep_last`` (>= 1, counting the new checkpoint) prunes complete checkpoints with a
    *lower* step once the new one is in place; newer ones are never touched.
    """
    keys = list(semantic_keys)
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"semantic_keys not present in config: {missing}")
    for name in files:
        _check_member_name(name)
    os.makedirs(root, exist_ok=True)
    final_dir = os.path.join(root, f"{CHECKPOINT_PREFIX}{int(step):09d}")
    staging = os.path.join(root, f".staging-{int(step):09d}-{os.getpid()}")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    try:
        entries = [_write_member(staging, name, data) for name, data in sorted(files.items())]
        env = hpclab_env()
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "epoch": epoch,
            "complete": True,
            "created_at": time.time(),
            "run_id": env["run_id"],
            "attempt_id": env["attempt_id"],
            "run_spec_hash": env["run_spec_hash"],
            "config": dict(config),
            "semantic_keys": keys,
            "files": entries,
            "framework": dict(framework) if framework else None,
            "resumable": bool(resumable),
        }
        atomic_write_json(os.path.join(staging, MANIFEST_NAME), manifest)
        _install(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # e.g. disk full: do not leave GBs behind
        raise
    if keep_last is not None:
        _prune(root, keep_last, below_step=int(step))
    return final_dir


def latest_checkpoint(root: str) -> str | None:
    """The complete checkpoint with the highest step under ``root`` (or ``root`` itself).

    Incomplete ones (no manifest, unreadable manifest, ``complete`` not true) are ignored.
    File digests are *not* verified here (cheap enough to call every startup); use
    ``read_checkpoint`` to load with verification.
    """
    if not root or not os.path.isdir(root):
        return None
    if _manifest_or_none(root) is not None:
        return root
    best: tuple[int, str] | None = None
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(path):
            continue
        manifest = _manifest_or_none(path)
        if manifest is None:
            continue
        step = manifest["step"]
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None


def read_checkpoint(path: str, verify: bool = True) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Load ``(manifest, {name: bytes})`` from one checkpoint directory.

    Raises ``CheckpointError`` if the manifest is missing or incomplete, or (with ``verify``)
    if any file is missing or its size/sha256 differs from the manifest.
    """
    manifest = _manifest_or_none(path)
    if manifest is None:
        raise CheckpointError(f"{path} is not a complete {CHECKPOINT_SCHEMA} checkpoint")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise CheckpointError(f"{path}: manifest 'files' must be a list")
    blobs: dict[str, bytes] = {}
    for entry in entries:
        name = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            raise CheckpointError(f"{path}: malformed manifest entry {entry!r}")
        try:
            _check_member_name(name)
        except ValueError as exc:
            raise CheckpointError(f"{path}: {exc}") from exc
        try:
            with open(os.path.join(path, name), "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise CheckpointError(f"checkpoint file {name} unreadable: {exc}") from exc
        if verify:
            if len(data) != entry.get("size"):
                raise CheckpointError(
                    "checkpoint file {} has size {}, manifest says {}".format(
                        name, len(data), entry.get("size")
                    )
                )
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            if digest != entry.get("sha256"):
                raise CheckpointError(f"checkpoint file {name} digest mismatch")
        blobs[name] = data
    return manifest, blobs


def check_compatible(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Semantic keys whose value differs between a checkpoint and the current config."""
    saved = manifest.get("config") or {}
    return [k for k in manifest.get("semantic_keys", []) if saved.get(k) != config.get(k)]


def _write_member(staging: str, name: str, data: bytes) -> dict[str, Any]:
    target = os.path.join(staging, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return {"path": name, "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}


def _check_member_name(name: str) -> None:
    parts = name.replace("\\", "/").split("/")
    if not name or name.startswith("/") or ".." in parts or name == MANIFEST_NAME:
        raise ValueError(f"invalid checkpoint member name {name!r}")


def _manifest_or_none(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, MANIFEST_NAME), encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("complete") is not True
        or not _is_int(manifest.get("step"))
    ):
        return None
    return manifest


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _install(staging: str, final_dir: str) -> None:
    """Rename ``staging`` to ``final_dir``; an existing ``final_dir`` is replaced safely.

    The old directory is renamed to a hidden name first (restored if the install fails) and
    deleted only afterwards: hidden directories are never offered by ``latest_checkpoint``,
    so a job killed mid-delete cannot leave a visible, manifest-complete, half-empty checkpoint.
    """
    root = os.path.dirname(final_dir)
    aside = None
    if os.path.lexists(final_dir):
        aside = os.path.join(root, f".replaced-{os.path.basename(final_dir)}-{os.getpid()}")
        os.rename(final_dir, aside)
    try:
        os.rename(staging, final_dir)
    except BaseException:
        if aside is not None:
            os.rename(aside, final_dir)
        raise
    _fsync_dir(root)
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)


def _prune(root: str, keep_last: int, below_step: int) -> None:
    """Delete all but the newest ``keep_last - 1`` complete checkpoints older than ``below_step``."""
    older = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        manifest = _manifest_or_none(path) if name.startswith(CHECKPOINT_PREFIX) else None
        if manifest is not None and manifest["step"] < below_step:
            older.append((manifest["step"], path))
    older.sort()
    for _step, path in older[: max(0, len(older) - (max(1, keep_last) - 1))]:
        _discard(path)


def _discard(path: str) -> None:
    """Hide a checkpoint (one rename) before deleting it, for the reason given in ``_install``."""
    hidden = os.path.join(os.path.dirname(path), f".discard-{os.path.basename(path)}-{os.getpid()}")
    try:
        os.rename(path, hidden)
    except OSError:
        return
    shutil.rmtree(hidden, ignore_errors=True)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Graceful stop
# ---------------------------------------------------------------------------


class GracefulStop:
    """Record SIGUSR1/SIGTERM instead of dying mid-step.

    Slurm sends SIGTERM on scancel/preemption and (with ``--signal=USR1@<lead>``) SIGUSR1
    shortly before the time limit. Poll ``requested`` once per step; when set, write a
    checkpoint and exit with ``exit_code``.

    If the wrapper exports ``HPCLAB_STOP_FILE``, the presence of that file also counts as a
    stop request: the wrapper touches it when *it* receives the warning signal, before
    forwarding the signal, so a program whose signal got lost (e.g. behind a launcher) still
    stops cleanly. Either way the stop is reported as 128 + the signal number; for the file
    that is the signal it names (e.g. ``USR1``), else SIGUSR1, Slurm's conventional warning.
    """

    _DEFAULT_STOP_FILE_SIGNAL = "SIGUSR1"

    def __init__(self, signals: Iterable[str] = ("SIGUSR1", "SIGTERM")) -> None:
        self._signal_name: str | None = None
        self._stop_file_seen = False
        self._stop_file = os.environ.get("HPCLAB_STOP_FILE") or None
        for name in signals:
            signum = getattr(signal, name, None)
            if signum is not None:
                signal.signal(signum, self._handle)

    def _handle(self, signum: int, _frame: Any) -> None:
        self._signal_name = signal.Signals(signum).name

    @property
    def signal_name(self) -> str | None:
        """The signal received (e.g. ``SIGTERM``), ``"STOP_FILE"``, or None if not stopped."""
        if self._signal_name is not None:
            return self._signal_name
        return "STOP_FILE" if self._stop_file_seen else None

    @property
    def requested(self) -> bool:
        if not self._stop_file_seen and self._stop_file and os.path.exists(self._stop_file):
            self._stop_file_seen = True
        return self._signal_name is not None or self._stop_file_seen

    @property
    def exit_code(self) -> int:
        """128 + signal number, the shell convention for 'terminated by signal'.

        A real signal wins over the stop file (the forwarded signal usually lands while the
        checkpoint is written); 1 only if nothing was requested.
        """
        name = self._signal_name
        if name is None and self._stop_file_seen:
            name = self._stop_file_signal()
        signum = getattr(signal, name or "", None)
        return 128 + int(signum) if isinstance(signum, signal.Signals) else 1

    def _stop_file_signal(self) -> str:
        try:
            with open(self._stop_file or "", encoding="utf-8", errors="replace") as fh:
                word = fh.read(32).strip().upper()
        except OSError:
            word = ""
        if word.isdigit():
            try:
                return signal.Signals(int(word)).name
            except ValueError:
                return self._DEFAULT_STOP_FILE_SIGNAL
        name = word if word.startswith("SIG") else "SIG" + word
        known = word and isinstance(getattr(signal, name, None), signal.Signals)
        return name if known else self._DEFAULT_STOP_FILE_SIGNAL
