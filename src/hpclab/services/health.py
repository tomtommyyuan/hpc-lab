"""Advisory health of a running attempt from the wrapper's telemetry sidecar (pure).

Inputs are bounded tails of ``<attempt>/telemetry/gpu.csv`` (nvidia-smi samples written by the
job wrapper every ~60 s: ``timestamp, index, uuid, utilization.gpu, memory.used, memory.total,
power.draw``) and ``<attempt>/telemetry/heartbeat`` (one epoch-seconds line per interval), plus
the mtimes of the stdout log and the metrics file.

Contracts:

* Advisory only: flags are *signals* for a human or agent to look at, never a reason to cancel
  a job automatically (evaluation and checkpointing legitimately pause logging and GPUs).
* Missing telemetry is ``source='none'`` with ``None`` numbers - never "0 % utilization". CPU-only
  attempts have no ``gpu.csv`` at all (the wrapper does not sample GPUs it does not own).
* Only the recent window counts (the last ``WINDOW_SAMPLES`` rows per GPU), so a slow start does
  not keep a busy job flagged as idle.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HEALTH_FLAGS",
    "Health",
    "HealthFlag",
    "flag_char",
    "health_from",
]

HealthFlag = Literal[
    "idle_allocation",
    "partial_gpu_use",
    "stalled_candidate",
    "heartbeat_frozen",
    "near_gpu_oom",
]
HEALTH_FLAGS: tuple[str, ...] = (
    "heartbeat_frozen",
    "near_gpu_oom",
    "idle_allocation",
    "stalled_candidate",
    "partial_gpu_use",
)

WINDOW_SAMPLES = 10  # per GPU: ~10 minutes at the wrapper's default 60 s interval
MIN_SAMPLES_FOR_IDLE = 3
IDLE_UTIL_PERCENT = 5.0
BUSY_UTIL_PERCENT = 20.0
NEAR_OOM_FRACTION = 0.95
HEARTBEAT_FROZEN_S = 300  # 5 missed beats at the default interval
STALL_S = 1800  # no log AND no metrics output for 30 min while the heartbeat is alive


class Health(BaseModel):
    """Derived, advisory view; ``source='none'`` means no GPU telemetry (numbers are None)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["telemetry", "none"] = "none"
    per_gpu_util_mean: dict[str, float] = Field(default_factory=dict)  # GPU index -> mean util %
    idle_gpus: int | None = None
    gpus_reporting: int | None = None
    gpus_allocated: int | None = None
    peak_gpu_mem_gib: float | None = None
    samples: int = 0
    heartbeat_age_s: int | None = None
    log_age_s: int | None = None
    metrics_age_s: int | None = None
    flags: list[HealthFlag] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    as_of: float | None = None  # epoch seconds the view was computed at

    def summary(self) -> str:
        if self.source == "none" and not self.flags:
            return "no GPU telemetry" + (
                f"; heartbeat {self.heartbeat_age_s}s ago" if self.heartbeat_age_s is not None else ""
            )
        parts = []
        if self.per_gpu_util_mean:
            mean = sum(self.per_gpu_util_mean.values()) / len(self.per_gpu_util_mean)
            parts.append(f"GPU util {mean:.0f}% over {len(self.per_gpu_util_mean)} GPU(s)")
        if self.peak_gpu_mem_gib is not None:
            parts.append(f"peak mem {self.peak_gpu_mem_gib:.1f} GiB")
        if self.flags:
            parts.append("flags: " + ", ".join(self.flags))
        return "; ".join(parts) or "ok"


_FLAG_CHARS = {
    "heartbeat_frozen": "H",
    "near_gpu_oom": "M",
    "idle_allocation": "I",
    "stalled_candidate": "S",
    "partial_gpu_use": "P",
}


def flag_char(health: Health | None) -> str:
    """One character for listings: ``-`` no data, ``.`` ok, else the most severe flag
    (H heartbeat frozen, M near GPU OOM, I idle allocation, S stalled, P partial GPU use)."""
    if health is None:
        return "-"
    for flag in HEALTH_FLAGS:
        if flag in health.flags:
            return _FLAG_CHARS[flag]
    if health.source == "none" and health.heartbeat_age_s is None:
        return "-"
    return "."


def _num(text: str) -> float | None:
    t = text.strip()
    if not t or t.startswith("[") or t.upper() in ("N/A", "ERR!"):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _parse_rows(csv_tail: str) -> dict[str, list[tuple[float | None, float | None, float | None]]]:
    """GPU index -> [(util, mem_used_mib, mem_total_mib)] in file order; malformed lines skipped
    (the first line of a tail window is usually cut)."""
    rows: dict[str, list[tuple[float | None, float | None, float | None]]] = {}
    for line in csv_tail.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        index = cols[1]
        if not index.isdigit():
            continue
        util, used, total = _num(cols[3]), _num(cols[4]), _num(cols[5])
        if util is None and used is None:
            continue
        rows.setdefault(index, []).append((util, used, total))
    return rows


def _last_epoch(heartbeat_tail: str) -> float | None:
    for line in reversed(heartbeat_tail.splitlines()):
        v = _num(line)
        if v is not None and v > 1e8:
            return v
    return None


def _age(now: float, then: float | None) -> int | None:
    if then is None:
        return None
    return max(0, int(now - then))


def health_from(
    telemetry_csv_tail: str,
    heartbeat_tail: str,
    *,
    now: float,
    gpus_allocated: int | None,
    log_mtime: float | None,
    metrics_mtime: float | None,
) -> Health:
    """Compute the advisory ``Health`` of a running attempt (``now`` and mtimes: epoch seconds)."""
    notes: list[str] = []
    flags: list[HealthFlag] = []
    rows = _parse_rows(telemetry_csv_tail or "")
    heartbeat_age = _age(now, _last_epoch(heartbeat_tail or ""))
    log_age = _age(now, log_mtime)
    metrics_age = _age(now, metrics_mtime)

    per_gpu: dict[str, float] = {}
    peak_mib: float | None = None
    near_oom = False
    samples = 0
    for index, series in sorted(rows.items(), key=lambda kv: int(kv[0])):
        window = series[-WINDOW_SAMPLES:]
        utils = [u for u, _, _ in window if u is not None]
        samples += len(window)
        if utils:
            per_gpu[index] = round(sum(utils) / len(utils), 1)
        for _, used, total in window:
            if used is not None:
                peak_mib = used if peak_mib is None else max(peak_mib, used)
                if total and used / total >= NEAR_OOM_FRACTION:
                    near_oom = True

    source: Literal["telemetry", "none"] = "telemetry" if per_gpu else "none"
    idle_gpus: int | None = None
    if source == "telemetry":
        enough = all(len(rows[i][-WINDOW_SAMPLES:]) >= MIN_SAMPLES_FOR_IDLE for i in per_gpu)
        idle = [i for i, u in per_gpu.items() if u < IDLE_UTIL_PERCENT]
        idle_gpus = len(idle)
        busy = [i for i, u in per_gpu.items() if u >= BUSY_UTIL_PERCENT]
        if enough and idle and len(idle) == len(per_gpu):
            flags.append("idle_allocation")
        elif enough and idle and busy:
            flags.append("partial_gpu_use")
        elif gpus_allocated and len(per_gpu) < gpus_allocated:
            notes.append(f"telemetry shows {len(per_gpu)} of {gpus_allocated} allocated GPU(s)")
        if not enough:
            notes.append("fewer than 3 samples per GPU: idle/partial flags not evaluated yet")
        if near_oom:
            flags.append("near_gpu_oom")
    elif gpus_allocated:
        notes.append("no GPU telemetry yet (sidecar samples every ~60 s)")

    if heartbeat_age is not None and heartbeat_age > HEARTBEAT_FROZEN_S:
        flags.append("heartbeat_frozen")
    output_ages = [a for a in (log_age, metrics_age) if a is not None]
    heartbeat_alive = heartbeat_age is not None and heartbeat_age <= HEARTBEAT_FROZEN_S
    if output_ages and min(output_ages) > STALL_S and heartbeat_alive:
        flags.append("stalled_candidate")
        notes.append(
            "no new log or metrics output for a while; may be evaluation/checkpointing (signal only)"
        )

    return Health(
        source=source,
        per_gpu_util_mean=per_gpu,
        idle_gpus=idle_gpus,
        gpus_reporting=len(per_gpu) if source == "telemetry" else None,
        gpus_allocated=gpus_allocated,
        peak_gpu_mem_gib=round(peak_mib / 1024.0, 2) if peak_mib is not None else None,
        samples=samples,
        heartbeat_age_s=heartbeat_age,
        log_age_s=log_age,
        metrics_age_s=metrics_age,
        flags=flags,
        notes=notes,
        as_of=now,
    )
