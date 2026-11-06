"""Training progress from a metrics log: step, rate, ETA, walltime risk and a stall *signal*.

Pure and deterministic (the caller passes ``now_epoch``). ``run_inspect`` / ``runs_list`` call it
over a bounded tail (<= 16 KiB) of ``$HPCLAB_METRICS_PATH`` (``hpclab.metrics/v1``, design §6.3);
generic JSONL from other loggers works too (``global_step``/``iteration``/``iter``/``step``,
``time``/``timestamp``/``wall_time``, ``total_steps``/``max_steps``).

Contracts:

* Never guesses: without enough data the numbers are ``None`` and ``timelimit_risk`` is
  ``"unknown"``. Partial last lines (the file is being appended), non-JSON lines, NaN/Inf values
  and step resets (a resumed run restarting its counter) are tolerated and noted.
* The rate comes from a recent window (last 20 timed records within 10 minutes of the newest,
  at least two points), so warm-up and compilation do not dominate it.
* ``stalled_suspect`` is a signal only: the newest record is older than
  ``max(10 min, 5 x median record interval)``. Evaluation and checkpointing legitimately pause
  logging; this must never be used as a reason to cancel a job automatically.
* Only the current segment counts: after the step counter goes backwards (a restart in the same
  file) the rate, ``final_seen``, ``epoch`` and ``latest_metrics`` come from the records after
  the reset, so an earlier run's final record never makes a restarted run look finished.
* Times: epoch seconds/milliseconds and ISO-8601 with an offset are absolute; ISO-8601 *without*
  an offset (``datetime.now().isoformat()`` on a compute node) is local time of an unknown zone,
  so it drives the rate but not the record age or the stall signal.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.domain.clock import format_slurm_duration

__all__ = [
    "METRICS_SCHEMA",
    "Progress",
    "Risk",
    "estimate_progress",
]

METRICS_SCHEMA = "hpclab.metrics/v1"
Risk = Literal["none", "low", "medium", "high", "unknown"]
MetricValue = float | int | str

RATE_WINDOW_RECORDS = 20
RATE_WINDOW_S = 600.0
STALL_FLOOR_S = 600.0
STALL_MEDIAN_FACTOR = 5.0
# eta / remaining walltime -> risk (upper bounds, exclusive)
RISK_NONE_BELOW = 0.75
RISK_LOW_BELOW = 0.90
RISK_MEDIUM_BELOW = 1.0
MAX_LINES = 20_000
MAX_METRICS = 50
MAX_NOTES = 10

_HPCLAB_STEP_KEYS = ("step", "global_step", "iteration", "iter")
_GENERIC_STEP_KEYS = ("global_step", "iteration", "iter", "step", "trainer/global_step")
_TIME_KEYS = ("time", "timestamp", "wall_time", "walltime", "_timestamp")
_TOTAL_KEYS = ("total_steps", "max_steps", "num_training_steps")
_RESERVED = frozenset(
    {
        *_GENERIC_STEP_KEYS,
        *_TIME_KEYS,
        *_TOTAL_KEYS,
        "schema",
        "epoch",
        "split",
        "final",
        "dataset",
        "seed",
        "units",
        "higher_is_better",
        "metrics",
        "nonfinite",
        "run_id",
        "attempt_id",
        "_step",
        "_runtime",
    }
)
# Plausible epoch seconds: 1973-03 .. 5138; larger values are treated as epoch milliseconds.
_EPOCH_MIN = 1e8
_EPOCH_MAX = 1e11
# Step/total counters above this are garbage (and would overflow float arithmetic).
_MAX_COUNT = 2**53
# Without timestamps the job's elapsed time only gives a rate when the log shows the run's start.
_START_VISIBLE_FRACTION = 0.05


class Progress(BaseModel):
    """Progress estimate for one Attempt; JSON-safe (non-finite metrics become strings)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int | None = None
    total_steps: int | None = None
    fraction: float | None = None
    rate_steps_per_s: float | None = None
    eta_s: int | None = None
    elapsed_s: int | None = None
    remaining_walltime_s: int | None = None
    timelimit_risk: Risk = "unknown"
    stalled_suspect: bool = False
    last_record_age_s: int | None = None
    latest_metrics: dict[str, MetricValue] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    epoch: float | None = None
    records: int = 0
    skipped_lines: int = 0
    final_seen: bool = False
    rate_basis: Literal["records", "elapsed", "none"] = "none"
    last_record_time: float | None = None

    def short_summary(self) -> str:
        """One line for listings, e.g. ``step 4100/10000 (41%) · 2.70 steps/s · ETA 00:36:00 · risk low``."""
        if self.step is None:
            return "no progress records yet"
        parts = [f"step {self.step}" + (f"/{self.total_steps}" if self.total_steps else "")]
        if self.fraction is not None:
            parts[0] += f" ({self.fraction:.0%})"
        if self.rate_steps_per_s is not None:
            parts.append(f"{self.rate_steps_per_s:.3g} steps/s")
        if self.eta_s is not None:
            parts.append(f"ETA {format_slurm_duration(self.eta_s)}")
        parts.append(f"risk {self.timelimit_risk}")
        if self.stalled_suspect:
            parts.append("no new records recently")
        return " · ".join(parts)


@dataclass(frozen=True)
class _Record:
    step: int | None
    time: float | None  # epoch seconds (for naive ISO: as if UTC, only differences are meaningful)
    time_absolute: bool  # False for ISO-8601 without an offset
    total: int | None
    epoch: float | None
    split: str | None
    final: bool
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and 0 <= value <= _MAX_COUNT:
        return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _as_epoch(value: Any) -> tuple[float, bool] | None:
    """(epoch seconds, absolute) from a number (s or ms) or an ISO-8601 string; None if implausible.

    ISO strings without an offset are read as UTC but flagged ``absolute=False``.
    """
    absolute = True
    if isinstance(value, str):
        text = value.strip()
        if text[-1:] in ("Z", "z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt, absolute = dt.replace(tzinfo=UTC), False
        t: float | None = dt.timestamp()
    else:
        t = _as_float(value)
        if t is not None and t > _EPOCH_MAX:
            t /= 1000.0
    if t is None or not _EPOCH_MIN <= t <= _EPOCH_MAX:
        return None
    return t, absolute


def _first(record: dict[str, Any], keys: tuple[str, ...], conv: Any) -> Any:
    for key in keys:
        if key in record:
            value = conv(record[key])
            if value is not None:
                return value
    return None


_NONFINITE = {"nan": math.nan, "inf": math.inf, "+inf": math.inf, "-inf": -math.inf}


def _metric_items(record: dict[str, Any], hpclab: bool) -> dict[str, Any]:
    metrics = record.get("metrics")
    out: dict[str, Any] = {}
    if isinstance(metrics, dict):
        out.update({k: v for k, v in metrics.items() if isinstance(k, str)})
    if not hpclab:
        out.update({k: v for k, v in record.items() if k not in _RESERVED})
    # hpclab_io.MetricsWriter writes NaN/Inf as null plus {"nonfinite": {"loss": "nan"}}
    nonfinite = record.get("nonfinite")
    if isinstance(nonfinite, dict):
        for name, kind in nonfinite.items():
            if isinstance(name, str) and isinstance(kind, str) and kind.lower() in _NONFINITE:
                out[name] = _NONFINITE[kind.lower()]
    return {k: v for k, v in out.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def _parse_record(obj: dict[str, Any]) -> _Record:
    hpclab = obj.get("schema") == METRICS_SCHEMA
    split = obj.get("split")
    when: tuple[float, bool] | None = _first(obj, _TIME_KEYS, _as_epoch)
    return _Record(
        step=_first(obj, _HPCLAB_STEP_KEYS if hpclab else _GENERIC_STEP_KEYS, _as_int),
        time=when[0] if when else None,
        time_absolute=when[1] if when else False,
        total=_first(obj, _TOTAL_KEYS, _as_int),
        epoch=_as_float(obj.get("epoch")),
        split=split if isinstance(split, str) and split else None,
        final=obj.get("final") is True,
        metrics=_metric_items(obj, hpclab),
    )


def _parse_lines(metrics_lines: list[str]) -> tuple[list[_Record], int, bool, bool]:
    """(records, skipped lines, last line partial, saw non-epoch time values)."""
    raw: list[str] = []
    for chunk in metrics_lines:
        raw.extend(chunk.splitlines() if "\n" in chunk or "\r" in chunk else [chunk])
    lines = [line.strip() for line in raw[-MAX_LINES:]]
    lines = [line for line in lines if line]
    records: list[_Record] = []
    skipped = 0
    partial_last = False
    bad_time = False
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            skipped += 1
            partial_last = partial_last or (i == len(lines) - 1 and line.startswith("{"))
            continue
        if not isinstance(obj, dict):
            skipped += 1
            continue
        rec = _parse_record(obj)
        if rec.time is None and any(k in obj for k in _TIME_KEYS):
            bad_time = True
        records.append(rec)
    return records, skipped, partial_last, bad_time


def _json_safe(value: Any) -> MetricValue:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _latest_metrics(records: list[_Record]) -> tuple[dict[str, MetricValue], list[str]]:
    last_seen: dict[str, tuple[int, Any]] = {}
    for idx, rec in enumerate(records):
        for name, value in rec.metrics.items():
            key = f"{rec.split}/{name}" if rec.split else name
            last_seen[key] = (idx, value)
    newest = sorted(last_seen.items(), key=lambda kv: kv[1][0], reverse=True)[:MAX_METRICS]
    out = {k: _json_safe(v) for k, (_, v) in sorted(newest, key=lambda kv: kv[1][0])}
    nonfinite = [k for k, v in out.items() if isinstance(v, str)]
    return out, nonfinite


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def _current_segment(records: list[_Record]) -> tuple[list[_Record], bool]:
    """Records after the last step reset (the counter going backwards), and whether one was seen."""
    start, prev = 0, None
    for i, r in enumerate(records):
        if r.step is None:
            continue
        if prev is not None and r.step < prev:
            start = i
        prev = r.step
    return records[start:], start > 0


def _timed_points(records: list[_Record]) -> list[tuple[float, int]]:
    return [(r.time, r.step) for r in records if r.time is not None and r.step is not None]


def _window_rate(points: list[tuple[float, int]]) -> float | None:
    window = points[-RATE_WINDOW_RECORDS:]
    if len(window) < 2:
        return None
    newest = window[-1][0]
    recent = [p for p in window if p[0] >= newest - RATE_WINDOW_S]
    if len(recent) < 2:
        recent = window[-2:]
    dt = recent[-1][0] - recent[0][0]
    ds = recent[-1][1] - recent[0][1]
    if dt <= 0 or ds <= 0:
        return None
    return ds / dt


def _elapsed_rate(first: int, step: int, elapsed_s: int) -> float | None:
    """Fallback when records carry no usable times: step / job elapsed time.

    Only valid when the records reach back to the start of the attempt. A first visible step
    far from 0 is either a resumed run or (the usual case) a bounded log tail of a long run;
    the two cannot be told apart, and assuming either makes the rate wrong by a large factor.
    """
    if elapsed_s <= 0 or step <= 0 or first > _START_VISIBLE_FRACTION * step:
        return None
    return step / elapsed_s


def _risk(eta_s: float | None, remaining_s: int | None) -> Risk:
    if eta_s is None or remaining_s is None:
        return "unknown"
    if remaining_s <= 0:
        return "high" if eta_s > 0 else "none"
    ratio = eta_s / remaining_s
    if ratio < RISK_NONE_BELOW:
        return "none"
    if ratio < RISK_LOW_BELOW:
        return "low"
    if ratio < RISK_MEDIUM_BELOW:
        return "medium"
    return "high"


def _stall(
    points_times: list[float], last_time: float | None, now_epoch: float | None, notes: list[str]
) -> tuple[bool, int | None]:
    if last_time is None or now_epoch is None:
        return False, None
    age = now_epoch - last_time
    if age < -60:
        notes.append(
            "Newest metrics record is from the future: clocks of the compute node and this host disagree."
        )
    age = max(0.0, age)
    recent = points_times[-50:]
    gaps = [b - a for a, b in itertools.pairwise(recent) if b > a]
    median_gap = _median(gaps)
    threshold = max(STALL_FLOOR_S, STALL_MEDIAN_FACTOR * (median_gap or 0.0))
    stalled = age > threshold
    if stalled:
        note = (
            f"No new metrics for {format_slurm_duration(int(age))} (threshold {format_slurm_duration(int(threshold))}). "
            "This is only a signal: check GPU utilisation and log activity before acting; evaluation and "
            "checkpointing can pause logging."
        )
        if gaps and max(gaps) >= age:
            note += " A gap this long already happened earlier in this run."
        notes.append(note)
    return stalled, int(age)


def _newest_absolute_time(records: list[_Record], notes: list[str]) -> float | None:
    """Epoch of the newest timed record, or None when that time has no time zone (its age
    relative to ``now`` would be off by the node's UTC offset)."""
    newest = next((r for r in reversed(records) if r.time is not None), None)
    if newest is None:
        return None
    if not newest.time_absolute:
        notes.append(
            "Record timestamps have no time zone: they give the rate, but the record age and the stall "
            "signal are not computed. Log epoch seconds (time.time()) or ISO-8601 with an offset."
        )
        return None
    return newest.time


def _nonneg(value: int | None) -> int | None:
    return value if value is not None and value >= 0 else None


def estimate_progress(
    metrics_lines: list[str],
    *,
    elapsed_s: int | None,
    time_limit_s: int | None,
    total_steps_hint: int | None = None,
    now_epoch: float | None = None,
) -> Progress:
    """Estimate progress from metrics lines (oldest first) and the job's elapsed/limit seconds.

    ``elapsed_s`` is the job's elapsed time as of ``now_epoch``; ``time_limit_s`` its walltime
    limit. ``total_steps_hint`` (e.g. from the run's params) is used only when the records carry
    no ``total_steps``.
    """
    elapsed_s, time_limit_s = _nonneg(elapsed_s), _nonneg(time_limit_s)
    records, skipped, partial_last, bad_time = _parse_lines(metrics_lines)
    notes: list[str] = []
    if partial_last:
        notes.append("The last line is incomplete (still being written) and was ignored.")
    if bad_time:
        notes.append("Some record times are not epoch seconds or ISO-8601 and were ignored.")
    remaining = (
        max(0, time_limit_s - elapsed_s) if elapsed_s is not None and time_limit_s is not None else None
    )
    steps = [r.step for r in records if r.step is not None]
    if not steps:
        if records:
            notes.append("Records carry no step counter (step/global_step/iteration/iter).")
        latest, _ = _latest_metrics(records)
        return Progress(
            elapsed_s=elapsed_s,
            remaining_walltime_s=remaining,
            latest_metrics=latest,
            notes=notes[:MAX_NOTES],
            records=len(records),
            skipped_lines=skipped,
            final_seen=any(r.final for r in records),
        )

    segment, reset = _current_segment(records)
    if reset:
        notes.append(
            "The step counter went backwards (restart/resume); only records after the reset are used."
        )
    seg_steps = [r.step for r in segment if r.step is not None]
    step = seg_steps[-1]
    total = next((r.total for r in reversed(segment) if r.total is not None), None)
    if total is None:  # a constant of the run: an older segment may be the only one that states it
        total = next((r.total for r in reversed(records) if r.total is not None), None)
    if total is None:
        total = total_steps_hint if total_steps_hint and 0 < total_steps_hint <= _MAX_COUNT else None
    elif total_steps_hint and total_steps_hint != total:
        notes.append(
            f"total_steps in the records ({total}) differs from the hint ({total_steps_hint}); using the records."
        )
    final_seen = any(r.final for r in segment)
    fraction = min(1.0, step / total) if total else None
    done = final_seen or (fraction is not None and fraction >= 1.0)

    points = _timed_points(segment)
    rate = _window_rate(points)
    basis: Literal["records", "elapsed", "none"] = "records" if rate is not None else "none"
    if rate is None and len(points) < 2 and elapsed_s:
        rate = _elapsed_rate(seg_steps[0], step, elapsed_s)
        if rate is not None:
            basis = "elapsed"
            notes.append("Rate estimated from job elapsed time (records have too few timestamps).")
        elif seg_steps[0] > _START_VISIBLE_FRACTION * step:
            notes.append(
                "Records carry no timestamps and do not reach back to the start of this attempt (a log tail or a "
                "resumed run), so the rate is unknown."
            )
    last_time = _newest_absolute_time(segment, notes)
    stalled, age = _stall([p[0] for p in points], last_time, now_epoch, notes) if not done else (False, None)
    if done and last_time is not None and now_epoch is not None:
        age = int(max(0.0, now_epoch - last_time))

    eta: float | None = None
    if done:
        eta = 0.0
    elif total is not None and rate:
        eta = max(0, total - step) / rate
        if age is not None and not stalled:
            eta = max(0.0, eta - age)  # progress likely continued since the newest record
    risk: Risk = "none" if done else _risk(eta, remaining)
    if not done and fraction is not None and fraction < 0.02 and eta is not None:
        notes.append("Very early in training: the ETA may still include warm-up and compilation.")
    if risk == "high" and not done:
        notes.append(
            "At the current rate the run will not finish before its time limit: make sure it checkpoints "
            "(and handles the pre-timeout signal) so it can be resumed."
        )
    latest, nonfinite = _latest_metrics(segment)
    if nonfinite:
        notes.append(f"Non-finite latest values: {', '.join(nonfinite[:5])}.")
    epoch = next((r.epoch for r in reversed(segment) if r.epoch is not None), None)
    return Progress(
        step=step,
        total_steps=total,
        fraction=round(fraction, 6) if fraction is not None else None,
        rate_steps_per_s=float(f"{rate:.6g}") if rate is not None else None,
        eta_s=math.ceil(eta) if eta is not None else None,
        elapsed_s=elapsed_s,
        remaining_walltime_s=remaining,
        timelimit_risk=risk,
        stalled_suspect=stalled,
        last_record_age_s=age,
        latest_metrics=latest,
        notes=notes[:MAX_NOTES],
        epoch=epoch,
        records=len(records),
        skipped_lines=skipped,
        final_seen=final_seen,
        rate_basis=basis,
        last_record_time=last_time,
    )
