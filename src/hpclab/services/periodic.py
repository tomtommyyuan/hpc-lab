"""Periodic background tasks run by the controller.

Service modules expose ``periodic_tasks() -> list[PeriodicTask]`` (cluster polling, reconcile
sweeps, notification flushes, ...). The controller feeds them to a ``PeriodicScheduler`` and
runs whatever is due on its own threads.

``PeriodicScheduler`` is a pure, thread-safe *timing table*: it never runs anything itself.
The caller asks ``due(now)``, calls ``mark_started`` before running a task and ``mark_ran``
when it finished. That keeps every timing rule unit-testable with a ``FakeClock``.

Timing rules
------------
* ``now`` is **monotonic** seconds (``clock.monotonic()``), immune to wall-clock changes.
* First run: at scheduler creation when ``run_at_start`` (the default), else one interval
  (+ jitter) later.
* Next run: ``finished + interval + uniform(0, jitter_s)`` — a fixed delay measured from
  completion, so a slow task never piles up behind itself. A task that is running is never
  returned by ``due`` (no overlapping runs of the same task).
* Consecutive failures back off: ``interval × 2^min(failures, 3)``, capped at
  ``max(interval, 300 s)``. A failing task keeps being retried; it is never dropped.
* ``force_due`` makes tasks due immediately (``controller.wake``, wake-from-sleep catch-up).
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hpclab.domain.clock import Clock, SystemClock, iso

if TYPE_CHECKING:
    from hpclab.services.context import ServiceContext

MAX_FAILURE_BACKOFF_S = 300.0
_MAX_BACKOFF_EXPONENT = 3


@dataclass(frozen=True)
class PeriodicTask:
    """A named function the controller runs every ``interval_s`` seconds.

    ``fn`` receives the controller's ``ServiceContext``; it must be safe to run concurrently with
    request handling and outbox workers (use ``ctx.db.tx()`` for writes, never hold a transaction
    across network I/O). Exceptions are logged by the controller and trigger backoff.
    """

    name: str
    interval_s: float
    fn: Callable[[ServiceContext], None]
    jitter_s: float = 0.0
    run_at_start: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PeriodicTask.name must be non-empty")
        # finite: an infinite interval/jitter would make the task silently never run again
        if not (math.isfinite(self.interval_s) and self.interval_s > 0):
            raise ValueError(f"PeriodicTask {self.name!r}: interval_s must be a finite number > 0")
        if not (math.isfinite(self.jitter_s) and self.jitter_s >= 0):
            raise ValueError(f"PeriodicTask {self.name!r}: jitter_s must be a finite number >= 0")

    @classmethod
    def coerce(cls, obj: Any) -> PeriodicTask:
        """Accept a ``PeriodicTask``, any object with ``name/interval_s/fn`` attributes (e.g. the
        NamedTuple stand-ins services used before this module existed) or a
        ``(name, interval_s, fn)`` tuple."""
        if isinstance(obj, cls):
            return obj
        if hasattr(obj, "name") and hasattr(obj, "interval_s") and hasattr(obj, "fn"):
            return cls(
                name=str(obj.name),
                interval_s=float(obj.interval_s),
                fn=obj.fn,
                jitter_s=float(getattr(obj, "jitter_s", 0.0) or 0.0),
                run_at_start=bool(getattr(obj, "run_at_start", True)),
            )
        if isinstance(obj, tuple) and len(obj) == 3:
            name, interval, fn = obj
            return cls(name=str(name), interval_s=float(interval), fn=fn)
        raise TypeError(f"cannot interpret {obj!r} as a PeriodicTask")


@dataclass
class _TaskState:
    task: PeriodicTask
    next_due: float
    running: bool = False
    started_mono: float | None = None
    runs: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_ok_at: str | None = None
    last_error: str | None = None
    last_duration_s: float | None = None


class PeriodicScheduler:
    """Thread-safe due-time bookkeeping for a fixed set of ``PeriodicTask``s."""

    def __init__(
        self,
        tasks: Iterable[Any],
        clock: Clock | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.clock: Clock = clock or SystemClock()
        self._rng = rng or random.Random()  # noqa: S311 - jitter only, not security relevant
        self._lock = threading.Lock()
        self._states: dict[str, _TaskState] = {}
        start = self.clock.monotonic()
        for raw in tasks:
            task = PeriodicTask.coerce(raw)
            if task.name in self._states:
                raise ValueError(f"duplicate periodic task name {task.name!r}")
            first = start if task.run_at_start else start + task.interval_s + self._jitter(task)
            self._states[task.name] = _TaskState(task=task, next_due=first)

    # -- queries -----------------------------------------------------------------
    @property
    def tasks(self) -> list[PeriodicTask]:
        with self._lock:
            return [s.task for s in self._states.values()]

    def due(self, now: float | None = None) -> list[PeriodicTask]:
        """Tasks whose due time has passed and that are not currently running,
        most overdue first. Pure: does not change any state."""
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            ready = [s for s in self._states.values() if not s.running and s.next_due <= t]
            ready.sort(key=lambda s: (s.next_due, s.task.name))
            return [s.task for s in ready]

    def next_due_in(self, now: float | None = None) -> float | None:
        """Seconds until the earliest non-running task is due (0 if overdue); None if every
        task is running or there are no tasks."""
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            pending = [s.next_due for s in self._states.values() if not s.running]
        if not pending:
            return None
        return max(0.0, min(pending) - t)

    def last_error(self, name: str) -> str | None:
        with self._lock:
            return self._state(name).last_error

    # -- transitions ---------------------------------------------------------------
    def mark_started(self, name: str, now: float | None = None) -> None:
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            st = self._state(name)
            st.running = True
            st.started_mono = t
            st.last_started_at = iso(self.clock.now())

    def mark_ran(
        self,
        name: str,
        now: float | None = None,
        ok: bool = True,
        error: str | None = None,
        *,
        duration_s: float | None = None,
    ) -> None:
        """Record completion at monotonic time ``now`` and schedule the next run."""
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            st = self._state(name)
            if duration_s is None and st.started_mono is not None:
                duration_s = max(0.0, t - st.started_mono)
            st.running = False
            st.started_mono = None
            st.runs += 1
            st.last_duration_s = duration_s
            wall = iso(self.clock.now())
            st.last_finished_at = wall
            interval = st.task.interval_s
            if ok:
                st.consecutive_failures = 0
                st.last_ok_at = wall
                st.last_error = None
            else:
                st.failures += 1
                st.consecutive_failures += 1
                st.last_error = error or "failed"
                factor = 2 ** min(st.consecutive_failures, _MAX_BACKOFF_EXPONENT)
                interval = min(interval * factor, max(interval, MAX_FAILURE_BACKOFF_S))
            st.next_due = t + interval + self._jitter(st.task)

    def force_due(self, names: Iterable[str] | None = None, now: float | None = None) -> list[str]:
        """Make the named tasks (default: all) due immediately and return the names pulled
        forward. Unknown names raise KeyError (nothing is changed). A task that is running
        right now is left alone: ``mark_ran`` reschedules it normally."""
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            targets = list(self._states) if names is None else list(names)
            for n in targets:
                self._state(n)  # validate before mutating anything
            forced: list[str] = []
            for n in targets:
                st = self._states[n]
                if not st.running:
                    st.next_due = min(st.next_due, t)
                    forced.append(n)
            return forced

    # -- reporting -------------------------------------------------------------------
    def stats(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        t = self.clock.monotonic() if now is None else now
        with self._lock:
            out: dict[str, dict[str, Any]] = {}
            for name, st in self._states.items():
                out[name] = {
                    "interval_s": st.task.interval_s,
                    "jitter_s": st.task.jitter_s,
                    "running": st.running,
                    "running_for_s": round(t - st.started_mono, 3) if st.started_mono is not None else None,
                    "runs": st.runs,
                    "failures": st.failures,
                    "consecutive_failures": st.consecutive_failures,
                    "last_started_at": st.last_started_at,
                    "last_finished_at": st.last_finished_at,
                    "last_ok_at": st.last_ok_at,
                    "last_error": st.last_error,
                    "last_duration_s": round(st.last_duration_s, 3)
                    if st.last_duration_s is not None
                    else None,
                    "next_due_in_s": None if st.running else round(max(0.0, st.next_due - t), 3),
                }
            return out

    # -- internals ----------------------------------------------------------------------
    def _state(self, name: str) -> _TaskState:
        try:
            return self._states[name]
        except KeyError:
            raise KeyError(f"unknown periodic task {name!r}") from None

    def _jitter(self, task: PeriodicTask) -> float:
        return self._rng.uniform(0.0, task.jitter_s) if task.jitter_s > 0 else 0.0
