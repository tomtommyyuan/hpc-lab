"""PeriodicScheduler timing rules (pure; driven by a FakeClock)."""

from __future__ import annotations

import random
from collections import namedtuple

import pytest

from hpclab.domain.clock import FakeClock
from hpclab.services.context import ServiceContext
from hpclab.services.periodic import MAX_FAILURE_BACKOFF_S, PeriodicScheduler, PeriodicTask


def noop(ctx: ServiceContext) -> None:
    return None


def names(tasks: list[PeriodicTask]) -> list[str]:
    return [t.name for t in tasks]


def test_first_run_interval_and_no_overlap(fake_clock: FakeClock) -> None:
    s = PeriodicScheduler(
        [PeriodicTask("a", 10, noop), PeriodicTask("b", 5, noop, run_at_start=False)], fake_clock
    )
    t0 = fake_clock.monotonic()
    assert names(s.due(t0)) == ["a"]
    s.mark_started("a", t0)
    assert s.due(t0 + 100) == [s.tasks[1]]  # a is running: never returned twice
    s.mark_ran("a", t0 + 1, ok=True)
    assert names(s.due(t0 + 5)) == ["b"]
    assert "a" not in names(s.due(t0 + 10.9))  # fixed delay from completion (t0+1)
    assert "a" in names(s.due(t0 + 11))


def test_due_is_pure_and_most_overdue_first(fake_clock: FakeClock) -> None:
    s = PeriodicScheduler([PeriodicTask("x", 3, noop), PeriodicTask("y", 1, noop)], fake_clock)
    t0 = fake_clock.monotonic()
    for n in ("x", "y"):
        s.mark_started(n, t0)
    s.mark_ran("x", t0, ok=True)  # due t0+3
    s.mark_ran("y", t0 + 1, ok=True)  # due t0+2
    assert names(s.due(t0 + 5)) == ["y", "x"]
    assert names(s.due(t0 + 5)) == ["y", "x"]  # asking again changes nothing


def test_failure_backoff_is_capped_and_resets_on_success(fake_clock: FakeClock) -> None:
    s = PeriodicScheduler([PeriodicTask("f", 10, noop)], fake_clock)
    now = fake_clock.monotonic()
    expected = [20, 40, 80, 80]  # 2^min(failures, 3)
    for delay in expected:
        s.mark_started("f", now)
        s.mark_ran("f", now, ok=False, error="boom")
        assert s.next_due_in(now) == pytest.approx(delay)
    st = s.stats(now)["f"]
    assert st["failures"] == 4 and st["consecutive_failures"] == 4 and st["last_error"] == "boom"
    s.mark_started("f", now)
    s.mark_ran("f", now, ok=True)
    assert s.next_due_in(now) == pytest.approx(10)
    st = s.stats(now)["f"]
    assert st["consecutive_failures"] == 0 and st["last_error"] is None and st["last_ok_at"]

    slow = PeriodicScheduler([PeriodicTask("slow", 100, noop)], fake_clock)
    for _ in range(3):
        slow.mark_started("slow", now)
        slow.mark_ran("slow", now, ok=False)
    assert slow.next_due_in(now) == pytest.approx(MAX_FAILURE_BACKOFF_S)


def test_jitter_stays_within_bounds(fake_clock: FakeClock) -> None:
    rng = random.Random(1)
    s = PeriodicScheduler([PeriodicTask("j", 10, noop, jitter_s=2, run_at_start=False)], fake_clock, rng=rng)
    now = fake_clock.monotonic()
    first = s.next_due_in(now)
    assert first is not None and 10 <= first <= 12
    seen = set()
    for _ in range(20):
        s.mark_started("j", now)
        s.mark_ran("j", now, ok=True)
        d = s.next_due_in(now)
        assert d is not None and 10 <= d <= 12
        seen.add(round(d, 3))
    assert len(seen) > 1


def test_force_due(fake_clock: FakeClock) -> None:
    s = PeriodicScheduler(
        [PeriodicTask("hourly", 3600, noop, run_at_start=False), PeriodicTask("busy", 1, noop)], fake_clock
    )
    now = fake_clock.monotonic()
    s.mark_started("busy", now)
    assert s.due(now) == []
    with pytest.raises(KeyError):
        s.force_due(["hourly", "nope"])
    assert s.due(now) == []  # nothing changed on error
    assert s.force_due(None, now) == ["hourly"]  # running task is left alone
    assert names(s.due(now)) == ["hourly"]


def test_stats_and_next_due_when_all_running(fake_clock: FakeClock) -> None:
    s = PeriodicScheduler([PeriodicTask("a", 5, noop)], fake_clock)
    now = fake_clock.monotonic()
    s.mark_started("a", now)
    assert s.next_due_in(now) is None
    fake_clock.advance(2)
    st = s.stats()["a"]
    assert st["running"] and st["running_for_s"] == pytest.approx(2) and st["next_due_in_s"] is None
    s.mark_ran("a", ok=True)
    st = s.stats()["a"]
    assert st["runs"] == 1 and st["last_duration_s"] == pytest.approx(2)
    assert st["last_started_at"].endswith("Z") and st["next_due_in_s"] == pytest.approx(5)


def test_validation_and_coercion() -> None:
    with pytest.raises(ValueError):
        PeriodicTask("x", 0, noop)
    with pytest.raises(ValueError):
        PeriodicTask("", 1, noop)
    with pytest.raises(ValueError):
        PeriodicTask("x", 1, noop, jitter_s=-1)
    with pytest.raises(ValueError, match="duplicate"):
        PeriodicScheduler([PeriodicTask("d", 1, noop), ("d", 2, noop)])
    legacy = namedtuple("legacy", "name interval_s fn")
    t = PeriodicTask.coerce(legacy("poll", 5, noop))
    assert (t.name, t.interval_s, t.jitter_s, t.run_at_start) == ("poll", 5.0, 0.0, True)
    assert PeriodicTask.coerce(("sweep", 2, noop)).interval_s == 2.0
    with pytest.raises(TypeError):
        PeriodicTask.coerce(42)
    with pytest.raises(KeyError):
        PeriodicScheduler([]).mark_ran("ghost")
