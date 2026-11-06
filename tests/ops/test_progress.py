"""Progress estimation over metrics JSONL: rate window, ETA, walltime risk, stall signal, robustness."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from hpclab.services.progress import (
    RATE_WINDOW_S,
    STALL_FLOOR_S,
    Progress,
    estimate_progress,
)

T0 = 1_790_000_000.0  # epoch seconds (2026-09)


def rec(step: int, t: float | None, *, total: int | None = 1000, **extra: Any) -> str:
    record: dict[str, Any] = {"schema": "hpclab.metrics/v1", "step": step, "split": "train"}
    if total is not None:
        record["total_steps"] = total
    if t is not None:
        record["time"] = t
    record["metrics"] = extra.pop("metrics", {"loss": 1.0 / (1 + step)})
    record.update(extra)
    return json.dumps(record)


def steady(n: int, *, every_steps: int = 10, every_s: float = 5.0, total: int | None = 1000) -> list[str]:
    """n records, one every ``every_s`` seconds, 2 steps/s."""
    return [rec(i * every_steps, T0 + i * every_s, total=total) for i in range(1, n + 1)]


def test_steady_rate_eta_and_fraction() -> None:
    lines = steady(20)  # steps 10..200 over 95 s -> 2 steps/s
    now = T0 + 20 * 5.0
    p = estimate_progress(lines, elapsed_s=200, time_limit_s=3600, now_epoch=now)
    assert p.step == 200 and p.total_steps == 1000
    assert p.fraction == pytest.approx(0.2)
    assert p.rate_steps_per_s == pytest.approx(2.0)
    assert p.rate_basis == "records"
    assert p.eta_s == 400  # 800 steps / 2 steps/s, newest record is 0 s old
    assert p.remaining_walltime_s == 3400
    assert p.timelimit_risk == "none"
    assert not p.stalled_suspect and p.last_record_age_s == 0
    assert p.records == 20 and p.skipped_lines == 0
    assert p.latest_metrics == {"train/loss": pytest.approx(1 / 201)}


def test_eta_accounts_for_time_since_newest_record() -> None:
    lines = steady(20)
    p = estimate_progress(lines, elapsed_s=260, time_limit_s=3600, now_epoch=T0 + 100 + 60)
    assert p.last_record_age_s == 60
    assert p.eta_s == 340  # 400 s at the newest record, 60 s of it presumably done


@pytest.mark.parametrize(
    ("remaining", "risk"),
    [(1000, "none"), (500, "low"), (420, "medium"), (400, "high"), (100, "high")],
)
def test_timelimit_risk_thresholds(remaining: int, risk: str) -> None:
    lines = steady(20)  # eta = 400 s
    p = estimate_progress(lines, elapsed_s=1000, time_limit_s=1000 + remaining, now_epoch=T0 + 100)
    assert p.eta_s == 400
    assert p.timelimit_risk == risk
    if risk == "high":
        assert any("will not finish" in n for n in p.notes)


def test_no_walltime_left_is_high_risk() -> None:
    p = estimate_progress(steady(20), elapsed_s=3600, time_limit_s=3600, now_epoch=T0 + 100)
    assert p.remaining_walltime_s == 0 and p.timelimit_risk == "high"


def test_unknown_limit_or_total_gives_unknown_risk() -> None:
    assert estimate_progress(steady(20), elapsed_s=100, time_limit_s=None).timelimit_risk == "unknown"
    no_total = estimate_progress(steady(20, total=None), elapsed_s=100, time_limit_s=3600)
    assert no_total.eta_s is None and no_total.fraction is None and no_total.timelimit_risk == "unknown"
    assert no_total.rate_steps_per_s == pytest.approx(2.0)


def test_total_steps_hint_used_when_records_lack_it() -> None:
    p = estimate_progress(
        steady(20, total=None), elapsed_s=100, time_limit_s=3600, total_steps_hint=400, now_epoch=T0 + 100
    )
    assert p.total_steps == 400 and p.eta_s == 100


def test_records_total_wins_over_conflicting_hint() -> None:
    p = estimate_progress(steady(5), elapsed_s=100, time_limit_s=3600, total_steps_hint=5000)
    assert p.total_steps == 1000
    assert any("differs from the hint" in n for n in p.notes)


def test_rate_uses_recent_window_not_warmup() -> None:
    warmup = [rec(i, T0 + i * 60.0) for i in range(1, 6)]  # 1 step/min while compiling
    fast_t0 = T0 + 400
    fast = [rec(100 + i * 50, fast_t0 + i * 10.0) for i in range(1, 21)]  # 5 steps/s
    p = estimate_progress(warmup + fast, elapsed_s=700, time_limit_s=3600, now_epoch=fast_t0 + 200)
    assert p.rate_steps_per_s == pytest.approx(5.0)


def test_window_falls_back_to_last_two_points_when_records_are_sparse() -> None:
    gap = RATE_WINDOW_S * 3
    lines = [rec(100, T0), rec(200, T0 + gap), rec(300, T0 + 2 * gap)]
    p = estimate_progress(lines, elapsed_s=int(2 * gap), time_limit_s=100_000, now_epoch=T0 + 2 * gap)
    assert p.rate_steps_per_s == pytest.approx(100 / gap, rel=1e-5)


def test_step_reset_uses_records_after_restart() -> None:
    before = [rec(s, T0 + s) for s in (100, 200, 300)]  # 1 step/s
    after = [rec(s, T0 + 1000 + (s - 200) / 4) for s in (200, 240, 280, 320)]  # resumed at 200, 4 steps/s
    p = estimate_progress(before + after, elapsed_s=1100, time_limit_s=3600, now_epoch=T0 + 1030)
    assert p.step == 320
    assert p.rate_steps_per_s == pytest.approx(4.0)
    assert any("went backwards" in n for n in p.notes)


def test_partial_last_line_and_garbage_are_tolerated() -> None:
    lines = [*steady(10), "Epoch 1: 10%|###  | 100/1000", "", '{"schema":"hpclab.metrics/v1","step":1']
    p = estimate_progress(lines, elapsed_s=60, time_limit_s=3600, now_epoch=T0 + 50)
    assert p.step == 100 and p.records == 10
    assert p.skipped_lines == 2
    assert any("incomplete" in n for n in p.notes)


def test_multiline_chunks_are_split() -> None:
    p = estimate_progress(["\n".join(steady(10)) + "\n"], elapsed_s=60, time_limit_s=3600)
    assert p.records == 10 and p.step == 100


def test_nan_metrics_are_json_safe_and_noted() -> None:
    lines = [
        *steady(5),
        '{"schema":"hpclab.metrics/v1","step":60,"time":'
        + str(T0 + 30)
        + ',"split":"train","metrics":{"loss":NaN,"grad":Infinity}}',
        rec(61, T0 + 31, metrics={"loss": None}, nonfinite={"loss": "-inf"}),
    ]
    p = estimate_progress(lines, elapsed_s=60, time_limit_s=3600, now_epoch=T0 + 31)
    assert p.latest_metrics["train/loss"] == "-Infinity"
    assert p.latest_metrics["train/grad"] == "Infinity"
    assert any("Non-finite" in n for n in p.notes)
    assert Progress.model_validate_json(p.model_dump_json()) == p


def test_nan_step_or_time_values_are_ignored() -> None:
    lines = ['{"step": NaN, "time": ' + str(T0) + ', "loss": 1.0}', '{"step": 5, "time": NaN, "loss": 0.5}']
    p = estimate_progress(lines, elapsed_s=10, time_limit_s=100)
    assert p.step == 5 and p.records == 2
    assert p.last_record_time == T0  # only the valid timestamp survives
    assert p.rate_steps_per_s is None and p.rate_basis == "none"


@pytest.mark.parametrize(
    ("line_a", "line_b", "step"),
    [
        ('{"global_step": 100, "timestamp": "2026-09-01T12:00:00Z", "loss": 1.0}',
         '{"global_step": 160, "timestamp": "2026-09-01T12:00:30Z", "loss": 0.9, "max_steps": 400}', 160),
        ('{"iteration": 10, "wall_time": 1790000000000, "loss": 1.0}',
         '{"iteration": 70, "wall_time": 1790000030000, "loss": 0.9, "max_steps": 400}', 70),
        ('{"iter": 10, "time": 1790000000.0, "lr": 0.1}', '{"iter": 70, "time": 1790000030.0, "lr": 0.1, "max_steps": 400}', 70),
        ('{"step": 10, "_timestamp": 1790000000.5}', '{"step": 70, "_timestamp": 1790000030.5, "total_steps": 400}', 70),
    ],
)  # fmt: skip
def test_generic_jsonl_keys(line_a: str, line_b: str, step: int) -> None:
    p = estimate_progress([line_a, line_b], elapsed_s=100, time_limit_s=10_000)
    assert p.step == step and p.total_steps == 400
    assert p.rate_steps_per_s == pytest.approx(2.0)
    assert p.eta_s is not None and p.timelimit_risk == "none"


def test_generic_records_prefer_global_step_and_keep_numeric_metrics() -> None:
    line = json.dumps(
        {"step": 3, "global_step": 3000, "epoch": 1.5, "loss": 0.2, "lr": 1e-4, "tag": "x", "ok": True}
    )
    p = estimate_progress([line], elapsed_s=None, time_limit_s=None)
    assert p.step == 3000 and p.epoch == 1.5
    assert p.latest_metrics == {"loss": 0.2, "lr": 1e-4}


def test_non_epoch_times_are_ignored_with_note() -> None:
    lines = ['{"step": 10, "time": 0.35}', '{"step": 20, "time": 0.36}']
    p = estimate_progress(lines, elapsed_s=None, time_limit_s=None)
    assert p.rate_steps_per_s is None and p.last_record_time is None
    assert any("not epoch seconds" in n for n in p.notes)


def test_elapsed_fallback_without_timestamps() -> None:
    lines = [json.dumps({"step": s, "total_steps": 1000}) for s in (10, 20, 500)]
    p = estimate_progress(lines, elapsed_s=250, time_limit_s=550)
    assert p.rate_basis == "elapsed" and p.rate_steps_per_s == pytest.approx(2.0)
    assert p.eta_s == 250 and p.timelimit_risk == "low"  # needs 250 s of the 300 s left (ratio 0.83)


def test_elapsed_fallback_does_not_guess_for_a_resumed_start() -> None:
    # A resumed attempt starting at 5000 looks exactly like a bounded tail of a long run; either
    # assumption can be wrong by a large factor, so without timestamps the rate is unknown.
    lines = [json.dumps({"step": s, "total_steps": 10_000}) for s in (5000, 5100, 5500)]
    p = estimate_progress(lines, elapsed_s=250, time_limit_s=10_000)
    assert p.rate_steps_per_s is None and p.timelimit_risk == "unknown"


def test_single_record_without_elapsed_has_no_rate() -> None:
    p = estimate_progress([rec(10, T0)], elapsed_s=None, time_limit_s=3600)
    assert p.step == 10 and p.rate_steps_per_s is None and p.eta_s is None
    assert p.timelimit_risk == "unknown"


def test_no_step_counter() -> None:
    p = estimate_progress(['{"loss": 0.5, "time": ' + str(T0) + "}"], elapsed_s=10, time_limit_s=100)
    assert p.step is None and p.timelimit_risk == "unknown"
    assert p.latest_metrics == {"loss": 0.5}
    assert any("no step counter" in n for n in p.notes)
    assert p.short_summary() == "no progress records yet"


def test_empty_input() -> None:
    p = estimate_progress([], elapsed_s=None, time_limit_s=None)
    assert p == Progress()


# ---------------------------------------------------------------------------
# Stall signal
# ---------------------------------------------------------------------------


def test_stall_floor_is_ten_minutes() -> None:
    lines = steady(20)  # records every 5 s
    last = T0 + 100
    fresh = estimate_progress(lines, elapsed_s=700, time_limit_s=36_000, now_epoch=last + STALL_FLOOR_S - 1)
    assert not fresh.stalled_suspect
    stale = estimate_progress(lines, elapsed_s=800, time_limit_s=36_000, now_epoch=last + STALL_FLOOR_S + 1)
    assert stale.stalled_suspect
    note = next(n for n in stale.notes if "No new metrics" in n)
    assert "only a signal" in note
    assert stale.eta_s == 400  # a stalled run gets no credit for time since the newest record


def test_stall_threshold_scales_with_median_interval() -> None:
    lines = [
        rec(i * 100, T0 + i * 300.0, total=10_000) for i in range(1, 11)
    ]  # every 5 min -> 25 min threshold
    last = T0 + 3000
    assert not estimate_progress(
        lines, elapsed_s=None, time_limit_s=None, now_epoch=last + 20 * 60
    ).stalled_suspect
    assert estimate_progress(
        lines, elapsed_s=None, time_limit_s=None, now_epoch=last + 30 * 60
    ).stalled_suspect


def test_stall_note_mentions_earlier_gaps_of_similar_length() -> None:
    lines = [rec(10, T0), rec(20, T0 + 10), rec(30, T0 + 20), rec(40, T0 + 20 + 1500), rec(50, T0 + 1530)]
    p = estimate_progress(lines, elapsed_s=None, time_limit_s=None, now_epoch=T0 + 1530 + 1200)
    assert p.stalled_suspect
    assert any("already happened earlier" in n for n in p.notes)


def test_no_stall_without_clock_or_after_final_record() -> None:
    assert not estimate_progress(steady(3), elapsed_s=None, time_limit_s=None).stalled_suspect
    final = [*steady(3), rec(1000, T0 + 20, final=True, split="val", metrics={"acc": 0.91})]
    p = estimate_progress(final, elapsed_s=500, time_limit_s=600, now_epoch=T0 + 99_999)
    assert p.final_seen and not p.stalled_suspect
    assert p.timelimit_risk == "none" and p.eta_s == 0 and p.fraction == 1.0
    assert p.latest_metrics["val/acc"] == 0.91


def test_future_timestamps_are_flagged_as_clock_skew() -> None:
    p = estimate_progress(steady(5), elapsed_s=None, time_limit_s=None, now_epoch=T0 - 3600)
    assert p.last_record_age_s == 0
    assert any("clocks" in n for n in p.notes)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_latest_metrics_are_bounded_and_notes_capped() -> None:
    wide = [rec(i, T0 + i, metrics={f"m{j}": float(j) for j in range(80)}) for i in range(1, 4)]
    p = estimate_progress(wide, elapsed_s=None, time_limit_s=None)
    assert len(p.latest_metrics) == 50
    assert len(p.notes) <= 10


def test_negative_inputs_are_treated_as_unknown() -> None:
    p = estimate_progress(steady(5), elapsed_s=-5, time_limit_s=-1)
    assert p.elapsed_s is None and p.remaining_walltime_s is None


def test_short_summary() -> None:
    p = estimate_progress(steady(20), elapsed_s=200, time_limit_s=3600, now_epoch=T0 + 100)
    assert p.short_summary() == "step 200/1000 (20%) · 2 steps/s · ETA 00:06:40 · risk none"


def test_round_trip_and_json_safety() -> None:
    p = estimate_progress(steady(20), elapsed_s=200, time_limit_s=3600, now_epoch=T0 + 100)
    data = json.loads(p.model_dump_json())
    assert Progress.model_validate(data) == p
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in data["latest_metrics"].values())


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------


def test_elapsed_fallback_refuses_to_guess_when_the_tail_does_not_show_the_start() -> None:
    # run_inspect passes a <=16 KiB tail: a long run's tail starts mid-run, not at a resume point.
    lines = [json.dumps({"step": s, "total_steps": 60_000}) for s in range(49_000, 50_001, 100)]
    p = estimate_progress(lines, elapsed_s=50_000, time_limit_s=72_000)
    assert p.rate_steps_per_s is None and p.eta_s is None
    assert p.timelimit_risk == "unknown"  # not a bogus "high" from a 50x too slow rate
    assert p.rate_basis == "none"
    assert any("start" in n for n in p.notes)


def test_elapsed_fallback_still_works_when_the_run_start_is_visible() -> None:
    lines = [json.dumps({"step": s, "total_steps": 1000}) for s in (1, 10, 500)]
    p = estimate_progress(lines, elapsed_s=250, time_limit_s=550)
    assert p.rate_basis == "elapsed" and p.rate_steps_per_s == pytest.approx(2.0)


@pytest.mark.parametrize("field", ["step", "total_steps"])
def test_absurd_integers_do_not_crash(field: str) -> None:
    huge = 10**400
    base = {"step": 5, "total_steps": 100, "time": T0}
    lines = [
        json.dumps({**base, field: huge}),
        json.dumps({**base, "step": 10, field: huge, "time": T0 + 10}),
    ]
    p = estimate_progress(lines, elapsed_s=10, time_limit_s=100, now_epoch=T0 + 10)
    assert p.records == 2  # parsed, the implausible value ignored
    assert p.eta_s is None or p.eta_s >= 0


def test_naive_iso_timestamps_give_a_rate_but_no_stall_or_age() -> None:
    # datetime.now().isoformat() on a compute node in UTC-7: interpreting it as UTC is 7 h off.
    lines = [
        json.dumps({"step": i * 10, "total_steps": 1000, "timestamp": f"2026-09-27T10:{i:02d}:00"})
        for i in range(10)
    ]
    now = 1_790_528_970.0  # 2026-09-27T17:09:30Z == 10:09:30 local (UTC-7)
    p = estimate_progress(lines, elapsed_s=600, time_limit_s=3600, now_epoch=now)
    assert p.rate_steps_per_s == pytest.approx(10 / 60, rel=1e-5)
    assert not p.stalled_suspect
    assert p.last_record_age_s is None
    assert any("time zone" in n for n in p.notes)


def test_aware_iso_timestamps_still_drive_the_stall_signal() -> None:
    lines = [json.dumps({"step": i * 10, "timestamp": f"2026-09-27T10:{i:02d}:00-07:00"}) for i in range(10)]
    now = 1_790_528_970.0 + 3600  # an hour after the newest record
    p = estimate_progress(lines, elapsed_s=None, time_limit_s=None, now_epoch=now)
    assert p.stalled_suspect and p.last_record_age_s == 3630


def test_final_record_of_an_earlier_run_does_not_mark_a_restarted_run_done() -> None:
    old = [rec(s, T0 + s) for s in (50, 100)]
    old.append(rec(100, T0 + 101, split="val", final=True, total=100, metrics={"acc": 0.9}))
    new = [rec(s, T0 + 1000 + s, total=100) for s in (10, 20)]
    p = estimate_progress([*old, *new], elapsed_s=500, time_limit_s=600, now_epoch=T0 + 5000)
    assert not p.final_seen
    assert p.step == 20 and p.fraction == pytest.approx(0.2)
    assert p.timelimit_risk != "none" or p.eta_s != 0
    assert p.stalled_suspect  # 3980 s without a record is a signal again once "done" no longer masks it
