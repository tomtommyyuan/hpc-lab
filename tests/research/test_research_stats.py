"""Descriptive statistics: t quantiles against published tables, hand-computed summaries."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from hpclab.domain.errors import InvalidArgument
from hpclab.services import stats
from hpclab.services.stats import paired_differences, summarize, t_critical, unpaired_difference

# Student-t critical values from standard tables (4 decimals), keyed by (two-sided level, df).
T_TABLE = {
    (0.95, 1): 12.7062,
    (0.95, 2): 4.3027,
    (0.95, 4): 2.7764,
    (0.95, 9): 2.2622,
    (0.95, 29): 2.0452,
    (0.95, 120): 1.9799,
    (0.90, 1): 6.3138,
    (0.90, 2): 2.9200,
    (0.90, 4): 2.1318,
    (0.90, 9): 1.8331,
    (0.90, 29): 1.6991,
    (0.90, 120): 1.6577,
    (0.99, 1): 63.6567,
    (0.99, 2): 9.9248,
    (0.99, 4): 4.6041,
    (0.99, 9): 3.2498,
    (0.99, 29): 2.7564,
    (0.99, 120): 2.6174,
}


@pytest.mark.parametrize(("level", "df"), sorted(T_TABLE))
def test_t_critical_matches_published_table(level: float, df: int) -> None:
    # Table values are rounded to 4 decimals, so agreement within 1e-4 means < 1e-3 accuracy.
    assert t_critical(df, level) == pytest.approx(T_TABLE[(level, df)], abs=1e-4)


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
def test_t_critical_closed_forms(level: float) -> None:
    q = (1 + level) / 2
    cauchy = math.tan(math.pi * (q - 0.5))  # df = 1 is the Cauchy distribution
    df2 = (2 * q - 1) / math.sqrt(2 * q * (1 - q))  # exact quantile for df = 2
    assert t_critical(1, level) == pytest.approx(cauchy, rel=1e-9)
    assert t_critical(2, level) == pytest.approx(df2, rel=1e-9)


def test_t_critical_approaches_normal_for_large_df() -> None:
    assert t_critical(10**6, 0.95) == pytest.approx(1.959966, abs=1e-5)
    assert t_critical(10**9, 0.95) == pytest.approx(1.959964, abs=1e-6)  # normal branch


def test_t_critical_fractional_df_is_between_neighbours() -> None:
    assert t_critical(2, 0.95) > t_critical(2.5, 0.95) > t_critical(3, 0.95)


@pytest.mark.parametrize(
    ("df", "level"),
    [(0, 0.95), (-1, 0.95), (math.inf, 0.95), (math.nan, 0.95), (True, 0.95), (3, 0.0), (3, 1.0), (3, 1.5)],
)
def test_t_critical_rejects_invalid_arguments(df: float, level: float) -> None:
    with pytest.raises(InvalidArgument):
        t_critical(df, level)


def test_summarize_hand_computed() -> None:
    s = summarize([1.0, 2.0, 3.0, 4.0])
    # var = ((1.5² + 0.5²) * 2) / 3 = 5/3
    assert s.n == 4 and s.mean == 2.5
    assert s.std == pytest.approx(math.sqrt(5 / 3), rel=1e-12)
    assert s.sem == pytest.approx(math.sqrt(5 / 3) / 2, rel=1e-12)
    half = 3.182446305284263 * math.sqrt(5 / 3) / 2  # t(0.975, 3)
    assert s.ci_low == pytest.approx(2.5 - half, rel=1e-9)
    assert s.ci_high == pytest.approx(2.5 + half, rel=1e-9)
    assert (s.min, s.max, s.values, s.ci_level) == (1.0, 4.0, [1.0, 2.0, 3.0, 4.0], 0.95)


def test_summarize_single_value_has_no_spread() -> None:
    s = summarize([0.9])
    assert (s.n, s.mean, s.min, s.max) == (1, 0.9, 0.9, 0.9)
    assert s.std is None and s.sem is None and s.ci_low is None and s.ci_high is None


def test_summarize_empty_is_all_none() -> None:
    s = summarize([])
    assert s.n == 0 and s.values == []
    assert all(getattr(s, f) is None for f in ("mean", "std", "sem", "ci_low", "ci_high", "min", "max"))


def test_summarize_identical_values_is_exactly_zero_spread() -> None:
    s = summarize([0.1, 0.1, 0.1])
    assert s.mean == 0.1 and s.std == 0.0 and s.ci_low == s.ci_high == 0.1


def test_summarize_is_order_independent() -> None:
    xs = [0.1, 1e8, -3.7, 2.5e-9, 42.0]
    a, b = summarize(xs), summarize(list(reversed(xs)))
    assert (a.mean, a.std, a.ci_low, a.ci_high) == (b.mean, b.std, b.ci_low, b.ci_high)


def test_summarize_ci_level_changes_width() -> None:
    narrow, wide = summarize([1, 2, 3], 0.90), summarize([1, 2, 3], 0.99)
    bounds = [b for b in (wide.ci_low, narrow.ci_low, narrow.ci_high, wide.ci_high) if b is not None]
    assert len(bounds) == 4
    assert bounds == sorted(bounds) and len(set(bounds)) == 4


@pytest.mark.parametrize("bad", [[1.0, math.nan], [math.inf], [1.0, -math.inf], [True], ["1.0"]])
def test_summarize_rejects_non_finite_or_non_numbers(bad: list[object]) -> None:
    with pytest.raises(InvalidArgument):
        summarize(bad)  # type: ignore[arg-type]


def test_paired_differences_hand_computed() -> None:
    a = {1: 0.92, 2: 0.90, 3: 0.95, 7: 0.5}
    b = {1: 0.90, 2: 0.89, 3: 0.91, 9: 0.1}
    p = paired_differences(a, b)
    diffs = [0.92 - 0.90, 0.90 - 0.89, 0.95 - 0.91]
    ref = summarize(diffs)
    assert p.n_pairs == 3 and p.paired_seeds == [1, 2, 3]
    assert p.unpaired_a == [7] and p.unpaired_b == [9]
    assert p.mean_diff == pytest.approx(sum(diffs) / 3)
    assert (p.std_diff, p.sem_diff, p.ci_low, p.ci_high) == (ref.std, ref.sem, ref.ci_low, ref.ci_high)


def test_paired_differences_without_overlap() -> None:
    p = paired_differences({1: 1.0}, {2: 2.0})
    assert p.n_pairs == 0 and p.mean_diff is None and p.ci_low is None
    assert p.unpaired_a == [1] and p.unpaired_b == [2]


def test_paired_differences_rejects_nan() -> None:
    with pytest.raises(InvalidArgument):
        paired_differences({1: math.nan}, {1: 1.0})


def test_unpaired_difference_equal_variance_case() -> None:
    # Equal n and variances: Welch df = 2n - 2 = 4, se = sqrt(1/3 + 1/3).
    u = unpaired_difference([1, 2, 3], [2, 3, 4])
    assert (u.n_a, u.n_b, u.mean_a, u.mean_b, u.mean_diff) == (3, 3, 2.0, 3.0, -1.0)
    assert u.df == pytest.approx(4.0)
    assert u.se_diff == pytest.approx(math.sqrt(2 / 3))
    assert u.ci_low == pytest.approx(-1 - 2.7764451 * math.sqrt(2 / 3), abs=1e-6)
    assert u.ci_high == pytest.approx(-1 + 2.7764451 * math.sqrt(2 / 3), abs=1e-6)


def test_unpaired_difference_welch_df_hand_computed() -> None:
    a, b = [10.0, 12.0, 14.0, 16.0], [1.0, 5.0]
    va, vb = (20 / 3) / 4, 8.0 / 2  # s²/n per side
    df = (va + vb) ** 2 / (va**2 / 3 + vb**2 / 1)
    u = unpaired_difference(a, b)
    assert u.mean_diff == 10.0
    assert u.df == pytest.approx(df, rel=1e-12)
    assert u.ci_high is not None
    assert u.ci_high - 10.0 == pytest.approx(t_critical(df, 0.95) * math.sqrt(va + vb), rel=1e-12)


def test_unpaired_difference_small_samples_have_no_interval() -> None:
    u = unpaired_difference([0.874], [0.912, 0.905, 0.918])
    assert u.mean_diff == pytest.approx(0.874 - (0.912 + 0.905 + 0.918) / 3)
    assert u.se_diff is None and u.df is None and u.ci_low is None and u.ci_high is None


def test_unpaired_difference_zero_variance_degenerates_to_point() -> None:
    u = unpaired_difference([1.0, 1.0], [0.5, 0.5])
    assert u.se_diff == 0.0 and u.df is None and u.ci_low == u.ci_high == 0.5


def test_unpaired_difference_empty_side() -> None:
    u = unpaired_difference([], [1.0, 2.0])
    assert u.mean_a is None and u.mean_diff is None and u.ci_low is None


def test_module_exposes_no_hypothesis_testing() -> None:
    public = {name for name in dir(stats) if not name.startswith("_")}
    assert not any("p_value" in n or "pvalue" in n or "ttest" in n or "signif" in n for n in public)
    assert set(stats.__all__) == {
        "PairedSummary",
        "Summary",
        "UnpairedSummary",
        "paired_differences",
        "summarize",
        "t_critical",
        "unpaired_difference",
    }


# ---------------------------------------------------------------------------
# Review regressions: extreme magnitudes and seed keys
# ---------------------------------------------------------------------------


def _exact_summary(xs: list[float]) -> tuple[float, float]:
    """Mean and sample std computed in exact rational arithmetic (reference values)."""
    fx = [Fraction(x) for x in xs]
    mean = sum(fx, Fraction(0)) / len(fx)
    var = sum(((x - mean) ** 2 for x in fx), Fraction(0)) / (len(fx) - 1)
    # sqrt of a huge/tiny Fraction: scale by an even power of two so float() stays in range.
    k = (var.numerator.bit_length() - var.denominator.bit_length()) // 2 * 2
    return float(mean), math.ldexp(math.sqrt(float(var / Fraction(2) ** k)), k // 2)


@pytest.mark.parametrize(
    "xs",
    [
        pytest.param([1e304, 20.0, 21.0], id="diverged-perplexity-seed"),
        pytest.param([3e200, 1e200, 2e200, 2.5e200], id="all-huge"),
        pytest.param([1e-170, 2e-170, 4e-170], id="tiny-spread-would-underflow"),
        pytest.param([-1e250, 1e250], id="symmetric-huge"),
    ],
)
def test_summarize_extreme_but_representable_values_are_exact(xs: list[float]) -> None:
    s = summarize(xs)
    mean, std = _exact_summary(xs)
    assert s.mean == pytest.approx(mean, rel=1e-12, abs=0.0)
    assert s.std is not None and s.std > 0.0
    assert s.std == pytest.approx(std, rel=1e-12)
    assert s.sem == pytest.approx(std / math.sqrt(len(xs)), rel=1e-12)
    assert s.ci_low is not None and s.ci_high is not None
    assert s.mean is not None
    assert math.isfinite(s.ci_low) and math.isfinite(s.ci_high) and s.ci_low < s.mean < s.ci_high


def test_summarize_normal_range_is_unchanged_by_scaling() -> None:
    xs = [0.912, 0.905, 0.918]
    s = summarize(xs)
    mean = math.fsum(xs) / 3
    assert s.mean == mean  # bit-identical to the plain formula
    assert s.std == math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / 2)


def test_summarize_unrepresentable_spread_is_a_clear_error() -> None:
    # std of [±1.7e308] is 2.4e308 > float max: report it, don't crash with OverflowError.
    with pytest.raises(InvalidArgument, match="float64"):
        summarize([1.7e308, -1.7e308])


def test_summarize_rejects_ints_too_large_for_float() -> None:
    with pytest.raises(InvalidArgument):
        summarize([10**400, 1])


def test_unpaired_difference_extreme_magnitudes() -> None:
    u = unpaired_difference([1e300, 2e300, 3e300], [1e300, 1e300, 1.5e300])
    assert u.mean_diff == pytest.approx(2e300 - (3.5e300 / 3), rel=1e-12)
    assert u.se_diff is not None and u.df is not None and u.ci_low is not None and u.ci_high is not None
    assert all(math.isfinite(v) for v in (u.se_diff, u.df, u.ci_low, u.ci_high))
    ref = unpaired_difference([1.0, 2.0, 3.0], [1.0, 1.0, 1.5])
    assert u.df == pytest.approx(ref.df, rel=1e-12)  # df is scale invariant

    tiny = unpaired_difference([1e-160, 2e-160], [1e-160, 3e-160])  # variances underflow when squared
    assert tiny.df == pytest.approx(unpaired_difference([1.0, 2.0], [1.0, 3.0]).df, rel=1e-12)


def test_paired_differences_accepts_string_and_mixed_seed_keys() -> None:
    # Evidence bundles key seeds as strings ("none" for seedless runs); stats must pair them.
    a = {"1": 0.9, "2": 0.8, "none": 0.7, "10": 0.6}
    b = {1: 0.85, 2: 0.75, 3: 0.1, 10: 0.5}
    p = paired_differences(a, b)
    assert p.n_pairs == 3
    assert p.paired_seeds == ["1", "2", "10"]  # a's spelling, numeric order
    assert p.unpaired_a == ["none"] and p.unpaired_b == [3]
    assert p.mean_diff == pytest.approx((0.05 + 0.05 + 0.1) / 3)


def test_paired_differences_rejects_ambiguous_seed_spellings() -> None:
    with pytest.raises(InvalidArgument):
        ambiguous: dict[int | str, float] = {1: 0.9, "1": 0.8}
        paired_differences(ambiguous, {1: 0.5})
