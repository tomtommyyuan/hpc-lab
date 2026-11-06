"""Descriptive statistics over seed-level results (``results_compare`` / ``evidence_export``).

Contract
--------
* Everything here is *descriptive*: n, mean, sample standard deviation (``ddof=1``),
  standard error of the mean, and a Student-t interval for a mean or a mean difference.
  The module deliberately offers no hypothesis tests and no p-values: the runtime reports
  numbers and their definitions; deciding what they mean is the researcher's job.
* Inputs must be finite. A NaN/inf metric (e.g. a diverged loss) is a *result* that the
  caller must surface explicitly (as a failed seed), so non-finite input raises
  ``InvalidArgument`` instead of being silently dropped from the denominator.
* ``n < 2`` makes std / sem / interval ``None`` — never 0. One seed says nothing about
  variability, and rendering "± 0" would claim the opposite.
* Extreme magnitudes are handled exactly: values whose largest magnitude lies outside
  ``[2**-400, 2**400]`` are rescaled by a power of two (an exact operation) before squaring,
  so a diverged seed's perplexity (~1e300) cannot overflow and tiny metrics cannot underflow
  their squared deviations to a false "± 0". A statistic that is itself beyond float64
  (e.g. the std of ±1.7e308) raises ``InvalidArgument`` instead of ``OverflowError``.
* Seeds (keys of the paired inputs) are ints or strings and are matched by their string
  form, as in the evidence bundle (``"none"`` for seedless runs); ``1`` and ``"1"`` pair.
* Pure stdlib (no numpy/scipy). The t quantile is computed from the regularized
  incomplete beta function (modified Lentz continued fraction) plus bisection; it is
  accurate to ~1e-10 for df >= 1, far beyond the 1e-3 the renderers need.

Formulas (the renderers print the same definitions next to every table)::

    mean = sum(x) / n
    std  = sqrt( sum((x - mean)^2) / (n - 1) )           # sample std, n >= 2
    sem  = std / sqrt(n)
    CI   = mean ± t_{(1+level)/2, n-1} * sem            # two-sided Student-t interval
    paired:   d_s = a_s - b_s for seeds s in both; summarize(d)
    unpaired: diff = mean_a - mean_b; se = sqrt(var_a/n_a + var_b/n_b);
              Welch–Satterthwaite df; CI = diff ± t_{(1+level)/2, df} * se
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from hpclab.domain.errors import InvalidArgument

__all__ = [
    "PairedSummary",
    "Summary",
    "UnpairedSummary",
    "paired_differences",
    "summarize",
    "t_critical",
    "unpaired_difference",
]

# Above this many degrees of freedom the t quantile equals the normal quantile to < 1e-6
# absolute, and the continued fraction would need thousands of iterations.
_NORMAL_DF_THRESHOLD = 1e7
_CF_MAX_ITER = 20_000
_CF_EPS = 1e-15
_CF_TINY = 1e-300
_BISECT_MAX_ITER = 300
_BISECT_REL_TOL = 1e-13
# Within this magnitude range squared deviations can neither overflow nor underflow.
_SCALE_MIN = 2.0**-400
_SCALE_MAX = 2.0**400
_INT_SEED = re.compile(r"-?\d+")

Seed = int | str


class _Stat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Summary(_Stat):
    """Descriptive summary of one group's values (one value per seed).

    ``values`` preserves the caller's order so the summary can be recomputed exactly.
    """

    n: int
    mean: float | None
    std: float | None
    sem: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    min: float | None
    max: float | None
    values: list[float]


class PairedSummary(_Stat):
    """Summary of per-seed differences ``a[s] - b[s]`` over seeds present in both inputs.

    Seeds present in only one input are listed in ``unpaired_a`` / ``unpaired_b`` rather
    than dropped silently.
    """

    n_pairs: int
    mean_diff: float | None
    std_diff: float | None
    sem_diff: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    paired_seeds: list[Seed]  # spelled as in ``a``; numeric seeds first, in numeric order
    unpaired_a: list[Seed]
    unpaired_b: list[Seed]


class UnpairedSummary(_Stat):
    """Difference of means ``mean(a) - mean(b)`` with a Welch-style t interval.

    ``df`` is the Welch–Satterthwaite degrees of freedom (non-integer in general). When
    either side has fewer than two values the spread is unknown and the interval is
    ``None``. When both sample variances are 0 the interval degenerates to the point.
    """

    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    mean_diff: float | None
    se_diff: float | None
    df: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize(values: Sequence[float], ci_level: float = 0.95) -> Summary:
    """Summarize finite values; std/sem/CI are ``None`` when ``n < 2``.

    ``summarize([])`` is allowed (n=0, everything ``None``) so a group whose seeds all
    failed still has a well-defined, visibly empty summary.
    """
    _check_level(ci_level)
    xs = _finite_list(values, what="values")
    n = len(xs)
    if n == 0:
        return Summary(
            n=0,
            mean=None,
            std=None,
            sem=None,
            ci_low=None,
            ci_high=None,
            ci_level=ci_level,
            min=None,
            max=None,
            values=[],
        )
    k = _scale_exponent(xs)
    ys = _scaled(xs, -k)
    mean = _mean(ys)
    std = _sample_std(ys, mean)
    sem = std / math.sqrt(n) if std is not None else None
    ci_low, ci_high = _interval(mean, sem, n - 1, ci_level)
    return Summary(
        n=n,
        mean=_unscaled(mean, k),
        std=_unscaled(std, k),
        sem=_unscaled(sem, k),
        ci_low=_unscaled(ci_low, k),
        ci_high=_unscaled(ci_high, k),
        ci_level=ci_level,
        min=min(xs),
        max=max(xs),
        values=xs,
    )


def paired_differences[SA: Seed, SB: Seed](
    a: Mapping[SA, float], b: Mapping[SB, float], ci_level: float = 0.95
) -> PairedSummary:
    """Summarize ``a[s] - b[s]`` over the seeds present in both mappings.

    Call as ``paired_differences(treatment, baseline)`` so a positive ``mean_diff`` means
    the treatment's metric is numerically larger (whether that is better depends on the
    metric's direction, which this module does not interpret). Seeds match by string form
    (``1`` pairs with ``"1"``); one mapping spelling a seed twice is ``InvalidArgument``.
    """
    _check_level(ci_level)
    _finite_list(a.values(), what="a")
    _finite_list(b.values(), what="b")
    ka, kb = _seed_index(a, what="a"), _seed_index(b, what="b")
    paired = _seed_sorted(set(ka) & set(kb))
    diffs = [float(a[ka[s]]) - float(b[kb[s]]) for s in paired]
    for seed, diff in zip(paired, diffs, strict=True):
        if not math.isfinite(diff):
            raise InvalidArgument(f"the difference for seed {seed} is not representable in float64")
    summary = summarize(diffs, ci_level)
    return PairedSummary(
        n_pairs=summary.n,
        mean_diff=summary.mean,
        std_diff=summary.std,
        sem_diff=summary.sem,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        ci_level=ci_level,
        paired_seeds=[ka[s] for s in paired],
        unpaired_a=[ka[s] for s in _seed_sorted(set(ka) - set(kb))],
        unpaired_b=[kb[s] for s in _seed_sorted(set(kb) - set(ka))],
    )


def unpaired_difference(a: Sequence[float], b: Sequence[float], ci_level: float = 0.95) -> UnpairedSummary:
    """Difference of means ``mean(a) - mean(b)`` with a Welch–Satterthwaite t interval."""
    _check_level(ci_level)
    xa = _finite_list(a, what="a")
    xb = _finite_list(b, what="b")
    k = _scale_exponent(xa, xb)  # one scale for both sides: differences stay exact
    ya, yb = _scaled(xa, -k), _scaled(xb, -k)
    mean_a = _mean(ya) if ya else None
    mean_b = _mean(yb) if yb else None
    mean_diff = mean_a - mean_b if mean_a is not None and mean_b is not None else None
    se, df = _welch(ya, yb, mean_a, mean_b)
    if mean_diff is None or se is None:
        ci_low: float | None = None
        ci_high: float | None = None
    elif df is None:  # both variances are 0: the interval degenerates to the point
        ci_low = ci_high = mean_diff
    else:
        half = t_critical(df, ci_level) * se
        ci_low, ci_high = mean_diff - half, mean_diff + half
    return UnpairedSummary(
        n_a=len(xa),
        n_b=len(xb),
        mean_a=_unscaled(mean_a, k),
        mean_b=_unscaled(mean_b, k),
        mean_diff=_unscaled(mean_diff, k),
        se_diff=_unscaled(se, k),
        df=df,
        ci_low=_unscaled(ci_low, k),
        ci_high=_unscaled(ci_high, k),
        ci_level=ci_level,
    )


def t_critical(df: float, level: float) -> float:
    """Two-sided Student-t critical value: ``t`` with ``P(|T_df| <= t) = level``.

    Equivalently the ``(1 + level) / 2`` quantile, e.g. ``t_critical(1, 0.95) = 12.7062``.
    ``df`` may be fractional (Welch); it must be finite and > 0.
    """
    _check_level(level)
    if isinstance(df, bool) or not isinstance(df, (int, float)) or not math.isfinite(df) or df <= 0:
        raise InvalidArgument(f"degrees of freedom must be a finite number > 0, got {df!r}")
    alpha = 1.0 - level
    if df > _NORMAL_DF_THRESHOLD:
        return _bisect_decreasing(lambda z: math.erfc(z / math.sqrt(2.0)), alpha)
    d = float(df)
    return _bisect_decreasing(lambda t: _t_two_sided_tail(t, d), alpha)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _check_level(level: float) -> None:
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not (0.0 < level < 1.0):
        raise InvalidArgument(f"confidence level must be in (0, 1), got {level!r}")


def _finite_list(values: Iterable[float], *, what: str) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InvalidArgument(f"{what} must contain numbers, got {type(v).__name__}")
        try:
            f = float(v)
        except OverflowError:  # an int beyond float range
            f = math.inf
        if not math.isfinite(f):
            raise InvalidArgument(
                f"{what} contains a non-finite value ({v!r}); report it as a failed seed instead",
                details={"value": repr(v)},
            )
        out.append(f)
    return out


def _seed_index[S: Seed](mapping: Mapping[S, float], *, what: str) -> dict[str, S]:
    """String form of each seed → the caller's key; rejects keys that are not seeds and
    seeds spelled twice (``1`` and ``"1"``), which would make the pairing ambiguous."""
    index: dict[str, S] = {}
    for key in mapping:
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise InvalidArgument(f"{what}: seeds must be ints or strings, got {type(key).__name__}")
        name = str(key)
        if name in index:
            raise InvalidArgument(f"{what}: seed {name} is given twice ({index[name]!r} and {key!r})")
        index[name] = key
    return index


def _seed_sorted(seeds: Iterable[str]) -> list[str]:
    """Numeric seeds first in numeric order, then the others by name (as the renderers do)."""
    return sorted(seeds, key=lambda s: (0, int(s), "") if _INT_SEED.fullmatch(s) else (1, 0, s))


def _scale_exponent(*groups: Sequence[float]) -> int:
    """``k`` such that ``x * 2**-k`` is O(1) for the largest ``|x|``; 0 when no rescaling is needed."""
    peak = max((abs(x) for xs in groups for x in xs), default=0.0)
    if peak == 0.0 or _SCALE_MIN <= peak <= _SCALE_MAX:
        return 0
    return math.frexp(peak)[1]


def _scaled(xs: Sequence[float], exponent: int) -> list[float]:
    """Multiply by ``2**exponent`` — exact, so statistics computed on the result are too."""
    return [math.ldexp(x, exponent) for x in xs] if exponent else list(xs)


def _unscaled(value: float | None, k: int) -> float | None:
    """Undo :func:`_scaled`; a result beyond float64 is a clear error, not ``OverflowError``."""
    if value is None:
        return None
    try:
        out = math.ldexp(value, k) if k else value
    except OverflowError:
        out = math.inf
    if not math.isfinite(out):
        raise InvalidArgument(
            "a summary statistic of these values is not representable in float64 (magnitude near "
            "1.8e308); report such seeds as failed or summarize a transformed metric (e.g. log scale)"
        )
    return out


def _mean(xs: Sequence[float]) -> float:
    first = xs[0]
    if all(x == first for x in xs):
        return first  # exact: avoids 1e-17 noise turning an identical-seed std into "non-zero"
    return math.fsum(xs) / len(xs)


def _sample_std(xs: Sequence[float], mean: float) -> float | None:
    if len(xs) < 2:
        return None
    return math.sqrt(_sample_var(xs, mean))


def _sample_var(xs: Sequence[float], mean: float) -> float:
    return math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


def _interval(center: float, se: float | None, df: float, level: float) -> tuple[float | None, float | None]:
    if se is None or df < 1:
        return None, None
    if se == 0.0:
        return center, center
    half = t_critical(df, level) * se
    return center - half, center + half


def _welch(
    xa: Sequence[float], xb: Sequence[float], mean_a: float | None, mean_b: float | None
) -> tuple[float | None, float | None]:
    """Return (standard error of the difference, Welch–Satterthwaite df).

    ``(None, None)`` when either side has < 2 values; ``(0.0, None)`` when both sample
    variances are 0 (df is 0/0, undefined, but the interval is the point itself).
    df = (va + vb)^2 / (va^2/(na-1) + vb^2/(nb-1)) is evaluated through the shares
    ``va/(va+vb)`` so that squaring can neither overflow nor underflow.
    """
    if len(xa) < 2 or len(xb) < 2 or mean_a is None or mean_b is None:
        return None, None
    va = _sample_var(xa, mean_a) / len(xa)
    vb = _sample_var(xb, mean_b) / len(xb)
    total = va + vb
    if total == 0.0:
        return 0.0, None
    share_a, share_b = va / total, vb / total
    df = 1.0 / (share_a * share_a / (len(xa) - 1) + share_b * share_b / (len(xb) - 1))
    return math.sqrt(total), df


def _t_two_sided_tail(t: float, df: float) -> float:
    """``P(|T_df| > t)`` for ``t >= 0`` via ``I_{df/(df+t^2)}(df/2, 1/2)``."""
    if t <= 0.0:
        return 1.0
    x = df / (df + t * t)
    return _betainc_regularized(df / 2.0, 0.5, x)


def _betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` (Numerical Recipes §6.4 formulation)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    # The continued fraction converges fast for x < (a+1)/(a+b+2); use symmetry otherwise.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _CF_TINY else _CF_TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _CF_TINY else _CF_TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def _bisect_decreasing(tail: Callable[[float], float], alpha: float) -> float:
    """Smallest ``t >= 0`` with ``tail(t) <= alpha`` for a decreasing tail function."""
    lo, hi = 0.0, 1.0
    while tail(hi) > alpha:
        lo, hi = hi, hi * 2.0
        if hi > 1e300:  # pragma: no cover - unreachable for levels representable in float
            raise ArithmeticError("t quantile search diverged")
    for _ in range(_BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= _BISECT_REL_TOL * hi:
            break
    return 0.5 * (lo + hi)
