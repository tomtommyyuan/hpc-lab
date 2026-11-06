"""Render an EvidenceBundle as Markdown, LaTeX (booktabs) or long-format CSV (``evidence_export``).

EvidenceBundle ``hpclab.evidence/v1`` — the contract between the evidence service and these
renderers. A bundle is a JSON object::

    {
      "schema": "hpclab.evidence/v1",
      "title": str, "purpose": str,
      "generated_at": str,                     # ISO-8601 UTC; renderers never read a clock
      "hpclab_version": str,
      "ci_level": float = 0.95,                # level of every interval in the bundle
      "experiment": {"id": str|null, "name": str|null, "question": str|null},
      "contract": {"treatment": [str], "invariants": [str], "baseline": {axis: scalar}|null},
      "metric_decls": [{"name": str, "higher_is_better": bool|null, "unit": str|null}],
      "groups": [{
        "key": {axis: scalar},                 # matrix coordinates without the seed
        "label": str,                          # unique within the bundle
        "is_baseline": bool,
        "metrics": {<metric>: {
          "n": int, "expected_n": int,
          "mean", "std", "sem", "ci_low", "ci_high", "min", "max": float|null,
          "values_by_seed": {"<seed>": float}, # included values only; keys are seeds as strings
          "missing_seeds": [seed],             # planned, no result (not run / not collected)
          "failed_seeds": [seed],              # run failed or value unusable (e.g. NaN)
          "excluded": [{"seed": seed, "reason": str}]   # explicitly excluded, with a reason
        }}
      }],
      "differences": [{
        "group_label": str, "baseline_label": str, "metric": str,
        "kind": "paired"|"unpaired",
        "n": int,                              # pairs (paired) or the group's n (unpaired)
        "n_baseline": int|null,                # the baseline's n (unpaired only)
        "mean_diff", "ci_low", "ci_high": float|null      # group − baseline
      }],
      "coverage": {"planned_runs", "included", "missing", "failed", "excluded": int,
                   "complete": bool, "notes": [str]},
      "provenance": [{"group_label", "seed", "metric", "value",
                      "evaluation_id", "evaluator", "artifact_id", "artifact_digest",
                      "attempt_id", "job_id", "cluster_id", "run_id", "run_spec_hash",
                      "code_digest", "git_commit", "env_gpu", "env_torch"}],   # ids: str|null
      "incomparable": [{"reason": str, "details": object|str|null}],
      "limitations": [str],
      "integrity": {                           # optional (evidence service); rendered when present
        "attempt_selection": str|null,         # latest_valid | first_valid (the contract's rule)
        "evaluations_pinned": bool = false,    # evaluation_refs chose the evaluations, not the rule
        "post_hoc": [{"group_label", "seed", "run_id"}],          # seeds added after results existed
        "late_exclusions": [{"group_label", "seed", "run_id", "reason"}],  # excluded after observation
        "heterogeneity": [{"dimension": str, "values": {value: runs}, "unknown": int,
                           "within_groups": [label]}],             # gpu / torch / code_digest
        "multiple_valid_attempts": [{"group_label", "seed", "run_id", "valid_attempts",
                                     "selected_attempt_id", "rule", "rule_attempt_id",
                                     "overrides_rule": bool}],   # overrides: pinned ≠ rule's pick
        "notes": [str]
      }
    }

A seed is an int or a string (``"none"`` for seedless runs); it is compared as a string.
Unknown keys are ignored (forward compatibility). The bundle is **rejected** with
``InvalidArgument`` — rather than rendered wrongly — unless:

* per metric cell: ``n == len(values_by_seed)``; included/missing/failed/excluded seeds are
  disjoint and ``n + |missing| + |failed| + |excluded| == expected_n`` (nothing leaves the
  denominator silently); all numbers finite;
* ``mean/std/sem/ci_low/ci_high/min/max`` equal :func:`hpclab.services.stats.summarize` of
  ``values_by_seed`` at ``ci_level`` (the renderers re-derive, they do not trust);
* each difference references existing groups/metrics and equals the recomputation: paired =
  :func:`hpclab.services.stats.paired_differences` of the two ``values_by_seed`` (seeds in
  both); unpaired = :func:`hpclab.services.stats.unpaired_difference` (Welch);
* provenance rows reference an existing group, metric and one of that cell's seeds, there is
  at most one row per (group, metric, seed) — two traces for one number would be ambiguous —
  and rows for included seeds carry the same value;
* coverage counts add up to ``planned_runs`` and ``complete`` implies no gaps;
* group labels are unique and each metric is declared at most once.

Rendering is deterministic (golden-tested): groups in bundle order; metrics in
``metric_decls`` order, then undeclared ones by name; seeds numerically. Output never contains
p-values or significance claims; the missing/failed/excluded section is never omitted.

Cell tags (``evidence_verify`` / ``trace``): :func:`cell_ids` gives every (group, metric) cell a
stable, slash- and brace-free id ``<group-slug>.<metric-slug>``; a tagged number is
``<base>.<stat>`` (``mean``, ``std``, ``sem``, ...) or ``<base>.seed-<seed>``. With
``render_latex(..., hpcval_bundle_id=<id>)`` every printed number becomes
``\\hpcval{<id>:<base>.<stat>}{<number>}`` and the table carries
``\\providecommand{\\hpcval}[2]{#2}``, so the tags print only the number and can be stripped.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hpclab.domain.errors import InvalidArgument
from hpclab.domain.hashing import canonical_json, short
from hpclab.services.stats import paired_differences, summarize, unpaired_difference

__all__ = [
    "CELL_STATS",
    "CSV_COLUMNS",
    "EVIDENCE_SCHEMA",
    "HPCVAL_PREAMBLE",
    "VALIDITY_LIMITATION",
    "EvidenceBundle",
    "IntegrityInfo",
    "cell_ids",
    "latex_escape",
    "parse_bundle",
    "parse_cell_id",
    "render_csv",
    "render_latex",
    "render_markdown",
]

EVIDENCE_SCHEMA = "hpclab.evidence/v1"
VALIDITY_LIMITATION = (
    "Record consistency is not proof of scientific validity: this bundle shows that each number "
    "traces to recorded code, configuration, jobs and artifacts, not that the experimental design, "
    "data or evaluation code are correct."
)
DASH = "—"
# Recomputed statistics must agree to 1e-9 relative to the data's own magnitude (so a
# metric living around 1e-12 is checked as strictly as one around 1).
_REL_TOL = 1e-9
_MAX_LISTED_GAPS = 20

Scalar = str | int | float | bool | None
Seed = int | str

# ---------------------------------------------------------------------------
# Bundle model (validation of the dict contract)
# ---------------------------------------------------------------------------


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class ExperimentInfo(_In):
    id: str | None = None
    name: str | None = None
    question: str | None = None


class ContractInfo(_In):
    treatment: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    baseline: dict[str, Scalar] | None = None


class MetricDeclInfo(_In):
    name: str
    higher_is_better: bool | None = None
    unit: str | None = None


class Exclusion(_In):
    seed: Seed
    reason: str


class MetricCell(_In):
    n: int = Field(ge=0)
    expected_n: int = Field(ge=0)
    mean: float | None = None
    std: float | None = None
    sem: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    min: float | None = None
    max: float | None = None
    values_by_seed: dict[str, float] = Field(default_factory=dict)
    missing_seeds: list[Seed] = Field(default_factory=list)
    failed_seeds: list[Seed] = Field(default_factory=list)
    excluded: list[Exclusion] = Field(default_factory=list)

    @field_validator("values_by_seed", mode="before")
    @classmethod
    def _str_keys(cls, v: Any) -> Any:
        return {str(k): val for k, val in v.items()} if isinstance(v, Mapping) else v

    @model_validator(mode="after")
    def _accounting(self) -> MetricCell:
        buckets = {
            "values_by_seed": list(self.values_by_seed),
            "missing_seeds": [str(s) for s in self.missing_seeds],
            "failed_seeds": [str(s) for s in self.failed_seeds],
            "excluded": [str(e.seed) for e in self.excluded],
        }
        seen: dict[str, str] = {}
        for bucket, seeds in buckets.items():
            for seed in seeds:
                if seed in seen:
                    raise ValueError(f"seed {seed} appears in both {seen[seed]} and {bucket}")
                seen[seed] = bucket
        if self.n != len(self.values_by_seed):
            raise ValueError(f"n={self.n} but values_by_seed has {len(self.values_by_seed)} values")
        if len(seen) != self.expected_n:
            raise ValueError(
                f"expected_n={self.expected_n} but included+missing+failed+excluded seeds = {len(seen)}"
            )
        return self

    def seed_status(self) -> dict[str, str]:
        """Seed → included | missing | failed | excluded, in seed order."""
        status = {s: "included" for s in self.values_by_seed}
        status.update({str(s): "missing" for s in self.missing_seeds})
        status.update({str(s): "failed" for s in self.failed_seeds})
        status.update({str(e.seed): "excluded" for e in self.excluded})
        return {s: status[s] for s in sorted(status, key=_seed_order)}

    def ordered_values(self) -> list[float]:
        return [self.values_by_seed[s] for s in sorted(self.values_by_seed, key=_seed_order)]

    def exclusion_reasons(self) -> dict[str, str]:
        return {str(e.seed): e.reason for e in self.excluded}

    def seeds(self) -> set[str]:
        """Every expected seed (included, missing, failed or excluded)."""
        return set(self.seed_status())


class Group(_In):
    key: dict[str, Scalar] = Field(default_factory=dict)
    label: str
    is_baseline: bool = False
    metrics: dict[str, MetricCell] = Field(default_factory=dict)


class Difference(_In):
    group_label: str
    baseline_label: str
    metric: str
    kind: Literal["paired", "unpaired"]
    n: int = Field(ge=0)
    n_baseline: int | None = None
    mean_diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class CoverageInfo(_In):
    planned_runs: int = Field(ge=0)
    included: int = Field(ge=0)
    missing: int = Field(ge=0)
    failed: int = Field(ge=0)
    excluded: int = Field(ge=0)
    complete: bool
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _adds_up(self) -> CoverageInfo:
        total = self.included + self.missing + self.failed + self.excluded
        if total != self.planned_runs:
            raise ValueError(
                f"coverage: included+missing+failed+excluded = {total} but planned_runs = {self.planned_runs}"
            )
        if self.complete and self.missing + self.failed + self.excluded:
            raise ValueError(
                "coverage.complete is true but some planned runs are missing, failed or excluded"
            )
        return self


class ProvenanceRow(_In):
    group_label: str
    seed: Seed | None = None
    metric: str
    value: float | None = None
    evaluation_id: str | None = None
    evaluator: str | None = None
    artifact_id: str | None = None
    artifact_digest: str | None = None
    attempt_id: str | None = None
    job_id: str | None = None
    cluster_id: str | None = None
    run_id: str | None = None
    run_spec_hash: str | None = None
    code_digest: str | None = None
    git_commit: str | None = None
    env_gpu: str | None = None
    env_torch: str | None = None

    @property
    def seed_key(self) -> str:
        return "none" if self.seed is None else str(self.seed)


class Incomparable(_In):
    reason: str
    details: dict[str, Any] | str | None = None


class PostHocSeed(_In):
    group_label: str
    seed: Seed | None = None
    run_id: str | None = None


class LateExclusion(_In):
    group_label: str
    seed: Seed | None = None
    run_id: str | None = None
    reason: str


class Heterogeneity(_In):
    """Distinct recorded values of one environment dimension over the included runs."""

    dimension: str
    values: dict[str, int] = Field(default_factory=dict)
    unknown: int = Field(default=0, ge=0)
    within_groups: list[str] = Field(default_factory=list)

    @property
    def mixed(self) -> bool:
        return len(self.values) > 1


_PINNED = "evaluation_refs"


class MultiAttempt(_In):
    group_label: str
    seed: Seed | None = None
    run_id: str | None = None
    valid_attempts: int = Field(ge=2)
    selected_attempt_id: str | None = None
    rule: str | None = None  # what chose the attempt: the contract's rule or "evaluation_refs"
    rule_attempt_id: str | None = None  # the attempt the contract's rule picks
    overrides_rule: bool = False  # the pinned attempt is not the one the contract's rule picks


class IntegrityInfo(_In):
    """Disclosures that change how numbers should be read (never judgements)."""

    attempt_selection: str | None = None
    evaluations_pinned: bool = False
    post_hoc: list[PostHocSeed] = Field(default_factory=list)
    late_exclusions: list[LateExclusion] = Field(default_factory=list)
    heterogeneity: list[Heterogeneity] = Field(default_factory=list)
    multiple_valid_attempts: list[MultiAttempt] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def pinned(self) -> bool:
        """The caller pinned the evaluations (``evaluation_refs``): the contract's rule did not choose."""
        return self.evaluations_pinned or any(m.rule == _PINNED for m in self.multiple_valid_attempts)

    def labels(self) -> set[str]:
        rows: list[PostHocSeed | LateExclusion | MultiAttempt] = [
            *self.post_hoc,
            *self.late_exclusions,
            *self.multiple_valid_attempts,
        ]
        return {r.group_label for r in rows} | {g for h in self.heterogeneity for g in h.within_groups}


class EvidenceBundle(_In):
    """Validated ``hpclab.evidence/v1`` bundle (see the module docstring)."""

    model_config = ConfigDict(
        extra="ignore", frozen=True, allow_inf_nan=False, validate_by_name=True, validate_by_alias=True
    )

    schema_id: str = Field(alias="schema")
    title: str
    purpose: str = ""
    generated_at: str
    hpclab_version: str
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    experiment: ExperimentInfo = Field(default_factory=ExperimentInfo)
    contract: ContractInfo = Field(default_factory=ContractInfo)
    metric_decls: list[MetricDeclInfo] = Field(default_factory=list)
    groups: list[Group]
    differences: list[Difference] = Field(default_factory=list)
    coverage: CoverageInfo
    provenance: list[ProvenanceRow] = Field(default_factory=list)
    incomparable: list[Incomparable] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    integrity: IntegrityInfo | None = None

    @field_validator("schema_id")
    @classmethod
    def _schema(cls, v: str) -> str:
        if v != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported evidence schema {v!r}; expected {EVIDENCE_SCHEMA!r}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> EvidenceBundle:
        labels = [g.label for g in self.groups]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(f"group labels must be unique; duplicated: {dupes}")
        declared = [d.name for d in self.metric_decls]
        dupes = sorted({x for x in declared if declared.count(x) > 1})
        if dupes:
            raise ValueError(f"metric(s) {dupes} declared more than once in metric_decls")
        for group in self.groups:
            for name, cell in group.metrics.items():
                _verify_cell(group.label, name, cell, self.ci_level)
        groups = {g.label: g for g in self.groups}
        for i, diff in enumerate(self.differences):
            _verify_difference(i, diff, groups, self.ci_level)
        traced: dict[tuple[str, str, str], int] = {}
        for i, row in enumerate(self.provenance):
            _verify_provenance(i, row, groups)
            key = (row.group_label, row.metric, row.seed_key)
            if key in traced:
                raise ValueError(
                    f"provenance[{i}]: more than one provenance row for group {key[0]!r} metric "
                    f"{key[1]!r} seed {key[2]} (also provenance[{traced[key]}])"
                )
            traced[key] = i
        if self.integrity is not None:
            unknown = sorted(self.integrity.labels() - set(groups))
            if unknown:
                raise ValueError(f"integrity references unknown group(s) {unknown}")
        return self

    def group(self, label: str) -> Group:
        return next(g for g in self.groups if g.label == label)

    def metric_order(self) -> list[str]:
        declared = [d.name for d in self.metric_decls]
        extra = sorted({m for g in self.groups for m in g.metrics} - set(declared))
        return [*declared, *extra]

    def gap_cells(self) -> int:
        """Metric cells with fewer included values than planned.

        Can be non-zero even when ``coverage.complete`` (run level): a finished, evaluated run
        whose evaluation lacks one metric leaves a gap in that metric's cell only.
        """
        return sum(1 for g in self.groups for c in g.metrics.values() if c.n < c.expected_n)

    def decl(self, metric: str) -> MetricDeclInfo:
        for d in self.metric_decls:
            if d.name == metric:
                return d
        return MetricDeclInfo(name=metric)


def parse_bundle(bundle: Mapping[str, Any] | EvidenceBundle) -> EvidenceBundle:
    """Validate a bundle dict; raises ``InvalidArgument`` listing the first problems."""
    if isinstance(bundle, EvidenceBundle):
        return bundle
    if not isinstance(bundle, Mapping):
        raise InvalidArgument(f"invalid evidence bundle: expected an object, got {type(bundle).__name__}")
    try:
        return EvidenceBundle.model_validate(dict(bundle))
    except ValidationError as exc:
        errors = [
            {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]} for err in exc.errors()[:20]
        ]
        first = f"{errors[0]['loc']}: {errors[0]['msg']}" if errors else str(exc)
        raise InvalidArgument(f"invalid evidence bundle: {first}", details={"errors": errors}) from None


# ---------------------------------------------------------------------------
# Cell ids (\hpcval tags, trace refs)
# ---------------------------------------------------------------------------

HPCVAL_PREAMBLE = r"\providecommand{\hpcval}[2]{#2}"
# Statistics a tag may name; ``seed-<seed>`` names one per-seed value.
CELL_STATS = ("mean", "std", "sem", "n", "min", "max", "ci_low", "ci_high")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.(?:[a-z_]+|seed-[A-Za-z0-9_-]+)$")


def _unique_slugs(names: Sequence[str], fallback: str) -> dict[str, str]:
    """Slug per name: only ``[A-Za-z0-9_-]`` (safe in URIs, LaTeX arguments and as the
    ``.``-separated parts of a cell id); collisions get a positional suffix."""
    raw = [(_SLUG_RE.sub("-", n).strip("-")[:40] or f"{fallback}{i + 1}") for i, n in enumerate(names)]
    out: dict[str, str] = {}
    used: set[str] = set()
    for i, (name, slug) in enumerate(zip(names, raw, strict=True)):
        candidate = slug
        if raw.count(slug) > 1 or candidate in used:
            candidate = f"{slug}-{i + 1}"
            while candidate in used:
                candidate += "x"
        used.add(candidate)
        out[name] = candidate
    return out


def cell_ids(bundle: Mapping[str, Any] | EvidenceBundle) -> dict[tuple[str, str], str]:
    """Stable base id ``<group-slug>.<metric-slug>`` for every (group label, metric) cell.

    Depends only on the group order and the metric order of the bundle, so the evidence
    service can store the map in the manifest and a later verifier resolves the same ids.
    """
    b = parse_bundle(bundle)
    groups = _unique_slugs([g.label for g in b.groups], "g")
    metrics = _unique_slugs(b.metric_order(), "m")
    return {(g.label, m): f"{groups[g.label]}.{metrics[m]}" for g in b.groups for m in b.metric_order()}


def parse_cell_id(cell_id: str) -> tuple[str, str]:
    """Split ``<group>.<metric>.<stat>`` into (base, stat); InvalidArgument when malformed."""
    if not _CELL_ID_RE.match(cell_id or ""):
        raise InvalidArgument(
            f"malformed cell id {cell_id!r}: expected <group>.<metric>.<stat> or <group>.<metric>.seed-<seed>"
        )
    base, stat = cell_id.rsplit(".", 1)
    if not stat.startswith("seed-") and stat not in CELL_STATS:
        raise InvalidArgument(f"unknown statistic {stat!r} in cell id {cell_id!r}; known: {list(CELL_STATS)}")
    return base, stat


def _close(got: float | None, want: float | None, scale: float) -> bool:
    """Equal within ``_REL_TOL`` of the value or of ``scale`` (the largest |input value|).

    The scale term absorbs harmless rounding of statistics near zero (a mean or interval
    bound of ~1e-17 for data of order 1) without accepting wrong values for tiny data.
    """
    if got is None or want is None:
        return got is None and want is None
    return math.isclose(got, want, rel_tol=_REL_TOL, abs_tol=_REL_TOL * scale)


def _scale(*cells: MetricCell) -> float:
    return max((abs(v) for c in cells for v in c.values_by_seed.values()), default=0.0)


def _verify_cell(label: str, metric: str, cell: MetricCell, ci_level: float) -> None:
    expected = summarize(cell.ordered_values(), ci_level)
    scale = _scale(cell)
    for field in ("mean", "std", "sem", "ci_low", "ci_high", "min", "max"):
        got, want = getattr(cell, field), getattr(expected, field)
        if not _close(got, want, scale):
            raise ValueError(
                f"group {label!r} metric {metric!r}: {field}={got!r} but values_by_seed give {want!r} "
                f"(ci_level={ci_level})"
            )


def _verify_difference(i: int, diff: Difference, groups: Mapping[str, Group], ci_level: float) -> None:
    where = f"differences[{i}]"
    cells = []
    for label in (diff.group_label, diff.baseline_label):
        if label not in groups:
            raise ValueError(f"{where}: unknown group {label!r}")
        if diff.metric not in groups[label].metrics:
            raise ValueError(f"{where}: group {label!r} has no metric {diff.metric!r}")
        cells.append(groups[label].metrics[diff.metric])
    group_cell, base_cell = cells
    if diff.kind == "paired":
        p = paired_differences(group_cell.values_by_seed, base_cell.values_by_seed, ci_level)
        want: dict[str, Any] = {
            "n": p.n_pairs,
            "n_baseline": None,
            "mean_diff": p.mean_diff,
            "ci_low": p.ci_low,
            "ci_high": p.ci_high,
        }
    else:
        u = unpaired_difference(group_cell.ordered_values(), base_cell.ordered_values(), ci_level)
        want = {
            "n": u.n_a,
            "n_baseline": u.n_b,
            "mean_diff": u.mean_diff,
            "ci_low": u.ci_low,
            "ci_high": u.ci_high,
        }
    for field, value in want.items():
        got = getattr(diff, field)
        ok = (
            got == value
            if field in ("n", "n_baseline")
            else _close(got, value, _scale(group_cell, base_cell))
        )
        if not ok:
            raise ValueError(f"{where} ({diff.kind}): {field}={got!r} but the values give {value!r}")


def _verify_provenance(i: int, row: ProvenanceRow, groups: Mapping[str, Group]) -> None:
    group = groups.get(row.group_label)
    if group is None:
        raise ValueError(f"provenance[{i}]: unknown group {row.group_label!r}")
    cell = group.metrics.get(row.metric)
    if cell is None:
        raise ValueError(f"provenance[{i}]: group {row.group_label!r} has no metric {row.metric!r}")
    seeds = cell.seeds()
    if row.seed_key not in seeds:
        raise ValueError(
            f"provenance[{i}]: seed {row.seed_key} is not one of the seeds of group {row.group_label!r} "
            f"metric {row.metric!r}: {sorted(seeds, key=_seed_order)}"
        )
    included = cell.values_by_seed.get(row.seed_key)
    if included is not None and not _close(row.value, included, abs(included)):
        raise ValueError(
            f"provenance[{i}]: value {row.value!r} differs from values_by_seed[{row.seed_key}]={included!r}"
        )


def _seed_order(seed: str) -> tuple[int, int, str]:
    return (0, int(seed), "") if re.fullmatch(r"-?\d+", seed) else (1, 0, seed)


# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------


def _fmt(x: float | None, digits: int) -> str:
    """Fixed decimals; scientific when the value would round to 0 or is huge; None → '—'."""
    if x is None:
        return DASH
    if x == 0:
        return f"{0.0:.{digits}f}"
    ax = abs(x)
    if ax < 0.5 * 10.0**-digits or ax >= 1e7:
        return f"{x:.{max(digits - 1, 1)}e}"
    text = f"{x:.{digits}f}"
    return text.lstrip("-") if float(text) == 0 else text


def _signed(x: float | None, digits: int) -> str:
    text = _fmt(x, digits)
    return f"+{text}" if x is not None and x > 0 and not text.startswith("+") else text


def _pct(level: float) -> str:
    return f"{level * 100:g}%"


def _arrow(decl: MetricDeclInfo) -> str:
    return {True: "↑", False: "↓", None: ""}[decl.higher_is_better]


def _details_text(details: dict[str, Any] | str | None) -> str:
    if details is None:
        return ""
    if isinstance(details, str):
        return details
    return "; ".join(
        f"{k}={v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)}"
        for k, v in sorted(details.items())
    )


def _limitations(b: EvidenceBundle) -> list[str]:
    out = [VALIDITY_LIMITATION]
    for item in b.limitations:
        if item not in out:
            out.append(item)
    return out


def _check_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 10:
        raise InvalidArgument(f"digits must be an integer in [0, 10], got {digits!r}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")


def _md(text: str) -> str:
    """Escape inline Markdown (safe inside table cells); newlines become spaces."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def _md_code(text: str | None) -> str:
    if not text:
        return DASH
    fence = "``" if "`" in text else "`"
    pad = " " if "`" in text else ""
    return f"{fence}{pad}{text}{pad}{fence}".replace("|", "\\|")


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _md_metric_header(b: EvidenceBundle, metric: str) -> str:
    decl = b.decl(metric)
    unit = f" ({_md(decl.unit)})" if decl.unit else ""
    arrow = f" {_arrow(decl)}" if decl.higher_is_better is not None else ""
    return f"{_md(metric)}{unit}{arrow}"


def _md_group_label(group: Group) -> str:
    return _md(group.label) + (" (baseline)" if group.is_baseline else "")


def render_markdown(
    bundle: Mapping[str, Any] | EvidenceBundle, *, digits: int = 4, max_provenance_rows: int = 50
) -> str:
    """Human-readable evidence summary (see the module docstring for guarantees)."""
    _check_digits(digits)
    b = parse_bundle(bundle)
    metrics = b.metric_order()
    sections = [
        _md_header(b),
        _md_results(b, metrics, digits),
        _md_gaps(b, metrics),
        _md_per_seed(b, metrics, digits),
        _md_statistics(b, metrics, digits),
        _md_differences(b, digits),
        _md_incomparable(b),
        *([_md_integrity(b.integrity)] if b.integrity is not None else []),
        _md_definitions(b),
        _md_provenance(b, metrics, max(0, max_provenance_rows)),
        "## Limitations\n\n" + "\n".join(f"- {_md(item)}" for item in _limitations(b)),
    ]
    return "\n\n".join(s.strip("\n") for s in sections) + "\n"


def _md_header(b: EvidenceBundle) -> str:
    lines = [f"# {_md(b.title)}", ""]
    if b.purpose.strip():
        lines += [_md(b.purpose), ""]
    lines.append(
        f"- **Generated:** {_md(b.generated_at)} by hpclab {_md(b.hpclab_version)} (`{EVIDENCE_SCHEMA}`)"
    )
    exp = b.experiment
    if exp.name or exp.id:
        parts = [_md(exp.name)] if exp.name else []
        if exp.id:
            parts.append(f"({_md_code(exp.id)})")
        lines.append("- **Experiment:** " + " ".join(parts))
    if exp.question:
        lines.append(f"- **Question:** {_md(exp.question)}")
    c = b.contract
    lines.append(f"- **Treatment (may differ):** {_md(', '.join(c.treatment)) or DASH}")
    lines.append(f"- **Invariants (must match):** {_md(', '.join(c.invariants)) or DASH}")
    if c.baseline:
        base = ", ".join(f"{k}={_scalar(v)}" for k, v in sorted(c.baseline.items()))
        lines.append(f"- **Baseline:** {_md(base)}")
    lines += ["", _md_coverage_banner(b)]
    return "\n".join(lines)


def _scalar(v: Scalar) -> str:
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    return str(v)


def _md_coverage_banner(b: EvidenceBundle) -> str:
    cov = b.coverage
    if cov.complete:
        line = f"> **Coverage: complete.** All {cov.planned_runs} planned runs are included."
        if gaps := b.gap_cells():
            line = (
                f"> **Coverage: runs complete, metrics incomplete.** All {cov.planned_runs} planned runs "
                f"are included, but {gaps} metric cell(s) have fewer values than planned; see "
                "*Missing, failed and excluded runs*."
            )
    else:
        line = (
            f"> **Coverage: incomplete.** {cov.included} of {cov.planned_runs} planned runs included "
            f"({cov.missing} missing, {cov.failed} failed, {cov.excluded} excluded). "
            "Every gap is listed under *Missing, failed and excluded runs*."
        )
    if b.incomparable:
        line += (
            f"\n>\n> **Warning:** {len(b.incomparable)} incomparability issue(s) flagged; "
            "see *Incomparable results* before reading the tables."
        )
    return line


def _md_results(b: EvidenceBundle, metrics: list[str], digits: int) -> str:
    intro = (
        "## Results\n\nEach cell is mean ± sample standard deviation over seeds, followed by "
        "(included/expected runs). ↑ higher is better, ↓ lower is better; "
        f"{DASH} means undefined (a standard deviation needs at least 2 seeds)."
    )
    if not b.groups or not metrics:
        return intro + "\n\nNo results: the bundle has no groups or no metrics."
    header = ["Group", *(_md_metric_header(b, m) for m in metrics)]
    rows = [[_md_group_label(g), *(_md_cell(g.metrics.get(m), digits) for m in metrics)] for g in b.groups]
    return intro + "\n\n" + _md_table(header, rows)


def _md_cell(cell: MetricCell | None, digits: int) -> str:
    if cell is None:
        return "not reported"
    counts = f"({cell.n}/{cell.expected_n})"
    if cell.n == 0:
        return f"{DASH} {counts}"
    return f"{_fmt(cell.mean, digits)} ± {_fmt(cell.std, digits)} {counts}"


def _md_gaps(b: EvidenceBundle, metrics: list[str]) -> str:
    cov = b.coverage
    lines = [
        "## Missing, failed and excluded runs",
        "",
        f"Planned runs: {cov.planned_runs}; included {cov.included}, missing {cov.missing}, "
        f"failed {cov.failed}, excluded {cov.excluded}. Nothing is dropped from a denominator silently: "
        "every expected seed is counted as included, missing, failed or excluded.",
        "",
    ]
    rows = []
    for g in b.groups:
        for m in metrics:
            cell = g.metrics.get(m)
            if cell is None or cell.n == cell.expected_n:
                continue
            excluded = ", ".join(f"{e.seed} ({e.reason})" for e in cell.excluded)
            rows.append(
                [
                    _md(g.label),
                    _md(m),
                    f"{cell.n}/{cell.expected_n}",
                    _md(_seed_list(cell.missing_seeds)) or DASH,
                    _md(_seed_list(cell.failed_seeds)) or DASH,
                    _md(excluded) or DASH,
                ]
            )
    not_reported = [f"{g.label}/{m}" for g in b.groups for m in metrics if m not in g.metrics]
    if rows:
        header = ["Group", "Metric", "Included", "Missing seeds", "Failed seeds", "Excluded seeds (reason)"]
        lines.append(_md_table(header, rows))
    elif cov.missing + cov.failed + cov.excluded == 0:
        lines.append("No gaps: every planned run contributed a value to every reported metric.")
    else:
        lines.append("No per-metric gaps are recorded; see the coverage counts and notes.")
    if not_reported:
        lines += ["", f"Metrics not reported at all: {_md(', '.join(not_reported))}."]
    if cov.notes:
        lines += ["", "Notes:", *(f"- {_md(n)}" for n in cov.notes)]
    return "\n".join(lines)


def _seed_list(seeds: Sequence[Seed]) -> str:
    return ", ".join(sorted((str(s) for s in seeds), key=_seed_order))


def _md_per_seed(b: EvidenceBundle, metrics: list[str], digits: int) -> str:
    blocks = ["## Per-seed values"]
    for m in metrics:
        cells = [(g, g.metrics[m]) for g in b.groups if m in g.metrics]
        seeds = sorted({s for _, c in cells for s in c.seed_status()}, key=_seed_order)
        if not seeds:
            continue
        rows = []
        for g, c in cells:
            status = c.seed_status()
            row = [_md(g.label)]
            for s in seeds:
                st = status.get(s)
                row.append(_fmt(c.values_by_seed[s], digits) if st == "included" else (st or ""))
            rows.append(row)
        header = ["Group", *(f"seed {_md(s)}" for s in seeds)]
        blocks += [f"### {_md_metric_header(b, m)}", _md_table(header, rows)]
    if len(blocks) == 1:
        blocks.append("No per-seed values.")
    return "\n\n".join(blocks)


def _md_statistics(b: EvidenceBundle, metrics: list[str], digits: int) -> str:
    level = _pct(b.ci_level)
    header = ["Group", "Metric", "n", "Mean", "Std", "SEM", f"{level} CI", "Min", "Max"]
    rows = []
    for g in b.groups:
        for m in metrics:
            c = g.metrics.get(m)
            if c is None:
                continue
            rows.append(
                [
                    _md(g.label),
                    _md(m),
                    f"{c.n}/{c.expected_n}",
                    _fmt(c.mean, digits),
                    _fmt(c.std, digits),
                    _fmt(c.sem, digits),
                    _interval(c.ci_low, c.ci_high, digits),
                    _fmt(c.min, digits),
                    _fmt(c.max, digits),
                ]
            )
    body = _md_table(header, rows) if rows else "No statistics."
    return "## Summary statistics\n\n" + body


def _interval(lo: float | None, hi: float | None, digits: int) -> str:
    if lo is None or hi is None:
        return DASH
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _md_differences(b: EvidenceBundle, digits: int) -> str:
    level = _pct(b.ci_level)
    intro = (
        "## Differences vs baseline (descriptive)\n\n"
        "Group − baseline, in the metric's units; read the sign together with the metric's direction. "
        "Paired rows use only seeds present in both groups; unpaired rows compare means (Welch). "
        f"The {level} interval describes the uncertainty of the estimate. These are descriptive "
        "summaries, not hypothesis tests, and they do not by themselves establish that one method is better."
    )
    if not b.differences:
        return intro + "\n\nNo differences were computed for this bundle."
    rows = []
    for d in b.differences:
        n = f"{d.n} pairs" if d.kind == "paired" else f"{d.n} vs {d.n_baseline}"
        decl = b.decl(d.metric)
        metric = _md(d.metric) + (f" {_arrow(decl)}" if decl.higher_is_better is not None else "")
        rows.append(
            [
                _md(d.group_label),
                _md(d.baseline_label),
                metric,
                d.kind,
                n,
                _signed(d.mean_diff, digits),
                _interval(d.ci_low, d.ci_high, digits),
            ]
        )
    header = ["Group", "Baseline", "Metric", "Kind", "n", "Mean diff", f"{level} CI"]
    return intro + "\n\n" + _md_table(header, rows)


def _md_incomparable(b: EvidenceBundle) -> str:
    if not b.incomparable:
        return "## Incomparable results\n\nNone flagged by the comparison contract checks."
    items = []
    for item in b.incomparable:
        details = _details_text(item.details)
        items.append(f"- **{_md(item.reason)}**" + (f": {_md(details)}" if details else ""))
    return (
        "## Incomparable results\n\n"
        "**Warning:** the comparison contract flagged the following; numbers affected by them should "
        "not be compared directly.\n\n" + "\n".join(items)
    )


def _seed_text(seed: Seed | None) -> str:
    return "none" if seed is None else str(seed)


def _heterogeneity_text(h: Heterogeneity) -> str:
    values = ", ".join(f"{v} ×{n}" for v, n in sorted(h.values.items(), key=lambda kv: (-kv[1], kv[0])))
    text = f"{h.dimension}: {values or 'not recorded'}"
    if h.unknown:
        text += f" (not recorded for {h.unknown} run(s))"
    if h.within_groups:
        text += f"; mixed within {', '.join(h.within_groups)}"
    return text


def _md_integrity(info: IntegrityInfo) -> str:
    lines = [
        "## Integrity",
        "",
        "Disclosures that affect how the numbers should be read. hpclab records them; it does not judge them.",
        "",
    ]
    post = "; ".join(f"{p.group_label} seed {_seed_text(p.seed)}" for p in info.post_hoc)
    lines.append(f"- **Seeds added after results existed (post hoc):** {_md(post) if post else 'none'}.")
    late = "; ".join(f"{e.group_label} seed {_seed_text(e.seed)} ({e.reason})" for e in info.late_exclusions)
    lines.append(f"- **Exclusions decided after the result was observed:** {_md(late) if late else 'none'}.")
    rule = info.attempt_selection or "unspecified"
    if info.pinned:
        lines.append(
            "- **Evaluations pinned by the caller (evaluation\\_refs):** the reported values come from "
            f"evaluations chosen by hand, not by the contract's attempt selection ({_md(rule)})."
        )
    if info.multiple_valid_attempts:
        multi = "; ".join(
            f"{m.group_label} seed {_seed_text(m.seed)} ({m.valid_attempts} valid attempts"
            + (f"; not the attempt {rule} picks)" if info.pinned and m.overrides_rule else ")")
            for m in info.multiple_valid_attempts
        )
        how = "the pinned evaluation picks one" if info.pinned else f"rule {rule} picks one"
        lines.append(f"- **Runs with several valid attempts** ({_md(how)}): {_md(multi)}.")
    elif info.pinned:
        lines.append("- **Attempt selection:** pinned evaluations; no run had more than one valid attempt.")
    else:
        lines.append(f"- **Attempt selection:** {_md(rule)}; no run had more than one valid attempt.")
    if info.heterogeneity:
        lines.append("- **Recorded environment of the included runs:**")
        lines += [
            f"  - {'**mixed** ' if h.mixed else ''}{_md(_heterogeneity_text(h))}" for h in info.heterogeneity
        ]
    lines += [f"- {_md(n)}" for n in info.notes]
    return "\n".join(lines)


def _md_definitions(b: EvidenceBundle) -> str:
    level = _pct(b.ci_level)
    q = f"{(1 + b.ci_level) / 2:g}"
    lines = [
        "## Definitions",
        "",
        "- **n / expected:** seeds with an included value / seeds planned for the group.",
        "- **Mean:** arithmetic mean of the included per-seed values.",
        "- **Std:** sample standard deviation with denominator n − 1; undefined (—) for n < 2, "
        "never reported as 0.",
        "- **SEM:** Std / √n.",
        f"- **{level} CI:** mean ± t(q={q}, df=n − 1) × SEM (two-sided Student-t interval).",
        "- **Paired difference:** d_s = group_s − baseline_s for each seed s present in both groups; "
        "summarized like a group (n = number of pairs). Seeds present in only one group are not paired.",
        "- **Unpaired difference:** mean(group) − mean(baseline); SE = √(s²_g/n_g + s²_b/n_b); "
        f"Welch–Satterthwaite df; {level} t interval.",
        "- **Missing:** planned but no result. **Failed:** the run failed or its value was unusable "
        "(e.g. non-finite). **Excluded:** removed explicitly, with the stated reason.",
    ]
    return "\n".join(lines)


def _md_provenance(b: EvidenceBundle, metrics: list[str], limit: int) -> str:
    intro = (
        "## Provenance\n\n"
        "Each value traces to evaluation → artifact digest → attempt → scheduler job → run (spec hash) "
        "→ code."
    )
    rows = _sorted_provenance(b, metrics)
    parts = [intro]
    if rows:
        header = [
            "Group",
            "Seed",
            "Metric",
            "Value",
            "Evaluation",
            "Artifact digest",
            "Attempt",
            "Job",
            "Run",
            "Code",
            "Environment",
        ]
        parts.append(_md_table(header, [_md_provenance_row(r) for r in rows[:limit]]))
        if len(rows) > limit:
            parts.append(
                f"Showing {limit} of {len(rows)} provenance rows; the complete list is in the bundle "
                "manifest and the CSV export."
            )
    else:
        parts.append("No provenance rows are recorded in this bundle.")
    untraced = _untraced(b, metrics)
    if untraced:
        listed = ", ".join(untraced[:_MAX_LISTED_GAPS]) + (" …" if len(untraced) > _MAX_LISTED_GAPS else "")
        parts.append(
            f"**Warning:** {len(untraced)} included value(s) have no provenance row and cannot be "
            f"traced: {_md(listed)}."
        )
    return "\n\n".join(parts)


def _md_provenance_row(r: ProvenanceRow) -> list[str]:
    job = f"{r.cluster_id}:{r.job_id}" if r.cluster_id and r.job_id else (r.job_id or "")
    code = r.git_commit[:12] if r.git_commit else (f"sha256:{short(r.code_digest)}" if r.code_digest else "")
    env = "; ".join(x for x in (r.env_gpu, f"torch {r.env_torch}" if r.env_torch else None) if x)
    digest = f"sha256:{short(r.artifact_digest)}" if r.artifact_digest else ""
    return [
        _md(r.group_label),
        _md(r.seed_key),
        _md(r.metric),
        repr(r.value) if r.value is not None else DASH,
        _md_code(r.evaluation_id),
        _md_code(digest),
        _md_code(r.attempt_id),
        _md_code(job),
        _md_code(r.run_id),
        _md_code(code),
        _md(env) or DASH,
    ]


def _sorted_provenance(b: EvidenceBundle, metrics: list[str]) -> list[ProvenanceRow]:
    group_index = {g.label: i for i, g in enumerate(b.groups)}
    metric_index = {m: i for i, m in enumerate(metrics)}
    return sorted(
        b.provenance,
        key=lambda r: (
            group_index[r.group_label],
            metric_index.get(r.metric, len(metrics)),
            r.metric,
            _seed_order(r.seed_key),
        ),
    )


def _untraced(b: EvidenceBundle, metrics: list[str]) -> list[str]:
    traced = {(r.group_label, r.metric, r.seed_key) for r in b.provenance}
    return [
        f"{g.label}/{m}/seed {s}"
        for g in b.groups
        for m in metrics
        if m in g.metrics
        for s in sorted(g.metrics[m].values_by_seed, key=_seed_order)
        if (g.label, m, s) not in traced
    ]


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}
_LABEL_RE = re.compile(r"^[A-Za-z0-9:._/-]+$")
_DAGGER = r"\textsuperscript{\dag}"


def _latex_row_head(text: str) -> str:
    r"""First cell of a table row (already escaped).

    It follows ``\\`` or a booktabs rule, which look ahead for ``*`` / ``[``: a label such as
    ``[ablation] mixup`` would be read as an optional length (compile error) and ``*mixup``
    as ``\\*`` (the star silently disappears). An empty group stops the look-ahead.
    """
    return "{}" + text if text.startswith(("[", "*")) else text


def latex_escape(text: str) -> str:
    """Escape LaTeX specials in one pass (no double escaping); newlines become spaces."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in " ".join(str(text).split()))


def _latex_number(x: float, digits: int) -> str:
    text = _fmt(x, digits)
    if "e" in text:
        mantissa, exp = text.split("e")
        return rf"{mantissa}\times10^{{{int(exp)}}}"
    return text


_WITHHELD = r"\textsuperscript{\S}"
_TAG_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class _LatexOptions:
    """Per-call LaTeX options that the cell/notes helpers need (defaults = plain table)."""

    def __init__(
        self,
        b: EvidenceBundle,
        *,
        digits: int,
        spread: Literal["std", "sem"],
        scale: float,
        min_n: int | None,
        bundle_id: str | None,
    ) -> None:
        self.digits = digits
        self.spread = spread
        self.scale = scale
        self.min_n = min_n
        self.bundle_id = bundle_id
        self.ids = cell_ids(b) if bundle_id is not None else {}

    def number(self, value: float, group: str, metric: str, stat: str) -> str:
        text = _latex_number(value * self.scale if self.scale != 1.0 else value, self.digits)
        if self.bundle_id is None:
            return text
        return rf"\hpcval{{{self.bundle_id}:{self.ids[(group, metric)]}.{stat}}}{{{text}}}"

    def withheld(self, cell: MetricCell) -> bool:
        return self.min_n is not None and 0 < cell.n < self.min_n


def render_latex(
    bundle: Mapping[str, Any] | EvidenceBundle,
    *,
    digits: int = 3,
    bold_best: bool = False,
    caption: str | None = None,
    label: str | None = None,
    metrics: Sequence[str] | None = None,
    group_header: str = "Method",
    wrap_table: bool = True,
    hpcval_bundle_id: str | None = None,
    groups: Sequence[str] | None = None,
    spread: Literal["std", "sem"] = "std",
    scale: float = 1.0,
    min_n: int | None = None,
) -> str:
    """booktabs table: rows = groups, columns = metrics, cells ``$mean \\pm std$``.

    Cells with fewer included runs than planned get a dagger; the footnote lists the
    missing / failed / excluded seeds. ``bold_best`` (off by default: bolding is a claim the
    author should make deliberately) bolds the best mean per metric at the displayed precision,
    only for metrics with a declared direction. ``caption`` is plain text (escaped); ``None``
    gives a caption stating the statistic. Requires ``\\usepackage{booktabs}``.

    Evidence options (all off by default): ``hpcval_bundle_id`` tags every number as
    ``\\hpcval{<bundle>:<cell id>}{<number>}`` (see :func:`cell_ids`); ``groups`` selects and
    orders rows by label; ``spread='sem'`` prints the standard error instead of the std;
    ``scale`` multiplies every printed value (e.g. 100 for percentages; stated in the header);
    ``min_n`` withholds means of cells with fewer included seeds (marked and footnoted).
    Integrity disclosures of the bundle (post-hoc seeds, late exclusions, mixed environment)
    are footnoted.
    """
    _check_digits(digits)
    if label is not None and not _LABEL_RE.match(label):
        raise InvalidArgument(f"LaTeX label {label!r} may only contain letters, digits and :._/-")
    if hpcval_bundle_id is not None and not _TAG_ID_RE.match(hpcval_bundle_id):
        raise InvalidArgument(f"bundle id {hpcval_bundle_id!r} may only contain letters, digits, _ and -")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not (math.isfinite(scale) and scale > 0)
    ):
        raise InvalidArgument(f"scale must be a finite number > 0, got {scale!r}")
    if min_n is not None and (isinstance(min_n, bool) or not isinstance(min_n, int) or min_n < 1):
        raise InvalidArgument(f"min_n must be an integer >= 1, got {min_n!r}")
    if spread not in ("std", "sem"):
        raise InvalidArgument(f"spread must be 'std' or 'sem', got {spread!r}")
    b = parse_bundle(bundle)
    cols = _latex_metrics(b, metrics)
    rows = _latex_groups(b, groups)
    opts = _LatexOptions(
        b, digits=digits, spread=spread, scale=float(scale), min_n=min_n, bundle_id=hpcval_bundle_id
    )
    best = _best_means(rows, b, cols, opts) if bold_best else set()
    header = [_latex_row_head(latex_escape(group_header)), *(_latex_metric_header(b, m, opts) for m in cols)]
    body = [
        " & ".join(
            [
                _latex_row_head(latex_escape(g.label)),
                *(_latex_cell(g, m, (g.label, m) in best, opts) for m in cols),
            ]
        )
        + r" \\"
        for g in rows
    ]
    tabular = [
        rf"\begin{{tabular}}{{l{'c' * len(cols)}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    notes = _latex_notes(b, rows, cols, opts)
    lines = _latex_preamble(b, opts)
    if wrap_table:
        cap = latex_escape(caption) if caption is not None else _default_caption(b, opts)
        lines += [r"\begin{table}[t]", r"\centering", rf"\caption{{{cap}}}"]
        if label:
            lines.append(rf"\label{{{label}}}")
        lines += [*tabular, *notes, r"\end{table}"]
    else:
        lines += [*tabular, *notes]
    return "\n".join(lines) + "\n"


def _latex_metrics(b: EvidenceBundle, metrics: Sequence[str] | None) -> list[str]:
    available = b.metric_order()
    if metrics is None:
        return available
    unknown = [m for m in metrics if m not in available]
    if unknown:
        raise InvalidArgument(f"unknown metrics {unknown}; the bundle has {available}")
    return list(dict.fromkeys(metrics))


def _latex_groups(b: EvidenceBundle, groups: Sequence[str] | None) -> list[Group]:
    if groups is None:
        return list(b.groups)
    labels = [g.label for g in b.groups]
    unknown = [g for g in groups if g not in labels]
    if unknown:
        raise InvalidArgument(f"unknown groups {unknown}; the bundle has {labels}")
    return [b.group(label) for label in dict.fromkeys(groups)]


def _latex_preamble(b: EvidenceBundle, opts: _LatexOptions) -> list[str]:
    cov = b.coverage
    gaps = b.gap_cells()
    if not cov.complete:
        state = "INCOMPLETE"
    elif gaps:
        state = f"all runs included; {gaps} metric cell(s) with fewer values than planned"
    else:
        state = "complete"
    # A LaTeX comment ends at a newline, so every interpolated value is flattened.
    title, version, when = (" ".join(x.split()) for x in (b.title, b.hpclab_version, b.generated_at))
    if opts.spread == "sem":
        cells = "% Cells: mean \\pm standard error of the mean (sample std / sqrt(n)) over seeds; --- = undefined (fewer than 2 seeds)."
    else:
        cells = "% Cells: mean \\pm sample std (n-1 denominator) over seeds; --- = undefined (fewer than 2 seeds)."
    lines = [
        f'% Generated by hpclab {version} from evidence bundle "{title}" ({EVIDENCE_SCHEMA}, {when}).',
        cells,
        f"% Coverage: {cov.included}/{cov.planned_runs} planned runs included ({state}); "
        f"missing {cov.missing}, failed {cov.failed}, excluded {cov.excluded}.",
        "% Requires \\usepackage{booktabs}.",
    ]
    if opts.scale != 1.0:
        lines.append(f"% Every printed value is multiplied by {opts.scale:g}.")
    if opts.bundle_id is not None:
        lines += [
            f"% Numbers are tagged \\hpcval{{{opts.bundle_id}:<cell>}}{{<value>}} for `hpclab evidence verify`;"
            " the macro below prints only the value.",
            HPCVAL_PREAMBLE,
        ]
    return lines


def _default_caption(b: EvidenceBundle, opts: _LatexOptions) -> str:
    if opts.spread == "sem":
        return f"{latex_escape(b.title)}. Mean $\\pm$ standard error of the mean over seeds."
    return f"{latex_escape(b.title)}. Mean $\\pm$ sample standard deviation over seeds ($n-1$ denominator)."


def _latex_metric_header(b: EvidenceBundle, metric: str, opts: _LatexOptions) -> str:
    decl = b.decl(metric)
    unit = f" ({latex_escape(decl.unit)})" if decl.unit else ""
    arrow = {True: r" $\uparrow$", False: r" $\downarrow$", None: ""}[decl.higher_is_better]
    scale = rf" ($\times {opts.scale:g}$)" if opts.scale != 1.0 else ""
    return f"{latex_escape(metric)}{unit}{scale}{arrow}"


def _latex_cell(group: Group, metric: str, bold: bool, opts: _LatexOptions) -> str:
    cell = group.metrics.get(metric)
    if cell is None:
        return "n/a"
    dagger = _DAGGER if cell.n < cell.expected_n else ""
    if cell.n == 0 or cell.mean is None:
        return "---" + dagger
    if opts.withheld(cell):
        return "---" + _WITHHELD + dagger
    mean = opts.number(cell.mean, group.label, metric, "mean")
    spread_value = cell.std if opts.spread == "std" else cell.sem
    if spread_value is None:
        core = rf"$\mathbf{{{mean}}} \pm$ ---" if bold else rf"${mean} \pm$ ---"
    else:
        spread = opts.number(spread_value, group.label, metric, opts.spread)
        core = rf"$\mathbf{{{mean} \pm {spread}}}$" if bold else rf"${mean} \pm {spread}$"
    return core + dagger


def _best_means(
    rows: Sequence[Group], b: EvidenceBundle, metrics: Sequence[str], opts: _LatexOptions
) -> set[tuple[str, str]]:
    """(group, metric) pairs holding the best mean at display precision (ties all win)."""
    best: set[tuple[str, str]] = set()
    for m in metrics:
        direction = b.decl(m).higher_is_better
        if direction is None:
            continue
        shown = {
            g.label: float(_fmt(c.mean * opts.scale, opts.digits))
            for g in rows
            if (c := g.metrics.get(m)) is not None and c.mean is not None and not opts.withheld(c)
        }
        if not shown:
            continue
        target = max(shown.values()) if direction else min(shown.values())
        best.update((label, m) for label, v in shown.items() if v == target)
    return best


def _latex_notes(
    b: EvidenceBundle, rows: Sequence[Group], metrics: Sequence[str], opts: _LatexOptions
) -> list[str]:
    gaps = []
    single_seed = False
    withheld = False
    for g in rows:
        for m in metrics:
            cell = g.metrics.get(m)
            if cell is None:
                continue
            single_seed = single_seed or cell.n == 1
            withheld = withheld or opts.withheld(cell)
            if cell.n < cell.expected_n:
                gaps.append(f"{latex_escape(g.label)}, {latex_escape(m)}: {_gap_text(cell)}")
    notes = []
    if gaps:
        notes.append(f"{_DAGGER}Fewer runs than planned: " + "; ".join(gaps) + ".")
    if single_seed:
        spread = "standard deviation" if opts.spread == "std" else "standard error"
        notes.append(f"---: {spread} undefined with a single seed.")
    if withheld:
        notes.append(f"{_WITHHELD}Fewer than {opts.min_n} included seeds; value withheld.")
    if b.incomparable:
        reasons = "; ".join(latex_escape(i.reason) for i in b.incomparable)
        notes.append(
            f"Note: the comparison flagged incomparable results ({reasons}); see the evidence bundle."
        )
    if b.integrity is not None:
        notes += _latex_integrity_notes(b.integrity)
    if not notes:
        return []
    return [r"\par\smallskip", r"{\footnotesize " + r" \\ ".join(notes) + r"\par}"]


def _latex_integrity_notes(info: IntegrityInfo) -> list[str]:
    notes = []
    if info.post_hoc:
        seeds = "; ".join(f"{p.group_label} seed {_seed_text(p.seed)}" for p in info.post_hoc)
        notes.append(f"Seeds added after results existed (post hoc): {latex_escape(seeds)}.")
    if info.late_exclusions:
        late = "; ".join(
            f"{e.group_label} seed {_seed_text(e.seed)} ({e.reason})" for e in info.late_exclusions
        )
        notes.append(f"Excluded after the result was observed: {latex_escape(late)}.")
    mixed = [h for h in info.heterogeneity if h.mixed]
    if mixed:
        text = "; ".join(_heterogeneity_text(h) for h in mixed)
        notes.append(f"Mixed recorded environment across included runs: {latex_escape(text)}.")
    multi = info.multiple_valid_attempts
    if info.pinned:
        rule = latex_escape(info.attempt_selection) if info.attempt_selection else None
        text = "Evaluations were chosen by hand (evaluation\\_refs), not by the " + (
            f"{rule} rule" if rule else "contract's attempt selection"
        )
        over = sum(m.overrides_rule for m in multi)
        if over:
            text += (
                f"; {len(multi)} run(s) had several valid attempts, and for {over} of them the reported "
                f"attempt is not the one {rule or 'the rule'} picks."
            )
        elif multi:
            text += f"; {len(multi)} run(s) had several valid attempts (the pinned one is reported)."
        else:
            text += "."
        notes.append(text)
    elif multi:
        rule = latex_escape(info.attempt_selection or "selected")
        notes.append(f"{len(multi)} run(s) had several valid attempts; the {rule} attempt is reported.")
    return notes


def _gap_text(cell: MetricCell) -> str:
    parts = [f"{cell.n} of {cell.expected_n} runs"]
    if cell.missing_seeds:
        parts.append(f"missing seeds {latex_escape(_seed_list(cell.missing_seeds))}")
    if cell.failed_seeds:
        parts.append(f"failed seeds {latex_escape(_seed_list(cell.failed_seeds))}")
    for e in cell.excluded:
        parts.append(f"seed {latex_escape(str(e.seed))} excluded ({latex_escape(e.reason)})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    "group_label",
    "group_key",
    "is_baseline",
    "metric",
    "unit",
    "higher_is_better",
    "seed",
    "status",
    "value",
    "exclusion_reason",
    "evaluation_id",
    "evaluator",
    "artifact_id",
    "artifact_digest",
    "attempt_id",
    "job_id",
    "cluster_id",
    "run_id",
    "run_spec_hash",
    "code_digest",
    "git_commit",
    "env_gpu",
    "env_torch",
)
_PROVENANCE_COLUMNS = CSV_COLUMNS[10:]


def render_csv(bundle: Mapping[str, Any] | EvidenceBundle) -> str:
    """Long format, one row per group × metric × expected seed (RFC 4180, CRLF).

    Missing/failed/excluded seeds get rows too (``status`` says which), so the CSV has the
    same denominator as the tables. Values use Python's shortest round-trip repr, so
    recomputing from the CSV reproduces the bundle exactly.
    """
    b = parse_bundle(bundle)
    metrics = b.metric_order()
    prov = {(r.group_label, r.metric, r.seed_key): r for r in b.provenance}
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for g in b.groups:
        for m in metrics:
            cell = g.metrics.get(m)
            if cell is not None:
                writer.writerows(_csv_rows(b, g, m, cell, prov))
    return out.getvalue()


def _csv_rows(
    b: EvidenceBundle,
    g: Group,
    metric: str,
    cell: MetricCell,
    prov: Mapping[tuple[str, str, str], ProvenanceRow],
) -> list[list[str]]:
    decl = b.decl(metric)
    reasons = cell.exclusion_reasons()
    rows = []
    for seed, status in cell.seed_status().items():
        p = prov.get((g.label, metric, seed))
        value = cell.values_by_seed.get(seed)
        if value is None and p is not None:
            value = p.value
        rows.append(
            [
                g.label,
                canonical_json(g.key),
                _csv_bool(g.is_baseline),
                metric,
                decl.unit or "",
                _csv_bool(decl.higher_is_better),
                seed,
                status,
                repr(value) if value is not None else "",
                reasons.get(seed, ""),
                *(("" if p is None else getattr(p, col) or "") for col in _PROVENANCE_COLUMNS),
            ]
        )
    return rows


def _csv_bool(v: bool | None) -> str:
    return "" if v is None else ("true" if v else "false")
