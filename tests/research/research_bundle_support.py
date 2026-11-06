"""Builders for a realistic, internally consistent EvidenceBundle (imported by the research tests)."""

from __future__ import annotations

import hashlib
from typing import Any

from hpclab.services.stats import paired_differences, summarize, unpaired_difference

CI = 0.95

# method × seed results for a CIFAR-style experiment with every kind of gap:
# mixup seed 3 failed (OOM); the lr=1e-3 cell has seed 2 missing and seed 3 excluded.
VALUES: dict[str, dict[str, dict[int, float]]] = {
    "erm": {"acc": {1: 0.912, 2: 0.905, 3: 0.918}, "loss": {1: 0.301, 2: 0.322, 3: 0.295}},
    "mixup": {"acc": {1: 0.921, 2: 0.917}, "loss": {1: 0.288, 2: 0.297}},
    "lr=1e-3 & wd_0.1 (50%)": {"acc": {1: 0.874}, "loss": {1: 0.41}},
}
KEYS: dict[str, dict[str, Any]] = {
    "erm": {"method": "erm"},
    "mixup": {"method": "mixup"},
    "lr=1e-3 & wd_0.1 (50%)": {"method": "mixup", "lr": 0.001, "wd": 0.1},
}
GAPS: dict[str, dict[str, Any]] = {
    "erm": {},
    "mixup": {"failed_seeds": [3]},
    "lr=1e-3 & wd_0.1 (50%)": {
        "missing_seeds": [2],
        "excluded": [{"seed": 3, "reason": "diverged: loss=nan at step 400"}],
    },
}
EXPECTED_SEEDS = [1, 2, 3]


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_cell(
    values: dict[int, float], *, expected: int, ci_level: float = CI, **gaps: Any
) -> dict[str, Any]:
    s = summarize([values[k] for k in sorted(values)], ci_level)
    return {
        "n": s.n,
        "expected_n": expected,
        "mean": s.mean,
        "std": s.std,
        "sem": s.sem,
        "ci_low": s.ci_low,
        "ci_high": s.ci_high,
        "min": s.min,
        "max": s.max,
        "values_by_seed": {str(k): v for k, v in values.items()},
        "missing_seeds": list(gaps.get("missing_seeds", [])),
        "failed_seeds": list(gaps.get("failed_seeds", [])),
        "excluded": list(gaps.get("excluded", [])),
    }


def provenance_row(group: str, seed: int, metric: str, value: float | None) -> dict[str, Any]:
    tag = f"{group}/{seed}"
    return {
        "group_label": group,
        "seed": seed,
        "metric": metric,
        "value": value,
        "evaluation_id": f"ev_{_hex(tag + metric)[:10]}",
        "evaluator": "hpclab_metrics@1",
        "artifact_id": f"art_{_hex(tag)[:10]}",
        "artifact_digest": f"sha256:{_hex('artifact ' + tag)}",
        "attempt_id": f"att_{_hex('attempt ' + tag)[:10]}",
        "job_id": str(4100 + seed + 10 * sorted(VALUES).index(group)),
        "cluster_id": "haic",
        "run_id": f"run_{_hex('run ' + tag)[:10]}",
        "run_spec_hash": f"sha256:{_hex('spec ' + tag)}",
        "code_digest": f"sha256:{_hex('code')}",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "env_gpu": "NVIDIA A100-SXM4-80GB",
        "env_torch": "2.8.0+cu128",
    }


def build_bundle() -> dict[str, Any]:
    groups = []
    provenance = []
    for label, metrics in VALUES.items():
        cells = {}
        for metric, values in metrics.items():
            cells[metric] = make_cell(values, expected=len(EXPECTED_SEEDS), **GAPS[label])
            provenance += [provenance_row(label, s, metric, v) for s, v in sorted(values.items())]
        groups.append({"key": KEYS[label], "label": label, "is_baseline": label == "erm", "metrics": cells})
    provenance.append(
        provenance_row("lr=1e-3 & wd_0.1 (50%)", 3, "loss", None)
    )  # the excluded, diverged seed
    differences = []
    for metric in ("acc", "loss"):
        p = paired_differences(VALUES["mixup"][metric], VALUES["erm"][metric], CI)
        differences.append(
            {
                "group_label": "mixup",
                "baseline_label": "erm",
                "metric": metric,
                "kind": "paired",
                "n": p.n_pairs,
                "n_baseline": None,
                "mean_diff": p.mean_diff,
                "ci_low": p.ci_low,
                "ci_high": p.ci_high,
            }
        )
    u = unpaired_difference(
        list(VALUES["lr=1e-3 & wd_0.1 (50%)"]["acc"].values()), list(VALUES["erm"]["acc"].values()), CI
    )
    differences.append(
        {
            "group_label": "lr=1e-3 & wd_0.1 (50%)",
            "baseline_label": "erm",
            "metric": "acc",
            "kind": "unpaired",
            "n": u.n_a,
            "n_baseline": u.n_b,
            "mean_diff": u.mean_diff,
            "ci_low": u.ci_low,
            "ci_high": u.ci_high,
        }
    )
    return {
        "schema": "hpclab.evidence/v1",
        "title": "CIFAR-10: mixup vs ERM",
        "purpose": "Table 2 of the workshop paper (draft).",
        "generated_at": "2026-09-27T12:00:00Z",
        "hpclab_version": "0.1.0",
        "ci_level": CI,
        "experiment": {
            "id": "exp_0001",
            "name": "cifar-mixup",
            "question": "Does mixup improve val accuracy?",
        },
        "contract": {
            "treatment": ["method", "lr", "wd"],
            "invariants": ["split", "evaluator", "dataset"],
            "baseline": {"method": "erm"},
        },
        "metric_decls": [
            {"name": "acc", "higher_is_better": True, "unit": "fraction"},
            {"name": "loss", "higher_is_better": False, "unit": "nats"},
        ],
        "groups": groups,
        "differences": differences,
        "coverage": {
            "planned_runs": 9,
            "included": 6,
            "missing": 1,
            "failed": 1,
            "excluded": 1,
            "complete": False,
            "notes": ["mixup seed 3: OUT_OF_MEMORY on gpu-07 (attempt att_oom)"],
        },
        "provenance": provenance,
        "incomparable": [
            {
                "reason": "environment differs",
                "details": {"env_torch": ["2.7.1", "2.8.0+cu128"], "group": "mixup"},
            }
        ],
        "limitations": ["Three seeds per method; intervals are wide."],
    }


def complete_bundle() -> dict[str, Any]:
    """Two groups, every seed included, no differences / incomparabilities."""
    groups = []
    provenance = []
    for label in ("erm", "mixup"):
        values = {1: 0.9 + 0.01 * len(label), 2: 0.91 + 0.01 * len(label)}
        groups.append(
            {
                "key": KEYS[label],
                "label": label,
                "is_baseline": label == "erm",
                "metrics": {"acc": make_cell(values, expected=2)},
            }
        )
        provenance += [provenance_row(label, s, "acc", v) for s, v in values.items()]
    bundle = build_bundle()
    bundle.update(
        groups=groups,
        differences=[],
        provenance=provenance,
        incomparable=[],
        limitations=[],
        metric_decls=[{"name": "acc", "higher_is_better": True, "unit": None}],
        coverage={
            "planned_runs": 4,
            "included": 4,
            "missing": 0,
            "failed": 0,
            "excluded": 0,
            "complete": True,
        },
    )
    return bundle
