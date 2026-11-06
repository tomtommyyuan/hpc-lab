"""The example's ExperimentSpecs are valid, self-consistent and fit their cluster profiles."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from tiny_classifier_support import EXAMPLE_DIR

from hpclab.domain.clock import parse_slurm_duration
from hpclab.domain.models import RUNTIME_PLACEHOLDERS, ExperimentSpec
from hpclab.profiles.loader import load_profile

SPECS = {"hpclab.yaml": "sim", "hpclab.hai.yaml": "hai"}


def _spec(name: str) -> ExperimentSpec:
    return ExperimentSpec.model_validate(yaml.safe_load((EXAMPLE_DIR / name).read_text()))


@pytest.mark.parametrize(("name", "cluster"), SPECS.items())
def test_spec_is_valid_and_targets_expected_cluster(name: str, cluster: str) -> None:
    spec = _spec(name)
    assert spec.cluster == cluster
    assert spec.cell_count() == 6
    assert spec.matrix == {"method": ["baseline", "mixup"]} and spec.seeds == [1, 2, 3]


def test_sim_spec_matches_the_documented_example() -> None:
    spec = _spec("hpclab.yaml")
    assert spec.name == "tiny-mixup-ablation" and spec.question
    assert spec.code.project_root == "."
    assert spec.runtime.setup == [] and spec.runtime.python == "python3"
    r = spec.resources
    assert (r.partition, r.gpus, r.gpu_type, r.cpus, r.mem, r.time) == ("gpu", 1, "a100", 2, "4G", "00:10:00")
    assert spec.budget.gpu_hours == 5 and spec.budget.max_parallel == 2
    assert spec.smoke is not None and spec.smoke.argv_append == ["--steps", "20"]
    assert spec.evaluation is not None and spec.evaluation.config == {"split": "val"}
    assert spec.comparison is not None and spec.comparison.baseline == {"method": "baseline"}
    assert {(m.name, m.higher_is_better) for m in spec.comparison.metrics} == {("acc", True), ("loss", False)}


@pytest.mark.parametrize("name", SPECS)
def test_argv_placeholders_are_all_resolvable(name: str) -> None:
    spec = _spec(name)
    known = set(spec.matrix) | set(spec.parameters) | {"seed"} | RUNTIME_PLACEHOLDERS
    used = {m for arg in spec.entrypoint.argv for m in re.findall(r"\{([A-Za-z_]+)\}", arg)}
    assert used == {"method", "seed", "output_dir", "checkpoint_dir"}
    assert used <= known
    assert (EXAMPLE_DIR / spec.entrypoint.argv[1]).is_file()


@pytest.mark.parametrize("name", SPECS)
def test_outputs_references_are_consistent(name: str) -> None:
    spec = _spec(name)
    outputs = {o.name: o for o in spec.outputs}
    assert spec.smoke is not None and set(spec.smoke.require_outputs) <= set(outputs)
    assert spec.evaluation is not None and spec.evaluation.output in outputs
    assert outputs["metrics"].role == "metrics" and outputs["metrics"].required
    assert outputs["metrics"].path == "{output_dir}/metrics.jsonl"  # where train.py --out writes it
    assert outputs["summary"].path == "{output_dir}/metrics.json"
    ckpt = outputs["checkpoints"]
    assert (ckpt.kind, ckpt.role, ckpt.required, ckpt.path) == (
        "dir",
        "checkpoint",
        False,
        "{checkpoint_dir}",
    )
    assert spec.comparison is not None and spec.comparison.baseline is not None
    for key, value in spec.comparison.baseline.items():
        assert value in spec.matrix[key]


@pytest.mark.parametrize(("name", "cluster"), SPECS.items())
def test_resources_fit_the_cluster_profile(name: str, cluster: str) -> None:
    spec = _spec(name)
    profile = load_profile(cluster, None)
    assert spec.resources.partition is not None
    rule = profile.partitions[spec.resources.partition]
    assert not rule.interactive_only
    assert rule.max_time is not None
    assert spec.resources.time_seconds <= (parse_slurm_duration(rule.max_time) or 0)
    if spec.resources.gpu_type:
        assert spec.resources.gpu_type in rule.gpu_types


@pytest.mark.parametrize("name", SPECS)
def test_budget_covers_matrix_and_smoke(name: str) -> None:
    spec = _spec(name)
    assert spec.smoke is not None and spec.budget.gpu_hours is not None
    smoke_gpus = spec.smoke.gpus if spec.smoke.gpus is not None else spec.resources.gpus
    smoke_hours = smoke_gpus * (parse_slurm_duration(spec.smoke.time) or 0) / 3600
    assert spec.cell_count() * spec.resources.requested_gpu_hours + smoke_hours <= spec.budget.gpu_hours


def test_example_ships_every_documented_file() -> None:
    names = {p.name for p in Path(EXAMPLE_DIR).iterdir()}
    assert {"README.md", "train.py", "hpclab_io.py", "hpclab.yaml", "hpclab.hai.yaml"} <= names
    ignored = (EXAMPLE_DIR / ".gitignore").read_text().split()
    assert {"outputs/", "checkpoints/", "__pycache__/"} <= set(ignored)
