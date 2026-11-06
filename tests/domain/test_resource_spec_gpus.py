"""ResourceSpec: GPUs are requested only through ``resources.gpus``.

Budget reservation, the GPU-hours the user confirms, partition/limit validation, the
compute-side guard and GPU telemetry all read ``gpus`` / ``total_gpus`` /
``requested_gpu_hours``. An ``extra_sbatch`` flag that makes Slurm allocate GPUs on its
own would make every one of those numbers wrong (e.g. report 0 GPU-h for a job holding
32 GPUs for two days), so such flags are refused at the model boundary.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hpclab.domain.models import EXTRA_SBATCH_ALLOWED, ResourceSpec


@pytest.mark.parametrize(
    "flag",
    [
        "--gpus-per-task=1",
        "--gpus-per-task=a100:1",
        "--gpus-per-task=8",
        "--exclusive",  # allocates every CPU *and GRES* on the node
    ],
)
@pytest.mark.parametrize("gpus", [0, 1])
def test_gpu_allocating_extra_sbatch_flags_are_rejected(flag: str, gpus: int) -> None:
    with pytest.raises(ValidationError) as err:
        ResourceSpec(gpus=gpus, nodes=4, ntasks_per_node=8, time="2-00:00:00", extra_sbatch=[flag])
    text = str(err.value)
    assert "allocate GPUs" in text
    assert "resources.gpus" in text


@pytest.mark.parametrize(
    "flag",
    [
        "--gres=gpu:8",
        "--gpus=8",
        "--gpus-per-node=8",
        "--gpus-per-socket=2",
        "--exclusive=user",
        "--exclusive=mcs",
    ],
)
def test_other_gpu_allocating_spellings_are_rejected_with_the_same_hint(flag: str) -> None:
    with pytest.raises(ValidationError) as err:
        ResourceSpec(extra_sbatch=[flag])
    assert "resources.gpus" in str(err.value)


def test_the_allow_list_admits_no_gpu_allocating_flag() -> None:
    for allowed in EXTRA_SBATCH_ALLOWED:
        name = allowed.split("=", 1)[0]
        assert name not in {"--gres", "--gpus", "--gpus-per-node", "--gpus-per-task", "--exclusive"}


def test_gpu_modifier_flags_stay_allowed_and_do_not_change_the_gpu_count() -> None:
    res = ResourceSpec(
        gpus=2,
        nodes=2,
        ntasks_per_node=2,
        time="10:00:00",
        extra_sbatch=["--gpu-bind=closest", "--cpus-per-gpu=4", "--mem-per-gpu=10G"],
    )
    assert res.total_gpus == 4
    assert res.requested_gpu_hours == pytest.approx(40.0)


def test_cpu_only_spec_reports_zero_gpu_hours_only_when_no_gpu_is_allocated() -> None:
    res = ResourceSpec(
        gpus=0, nodes=4, ntasks_per_node=8, time="2-00:00:00", extra_sbatch=["--hint=nomultithread"]
    )
    assert res.total_gpus == 0
    assert res.requested_gpu_hours == 0.0
