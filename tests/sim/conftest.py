"""Fixtures for simulator tests. Every live simulator is shut down after the test."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from hpclab.adapters.slurm import SlurmScheduler
from hpclab.sim import cli
from hpclab.sim.cluster import SimCluster
from hpclab.sim.testing import sim_cluster, sim_scheduler
from hpclab.sim.transport import SimTransport


@dataclass
class CliResult:
    rc: int
    stdout: str
    stderr: str


RunCli = Callable[..., CliResult]
WriteScript = Callable[..., Path]


@pytest.fixture
def sim(tmp_path: Path) -> Iterator[SimCluster]:
    with sim_cluster(tmp_path) as cluster:
        yield cluster


@pytest.fixture
def transport(sim: SimCluster) -> SimTransport:
    return sim_scheduler(sim)[0]


@pytest.fixture
def slurm(sim: SimCluster) -> SlurmScheduler:
    """A scheduler whose capability probe already ran, as in a long-lived controller.

    Queue/accounting calls probe lazily on first use (six short commands); with the fast
    simulator timings (MinJobAge 1 s) that one-time latency would race the observations.
    """
    scheduler = sim_scheduler(sim)[1]
    scheduler.probe_capabilities()
    return scheduler


@pytest.fixture
def workdir(sim: SimCluster) -> Path:
    path = sim.fs_root / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def run_cli(sim: SimCluster) -> RunCli:
    """Run a fake Slurm command in-process (no interpreter start-up): ``run_cli("squeue", "-h")``."""

    def run(*argv: str, cwd: Path | None = None, stdin: str = "", time_format: str = "%s") -> CliResult:
        out, err = io.StringIO(), io.StringIO()
        env = {"HPCLAB_SIM_ROOT": str(sim.root), "SLURM_TIME_FORMAT": time_format, "PATH": str(sim.bin_dir)}
        rc = cli.main(
            list(argv), stdout=out, stderr=err, stdin=io.StringIO(stdin), env=env, cwd=str(cwd or sim.root)
        )
        return CliResult(rc, out.getvalue(), err.getvalue())

    return run


@pytest.fixture
def write_script(workdir: Path) -> WriteScript:
    """``write_script("echo hi", "--gres=gpu:1", name="job.sbatch")`` -> path of a batch script."""

    def write(body: str, *directives: str, name: str = "job.sbatch") -> Path:
        path = workdir / name
        lines = ["#!/bin/bash", *(f"#SBATCH {d}" for d in directives), body, ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    return write
