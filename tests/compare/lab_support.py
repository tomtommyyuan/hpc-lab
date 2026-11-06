"""Ledger builder for comparison / evidence tests (imported by tests/compare and tests/evidence).

It writes realistic rows through ``hpclab.testing.factories``: runs with a RunSpec-like spec
(code digest + git commit), terminal attempts with a wrapper ``env.json``, a metrics artifact
with its digest, and an evaluation whose metric rows carry split/dataset/unit/direction and the
artifact digest as ``source.digest`` (what ``evaluators.run_evaluator`` produces).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpclab.domain.clock import FakeClock
from hpclab.domain.ids import new_id
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jdump
from hpclab.testing.factories import (
    make_attempt,
    make_evaluation,
    make_experiment,
    make_plan,
    make_run,
)

GIT_COMMIT = "0123456789abcdef0123456789abcdef01234567"
CODE_A = "sha256:" + "a" * 64
CODE_B = "sha256:" + "b" * 64


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def env_json(gpu: str | None = "NVIDIA A100-SXM4-80GB", torch: str | None = "2.8.0+cu128") -> str:
    env: dict[str, Any] = {"schema": "hpclab.env/v1", "host": "gpu-01"}
    env["nvidia_smi"] = (
        {"driver_version": "570.1", "gpus": [{"name": gpu, "memory_total_mib": 81920}]} if gpu else None
    )
    env["torch"] = {"version": torch, "cuda": "12.8", "device_names": [gpu] if gpu else []} if torch else None
    return jdump(env)


@dataclass
class Lab:
    ctx: ServiceContext
    clock: FakeClock

    # -- structure -----------------------------------------------------------
    def experiment(
        self,
        *,
        name: str = "cifar",
        comparison: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
    ) -> str:
        spec: dict[str, Any] = {
            "name": name,
            "cluster": "sim",
            "entrypoint": {"argv": ["python", "train.py"]},
        }
        if comparison is not None:
            spec["comparison"] = comparison
        if evaluation is not None:
            spec["evaluation"] = evaluation
        with self.ctx.db.tx() as tx:
            return make_experiment(tx, name=name, spec=spec, question="Does mixup help?")

    def plan(self, experiment_id: str, *, kind: str = "matrix") -> str:
        with self.ctx.db.tx() as tx:
            return make_plan(tx, experiment_id, kind=kind)

    def run(
        self,
        experiment_id: str,
        plan_id: str,
        *,
        method: str | None = "erm",
        seed: int | None = 1,
        code_digest: str | None = CODE_A,
        added_post_hoc: bool = False,
        kind: str = "main",
        extra_cell: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        cell: dict[str, Any] = {} if method is None else {"method": method}
        cell.update(extra_cell or {})
        spec_cell = dict(cell)
        if seed is not None:
            spec_cell["seed"] = seed
        spec: dict[str, Any] = {
            "experiment": "cifar",
            "kind": kind,
            "cell": spec_cell,
            "seed": seed,
            "argv": ["python", "train.py", f"--seed={seed}"],
            "params": {**cell, "lr": 0.1, **(params or {})},
            "code": {
                "digest": code_digest,
                "file_count": 3,
                "total_bytes": 1000,
                "git_commit": GIT_COMMIT,
                "git_dirty": False,
            }
            if code_digest
            else None,
        }
        with self.ctx.db.tx() as tx:
            return make_run(
                tx,
                experiment_id,
                plan_id,
                cell=cell,
                seed=seed,
                kind=kind,
                spec=spec,
                added_post_hoc=1 if added_post_hoc else 0,
            )

    # -- outcomes --------------------------------------------------------------
    def valid(
        self,
        run_id: str,
        *,
        acc: float | None = None,
        metrics: dict[str, float | None] | None = None,
        split: str | None = "val",
        dataset: str | None = "cifar10",
        gpu: str | None = "NVIDIA A100-SXM4-80GB",
        torch: str | None = "2.8.0+cu128",
        evaluator_id: str = "hpclab_metrics",
        evaluator_version: str = "1",
        config_hash: str | None = None,
        issues: list[dict[str, Any]] | None = None,
        job_id: str | None = None,
        advance: float = 1.0,
        remote_dir: str | None = None,
    ) -> tuple[str, str, str]:
        """A COMPLETED attempt with valid results + a valid evaluation. Returns (attempt, artifact, evaluation)."""
        values = dict(metrics or {"acc": acc if acc is not None else 0.9})
        self.clock.advance(advance)
        with self.ctx.db.tx() as tx:
            job = job_id or str(1000 + int(tx.one("SELECT COUNT(*) AS n FROM attempts")["n"]))  # type: ignore[index]
            aid = make_attempt(
                tx,
                run_id,
                phase="terminal",
                scheduler_state="COMPLETED",
                result_state="valid",
                app_state="exited_ok",
                job_id=job,
                job_owner="alice",
                job_submit_time=tx.now,
                terminal_at=tx.now,
                env_json=env_json(gpu, torch),
                gpu_type="a100",
                partition="gpu",
            )
            remote_path = f"/remote/attempts/{aid}/outputs/metrics.jsonl"
            content = jdump({"schema": "hpclab.metrics/v1", "final": True, "metrics": values}).encode()
            if remote_dir is not None:  # a real file on the (simulated) cluster filesystem
                path = Path(remote_dir) / aid / "metrics.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                remote_path = str(path)
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            art = new_id("artifact")
            tx.insert(
                "artifacts",
                {
                    "artifact_id": art,
                    "attempt_id": aid,
                    "run_id": run_id,
                    "name": "metrics",
                    "role": "metrics",
                    "kind": "file",
                    "format": "jsonl",
                    "remote_path": remote_path,
                    "size_bytes": len(content),
                    "mtime": tx.now,
                    "digest": digest,
                    "digest_state": "computed",
                    "digest_source": "compute_node",
                    "integrity": "finalized",
                    "required": 1,
                    "observed_at": tx.now,
                    "version": 1,
                },
            )
            rows = [
                {
                    "name": name,
                    "value": value,
                    "unit": "fraction" if name == "acc" else None,
                    "split": split,
                    "step": 1000,
                    "dataset": dataset,
                    "higher_is_better": name == "acc",
                    "source": {"input_name": "metrics", "digest": digest, "locator": "L10#/metrics/" + name},
                }
                for name, value in values.items()
            ]
            ev = make_evaluation(
                tx,
                aid,
                evaluator_id=evaluator_id,
                evaluator_version=evaluator_version,
                metrics_json=jdump(rows),
                input_refs_json=jdump([{"artifact_id": art, "digest": digest, "name": "metrics"}]),
                validation_json=jdump({"status": "valid", "issues": issues or []}),
                config_hash=config_hash or "sha256:" + "c" * 64,
                provenance_json=jdump({"attempt_id": aid, "job_id": job}),
            )
            tx.event("evaluation", ev, "evaluation.created", {"run_id": run_id, "attempt_id": aid})
        return aid, art, ev

    def failed(self, run_id: str, *, state: str = "OUT_OF_MEMORY") -> str:
        self.clock.advance(1)
        with self.ctx.db.tx() as tx:
            return make_attempt(
                tx,
                run_id,
                phase="terminal",
                scheduler_state=state,
                result_state="incomplete",
                env_json=env_json(),
                diagnosis_json=jdump({"primary_code": "cuda_oom"}),
            )

    def running(self, run_id: str) -> str:
        self.clock.advance(1)
        with self.ctx.db.tx() as tx:
            return make_attempt(tx, run_id, phase="observing", scheduler_state="RUNNING")


@dataclass
class Scenario:
    """2 methods × 3 seeds: erm all valid; mixup seed 1 valid, seed 2 failed (OOM), seed 3 never run."""

    lab: Lab
    experiment_id: str
    plan_id: str
    runs: dict[tuple[str, int], str]
    evaluations: dict[tuple[str, int], str]
    attempts: dict[tuple[str, int], str]


ERM = {1: 0.80, 2: 0.82, 3: 0.84}
MIXUP = {1: 0.85, 2: 0.86, 3: 0.90}


def build_scenario(lab: Lab, *, complete: bool = False, comparison: dict[str, Any] | None = None) -> Scenario:
    contract = (
        comparison if comparison is not None else {"baseline": {"method": "erm"}, "treatment": ["method"]}
    )
    exp = lab.experiment(comparison=contract)
    plan = lab.plan(exp)
    runs: dict[tuple[str, int], str] = {}
    evals: dict[tuple[str, int], str] = {}
    attempts: dict[tuple[str, int], str] = {}
    for method in ("erm", "mixup"):
        for seed in (1, 2, 3):
            runs[(method, seed)] = lab.run(exp, plan, method=method, seed=seed)
    for seed, acc in ERM.items():
        attempts[("erm", seed)], _, evals[("erm", seed)] = lab.valid(runs[("erm", seed)], acc=acc)
    if complete:
        for seed, acc in MIXUP.items():
            attempts[("mixup", seed)], _, evals[("mixup", seed)] = lab.valid(runs[("mixup", seed)], acc=acc)
    else:
        attempts[("mixup", 1)], _, evals[("mixup", 1)] = lab.valid(runs[("mixup", 1)], acc=MIXUP[1])
        attempts[("mixup", 2)] = lab.failed(runs[("mixup", 2)])
    return Scenario(lab, exp, plan, runs, evals, attempts)
