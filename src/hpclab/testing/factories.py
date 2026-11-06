"""Minimal ledger row factories for tests.

They write *valid* rows (all NOT NULL columns filled, foreign keys satisfied) with sensible
defaults so a test can say ``make_attempt(tx, run_id, phase="terminal",
scheduler_state="COMPLETED")`` without caring about the rest. They bypass service logic on
purpose: use them to set up state, then exercise the service under test.
"""

from __future__ import annotations

from typing import Any

from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id, submit_token
from hpclab.storage.db import Tx, jdump


def make_experiment(
    tx: Tx,
    *,
    name: str = "exp",
    cluster_id: str = "sim",
    budget_gpu_hours: float | None = None,
    max_parallel: int | None = None,
    spec: dict[str, Any] | None = None,
    **extra: Any,
) -> str:
    eid = new_id("experiment")
    spec = spec or {"name": name, "cluster": cluster_id, "entrypoint": {"argv": ["python", "train.py"]}}
    tx.insert(
        "experiments",
        {
            "experiment_id": eid,
            "name": name,
            "cluster_id": cluster_id,
            "question": extra.pop("question", None),
            "spec_json": jdump(spec),
            "spec_hash": digest(spec),
            "project_root": extra.pop("project_root", None),
            "budget_gpu_hours": budget_gpu_hours,
            "max_parallel": max_parallel,
            "status": extra.pop("status", "ready"),
            "tags_json": "[]",
            "version": 1,
            "created_at": tx.now,
            "updated_at": tx.now,
            **extra,
        },
    )
    return eid


def make_plan(
    tx: Tx, experiment_id: str, *, kind: str = "matrix", status: str = "ready", **extra: Any
) -> str:
    pid = new_id("plan")
    tx.insert(
        "plans",
        {
            "plan_id": pid,
            "experiment_id": experiment_id,
            "kind": kind,
            "status": status,
            "plan_hash": extra.pop("plan_hash", digest({"plan": pid})),
            "snapshot_digest": extra.pop("snapshot_digest", None),
            "validation_json": "{}",
            "created_at": tx.now,
            "updated_at": tx.now,
            **extra,
        },
    )
    return pid


def make_run(
    tx: Tx,
    experiment_id: str,
    plan_id: str,
    *,
    cell: dict[str, Any] | None = None,
    seed: int | None = 1,
    kind: str = "main",
    label: str | None = None,
    status: str = "prepared",
    spec: dict[str, Any] | None = None,
    **extra: Any,
) -> str:
    rid = new_id("run")
    cell = dict(cell or {})
    if seed is not None:
        cell.setdefault("seed", seed)
    spec = spec or {
        "experiment": "exp",
        "kind": kind,
        "cell": cell,
        "seed": seed,
        "argv": ["python", "train.py"],
    }
    tx.insert(
        "runs",
        {
            "run_id": rid,
            "experiment_id": experiment_id,
            "plan_id": plan_id,
            "kind": kind,
            "label": label or ".".join(f"{k}-{v}" for k, v in sorted(cell.items())) or "run",
            "cell_key": jdump(cell),
            "seed": seed,
            "spec_json": jdump(spec),
            "spec_hash": digest(spec),
            "status": status,
            "version": 1,
            "created_at": tx.now,
            "updated_at": tx.now,
            **extra,
        },
    )
    return rid


def make_exec_plan(tx: Tx, run_id: str, *, plan: dict[str, Any] | None = None, status: str = "ready") -> str:
    plan = plan or {
        "run_id": run_id,
        "resources": {"gpus": 1, "time": "01:00:00"},
        "nonce": new_id("request"),
    }
    h = digest(plan)
    tx.insert(
        "execution_plans",
        {
            "exec_plan_hash": h,
            "run_id": run_id,
            "plan_json": jdump(plan),
            "status": status,
            "validation_json": "{}",
            "created_at": tx.now,
        },
    )
    tx.execute("UPDATE runs SET current_exec_plan_hash=? WHERE run_id=?", (h, run_id))
    return h


def make_attempt(
    tx: Tx,
    run_id: str,
    *,
    exec_plan_hash: str | None = None,
    seq: int | None = None,
    phase: str = "observing",
    scheduler_state: str = "RUNNING",
    result_state: str = "unchecked",
    cluster_id: str = "sim",
    job_id: str | None = "1001",
    **extra: Any,
) -> str:
    aid = new_id("attempt")
    if exec_plan_hash is None:
        exec_plan_hash = make_exec_plan(tx, run_id)
    if seq is None:
        row = tx.one("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM attempts WHERE run_id=?", (run_id,))
        seq = int(row["n"]) if row else 1
    token = submit_token()
    tx.insert(
        "attempts",
        {
            "attempt_id": aid,
            "run_id": run_id,
            "exec_plan_hash": exec_plan_hash,
            "seq": seq,
            "cluster_id": cluster_id,
            "submit_token": token,
            "job_name": f"hl.run.{token[:8]}",
            "remote_dir": f"/remote/attempts/{aid}",
            "phase": phase,
            "scheduler_state": scheduler_state,
            "result_state": result_state,
            "job_id": job_id,
            "version": 1,
            "created_at": tx.now,
            "updated_at": tx.now,
            **extra,
        },
    )
    tx.execute("UPDATE runs SET latest_attempt_id=? WHERE run_id=?", (aid, run_id))
    return aid


def make_evaluation(
    tx: Tx,
    attempt_id: str,
    *,
    metrics: dict[str, float | None] | None = None,
    status: str = "valid",
    evaluator_id: str = "hpclab_metrics",
    evaluator_version: str = "1",
    split: str = "val",
    **extra: Any,
) -> str:
    row = tx.one("SELECT run_id FROM attempts WHERE attempt_id=?", (attempt_id,))
    assert row is not None, attempt_id
    evid = new_id("evaluation")
    metric_rows = [{"name": k, "value": v, "split": split} for k, v in (metrics or {"acc": 0.9}).items()]
    tx.insert(
        "evaluations",
        {
            "evaluation_id": evid,
            "attempt_id": attempt_id,
            "run_id": row["run_id"],
            "evaluator_id": evaluator_id,
            "evaluator_version": evaluator_version,
            "config_hash": digest({}),
            "input_key": digest({"evaluation": evid}),
            "input_refs_json": "[]",
            "metrics_json": jdump(metric_rows),
            "provenance_json": "{}",
            "validation_json": jdump({"status": status, "issues": []}),
            "status": status,
            "created_at": tx.now,
            **extra,
        },
    )
    return evid
