---
name: hpclab-run-doctor
description: Triage and recover HPC Lab runs on Slurm — failed, timed-out, OOM, preempted, pending-forever, "stuck", or submission-uncertain jobs, a failed smoke gate or a paused sweep — using deterministic diagnosis and recovery plans instead of guessing or blind resubmits. Use when a run failed, is pending too long, looks hung, or the user asks "what happened to my job?".
---

# Run doctor: evidence first, then the smallest correct fix

The two most expensive mistakes on a shared cluster are **resubmitting a job that is actually already
queued** (double GPU spend, two writers on one output dir) and **cancelling the wrong or a healthy
job**. HPC Lab's runtime prevents both mechanically; this skill keeps your reasoning on the same rails.
Each tool result ends with `Next:` suggestions carrying the exact ids — prefer them over re-deriving.

## Triage order

1. `runs_list(experiment_id=..., view="grid")` — which cells are not `V`? The grid lists them under
   "needs attention" with the reason and run id, and its notes say when a smoke gate or a stopped
   workflow keeps `P` cells from starting.
2. `run_inspect(run_id=...)` for each. Read, in order: the attempt's `phase`, `scheduler_state`,
   `app_state`, `result_state` (three different questions!), then `diagnosis` (with evidence lines and
   `resource_hint`), `pending`, `progress`, `health`, and `recommended_next`.
3. `run_logs(attempt_id=..., stream="stderr", tail_lines=80)` only if the diagnosis evidence is not
   enough. Quote the evidence line when you explain.

## Phase rules you must never break

| What you see | What it means | What to do |
|---|---|---|
| `submission_uncertain` / `U` | sbatch may or may not have reached Slurm (reply lost). | **Wait.** The controller reconciles via the on-cluster receipt, `squeue`, `sacct`. Never resubmit, never prepare a "retry". If it stays unresolved past the window, tell the user; resolving it is a human decision (`hpclab resolve <attempt_id> --bind-job <job_id>` or `--abandon`, with `--reason`). |
| `duplicate_detected` | Two queued jobs carry this attempt's token. | Show both candidates; the user decides. Do not cancel either yourself. |
| `observation_gap` | Job left the queue, accounting not updated yet. | Not finished, not failed. Check again later. |
| `S` for more than a minute (`staging`) | The submission has not reached Slurm yet. `run_inspect(run_id=...)` lists a **blocked operation** when something needs a person: `auth_required` (SSH/MFA expired) or `cluster_access`. | Ask the user to run the shown command (e.g. `hpclab auth <cluster>`) in their own terminal; the submission resumes by itself. Never resubmit. |
| `E` (`COMPLETED`, evaluation pending) | Slurm exited 0; outputs are being checked or the automatic evaluation has not run yet. **Not a failure.** | Check again shortly; if it stays `E`, `evaluation_collect(run_set_ref=..., idempotency_key=...)`. Never recover or resubmit it. |
| `C!` (`COMPLETED`, bad result) | Slurm exited 0 but the declared outputs are incomplete/invalid, or the latest evaluation is partial/invalid. | `artifacts_list(attempt_id=...)`; it's usually a wrong output path, a crash after the last eval, or a metric the evaluator cannot find. |

## Pending jobs

`pending.category` from `run_inspect`:

* `cluster_busy` (Priority/Resources) — normal. For a start-time guess use
  `cluster_inspect(cluster_id=..., checks=["capacity"], candidate_shapes=[...])`; its `est_start` is a
  low-confidence hint, say so.
* `self_limit` (QOSMax…PerUser) — the user's *own* other jobs hold the quota; list them.
* `account_limit` — the lab allocation is saturated; a PI conversation, not a code fix.
* `never_starts` (PartitionTimeLimit, DependencyNeverSatisfied, BadConstraints) — it will never run.
  Cancel it (`run_cancel`) and fix the plan (e.g. time above partition max).
* A preemptible partition (e.g. `hai-lo`) is only OK for runs declared resumable.

## "It looks stuck"

`health.flags` (`stalled_candidate`, `heartbeat_frozen`, `idle_allocation`, `partial_gpu_use`) are
**signals, not verdicts**. Checkpoint writes and evals legitimately go quiet for minutes. Give it five
minutes of evidence (log size, metrics step, GPU utilization via `run_inspect(run_id=..., probe="gpu")`)
before suggesting a cancel, and prefer `run_cancel(mode="request_checkpoint")` when the run declares
`resilience.stop_behavior: checkpoint_and_exit`. `heartbeat_frozen` on one node plus NFS errors in the
log points at a sick node — say so, and let recovery exclude it.

## Failure → recovery mapping

Call `run_recover(attempt_id=..., idempotency_key=...)` (optionally `strategy=...`,
`resource_patch={...}`); it returns a *plan* and never submits. Check `eligible`, `blocking` and
`hints`, then submit the new plan with `run_submit(run_id=..., execution_plan_hash=<exec_plan_hash>,
idempotency_key=...)` after the user agrees to the spend.

| diagnosis | strategy | notes |
|---|---|---|
| `timeout` with valid checkpoint | `resume` | May first create a checkpoint **probe run** (loads the checkpoint on a compute node). Submit it, then call `run_recover` again. |
| `timeout` without checkpoint | `adjust_resources` (more `time`, capped at the partition max) | Suggest adding `resilience` + checkpointing for next time. |
| `host_oom` | `adjust_resources` (more `mem`; the diagnosis `resource_hint` proposes a value) | |
| `cuda_oom` | **derive a new run** (smaller batch / grad accumulation) | Changing batch size changes the science → new Run via `experiment_prepare(derive_from={...}, idempotency_key=...)`, unless the spec declares `batch_invariant_to_gpus` and you add GPUs. |
| `node_fail` / `nfs_stale` / `gpu_hardware_error` | `retry` | The plan excludes a node only when ≥ 2 attempts implicate it recently. |
| `preempted` | `resume` (resumable) or move partition | |
| `nan_loss` | fix code / derive (lower lr) | Never "retry the same config". |
| `import_error`, `cuda_driver_mismatch`, `env_guard_failed`, `command_not_found` | fix the environment, then prepare again | These also trip the workflow circuit breaker. |
| `disk_quota` | clean scratch / move caches off home | |

For several broken cells at once: `run_recover(selector={"workflow_id": "wf_...", "cells": "missing_valid"},
idempotency_key=...)` builds one recovery plan; dispatch it with `workflow_submit(plan_id=<returned plan_id>,
idempotency_key=...)`.

## A sweep that does not move

* **Smoke gate failed** (grid note "smoke run …: F …", workflow state `gate_failed`): no main cell is
  dispatched. Diagnose the smoke run like any failed run, fix the cause, recover and submit the smoke
  run, and once it passed call `workflow_submit(workflow_id=..., resume_dispatch=true,
  expected_version=..., idempotency_key=...)`.
* **Circuit breaker** (workflow `dispatch_state: paused`): ≥ 2 cells failed with the same environment
  error. Fix it (prove it with one run), then resume the same way.
* The current `version` of a workflow is in resource `hpclab://workflows/{workflow_id}`. It moves while
  the sweep runs; a `version_conflict` error names the current one — re-check the situation, then retry
  with it.

## Cancelling

Only `run_cancel(attempt_id, expected_version, idempotency_key)` — `expected_version` is the attempt's
`version` from `run_inspect`. It re-verifies job id, owner, token and submit time immediately before
`scancel` and refuses on any mismatch. Never suggest raw `scancel <id>` from memory or from a job name.
To stop a whole sweep: `workflow_cancel(workflow_id=..., expected_version=..., scope="queued_only",
idempotency_key=...)` first; `scope="all_managed"` also cancels running cells. Confirm with the user.
