---
name: hpclab-lab-overview
description: Give a PI or student an honest overview of GPU usage and experiment progress on a shared Slurm allocation — usage by user/partition/state with waste by cause, current queue, experiment grids, weekly digests, and verification of a student's evidence bundle. Use for lab meetings, allocation reviews, "where did our GPU hours go?", or checking results before a paper goes out.
---

# Lab overview for PIs (and students preparing for lab meeting)

## Usage: honest, not a leaderboard

`usage_report(cluster_id="hai", scope="account", account=<lab account>, since="30d", group_by=["user", "state"])`
(`include=["now"]` adds the current queue).

* Read `visibility` first. On many clusters (`PrivateData=jobs`) a non-coordinator only sees their own
  jobs: then the report is `self_only`, `totals` has only the `self` key (**no lab total**), and you
  must say so instead of presenting a partial sum as the lab's usage.
* GPU-hours are *allocated* hours (GPUs × elapsed), not utilization. Say "allocated".
* `wasted_gpu_hours_by_cause` groups failed/timed-out/cancelled time by diagnosis class for
  hpclab-managed jobs (e.g. "31 GPU-h lost to CUDA OOM in 4 runs") — useful for deciding what to fix,
  not for ranking people. Do not rank or score users.
* `walltime_accuracy` (elapsed / requested) flags habitual 7-day requests that hurt everyone's backfill.
* An incomplete result (long windows are cut to stay fast) says so; call again to continue — finished
  days are cached.

## Progress

* Experiments: resource `hpclab://experiments` (newest first); per experiment
  `runs_list(experiment_id=..., view="grid")`: cells valid / running / failed / missing (`E` = completed,
  evaluation pending — not a failure; `C!` = completed with invalid outputs or evaluation).
* `hpclab digest --since 7d [--account <acct>] [--cluster <id>]` (CLI) renders a Markdown weekly summary
  from the ledger: finished and failed runs with causes, settled GPU-hours, new evidence bundles,
  attempts needing a human decision, long-pending jobs with decoded reasons.

## Checking results before they go into a paper

`evidence_verify(bundle_ref=<bundle_id>, tex_paths=["paper/main.tex"])` recomputes aggregates from
per-seed evaluations and checks the manuscript's `\hpcval` numbers against the bundle (stale or
hand-edited numbers, missing-seed footnotes, mixed code versions or GPU types, selection leaks). The
bundle id is inside the tags: `\hpcval{<bundle_id>:<cell>}{...}`. It verifies record consistency —
whether the experiment answers the question is still the PI's call.

## What HPC Lab will not do (by design)

No killing other people's jobs, no auto-cancel of "idle" runs (idle flags are advisory), no lab-wide
budget enforcement (only the scheduler can enforce allocations; hpclab budgets cover jobs submitted
through it), no significance claims.
