---
name: hpclab-experiment-design
description: Turn a research question into a frozen, budgeted HPC Lab experiment (baseline, treatments × seeds, metrics, comparison contract, smoke gate, resources) and launch it on a Slurm cluster through the hpclab MCP tools. Use when the user wants to run an ablation, a method comparison, a seed sweep, or "try X vs Y on the cluster".
---

# Designing and launching an experiment with HPC Lab

HPC Lab is the deterministic runtime; you are the experimental designer. The runtime freezes code,
config and resources, enforces budgets, and refuses unsafe submissions. Your job is to produce a
*good* experiment and to explain the runtime's findings to the user — never to route around them.

Every tool answers with a short text block (summary line, the ids you need next, and a `Next:` list of
exact follow-up calls) plus the complete result as structured content. Chain ids from the result;
never invent them.

## 1. Pin down the science before touching the cluster

Write these down with the user (ask only for what you cannot infer from the repo):

| Decision | Rule of thumb |
|---|---|
| Hypothesis | One sentence, falsifiable: "mixup improves val accuracy of ResNet-18 on CIFAR-10". |
| Baseline | The exact config the treatment is compared against — a matrix value, not a separate script. |
| Treatment axes | Only what the hypothesis varies (`method`, `lr`). Everything else is an **invariant**. |
| Seeds | ≥ 3 per cell (5 if the effect is small). Seeds are part of the plan; never "add seeds until it looks good". |
| Metrics | Name, direction (`higher_is_better`), unit, and the **split** they are reported on. |
| Selection | How the reported number is picked (`final` record by default). Never select the best checkpoint on the split you report (`selection_leak`). |
| Budget | GPU-hours cap for the whole matrix incl. smoke and retries. Ask; do not invent generous numbers. |

## 2. Make the training script hpclab-friendly (small, optional edits)

* Read `HPCLAB_OUTPUT_DIR`, `HPCLAB_CHECKPOINT_DIR`, `HPCLAB_RESUME_FROM`, `HPCLAB_SEED` or pass
  them via argv placeholders `{output_dir} {checkpoint_dir} {resume_from}`; matrix values and
  `{seed}` are filled in at prepare time.
* Log metrics as `hpclab.metrics/v1` JSONL (`step`, `total_steps`, `split`, `metrics`, `final: true` on
  the last eval). `total_steps` enables ETA and time-limit risk warnings.
* Write checkpoints with a `hpclab-ckpt.json` manifest written **last**. The single-file helper
  `hpclab_io.py` of the tiny-classifier example does both; vendor it rather than reinventing it
  (`hpclab examples copy tiny-classifier <dir>` gets it from an installed hpclab).
* If the script can checkpoint on `SIGUSR1` and resume, declare it (`resilience`, below) — that is what
  turns a TIMEOUT into a cheap resume instead of a lost day.

## 3. Write the ExperimentSpec (YAML next to the code, e.g. `hpclab.yaml`)

```yaml
name: resnet18-mixup-ablation
question: Does mixup improve val accuracy of ResNet-18 on CIFAR-10?
cluster: hai                      # a registered cluster id (hpclab cluster list)
code: {project_root: ".", include_untracked: []}
entrypoint:
  argv: [python, train.py, --method, "{method}", --seed, "{seed}", --out, "{output_dir}",
         --ckpt-dir, "{checkpoint_dir}", --resume, "{resume_from}"]
runtime:
  venv: /hai/scratch/<user>/envs/resnet     # lets prepare check torch CUDA build vs driver
  lockfiles: [uv.lock]
  env: {HF_HOME: /hai/scratch/<user>/hf}
matrix: {method: [baseline, mixup]}
seeds: [1, 2, 3]
resources: {partition: hai, gpus: 1, gpu_type: h100, cpus: 8, mem: 64G, time: "06:00:00"}
outputs:
  - {name: metrics, path: "{output_dir}/metrics.jsonl", role: metrics}
  - {name: checkpoints, path: "{checkpoint_dir}", role: checkpoint, kind: dir, required: false}
smoke: {argv_append: ["--max-steps", "50"], time: "00:20:00", require_outputs: [metrics]}
resilience: {stop_behavior: checkpoint_and_exit, resumable: true, lead_s: 900}
budget: {gpu_hours: 60, max_parallel: 4}
evaluation: {evaluator_id: hpclab_metrics, output: metrics, config: {split: val}}
comparison:
  treatment: [method]
  baseline: {method: baseline}
  metrics: [{name: acc, higher_is_better: true}, {name: loss, higher_is_better: false}]
```

`examples/tiny-classifier/hpclab.yaml` is a complete, runnable example (cluster `sim`); outside a source
checkout, `hpclab examples copy tiny-classifier <dir>` copies the whole project (script, helper, specs).

Time limit rule: estimated runtime × 1.15 + eval/checkpoint overhead. Never "just ask for 7 days" —
it kills backfill and inflates the budget reservation. `experiment_prepare` returns
`resource_advice` from the user's own history; use it.

## 4. Prepare, then read every finding

`experiment_prepare(spec_path="hpclab.yaml", idempotency_key="prep-resnet18-mixup")` (relative paths
resolve against your working directory; `spec={...}` works too). Each new key prepares a **new**
experiment; reuse the key only to retry the same call.

* `status`: `ready` (has `plan_hash`), `preparing` (large project: read the returned `operation_ref`
  after a few seconds), or `invalid`.
* `validation.findings` with `severity: block` must be fixed, not overridden. Explain each one in one
  line (e.g. *"torch was built for CUDA 13.0 but the cluster driver is 12.8 — the job would die on
  start; reinstall with the cu128 wheel"*).
* `status: unchecked` means the runtime could not verify something (cluster unreachable, auth
  expired, no venv declared). Say so; do not call it a pass.
* `overrides=[{"check": ..., "reason": ...}]` exist for genuine exceptions; only use one when the
  user explicitly agrees, and pass their reason.
* Show the user the plan: cells, seeds, requested GPU-hours vs budget, the smoke gate; offer the job
  script (resource `hpclab://plans/{plan_id}/script`, run-specific part; `/script/full` for all of it).

## 5. Launch

* Matrix → `workflow_submit(plan_id=..., plan_hash=..., idempotency_key=...)` (optionally
  `max_parallel=...`); a single run → `run_submit(run_id=..., execution_plan_hash=..., idempotency_key=...)`
  with the run's `exec_plan_hash` from the prepare result. Both return receipts immediately; the
  controller keeps working if the chat ends.
* With a smoke gate the workflow first runs only the smoke run; main cells are dispatched after it
  COMPLETED with valid outputs.
* Spending GPU-hours is the user's decision: confirm once with the numbers (cells × GPUs × time limit,
  budget) before the first submit of a new plan.
* Reuse the same `idempotency_key` when retrying a call whose reply you did not see; use a new key only
  for a genuinely new request. A different key cannot double-submit — the runtime refuses a second
  execution claim (and a second workflow for the same plan) — but it will tell you so.

## 6. Watch without hammering the scheduler

`runs_list(experiment_id=..., view="grid")` shows the method × seed matrix of main runs with one code
per cell (`V` valid, `R` running, `Q` queued, `S` submitting, `P` not dispatched, `E` completed with
evaluation pending — not a failure, normally `V` within seconds, `F` failed, `C!` completed but its outputs
or evaluation are invalid, `U` submission uncertain, `X` cancelled, rejected or abandoned (incl. cells a
cancelled workflow never dispatched), `?` stale) and notes when a
smoke gate or a stopped workflow holds `P` cells back. `runs_list(experiment_id=...)` (list view) also
shows the smoke run. Jobs take minutes to days: check back when the user asks or at a sensible
interval, not in a tight loop — the controller already polls Slurm on a shared, rate-limited schedule.
Hand failures to the **hpclab-run-doctor** skill and results to **hpclab-paper-evidence**.

## Adding seeds or cells later

`experiment_prepare(extends=<experiment_id>, add_seeds=[4, 5], idempotency_key=...)` (or
`add_matrix={"method": ["cutmix"]}`). Existing cells are reused; new runs are marked post-hoc if
the experiment already had a result (a main run with an evaluation or with validated outputs), and
code drift since the original snapshot is reported (`code_drift`). Tell the user
both facts — they belong in the paper's disclosure.
