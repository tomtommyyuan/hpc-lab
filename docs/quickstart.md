# Quickstart: five minutes on the simulator

No cluster needed. You will run a real 2 methods × 3 seeds ablation through the bundled Slurm
simulator (fake `sbatch`/`squeue`/`sacct` executables, jobs run as local processes), get a paper
table whose every number traces back to a job, and connect Claude Code.

All output below is real, from hpclab 0.1.0 on macOS; only the username (`alice`), the host
name and long paths were replaced, and some lines were cut (`...`). Your ids, hashes and times
will differ.

## 0. Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/). hpclab is not on PyPI yet; install it from a
checkout of this repository:

```bash
cd hpc-lab
uv tool install .          # puts `hpclab` on your PATH (or: pip install .)
hpclab --version           # hpclab 0.1.0
```

For development use `uv sync` and prefix commands with `uv run` instead (see
[CONTRIBUTING.md](../CONTRIBUTING.md)).

Everything hpclab stores lives in one **home**, `~/.hpclab` by default. Set `HPCLAB_HOME` (or
`--home`) to use another one, e.g. a throwaway home for this tutorial:

```bash
export HPCLAB_HOME=/tmp/hpclab-tutorial
```

## 1. Register the simulator

```console
$ hpclab init --sim
hpclab home: /tmp/hpclab-tutorial
cluster sim registered
cluster sim (Local Slurm simulator) · transport sim · user alice · account lab
remote root: /tmp/hpclab-tutorial/sim/sim/fs/scratch/alice/hpclab
next: copy the example project: hpclab examples copy tiny-classifier ./tiny-classifier
next: then freeze it into a validated plan: hpclab prepare tiny-classifier/hpclab.yaml
next: or take the guided tour first: hpclab demo
```

This also started the **controller**, the one background process per home that owns the ledger
(`hpclab status` shows it). The CLI and every MCP server are clients of it.

The simulated cluster has partitions `gpu` (4 nodes × 4 A100, default), `interactive` and
`scavenge` (preemptible H100s), and accounts `lab` and `other`.

## 2. The guided tour (about a minute)

```console
$ hpclab demo
HPC Lab demo — a guided tour on a simulated Slurm cluster (about a minute)
You will see: prepare → smoke gate → a 2×3 sweep → a node failure → diagnosis → recovery → a paper table where
every number traces back to a Slurm job.
It runs in its own throwaway hpclab home (...); your clusters, experiments and controller are not touched.
...
[4/8] Submit the matrix: smoke gate first, at most 2 jobs at a time ─────────────────────────────
  ✓ workflow wf_01M3KP468S7RPC0VZRSP81THAM: 7 runs, max_parallel 2
  ⚡ injected: the node under the next job will fail (simulator fault node_fail_next=1)
   0:07  gating · smoke gate pending   baseline: P P P   mixup: P P P
   0:15  dispatching · smoke gate passed   baseline: Q Q P   mixup: P P P
   0:20  dispatching · smoke gate passed   baseline: F R S   mixup: P P P
   ...
   0:47  completed · smoke gate passed   baseline: F V V   mixup: V V V

[5/8] A cell failed — what hpclab did, and did not do ────────────────────────────────────────────
  ✓ failed cell: method-baseline.seed-1 — Slurm job 1001 ended NODE_FAIL on sim-a100-01
  hpclab recorded the failure as Slurm reported it and kept dispatching the other cells
  (infrastructure failures never trip the circuit breaker); it did NOT resubmit anything;
  ...
  the table right now: baseline n=2/3 (seed 1 failed); mixup n=3/3
  The failed seed stays in the denominator — it is never silently dropped from a mean.

[6/8] Recovery plan ──────────────────────────────────────────────────────────────────────────────
  ✓ method-baseline.seed-1: retry (diagnosis node_fail)
  ✓ recovery plan pln_01M3KP5GYD79A4TB2DBNR0209M dispatched as workflow wf_01M3KP5GYKGT5NV8186VX33643
   1:04  complete   baseline: V V V   mixup: V V V
...
demo complete in 1:07
```

Now do the same by hand, step by step.

## 3. Copy the example project

```console
$ hpclab examples copy tiny-classifier
copied tiny-classifier → tiny-classifier (7 file(s))
next:
  hpclab init --sim   # once: the local Slurm simulator, cluster 'sim'
  hpclab prepare tiny-classifier/hpclab.yaml   # validates and freezes the runs
  hpclab submit <plan id>   # printed by prepare: smoke gate first, then the matrix
  hpclab grid <workflow id> --watch
  hpclab prepare tiny-classifier/hpclab.hai.yaml   # for cluster hai
$ cd tiny-classifier && git init -q && git add . && git commit -qm "tiny-classifier" && cd ..
```

`tiny-classifier` is a stdlib-only training script that behaves like a real one (metrics log,
checkpoints, exact resume, graceful stop on `SIGUSR1`, injectable failures) plus
`hpclab.yaml`, its **ExperimentSpec**: the research question, a `method: [baseline, mixup]` ×
`seeds: [1, 2, 3]` matrix, resources, declared outputs, a smoke gate, a 5 GPU-hour budget and the
comparison contract. See [experiment-spec.md](experiment-spec.md). The `git init` is optional;
with it, every result records the commit.

## 4. Prepare: freeze and validate (nothing is submitted)

```console
$ hpclab prepare tiny-classifier/hpclab.yaml
plan pln_01M3KP6DR8M8T73M2T9KYMGSDW ready (matrix) · experiment exp_01M3KP6DR81N2KS3E59PF3J3JM
validation: warn — 24 pass, 1 warn, 4 unchecked, 0 fail
validation findings
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ status    ┃ severity ┃ check             ┃ finding                                                         ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ warn      │ info     │ lockfiles_present │ no lockfile in the snapshot: the Python environment is unpinned │
│           │          │                   │ → commit uv.lock / requirements.txt and list it in              │
│           │          │                   │ runtime.lockfiles                                               │
│ unchecked │ info     │ gpus_limits       │ main: GPU limits of partition 'gpu' unknown; smoke: GPU limits  │
│           │          │                   │ of partition 'gpu' unknown                                      │
│ unchecked │ info     │ resource_advice   │ fewer than 3 comparable COMPLETED+valid attempts in the ledger  │
│ unchecked │ info     │ env_guard_static  │ no runtime.venv declared: the environment is only checked on    │
│           │          │                   │ the compute node (runtime.guard)                                │
│ unchecked │ info     │ snapshot_bypass   │ no runtime.venv declared: the environment is only checked on    │
│           │          │                   │ the compute node (runtime.guard)                                │
└───────────┴──────────┴───────────────────┴─────────────────────────────────────────────────────────────────┘
24 other check(s) passed (--all-findings to list them)
runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ label                        ┃ kind  ┃ seed ┃ GPU-h ┃ run id                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ smoke.method-baseline.seed-1 │ smoke │ 1    │ 0.25  │ run_01M3KP6DY0BY9TV871DEZJ2QZZ │
│ method-baseline.seed-1       │ main  │ 1    │ 0.17  │ run_01M3KP6DY0PAX25189MPN05G56 │
│ method-baseline.seed-2       │ main  │ 2    │ 0.17  │ run_01M3KP6DY0NZZ63CB5W9QZAF9H │
│ method-baseline.seed-3       │ main  │ 3    │ 0.17  │ run_01M3KP6DY0N2C9GBCW45Q0VEGR │
│ method-mixup.seed-1          │ main  │ 1    │ 0.17  │ run_01M3KP6DY0K662RKHVPBR4XSY9 │
│ method-mixup.seed-2          │ main  │ 2    │ 0.17  │ run_01M3KP6DY0CPQNW74S1AB2VPM6 │
│ method-mixup.seed-3          │ main  │ 3    │ 0.17  │ run_01M3KP6DY04HN30SQ5ATEE9XWK │
└──────────────────────────────┴───────┴──────┴───────┴────────────────────────────────┘
budget: 1.25 GPU-h requested (time limit × GPUs, incl. smoke 0.25) of 5.00 GPU-h budget
code snapshot: sha256:80a01e9f11ee · 7 files, 54.0 KiB · git e3f04031b36b
smoke gate: run_01M3KP6DY0BY9TV871DEZJ2QZZ runs first; the matrix is dispatched only if it passes
next:
  hpclab submit pln_01M3KP6DR8M8T73M2T9KYMGSDW
  hpclab plan pln_01M3KP6DR8M8T73M2T9KYMGSDW --script    # review the exact sbatch script
```

What just happened:

* The project was captured into a **content-addressed snapshot** (`sha256:80a01e9f11ee`). Queued
  jobs run exactly this code, whatever you edit afterwards.
* Six immutable **Runs** (one per cell) plus a smoke run, each with a frozen execution plan.
* 29 checks. `unchecked` means hpclab could not verify it, never "pass". A `fail` with severity
  `block` makes the plan `invalid` (for example a torch wheel built for a newer CUDA than the
  cluster's driver; see [sites/hai.md](sites/hai.md#the-prepare-time-driver-check)).
* The budget counts GPUs × time limit, never an optimistic runtime.

## 5. Submit: smoke gate first, then the matrix

```console
$ hpclab submit pln_01M3KP6DR8M8T73M2T9KYMGSDW --max-parallel 2
submit: plan pln_01M3KP6DR8M8T73M2T9KYMGSDW (matrix) of experiment exp_01M3KP6DR81N2KS3E59PF3J3JM · 7 run(s)
  smoke: 1 run(s) × 1 GPU (a100) × 00:15:00 = 0.25 GPU-h
  main: 6 run(s) × 1 GPU (a100) × 00:10:00 = 1.00 GPU-h
total: 1.25 GPU-h at most (time limit × GPUs; reserved as cells are admitted, settled from accounting when
they end)
budget: 5.00 GPU-h available of 5.00 GPU-h
  the smoke run goes first; the matrix is dispatched only if it passes
Dispatch this plan (spends GPU-hours)? [y/N]: y
workflow wf_01M3KP6J3HHRC189DP3Z0RWTBW gating · dispatch open · gate pending · at most 2 at once
cells: 7 planned · at most 2 at once
smoke gate run_01M3KP6DY0BY9TV871DEZJ2QZZ runs first; the other cells are dispatched only if it passes
watch:
  hpclab grid wf_01M3KP6J3HHRC189DP3Z0RWTBW --watch
  hpclab workflow show wf_01M3KP6J3HHRC189DP3Z0RWTBW
```

The command returns immediately. The controller dispatches, observes, validates outputs and
evaluates metrics in the background; you can close the terminal.

## 6. Watch the grid

In a terminal, `--watch` redraws one table in place until nothing more happens on its own (every
cell terminal). Piped into a file, as here, it prints one count line per refresh and then the
final table:

```console
$ hpclab grid wf_01M3KP6J3HHRC189DP3Z0RWTBW --watch
02:40:43: 0/6 valid · Q=2 · P=4
02:40:48: 1/6 valid · V=1 · S=2 · P=2 · E=1
02:40:53: 2/6 valid · V=2 · Q=2 · P=2
02:40:58: 4/6 valid · V=4 · Q=2
wf_01M3KP6J3HHRC189DP3Z0RWTBW  ·  as of 02:41:03
workflow wf_01M3KP6J3HHRC189DP3Z0RWTBW completed · dispatch open · gate passed · slots 0/2
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ method   ┃ seed=1 ┃ seed=2 ┃ seed=3 ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ baseline │ V      │ V      │ V      │
│ mixup    │ V      │ V      │ V      │
└──────────┴────────┴────────┴────────┘
legend: V valid result
cells: 6/6 valid · V=6
done: every cell is terminal
```

`hpclab runs --experiment tiny-mixup-ablation` lists the same runs one per line with the Slurm
job id, scheduler state and pending reason (`awaiting first observation` right after
submission).

One code per cell: **V** valid result · **R** running · **Q** queued · **S** submitting ·
**P** not dispatched · **E** completed, evaluation pending (not a failure) · **C!** completed but
outputs or evaluation invalid · **F** failed (with the diagnosis) · **U** submission uncertain
(never resubmit it; hpclab is reconciling) · **X** cancelled/rejected · **?** stale observation.
Details: [architecture.md](architecture.md#cell-codes).

```console
$ hpclab grid tiny-mixup-ablation --metric acc
tiny-mixup-ablation  ·  as of 02:41:03
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ method   ┃ seed=1   ┃ seed=2   ┃ seed=3   ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ baseline │ V 0.8440 │ V 0.8700 │ V 0.8540 │
│ mixup    │ V 0.8360 │ V 0.8460 │ V 0.8440 │
└──────────┴──────────┴──────────┴──────────┘
legend: V valid result
values: acc from the selected valid evaluation only (V cells)
cells: 6/6 valid · V=6
```

`hpclab inspect <run id>` shows everything about one run: the attempts with their Slurm job and
three separate state axes (scheduler, application, result), progress/ETA, health, diagnosis.

## 7. Compare over the planned runs

```console
$ hpclab compare exp_01M3KP6DR81N2KS3E59PF3J3JM
comparison cmp_01M3KP7FNBHTW6S1M3KYCPT544 · experiment tiny-mixup-ablation (exp_01M3KP6DR81N2KS3E59PF3J3JM)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ group                ┃ acc (fraction) ↑      ┃ loss (nats) ↓         ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ baseline (reference) │ 0.8560 ± 0.0131 (3/3) │ 0.3478 ± 0.0104 (3/3) │
│ mixup                │ 0.8420 ± 0.0053 (3/3) │ 0.3831 ± 0.0064 (3/3) │
└──────────────────────┴───────────────────────┴───────────────────────┘
cells: mean ± sample std over seeds (included/expected); — = undefined
differences vs baseline (descriptive; not a significance test):
  mixup − baseline  acc ↑: -0.0140  CI [-0.0357, +0.0077] (includes 0) · paired, 3 seed(s)
  mixup − baseline  loss ↓: +0.0353  CI [-0.0054, +0.0759] (includes 0) · paired, 3 seed(s)
  intervals are 95% confidence intervals of the difference
coverage: 6/6 planned runs included · missing 0 · failed 0 · excluded 0  complete
next:
  hpclab export cmp_01M3KP7FNBHTW6S1M3KYCPT544 --out <dir>
  hpclab grid exp_01M3KP6DR81N2KS3E59PF3J3JM
```

`(3/3)` is included/expected: had a seed failed, the cell would read `(2/3)` and the failed run
would be listed, never silently dropped. The honest answer at this scale: mixup does not help.

## 8. Export an evidence bundle

```console
$ hpclab export cmp_01M3KP7FNBHTW6S1M3KYCPT544 --out evidence --purpose "Table 1 draft"
evidence bundle evb_01M3KP7Q7P07RMJFDASWYG1MQR: 2 group(s) × 2 metric(s), 6/6 planned runs included
bundle: evb_01M3KP7Q7P07RMJFDASWYG1MQR · comparison cmp_01M3KP7FNBHTW6S1M3KYCPT544 · Table 1 draft
coverage: 6/6 planned runs included · missing 0 · failed 0 · excluded 0
written to: ./evidence
┏━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ name          ┃ bytes ┃ sha256              ┃ status  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ manifest.json │ 21965 │ sha256:b1c93c10d0c0 │ written │
│ summary.md    │ 7467  │ sha256:fe72deb930e9 │ written │
│ table.csv     │ 6376  │ sha256:44d8ac70d242 │ written │
│ table.tex     │ 1384  │ sha256:614d7e83f35c │ written │
└───────────────┴───────┴─────────────────────┴─────────┘
```

`table.tex` (booktabs) tags every number so it stays verifiable; the preamble line makes the
macro print only the value:

```latex
\providecommand{\hpcval}[2]{#2}
...
baseline & $\hpcval{evb_01M3KP7Q7P07RMJFDASWYG1MQR:baseline.acc.mean}{0.856} \pm \hpcval{evb_01M3KP7Q7P07RMJFDASWYG1MQR:baseline.acc.std}{0.013}$ & ...
mixup & $\hpcval{evb_01M3KP7Q7P07RMJFDASWYG1MQR:mixup.acc.mean}{0.842} \pm ...
```

Trace any number back to its sources:

```console
$ hpclab trace evb_01M3KP7Q7P07RMJFDASWYG1MQR:mixup.acc.mean
evb_01M3KP7Q7P07RMJFDASWYG1MQR:mixup.acc.mean: mixup / acc mean = 0.8420  (n=3/3)  chain complete
bundle: evb_01M3KP7Q7P07RMJFDASWYG1MQR · comparison cmp_01M3KP7FNBHTW6S1M3KYCPT544 · run set
hpclab://experiments/exp_01M3KP6DR81N2KS3E59PF3J3JM
seed 1: included value 0.8360
   evaluation ev_01M3KP71EPT4FFQ5NNYP5MTM9H hpclab_metrics@1 valid
   artifact   art_01M3KP71CHT45DSX189NFAEMH7 metrics sha256:841b0794b850 finalized
   attempt    att_01M3KP6VX5KY0F9E6ZM7NHJ77D #1 COMPLETED valid
   job        sim:1004 owner alice submitted 2026-09-28T09:40:48.000Z
   run        run_01M3KP6DY0K662RKHVPBR4XSY9 method-mixup.seed-1 spec sha256:129460cc1445
   code       sha256:80a01e9f11ee git e3f04031b36b
   env        NVIDIA A100-SXM4-80GB · 570.124.06 · 3.14.7 · laptop.local
seed 2: ...
```

(The simulator fakes `nvidia-smi`, hence an A100 on a laptop; on a real cluster the `env` line
is what the job's own node reported in `env.json`.)

And check a manuscript before you submit it. Here `paper.tex` inputs the table and quotes one
number by hand, wrongly:

```console
$ hpclab paper check paper.tex
bundle evb_01M3KP7Q7P07RMJFDASWYG1MQR: 1 error(s), 0 warning(s), 0 info; 12 value(s) recomputed, 9 tex tag(s)
checked (record consistency only)
scanned 2 file(s):
  ./paper.tex
  ./evidence/table.tex
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ severity ┃ code           ┃ message                                ┃ where                               ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ error    │ VALUE_MISMATCH │ mixup/acc mean: tex shows 0.850 but    │ ./paper.tex:3,                      │
│          │                │ the bundle has 0.842                   │ evb_01M3KP7Q7P07RMJFDASWYG1MQR:mix… │
└──────────┴────────────────┴────────────────────────────────────────┴─────────────────────────────────────┘
9 tag(s) in 2 scanned file(s), 1 bundle(s): 1 error(s)
record consistency only: this does not verify scientific correctness
```

It exits non-zero on errors, so it can gate a CI job or a pre-submission script.

## 9. Break something on purpose

The simulator can inject the failures that cost research weeks. Lose the reply of the next
`sbatch` (the job *is* accepted, the answer never arrives), then run the experiment again under
a new name:

```bash
hpclab sim fault sbatch_lose_reply=1
hpclab prepare tiny-classifier/hpclab.yaml --experiment lost-reply-demo
hpclab submit <plan id> --yes
hpclab events --entity <attempt id of the smoke run>     # from `hpclab inspect <run id>`
```

The attempt's audit trail shows hpclab refusing to guess (condensed: event, source and the start
of the payload, oldest first; `hpclab events` prints the same rows as a table, newest first):

```text
attempt.admitted        worker     {"cluster_id": "sim", "gpus": 1, ...}
attempt.staged          worker     {"plan_file_sha256": ..., "script_sha256": ...}
attempt.dispatching     worker     {"from": "staging", "owner": "alice", "receipt": ".../submit.receipt"}
attempt.uncertain       worker     {"cause": "sbatch", "error": {"code": "unavailable", "details": {"maybe_executed": true, ...}}}
attempt.job_identified  reconcile  {"evidence": {"job": {"job_id": "1007", "from_receipt": true, ...}}}
attempt.observing       reconcile  {"from": "job_identified", "job_id": "1007", "to": "observing"}
attempt.terminal        observer   {"changes": ["state:UNKNOWN->COMPLETED", ...], "source": "sacct"}
attempt.finalized       worker     {"accounting": "ok", "app_state": "exited_ok", "budget": "settled", ...}
attempt.result_state_changed worker {"from": "unchecked", "reason": "artifacts indexed", "to": "valid"}
```

While it was uncertain, `hpclab inspect` said *"submission outcome uncertain ...: hpclab is
reconciling and will never resubmit on its own"*; `hpclab sim jobs` (the simulator's ground
truth) lists exactly one job for it. Other faults: `sbatch_ambiguous_error`, `node_fail_next`,
`preempt_next`, `oom_next`, `requeue_next`, `accounting_lag_s`, `squeue_unavailable`,
`sacct_unavailable`, `auth_required`, `transport_timeout_cmds`, ... (the full list is
`SimFaults` in `src/hpclab/sim/config.py`; `hpclab sim clear` resets them).

## 10. Connect Claude Code

```bash
claude mcp add hpclab -- hpclab mcp serve
hpclab skills install          # copies the five hpclab skills into ~/.claude/skills
```

If you use a non-default home, or `hpclab` is not on the PATH Claude Code sees, let hpclab print
the exact command. It uses the absolute path of `hpclab` (or `python -m hpclab.cli` when
`hpclab` is not on your PATH) and passes `-e HPCLAB_HOME=...` when you chose a home:

```console
$ hpclab mcp config
Claude Code:
  claude mcp add hpclab -e HPCLAB_HOME=/tmp/hpclab-tutorial -- /Users/alice/.local/bin/hpclab mcp serve
JSON config (Claude Desktop, Cursor, ...):
{
  "mcpServers": {
    "hpclab": {
      "command": "/Users/alice/.local/bin/hpclab",
      "args": [
        "mcp",
        "serve"
      ],
      "env": {
        "HPCLAB_HOME": "/tmp/hpclab-tutorial"
      }
    }
  }
}
```

```console
$ hpclab skills install
installed hpclab-cluster-onboarding -> ~/.claude/skills/hpclab-cluster-onboarding
installed hpclab-experiment-design -> ~/.claude/skills/hpclab-experiment-design
installed hpclab-lab-overview -> ~/.claude/skills/hpclab-lab-overview
installed hpclab-paper-evidence -> ~/.claude/skills/hpclab-paper-evidence
installed hpclab-run-doctor -> ~/.claude/skills/hpclab-run-doctor
```

The MCP server exposes 17 tools, `hpclab://` resources and 5 prompts (the same playbooks as the
skills, for clients without skill support); see [tools.md](tools.md). Then, in Claude Code:

> Run the example in tiny-classifier on the sim cluster: baseline vs mixup, 3 seeds, at most 2 jobs
> at a time. Smoke-test first. When it is done, give me mean ± std, which seeds are missing and where
> each number came from.

> What happened to my runs in experiment tiny-mixup-ablation? Plan a recovery for anything broken.

> Check that the numbers in paper/main.tex are still current.

The agent reasons about the experiment; hpclab executes it deterministically and refuses unsafe
actions (a second submission of an uncertain run, a cancel whose job identity does not match).

## 11. Clean up

```bash
hpclab controller stop                # jobs on a real cluster would keep running
chmod -R u+w "$HPCLAB_HOME" && rm -rf "$HPCLAB_HOME"   # snapshots are read-only on purpose
```

## Next

* A real cluster: [sites/hai.md](sites/hai.md) (Stanford HAI) or start from
  `hpclab cluster add <id> --profile generic-slurm ...`, then `hpclab onboard <id>` runs a 1-GPU
  smoke job end to end.
* Your own project: [experiment-spec.md](experiment-spec.md); copy
  `tiny-classifier/hpclab_io.py` for the metrics and checkpoint formats.
* For PIs: [for-pis.md](for-pis.md). What is and is not promised: [guarantees.md](guarantees.md).
