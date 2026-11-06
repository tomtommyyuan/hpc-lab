# HPC Lab for PIs

You run a lab on a shared Slurm allocation. Your students and postdocs run the experiments; you
need to know where the GPU-hours go, how the experiments are progressing, and whether the numbers
in a draft can be trusted. HPC Lab gives you those answers from records, not recollection, and it
says plainly what it cannot see.

Everything below works from the CLI or by asking Claude Code (the `hpclab-lab-overview` skill,
or the `lab_overview` MCP prompt, encodes the same practice). Each person runs hpclab under their
own Unix user with their own ledger; there is no shared server to operate.

## Where did the GPU-hours go? `usage_report`

```bash
hpclab usage --cluster hai --account <slurm-account> --by user,state --since 30d
```

or ask the agent: *"GPU usage of account `<slurm-account>` on hai over the last 30 days, by user
and state."* (MCP tool `usage_report`, [reference](tools.md#usage_report)).

What you get, per group (user, partition, QoS, GPU type, state, diagnosis class):

* **Allocated GPU-hours**: GPUs × wall time from Slurm accounting. This is what the scheduler
  charged, not GPU utilization; hpclab never presents it as utilization.
* **GPU-hours by state** and **wasted GPU-hours by cause**: failed, timed-out, OOM, node-failure,
  preempted and cancelled time. For jobs submitted through hpclab the cause is the diagnosis
  (e.g. "31 GPU-h lost to CUDA OOM in 4 runs"); for other jobs, the terminal state. A timeout that
  left a finalized checkpoint is *resumable progress*, not waste.
* **Median queue wait**, **walltime accuracy** (elapsed / requested; habitual 7-day requests for
  2-hour jobs hurt everyone's backfill) and interactive hours.
* Requeued jobs count every segment; running jobs are charged up to now. A report window spans
  at most 93 days (longer ones are refused: split them yourself); hpclab queries `sacct` in
  chunks of at most 31 days and caches finished days, so repeating a report is cheap.

**Visibility first.** Whether you can see other users' jobs is decided by Slurm's `PrivateData`
setting and your role (account coordinators usually see the account). The report states it:

| `visibility` | Meaning | What hpclab shows |
|---|---|---|
| `full` | Accounting returned other users' jobs | Per-user rows and an account total |
| `self_only` | Only your own jobs came back (hidden by `PrivateData`, or nobody else ran) | Your rows, **no account total**, coverage marked incomplete, and a note to ask an account coordinator |
| `unavailable` | Accounting could not be read | An error or an incomplete result, never "0 GPU-hours" |

Real output of an account report where only the caller's jobs were visible:

```text
usage on sim · account lab · 2026-09-21T09:41 → 2026-09-28T09:41
visibility: only your jobs are visible (Slurm PrivateData hides other users' jobs) — no lab totals can be
computed from here; ask an account coordinator for a lab-wide report
...
your total: 7 job(s) · <0.01 GPU-h allocated
accounting returned no other user's jobs for this account: either nobody else ran in the window or slurmdbd's
PrivateData hides them (it cannot be read from here), so these numbers are NOT account totals
GPU-hours are allocated hours (GPUs x wall time), not measured utilization
```

It is not a leaderboard: there are no scores or rankings, and waste is grouped by cause so you can
decide what to fix (a bad environment, time limits, a flaky node), not whom to blame.

## The weekly digest

```bash
hpclab digest --since 7d --out digest.md                          # from the local ledger only
hpclab digest --since 7d --cluster hai --account <slurm-account>   # + the account usage section
```

A Markdown summary built from the ledger (every number is a stored record; nothing is
estimated): experiment progress by cell code, failures by cause with the GPU-hours lost, settled
GPU-hours (and how many attempts settled without accounting, which count as *unknown*, not zero),
new evidence bundles, attempts that need a human decision (uncertain submissions, duplicates) and
jobs pending for more than 4 hours with their decoded reasons. With `--account` it adds that
account's usage report, labelled with its visibility. The usage section needs `--account`;
`--cluster` alone only scopes the ledger sections to that cluster.

A digest describes the ledger of whoever runs it. A student can send you theirs; your own shows
your own experiments.

## Experiment progress: the grid

```bash
hpclab experiments                       # newest first
hpclab grid <experiment> --metric acc    # method × seed, one code per cell
```

```text
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ method   ┃ seed=1   ┃ seed=2   ┃ seed=3   ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ baseline │ V 0.8440 │ V 0.8700 │ V 0.8540 │
│ mixup    │ V 0.8360 │ V 0.8460 │ V 0.8440 │
└──────────┴──────────┴──────────┴──────────┘
```

`V` valid · `R` running · `Q` queued · `S` submitting · `P` not dispatched · `E` completed,
evaluation pending · `C!` completed but the result is bad · `F` failed · `U` submission
uncertain · `X` cancelled/rejected · `?` stale. The grid, the comparison tables and the digest use
the same cell-status function, so the grid and the paper's denominators cannot disagree.
Details: [architecture.md](architecture.md#cell-codes).

Seeds are part of the plan. Seeds added after results already exist
(`hpclab prepare --extends <exp> --add-seeds 4,5`) are recorded as **post hoc** and disclosed in
every comparison and bundle of that experiment. A result exists as soon as a main run has an
evaluation or finished with validated outputs (its metrics are readable then); the smoke run does
not count.

## Can I trust this table? `evidence_verify`

Numbers for a paper come from **evidence bundles**, not from logs or notebooks. A bundle is an
immutable export of a comparison over the *planned* runs: Markdown summary, LaTeX table, CSV and a
JSON manifest in which every value carries its chain: evaluation → artifact digest (computed on
the compute node) → attempt → Slurm job (cluster, id, owner, submit time) → run spec → code
commit → environment (GPU, driver, Python, torch).

The LaTeX table tags every number as `\hpcval{<bundle>:<cell>}{0.842}`. Before submission:

```bash
hpclab paper check paper/main.tex       # follows \input/\include; exit 1 on any error
```

or ask the agent to run `evidence_verify` with the `.tex` path. It recomputes the aggregates from
the per-seed evaluations and checks every tagged number against the bundle and the current
ledger. Findings include:

| Code | What it catches |
|---|---|
| `VALUE_MISMATCH` | A number in the `.tex` differs from the bundle (hand-edited) |
| `STALE_TEX_NUMBER` | A newer comparison over the same runs has a different value |
| `AGGREGATE_MISMATCH` | The bundle's mean/std no longer match the per-seed evaluations |
| `N_BELOW_EXPECTED_UNFOOTNOTED` | A cell uses fewer seeds than planned without saying so |
| `SEED_SELECTION` | Seeds added after results existed, exclusions decided after seeing the result, evaluations pinned by hand (`evaluation_refs`) instead of the contract's attempt selection (info: several valid attempts per run) |
| `MIXED_CODE_DIGEST`, `MIXED_GPU_TYPE` | Cells mix code versions or GPU models |
| `INVARIANT_MISMATCH`, `SELECTION_LEAK` | Different split/evaluator/dataset; best-checkpoint selection on the reported split |
| `ARTIFACT_DIGEST_CHANGED` | The artifact behind a value changed (ledger record; the file on the cluster with `--recheck-remote`) |
| `UNTAGGED_TEX_NUMBER`, `NO_TAGS_FOUND`, `UNRESOLVED_REF`, `UNRESOLVED_INPUT` | Numbers or files hpclab could not check (an untagged number, a manuscript without tags, an unknown bundle/cell, an `\input` it could not read), reported as such |

It verifies **record consistency, not scientific correctness**: whether the experiment answers the
question is still your call.

Verification runs against the ledger that produced the bundle, which lives on the student's
machine. Either ask the student to run `hpclab paper check` and send you the output, or ask for a
backup and verify it on your machine:

```bash
# student
hpclab backup ~/lab-share/            # consistent online copy of the ledger + vault
# you: open the copy as a separate hpclab home, verify, stop its controller
HPCLAB_HOME=/path/to/hpclab-backup-20260928T095957Z hpclab verify evb_01M3KP7Q7P07RMJFDASWYG1MQR paper/main.tex
HPCLAB_HOME=/path/to/hpclab-backup-20260928T095957Z hpclab controller stop
```

The exported `manifest.json` and `table.csv` also carry the full chain per value, readable without
hpclab.

## Grids of experiments, done right

What the tooling enforces so that a lab's results hold up:

* **Planned denominators.** A comparison starts from the planned matrix: each planned run is
  included, missing, failed or excluded with a stated reason. `mean ± std (2/3)` means one of three
  planned seeds is missing, and the table footnotes it.
* **Paired, descriptive statistics.** Mean, sample std, SEM, t-intervals and differences to the
  baseline paired by seed; no significance claims, and no bold "winner" unless the contract asks for
  it (`bold_rule: best_mean`, footnoted).
* **Comparability checks.** Split, evaluator and dataset must match across compared results;
  selecting the best checkpoint on the split you report is refused (`selection_leak`).
* **Smoke gates and circuit breakers** keep a broken environment from burning a sweep's
  allocation: the matrix waits for a short smoke job, and dispatch pauses after two identical
  environment failures.
* **Budgets per experiment**: GPUs × time limit reserved per attempt, settled from accounting;
  unknown consumption counts at its reservation, never as zero.

## What hpclab deliberately does not do

* **It does not act on other people's jobs.** No cancelling, requeueing or "cleaning up" jobs
  that were not submitted through your own hpclab; cancel always re-verifies that the job is the
  caller's own.
* **It does not auto-cancel "idle" runs.** GPU-idle, stalled-log and heartbeat flags are advisory
  evidence for a human.
* **It does not enforce lab-wide budgets or quotas.** Only Slurm can enforce an allocation.
  hpclab budgets cover the jobs submitted through it, per experiment.
* **It does not rank people** or produce productivity scores.
* **It does not guess.** Uncertain submissions, duplicates and unreadable accounting are reported
  as such and wait for a human; hidden usage is labelled `self_only`, not summed.
* **It does not judge the science.** Verification is about records; statistics are descriptive.
* **It does not centralize your lab's data.** Each person's ledger stays on their machine; nothing
  is uploaded anywhere. Sharing is explicit (evidence bundles, digests, backups).
* **It does not run on the login node** or do compute there, and it does not bypass site policy
  (`--qos`, partitions, limits are the scheduler's).

See [guarantees.md](guarantees.md) for the exact promises and their limits.
