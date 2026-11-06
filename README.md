# HPC Lab

**Reliable Slurm execution and traceable results for ML research: MCP tools for your coding agent,
a CLI, and a Python client.**

You ask Claude Code (or any MCP client):

> Run baseline and mixup on the cluster, 3 seeds each, at most 2 jobs at a time, under 40 GPU-hours.
> Smoke-test first. If something fails, tell me why and give me a recovery plan. When it's done, give me
> mean ± std, which seeds are missing, and where every number in the table came from.

HPC Lab is the deterministic runtime underneath that request. The agent reasons about the
experiment; HPC Lab freezes the code, submits to Slurm, watches the jobs, refuses unsafe actions and
keeps a ledger that links every table cell to a code snapshot, a config, a Slurm job and an output
file digest.

It targets the failure modes that eat research weeks on shared university clusters:

| What goes wrong today | What HPC Lab does |
|---|---|
| SSH drops right after `sbatch`, you resubmit, now two jobs write the same output dir | Records the submission intent first. A lost reply makes the attempt `submission_uncertain`, and hpclab **reconciles** it from an on-cluster receipt, `squeue` and `sacct`. It never resubmits blindly. |
| You edit the code while the job waits in the queue, and the job runs the new code | Every run executes a **content-addressed, read-only snapshot** captured at prepare time. |
| `scancel` of the wrong job id, typed from memory | Cancel re-verifies job id, owner, submit token and submit time **right before** `scancel`; on any mismatch nothing is cancelled. |
| The job waits a day, then dies in 10 s (`driver too old`, wrong venv, missing module) | Checks at prepare time (torch CUDA build vs the driver, GPU architecture, editable installs, caches on a small home), a fail-fast guard on the compute node, a **smoke gate** before the sweep, and a **circuit breaker** that pauses a sweep after 2 identical environment failures. |
| TIMEOUT at 98 % | Progress/ETA against the remaining walltime, `--signal` checkpoint-and-exit support, and a **resume plan** that validates the checkpoint first. |
| "Which job produced this number?" / a failed seed silently dropped from the mean | Comparisons use the **planned** denominator (missing and failed seeds stay visible). Evidence bundles export **LaTeX / CSV / Markdown** tables whose cells trace to evaluation → artifact digest → attempt → Slurm job → run spec → code → environment, and `hpclab paper check` flags stale or hand-edited numbers in your `.tex`. |
| PIs can't see where the lab's GPU-hours went | `usage_report` by user / partition / state with waste by cause, honest about Slurm's `PrivateData` visibility. |

> **Status: 0.1.0 alpha.** Tested with a process-level Slurm simulator, recorded real Slurm output
> and 3,863 tests. Stanford HAI is the first real site profile; no measurements from a real
> cluster are published yet. Read [what is and is not guaranteed](docs/guarantees.md).

## Quick start (no cluster needed)

Python 3.12+ and [uv](https://docs.astral.sh/uv/). hpclab is not on PyPI yet; install it from a
checkout of this repository:

```bash
uv tool install .                   # puts `hpclab` on your PATH (or: pip install .)
hpclab init --sim                   # local Slurm simulator as cluster "sim"
hpclab demo                         # prepare → smoke gate → 2×3 sweep → a node failure → diagnosis
                                    # → recovery → evidence table, in about a minute
```

Connect Claude Code and install the workflow skills:

```bash
claude mcp add hpclab -- hpclab mcp serve     # `hpclab mcp config` prints the exact command for your setup
hpclab skills install                         # copies the hpclab skills into ~/.claude/skills
```

The [five-minute quickstart](docs/quickstart.md) runs the same thing by hand: `examples copy`,
`prepare`, `submit`, `grid`, `compare`, `export`, `trace`, `paper check`, and a lost `sbatch` reply
injected on purpose.

## Using a real cluster

```bash
hpclab cluster profiles                     # bundled site profiles: generic-slurm, hai, sim
hpclab cluster add mycluster --profile generic-slurm --ssh-host login.example.edu --user <you> --account <slurm-account>
hpclab auth mycluster                       # opens an SSH master in *your* terminal (MFA stays with you)
hpclab cluster inspect mycluster            # auth, scheduler, storage, limits, capacity, nodes, site facts
hpclab onboard mycluster                    # the same checks, then one 1-GPU, 10-minute job end to end
```

HPC Lab works with any Slurm cluster you can reach over SSH; a site profile only adds local rules
(partitions, storage roles, polling limits). `generic-slurm` is the template, `hai` is a ready-made
profile for Stanford HAI, `sim` is the local simulator.

The public **site profile** holds site rules: partitions, storage roles, polling limits, and known
issues and facts with an expiry date. Your private **overlay** (`~/.hpclab/clusters/<id>.yaml`,
mode 0600) holds your username, account and paths. On the login node hpclab only runs bounded
metadata commands (`squeue`, `sacct`, `sinfo`, `stat`, small reads); training, checkpoint loading
and big hashes run as Slurm jobs. Stanford HAI setup: [docs/sites/hai.md](docs/sites/hai.md).

## How it works

```
Claude Code ─ MCP (stdio) ─┐                                    ┌─ Slurm adapter ─ OpenSSH (ControlMaster) ─ login node
hpclab CLI ────────────────┼─ Unix socket ─ controller ─ ledger ─┤                                          └ sbatch → compute node
Python client ─────────────┘   (single writer)  (SQLite WAL)     └─ snapshots · evidence · vault             (job wrapper)
```

* One **controller** per user and hpclab home owns the ledger (SQLite in WAL mode, on local disk)
  and a durable work queue. MCP servers, the CLI and Python scripts are reconnectable clients:
  closing the chat does not stop the bookkeeping.
* **Experiment** (question + planned matrix) → **Run** (immutable scientific config, one per cell)
  → **Attempt** (one execution; retries and resumes are new attempts) → **Slurm job** (cluster, job
  id, owner, submit time; a job id alone is never an identity) → **Artifact** → **Evaluation** →
  **Comparison** → **Evidence bundle**.
* Each attempt has three independent state axes: scheduler state, application exit, and result
  validation. `COMPLETED` does not mean "the results are valid".

Details: [docs/architecture.md](docs/architecture.md).

## MCP surface

17 tools, `hpclab://` resources and 5 prompts. Full schemas: [docs/tools.md](docs/tools.md)
(generated from the code).

| Tool | Purpose |
|---|---|
| `cluster_inspect` | Auth, identity, scheduler, storage, runtime, queue, GPU capacity, limits, drained/suspect nodes, site facts; optional `sbatch --test-only` start estimates |
| `experiment_prepare` | Freeze a code snapshot and configs into Runs and execution plans, with validation findings (block / warn / unchecked); add seeds or cells; derive a run |
| `run_submit` | Submit one run (idempotent; one execution claim per run; budget reservation) |
| `workflow_submit` | Dispatch a whole matrix with `max_parallel`, smoke gate and circuit breaker |
| `runs_list` | Runs, or the method × seed **grid** with one code per cell, plus a change feed |
| `run_inspect` | Attempts with job identity and the three state axes, diagnosis, pending reason, progress/ETA, GPU health, environment drift |
| `run_logs` | Bounded, cursor-based log reads (rotation- and UTF-8-safe) |
| `run_cancel` | Identity-verified cancel, or ask the job to checkpoint and exit (`request_checkpoint`) |
| `run_recover` | Diagnosis-driven retry / resume / resource-adjust plan (one run or every broken cell); never submits |
| `artifacts_list` | Declared outputs with digests and integrity state |
| `artifact_read` | Bounded, stale-safe read of one artifact |
| `evaluation_collect` | Extract metrics with a registered evaluator (idempotent, provenance-carrying) |
| `results_compare` | Planned-denominator comparison: coverage, mean ± std, paired differences with intervals, integrity disclosure |
| `evidence_export` | Evidence bundle: Markdown summary, LaTeX (booktabs) and CSV tables, JSON manifest |
| `evidence_verify` | Recompute a bundle and check a manuscript's numbers (record consistency, not science) |
| `workflow_cancel` | Stop dispatch; cancel queued or all managed cells with identity checks |
| `usage_report` | GPU-hours by user / partition / state, waste by cause, visibility-honest |

Resources: `hpclab://experiments` (start here in a new session), `experiments/{id}`,
`operations/{id}`, `plans/{id}` (+ `/script`, `/script/full`), `workflows/{id}` (+ `/grid`),
`runs/{id}`, `artifacts/{id}`, `evaluations/{id}`, `comparisons/{id}`, `evidence/{id}` (+
`/summary.md`, `/table.tex`, `/table.csv`, `/manifest.json`), `trace/{bundle}:{cell}`.

Prompts and skills: `plan_experiment`, `diagnose_run`, `write_results`, `onboard_cluster` and
`lab_overview` return the playbooks in [`skills/`](skills) (`hpclab-experiment-design`,
`hpclab-run-doctor`, `hpclab-paper-evidence`, `hpclab-cluster-onboarding`,
`hpclab-lab-overview`) for clients without skill support.

## Guarantees and limits

* **No blind resubmission, no unverified cancel, no unknown reported as success or zero.** Enforced
  by the attempt state machine and covered by fault injection (14 scenarios: lost replies,
  ambiguous errors, controller crashes at each step, accounting lag, expired auth, duplicate jobs,
  job-id reuse). The latest run (1 repetition each) met the expected outcome in **14 of 14**
  scenarios with 17 `sbatch` calls for 17 attempts; the earlier full run (5 repetitions each) in
  **70 of 70** repetitions with 85 `sbatch` calls for 85 attempts. Both: 0 duplicates, 0 wrong-job
  cancels, every unresolved attempt flagged for a human.
* **Exactly-once submission is not claimed.** The window between Slurm accepting a job and the
  reply being recorded cannot be closed; hpclab detects and reconciles it, and asks a human when it
  cannot identify the job uniquely.
* Cancel identity checks narrow but cannot eliminate a check-then-cancel race (Slurm has no
  compare-and-swap). Budgets govern only jobs submitted through hpclab. Read-only snapshots are a
  safeguard, not a sandbox. The simulator is not Slurm.
* The controller runs where you run it: if your laptop sleeps, Slurm jobs keep running but dispatch
  and observation pause until it wakes. Use an always-on machine for long sweeps.
* v0.1: single-node jobs (multi-GPU is fine), independent matrix cells (no job arrays or DAGs),
  JSON/JSONL metrics, one checkpoint manifest format.

Everything, with the evidence: [docs/guarantees.md](docs/guarantees.md).

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Five minutes on the simulator, then Claude Code |
| [Stanford HAI](docs/sites/hai.md) | VPN, Duo, overlay, scratch, CUDA 12.8 wheels, partitions and preemption, known issues |
| [For PIs](docs/for-pis.md) | Usage reports, the weekly digest, grids, verifying a paper's numbers, what hpclab will not do |
| [ExperimentSpec](docs/experiment-spec.md) | Every field of `hpclab.yaml` |
| [Architecture](docs/architecture.md) | Controller, ledger, outbox, state machine, reconciliation, provenance, trust boundaries |
| [Guarantees](docs/guarantees.md) | What is promised, what is not, and the measurements |
| [MCP reference](docs/tools.md) · [CLI reference](docs/cli.md) | Generated from the code by `scripts/gen_docs.py` |
| [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md) | |

## Development

```bash
uv sync
uv run pytest -m "not sim and not slow"      # fast suite (< 3 min); `-m sim` runs the simulator end-to-end tests
uv run ruff check src tests scripts && uv run mypy
uv run python scripts/gen_docs.py            # regenerate docs/tools.md and docs/cli.md after an interface change
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture rules and invariants.

License: [Apache-2.0](LICENSE).
