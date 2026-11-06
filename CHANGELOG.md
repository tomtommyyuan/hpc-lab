# Changelog

All notable changes to HPC Lab are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (while at 0.x, minor versions may break
compatibility).

## 0.1.0 - 2026-09-28

First public release (alpha). Tested against a process-level Slurm simulator, recorded real
Slurm output and 3,861 tests (3,842 without the slow ones); Stanford HAI is the
first real site profile. No measurements from a real cluster are published yet.

### Added

- **Frontends.** A stdio MCP server with 17 tools, `hpclab://` resources and 5 prompts
  (`docs/tools.md`); the `hpclab` CLI (`docs/cli.md`); a Python client (`HpcLabClient`). All
  three are reconnectable clients of one local controller and return the same result envelope
  (summary, data, coverage, warnings, typed errors).
- **Controller and ledger.** One controller per hpclab home as the single writer of a SQLite
  (WAL) ledger with an append-only event log, idempotency records, optimistic versions and a
  durable, leased outbox; autostart, crash recovery of leases, periodic polling and sweeps.
- **Submission protocol.** One execution claim per run and a single admission gate
  (claim, consume-once plans, concurrency slots, budget). The intent is committed before the one
  `sbatch` call, with an on-cluster receipt. Lost replies, ambiguous errors and crashes become
  `submission_uncertain` and are reconciled from the receipt, `squeue` and `sacct` (unique match
  on owner, token and submit time). Duplicates are detected and never auto-cancelled;
  `hpclab resolve` records a human decision.
- **Identity-verified cancel** (job id, owner, job-name token, submit time, restart count
  re-read right before `scancel`) and `request_checkpoint` (send the declared signal so the
  program can checkpoint and exit).
- **Prepare.** Content-addressed, git-aware code snapshots (secrets, data, weights and
  virtualenvs excluded; read-only on the cluster). Immutable Runs per matrix cell, frozen
  execution plans, smoke runs. Validation at prepare time: partitions and limits, account and
  QoS, storage roles, preemption and walltime rules, budget, plus bounded remote checks of the
  environment (torch CUDA build vs the driver fact, GPU architecture, editable installs, cache
  locations) and an optional `sbatch --test-only`. Resource advice from your own history.
  `--extends` (post-hoc seeds and cells, disclosed) and `--derive-from`.
- **Job wrapper.** `status.json`, `env.json`, `outputs.json` with digests computed on the
  compute node; fail-fast environment guard (exit 3); GPU telemetry and heartbeat sidecar;
  signal forwarding and a stop file; refusal to run in an attempt directory another job owns.
- **Workflows.** Matrix dispatch with `max_parallel`, a smoke gate and a circuit breaker for
  repeated environment failures; `workflow cancel` / `resume`.
- **Observation.** Batched, rate-limited polling shared by all clients; three state axes
  (scheduler, application, result); decoded pending reasons; progress/ETA against the
  remaining walltime; advisory health flags; deterministic diagnosis of failures with evidence.
- **Recovery.** Diagnosis-driven `retry`, `resume` (from a validated checkpoint, proven
  loadable by a probe run) and `adjust_resources` plans for one run or every broken cell;
  automatic exclusion of nodes that failed repeatedly. Never submits by itself.
- **Results and evidence.** Declared artifacts with integrity states; evaluators
  `hpclab_metrics@1`, `json_path@1`, `jsonl_record@1`; comparisons over the planned run set
  (included / missing / failed / excluded; mean, sample std, SEM, t-intervals, paired
  differences; comparability and selection-leak checks); evidence bundles (Markdown, LaTeX with
  `\hpcval` tags, CSV, JSON manifest); `trace`, `verify` and `paper check` for manuscripts.
- **Lab tools.** Per-experiment GPU-hour budgets; `usage_report` with waste by cause and
  explicit `PrivateData` visibility; the weekly `digest`; a local vault of raw job records;
  consistent online `backup`; notifications (desktop, ntfy, Slack webhook, command, log).
- **Sites.** Profiles `generic-slurm`, `hai` (Stanford HAI) and `sim`, with private 0600
  overlays for identity and expiring facts and known issues instead of hard-coded site lore;
  `cluster inspect`, `auth` (MFA stays in your terminal) and `onboard`.
- **Simulator.** Fake `sbatch`/`squeue`/`sacct`/`scancel`/`sinfo`/`sacctmgr`/`sstat`/`srun`/
  `scontrol` running the real `SlurmScheduler` unmodified, jobs as local processes, and fault
  injection (lost replies, ambiguous errors, accounting lag, timeouts, auth expiry, node
  failure, preemption, OOM, requeue).
- **Getting started.** `hpclab init --sim`, `hpclab demo`, the `tiny-classifier` example
  (with a vendorable `hpclab_io.py` for metrics and checkpoints) and five Claude Code skills
  (`hpclab skills install`).
- **Benchmarks.** `benchmarks/fault_matrix.py`: 14 fault scenarios against a real controller
  and the simulator's ground truth. Full run (5 repetitions): 70/70 repetitions met their
  expected outcome (85 attempts, 85 `sbatch` calls, 0 duplicates, 0 wrong-job cancels, 15 of 15
  unresolved attempts flagged for a human); quick re-run on the release code (1 repetition):
  14/14 (17 attempts, 17 `sbatch` calls, 0 duplicates, 0 wrong-job cancels, 3 of 3 unresolved
  attempts flagged).
- **Docs.** Quickstart, Stanford HAI guide, guide for PIs, architecture, guarantees,
  ExperimentSpec reference, and MCP and CLI references generated from the code.

### Known limitations

- Single-node jobs (multi-GPU is fine); independent matrix cells (no job arrays or DAGs);
  JSON/JSONL metrics; one checkpoint manifest format; no container runtime field; Slurm only.
- No exactly-once submission (detected and reconciled instead); the cancel identity check
  narrows but cannot close a check-then-act race; budgets cover only jobs submitted through
  hpclab. See [docs/guarantees.md](docs/guarantees.md).
- The controller runs where you start it: dispatch and observation pause while that machine
  sleeps.
