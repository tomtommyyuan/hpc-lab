# Guarantees and limits

What HPC Lab promises, how it keeps each promise, what it deliberately does *not* promise, and the
measurements behind it. The mechanisms are described in [architecture.md](architecture.md).

## What is guaranteed

| Guarantee | Mechanism | Evidence |
|---|---|---|
| **`sbatch` is never called twice for one attempt.** A lost reply, an ambiguous error or a controller crash never leads to a blind resubmission. | The dispatch handler commits `submission_in_flight` *before* its single `sbatch` call; a handler that finds the attempt already in flight turns it `submission_uncertain` and reconciles (on-cluster receipt → `squeue` → `sacct`). Nothing resubmits automatically, ever. | Fault matrix scenarios `lost_reply`, `ambiguous_error_*`, `crash_*`, `kill9_random_workflow`; `tests/submission`, `tests/e2e` |
| **An uncertain submission is bound only to a job that is provably its own.** | Binding needs a *unique* job with owner = your cluster user, this attempt's token in the job name (and comment), submitted after `dispatch_at − skew`. More than one match → `duplicate_detected`, nothing cancelled. No match before the reconcile window closes → the operation is `blocked` for a human (`hpclab resolve`). | `duplicate_same_token`, `ambiguous_error_not_queued`, `crash_after_inflight` |
| **At most one active execution per run.** Two agents, a retry with another idempotency key, a workflow and a manual `run_submit` cannot run the same cell twice. | One execution claim per run, set in the transaction that creates the attempt and released only when it is terminal, rejected, cancelled before submission or abandoned. Every entry point goes through the same admission gate; an execution plan is consumed once. | `tests/submission`, `tests/workflows/test_workflows_invariants.py` |
| **No unverified cancel.** | Right before `scancel` the job is re-read live and job id, owner, job-name token, submit time and restart count must match the attempt. Any mismatch → `cancel_state: refused`, zero `scancel` calls. | `cancel_foreign_owner` (job id reused by another user: refused), `cancel_genuine` (positive control) |
| **Unknown is never reported as success or as zero.** | "Not in `squeue`" → ask `sacct`; no row yet → `observation_gap`, state unchanged. No accounting → budget settles as *unknown* (still counted at its reservation). An unreachable source → `coverage.complete: false` or a typed error, never an empty success. A check that could not run is `unchecked`, never `pass`. | `accounting_gap`, `squeue_timeouts`; `tests/observe`, `tests/coverage` |
| **Terminal states are not rewritten by stale data.** | Out-of-order observations are ignored; only a higher restart count (a Slurm requeue) reopens an attempt, as a new recorded segment. | `tests/observe` |
| **A queued job runs the code you prepared.** | The project is captured into a content-addressed snapshot at prepare time, published read-only on the cluster, and each attempt runs in its own hardlinked work directory. Edits after `prepare` cannot reach it. | `tests/snapshot`, `tests/e2e` |
| **Failed and missing seeds stay in the denominator.** | `results_compare` starts from the *planned* run set; every planned run is included, missing, failed or excluded (exclusions need a reason and are disclosed). The grid, comparison, evidence and digest share one cell-status function. | `tests/compare`, `tests/coverage` |
| **Every reported number traces to its sources.** | Evidence bundles record evaluation → artifact digest (computed on the compute node) → attempt → Slurm job → run spec → code digest/commit → environment; LaTeX cells carry `\hpcval{<bundle>:<cell>}` tags that `evidence_verify` / `hpclab paper check` recheck (stale, hand-edited or unfootnoted numbers). | `tests/evidence` |
| **Retries of mutations are safe.** | Idempotency records `(principal, tool, key)` with the payload hash and the stored response, written in the same transaction as the mutation. | `tests/submission`, `tests/mcp` |
| **Bookkeeping survives crashes and restarts.** | Durable outbox with leases; the restarted controller reclaims a dead controller's leases; handlers are re-entrant. | `crash_*`, `kill9_random_workflow`; `tests/controller` |

## What is not guaranteed

* **No exactly-once submission.** There is an unavoidable window between Slurm accepting a job
  and hpclab recording the reply. hpclab *detects* it (the attempt becomes
  `submission_uncertain`) and *reconciles* it; if it cannot uniquely identify the job, it stops
  and asks a human instead of guessing. The price of never duplicating is that a submission
  whose `sbatch` may or may not have run (a crash right after the in-flight commit) waits for
  your decision (`hpclab resolve <attempt> --abandon --reason ...` or `--bind-job <job id>`) rather than being
  retried automatically.
* **Cancel narrows, but cannot close, a check-then-act race.** Slurm has no compare-and-swap:
  between the live identity check and `scancel` the job could, in theory, end and its id be
  reused. The window is one round trip; the check makes a wrong cancel require that coincidence,
  instead of a typo.
* **Budgets cover only jobs submitted through hpclab.** A budget reserves GPUs × time limit per
  attempt at admission and settles from accounting. Jobs you submit by hand, other people's jobs
  and your site allocation are outside it; the site's quotas are enforced by Slurm, not by
  hpclab. `usage_report` shows all of your jobs (and your lab's, when Slurm lets you see them).
* **The simulator is not Slurm.** It reproduces Slurm's command-line behaviour and output formats
  (the parsers are also tested against recorded real Slurm output in `tests/fixtures/slurm`),
  not slurmctld/slurmdbd internals: scheduling latency, RPC timeouts under load, accounting-lag
  distributions, preemption/requeue races, federation, NFS attribute caching, quota errors and
  SSH ControlMaster failures are approximated or absent. Simulator timings are not predictions
  for a real cluster.
* **Snapshots are not a sandbox.** Read-only permissions and content addressing prevent
  accidents; they do not protect against code running as your own Unix user. `runtime.setup`
  lines are executed verbatim.
* **Environments are recorded, not frozen.** A shared virtualenv or module can change between
  prepare and run. hpclab statically checks the venv at prepare time (torch CUDA build vs the
  driver, GPU architecture, editable installs), fingerprints declared lockfiles, records what
  actually ran in `env.json` and reports drift; it does not pin or rebuild environments. Large
  datasets are referenced with a pin level (`pinned`, `fingerprinted`, `unpinned`, `unresolved`),
  not hashed.
* **A checkpoint's existence is not proof it loads.** Checkpoint manifests are validated
  deterministically (files, sizes, digests, config compatibility); loadability needs a probe run
  on a compute node, which `run_recover` creates for you.
* **The controller needs to be running.** If your laptop sleeps, Slurm jobs continue but
  dispatch, observation and recovery pause until it wakes.
* **Accounting is required for confirmed outcomes.** Without `sacct`, a job that left the queue
  stays in `observation_gap` and its GPU-hours stay *unknown*.
* **Visibility is Slurm's.** With `PrivateData=jobs` a non-coordinator sees only their own jobs;
  the usage report then says `visibility: self_only` and gives no lab totals.
* **Advisory signals are advisory.** GPU-idle and stalled-log flags never cancel anything.
  Notifications are delivered at least once (a crash can re-send one).
* **Record consistency, not scientific correctness.** `evidence_verify` checks that the numbers
  match their records. Statistics are descriptive (mean, sample std, SEM, t-intervals, paired
  differences); there are no significance claims and no automatic "best" bolding unless you ask.
* **v0.1 scope.** Single-node jobs (multi-GPU fine), independent matrix cells (no job arrays or
  DAGs), `hpclab.metrics/v1` / JSON / JSONL metrics, one checkpoint manifest format
  (`hpclab.checkpoint/v1`), no container runtime field (put `apptainer exec` in the argv), Slurm
  only.

## Measurements

### Fault-injection matrix

`benchmarks/fault_matrix.py` runs each scenario in a fresh hpclab home with a real controller
subprocess, the real services and the real `SlurmScheduler` against the simulator. Ground truth
(which jobs `sbatch` really accepted, which received `scancel`, how each ended) comes from the
simulator's own state, not from hpclab's ledger. Methodology and per-repetition rows:
[benchmarks/README.md](../benchmarks/README.md).

**Latest run** (`benchmarks/results/fault_matrix_quick.md`, generated 2026-09-28T14:36:18Z by
`python benchmarks/fault_matrix.py --quick`, hpclab 0.1.0, Python 3.12.4, macOS arm64,
14 scenarios × 1 repetition, reconcile window shortened to 20 s): **14/14 repetitions met their
expected outcome.**

| Scenario | Injected fault | Pass | sbatch calls / attempts | Jobs accepted by Slurm | Duplicates | Cancels (wrong job) | Unresolved (flagged for a human) | Wrong final states | Decided, s |
|---|---|---|---|---|---|---|---|---|---|
| `lost_reply` | job accepted, reply lost | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.2 |
| `ambiguous_error_queued` | socket timeout, but queued | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.0 |
| `ambiguous_error_not_queued` | socket timeout, nothing queued | 1/1 | 1 / 1 | 0 | 0 | 0 (0) | 1 (1) | 0 | 26.6 |
| `crash_before_inflight` | SIGKILL before the in-flight commit | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.5 |
| `crash_after_inflight` | SIGKILL after the commit, before sbatch | 1/1 | 1 / 1 | 0 | 0 | 0 (0) | 1 (1) | 0 | 26.9 |
| `crash_after_sbatch` | SIGKILL after sbatch accepted | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 3.4 |
| `crash_before_identity` | SIGKILL before the job id is committed | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 3.5 |
| `accounting_gap` | sacct lags 10 s, squeue forgets after 0.2 s | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.2 |
| `squeue_timeouts` | squeue times out for 12 s | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.6 |
| `auth_required_window` | auth required for 6 s from submit | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 121.5 |
| `duplicate_same_token` | lost reply + a manual resubmission | 1/1 | 1 / 1 | 1 (+1 external) | 0 | 0 (0) | 1 (1) | 0 | 3.1 |
| `cancel_genuine` | run_cancel of a running job (control) | 1/1 | 1 / 1 | 1 | 0 | 1 (0) | 0 (0) | 0 | 5.2 |
| `cancel_foreign_owner` | job id now names another user's job | 1/1 | 1 / 1 | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.2 |
| `kill9_random_workflow` | kill -9 at a random moment of a 2×2 workflow | 1/1 | 4 / 4 | 4 | 0 | 0 (0) | 0 (0) | 0 | 7.4 |
| **Total** | | **14/14** | **17 / 17** | **15 (+1 external)** | **0** | **1 (0)** | **3 (3)** | **0** | |

How to read it:

* 17 attempts, 17 `sbatch` calls, 15 jobs created by hpclab: the 2 attempts without a job are
  the two scenarios where nothing was queued. Zero duplicates, zero wrong-job cancels, zero wrong
  final states.
* The 3 unresolved attempts are the *correct* outcome of `ambiguous_error_not_queued`,
  `crash_after_inflight` and `duplicate_same_token`: hpclab cannot prove what happened, so every
  one of them (3 of 3) is flagged for a human with a remediation. None is silently left open.
* `auth_required_window` takes about 2 minutes because a step blocked on authentication is
  retried every 120 s and the benchmark clears the fault without running `hpclab auth`. In real
  use `hpclab auth <cluster>` wakes the blocked work immediately.
* Times depend on the controller's 5 s observation tick and machine load; they are not
  predictions for a real cluster.

**Earlier full run** (`benchmarks/results/fault_matrix.md`, generated 2026-09-28T07:24:39Z,
same machine, 14 scenarios × 5 repetitions, before the last changes to the services, CLI and job
wrapper): **70/70 repetitions met their expected outcome**: 85 `sbatch` calls for 85 attempts,
75 jobs (+5 external), 0 duplicates, 5 cancels (0 wrong), 15 unresolved (15 of 15 flagged for a
human), 0 wrong final states. Per-scenario rows are in that report; re-run it with
`python benchmarks/fault_matrix.py` (about 5 minutes).

One repetition per scenario (five in the full run) shows that the protocol handles each fault as
designed on this machine; it does not bound rare interleavings. The deterministic crash points
and the test suite carry that burden.

### Test suite

3,861 tests at 0.1.0 (`uv run pytest --co | tail -1`): 116 are marked `sim` (real local
processes through the simulator), 19 of which are also `slow`. They include the state-machine
invariants, controller crashes at each step of a submission, parsers against recorded real Slurm
output (`tests/fixtures/slurm`), MCP contract tests (every parameter and nested field
documented), end-to-end CLI runs, and checks that these docs stay true (generated references
fresh, links and `hpclab` commands valid, no personal data). The fast subset
(`-m "not sim and not slow"`, 3,745 tests) passes in under 3 minutes on the machine above.

### What has not been measured yet

There are no published hpclab measurements from a real cluster yet. The Stanford HAI profile
([sites/hai.md](sites/hai.md)) encodes site facts measured by hand in 2026-08, each with a source
and an expiry; a controlled real-cluster benchmark (submission counts, reconcile latency under
real `sacct` lag) is future work.
