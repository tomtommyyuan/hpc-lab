# Benchmarks: the fault-injection matrix

`fault_matrix.py` measures what hpclab's submission, observation and cancel protocol does when
the things that go wrong on shared clusters go wrong: lost `sbatch` replies, ambiguous `sbatch`
errors, controller crashes at every step of a submission, accounting lag, `squeue` timeouts,
expired SSH authentication, duplicate jobs and job-id reuse before a cancel.

```bash
uv run python benchmarks/fault_matrix.py --out benchmarks/results          # 14 scenarios x 5 reps (~4-5 min)
uv run python benchmarks/fault_matrix.py --out benchmarks/results --quick  # 14 scenarios x 1 rep (~2 min)
uv run python benchmarks/fault_matrix.py --list                            # scenario names
uv run python benchmarks/fault_matrix.py --only lost_reply,crash_after_inflight --reps 10 --jobs 4
```

Outputs `results/fault_matrix.{json,md}` (quick mode: `fault_matrix_quick.*`; a run with `--only` or
`--reps` writes `fault_matrix_custom.*`, so a subset never overwrites the full report). Ctrl-C stops
every repetition, stops its controller and simulated jobs, removes the work directory and writes no
report (exit status 130). The Markdown report
has one summary row per scenario with sums over **all** repetitions, the expected outcome of
every scenario, every failing repetition with its counters and notes, and one row per repetition.
The JSON has the same records plus the environment (versions, platform, random seed, settings).
The exit status is non-zero if any repetition missed its expected outcome.

## Methodology

**Everything real except the cluster.** Each repetition gets a fresh hpclab home, the local
simulator registered through `init_sim`, and a real controller *subprocess* started by
`HpcLabClient` autostart exactly as for a user. Requests go over the controller socket to the
real services; the real `SlurmScheduler` runs its real `sbatch`/`squeue`/`sacct`/`scancel`
command lines against the simulator's executables. The workload is the bundled
`examples/tiny-classifier` (one cell, no smoke gate, except the workflow scenario: 2 methods x 2
seeds).

**Ground truth comes from outside hpclab.** The simulator's own state says which jobs `sbatch`
accepted, which received `scancel`, and how each job ended. hpclab's ledger is only read
(read-only SQLite) to see what hpclab *believes*; the two are compared.

**Faults.** Simulator faults (`SimCluster.set_faults`) inject lost replies, ambiguous errors,
accounting lag, command timeouts and authentication failures at the transport/command level, the
way they surface from a real login node. Controller crashes at precise points of
`submit.dispatch` use `crash_controller.py`: the stock controller plus one hook that appends to a
marker file and sends `SIGKILL` to its own process (no cleanup, no commit of an open SQLite
transaction -- the same as `kill -9`). The hook is disarmed once the marker exists and the
harness restarts the stock controller. A random `kill -9` of a whole workflow complements the
deterministic crash points (the in-flight window is milliseconds wide, so random kills rarely
hit it -- which is why the deterministic points exist).

**Repetitions are independent and run in parallel** (`--jobs`, default 5): own home, own
simulator, own controller. The random seed of the `kill -9` timing is recorded in the JSON.

### Metrics (per repetition, summed in the report)

| Metric | Definition |
|---|---|
| sbatch calls by hpclab | `attempt.dispatching` events: hpclab commits `submission_in_flight` immediately before its single `sbatch` call, so this counts the sbatch calls it made (a crash before that commit makes none). |
| Jobs accepted by Slurm | Jobs in the simulator (ground truth). Jobs the scenario itself submitted are shown separately as *external*. |
| Duplicates | Jobs beyond one per attempt (same job name = same attempt token), not counting external ones. |
| Cancels (wrong job) | Jobs that received `scancel` (simulator `cancel_requested`); in parentheses, cancels of a job the scenario did not ask hpclab to cancel. |
| Unresolved (flagged) | Attempts left `submission_in_flight` / `submission_uncertain` / `duplicate_detected` at the end; in parentheses, those whose operation is `blocked` with a remediation for a human. Unresolved and *not* flagged would be a silent failure. |
| Wrong final states | Terminal attempts whose recorded scheduler state differs from the job's real state. |
| Decided | Seconds from the fault (or crash) until the outcome was decided: job identity bound, attempt flagged for a human, or cancel confirmed/refused. For timeouts/auth scenarios: from the moment the fault ended. |
| Settled | Seconds until the attempt was terminal *and* its outputs validated. |

A repetition **passes** when its scenario-specific expectation holds (listed in the report),
which always includes zero duplicates caused by hpclab, zero wrong-job cancels and zero wrong
final states.

### Scenarios

| Scenario | Fault | Expected |
|---|---|---|
| `lost_reply` | `sbatch` accepts the job, the reply is lost | uncertain -> reconciled via receipt/`squeue`/`sacct` -> bound to the one job; never resubmitted |
| `ambiguous_error_queued` | `sbatch` prints a socket timeout and exits 1, but queued the job | same as a lost reply |
| `ambiguous_error_not_queued` | same error, nothing queued | never resubmitted; after the reconcile window the attempt is flagged for a human (`hpclab attempt resolve`) |
| `crash_before_inflight` | controller killed at the start of `submit.dispatch` | nothing sent; the restarted controller dispatches once |
| `crash_after_inflight` | killed after the in-flight commit, before `sbatch` | hpclab cannot know whether `sbatch` ran: reconcile finds nothing -> flagged for a human; 0 jobs |
| `crash_after_sbatch` | killed after the simulator accepted the job, reply unread | uncertain -> reconciled and bound |
| `crash_before_identity` | killed inside the transaction that records the job id | rolled back -> uncertain -> reconciled and bound |
| `accounting_gap` | job leaves `squeue` after 0.2 s, `sacct` lags 10 s | observation gap recorded, no state guessed; COMPLETED once `sacct` shows it |
| `squeue_timeouts` | `squeue` times out for 12 s while the job runs | no state invented during the outage; completes afterwards |
| `auth_required_window` | every cluster call needs interactive auth for 6 s from submit | operation `blocked` with `hpclab auth sim`; completes after auth returns |
| `duplicate_same_token` | lost reply, then the same job script is resubmitted outside hpclab | `duplicate_detected`, flagged, **nothing cancelled automatically** |
| `cancel_genuine` | `run_cancel` of a running job (positive control) | identity verified, exactly that job gets `scancel`, cancel confirmed |
| `cancel_foreign_owner` | the job id now names another user's job (id reuse), then `run_cancel` | identity check fails -> refused, zero `scancel` |
| `kill9_random_workflow` | `kill -9` at a random moment (0-12 s) of a 2x2 workflow | at most one job per attempt; every cell valid or flagged |

## Reading the current results

`results/fault_matrix.md` is the latest full run (the commit that changes the code should also
update it); `results/fault_matrix_quick.md` is the latest quick run (1 repetition per scenario),
the one [docs/guarantees.md](../docs/guarantees.md#fault-injection-matrix) quotes, with the full
run as the earlier totals. Two numbers need context:

* **Unresolved attempts are the correct outcome** of `ambiguous_error_not_queued`,
  `crash_after_inflight` and `duplicate_same_token`: hpclab cannot prove what happened, so it
  stops and asks (operation `blocked`, remediation `hpclab attempt resolve <id> --bind-job ... |
  --abandon`). A duplicate is flagged at the first reconcile pass (seconds); an attempt with no
  job found is flagged when the reconcile window closes (20 s here; 120 s in the bundled sim
  profile, 900 s for HAI), plus one retry interval.
* **`auth_required_window` takes ~2 minutes** after authentication works again: a step blocked on
  authentication is retried every `auth_retry_s` = 120 s, and nothing wakes it early when the user
  re-authenticates. Correct (nothing is lost or duplicated), but slow; waking blocked work when
  `hpclab auth` succeeds is future work.

## Limits (simulator != real Slurm)

* The simulator reproduces Slurm's command-line behaviour and output formats (the parsers are
  also tested against recorded real Slurm output in `tests/fixtures/slurm`), not
  slurmctld/slurmdbd internals: scheduling latency, RPC timeouts under load, `sacct` database lag distributions,
  requeue/preemption races and federation are approximations or absent.
* The "remote" filesystem is local: NFS attribute caching, partial writes on a network
  filesystem and quota errors are not exercised. SSH ControlMaster behaviour (MFA prompts,
  master death, host-key changes) is modelled only as `AuthRequiredError` from the transport.
* Times are dominated by the controller's 5 s observation tick, the 60 s periodic sweeps and the
  retry constants, not by Slurm; they vary with machine load (repetitions run in parallel) and are
  not predictions for a real cluster.
* Five repetitions (one in quick mode) show that the protocol handles each fault as designed on
  this machine; they do not bound rare interleavings. The deterministic crash points and the unit/integration
  suites (`tests/submission`, `tests/workflows`, `tests/e2e`) carry that burden.
* Exactly-once submission is not claimed: the window between Slurm accepting a job and hpclab
  recording it cannot be closed, only detected and reconciled (see `crash_after_sbatch`,
  `crash_before_identity`, `duplicate_same_token`).

## Files

* `fault_matrix.py` -- the matrix runner (standalone script; imports hpclab).
* `crash_controller.py` -- `python benchmarks/crash_controller.py --home H --crash-at POINT
  [--nth N] --marker FILE`: a controller that SIGKILLs itself at `before_inflight`,
  `after_inflight`, `after_sbatch` or `before_identity`. Also used by `tests/e2e`.
* `results/` -- the latest reports.
