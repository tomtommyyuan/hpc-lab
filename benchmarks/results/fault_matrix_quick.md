# hpclab fault-injection matrix

Generated 2026-09-28T14:36:18Z · hpclab 0.1.0 · Python 3.12.4 · Darwin 24.1.0 (arm64) · mode **quick** · 1 rep(s) per scenario · 5 parallel workers · wall time 128 s

**14/14 repetitions met their expected outcome.** Every repetition ran in a fresh hpclab home with a real controller subprocess, the real services and `SlurmScheduler` against the Slurm simulator (ground truth). Methodology and limits: [README.md](../README.md). Site-policy change vs the bundled sim profile: `{"reconcile_window_s": 20}`.

## Summary (sums over all repetitions; nothing is omitted)

| Scenario | Injected fault | Pass | sbatch calls by hpclab | Jobs accepted by Slurm | Duplicates | Cancels (wrong job) | Unresolved (flagged for a human) | Wrong final states | Decided, s (median / max) | Settled, s (median / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lost_reply` | sbatch_lose_reply=1 (job accepted, reply lost) | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.2 / 4.2 | 4.6 / 4.6 |
| `ambiguous_error_queued` | sbatch_ambiguous_error=1, job queued | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.0 / 4.0 | 5.0 / 5.0 |
| `ambiguous_error_not_queued` | sbatch_ambiguous_error=1, nothing queued | 1/1 | 1 / 1 attempts | 0 | 0 | 0 (0) | 1 (1) | 0 | 26.6 / 26.6 | — |
| `crash_before_inflight` | controller SIGKILL before the in-flight commit | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.5 / 0.5 | 6.3 / 6.3 |
| `crash_after_inflight` | controller SIGKILL after the in-flight commit, before sbatch | 1/1 | 1 / 1 attempts | 0 | 0 | 0 (0) | 1 (1) | 0 | 26.9 / 26.9 | — |
| `crash_after_sbatch` | controller SIGKILL after sbatch accepted, reply unread | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 3.4 / 3.4 | 5.9 / 5.9 |
| `crash_before_identity` | controller SIGKILL before the job identity is committed | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 3.5 / 3.5 | 6.2 / 6.2 |
| `accounting_gap` | accounting_lag_s=10 + min_job_age_s=0.2 | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.2 / 0.2 | 12.1 / 12.1 |
| `squeue_timeouts` | squeue transport timeouts for 12 s | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 4.6 / 4.6 | 4.6 / 4.6 |
| `auth_required_window` | auth_required for 6 s from submit | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 121.5 / 121.5 | 121.5 / 121.5 |
| `duplicate_same_token` | lost reply + external resubmission (same token) | 1/1 | 1 / 1 attempts | 1 (+1 external) | 0 | 0 (0) | 1 (1) | 0 | 3.1 / 3.1 | — |
| `cancel_genuine` | run_cancel on a running job (control) | 1/1 | 1 / 1 attempts | 1 | 0 | 1 (0) | 0 (0) | 0 | 5.2 / 5.2 | — |
| `cancel_foreign_owner` | run_cancel after the job id names another user's job | 1/1 | 1 / 1 attempts | 1 | 0 | 0 (0) | 0 (0) | 0 | 0.2 / 0.2 | — |
| `kill9_random_workflow` | kill -9 at a random moment of a 2x2 workflow | 1/1 | 4 / 4 attempts | 4 | 0 | 0 (0) | 0 (0) | 0 | 7.4 / 7.4 | 7.4 / 7.4 |

## Expected outcome per scenario

- `lost_reply`: 1 job; reconciled (receipt/squeue/sacct) and bound; completes; no resubmission
- `ambiguous_error_queued`: sbatch errors ambiguously but queued the job: 1 job, bound, completes
- `ambiguous_error_not_queued`: sbatch errors ambiguously and queued nothing: 0 jobs, never resubmitted, flagged for a human
- `crash_before_inflight`: nothing was sent yet; the restarted controller re-runs the step and dispatches once: 1 job, completes
- `crash_after_inflight`: hpclab cannot know whether sbatch ran: it reconciles, finds no job and flags the attempt for a human -- never resubmits: 0 jobs
- `crash_after_sbatch`: Slurm has the job but hpclab never read the reply: uncertain -> reconciled and bound to that one job
- `crash_before_identity`: the transaction recording the job id rolled back: uncertain -> reconciled and bound to the one job
- `accounting_gap`: job leaves squeue before sacct knows it: gap flagged, never guessed; COMPLETED once sacct shows it
- `squeue_timeouts`: squeue times out for 12 s while the job runs: no state is invented; completes after
- `auth_required_window`: auth expires for 6 s at submit: operation blocked with 'hpclab auth sim'; completes after; 1 job
- `duplicate_same_token`: a 2nd job with the same token appears (external resubmission after a lost reply): flagged, nothing cancelled
- `cancel_genuine`: positive control: identity verified, exactly this job receives scancel, cancel confirmed
- `cancel_foreign_owner`: the job id now names another user's job (id reuse): cancel refused, zero scancel
- `kill9_random_workflow`: kill -9 at a random moment of a 2x2 dispatch: at most 1 job per attempt; every cell valid or flagged

## Failure cases (0)

None in this run.

## Every repetition

| Scenario | Rep | Pass | Outcome | Decided s | Settled s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `lost_reply` | 1 | yes | terminal/COMPLETED/valid job 1000 | 4.2 | 4.61 |  |
| `ambiguous_error_queued` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.96 | 5.0 |  |
| `ambiguous_error_not_queued` | 1 | yes | submission_uncertain (op unresolved) | 26.62 | — |  |
| `crash_before_inflight` | 1 | yes | terminal/COMPLETED/valid job 1000 | 0.5 | 6.35 | phase at crash: staging |
| `crash_after_inflight` | 1 | yes | submission_uncertain (op unresolved) | 26.86 | — | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.4 | 5.93 | phase at crash: submission_in_flight |
| `crash_before_identity` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.45 | 6.18 | phase at crash: submission_in_flight |
| `accounting_gap` | 1 | yes | terminal/COMPLETED/valid job 1000 | 0.21 | 12.14 | observation_gap flagged: True; terminal state from: sacct |
| `squeue_timeouts` | 1 | yes | terminal/COMPLETED/valid job 1000 | 4.59 | 4.59 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `auth_required_window` | 1 | yes | terminal/COMPLETED/valid job 1000 | 121.51 | 121.51 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `duplicate_same_token` | 1 | yes | duplicate_detected (duplicate_detected) | 3.13 | — |  |
| `cancel_genuine` | 1 | yes | cancel confirmed, CANCELLED | 5.21 | — |  |
| `cancel_foreign_owner` | 1 | yes | cancel refused | 0.21 | — |  |
| `kill9_random_workflow` | 1 | yes | workflow completed {'succeeded': 4} | 7.44 | 7.44 | killed 2.6s after submit; attempt phases then: {'observing': 2} |

## Limits

The simulator reproduces Slurm's command-line behaviour and output formats, not its internals: real slurmctld/slurmdbd timing, multi-cluster federation, SSH ControlMaster failures and site-specific wrappers are not exercised. Timings depend on the controller's 5 s observation tick and this machine's load (repetitions run in parallel). See README.md.
