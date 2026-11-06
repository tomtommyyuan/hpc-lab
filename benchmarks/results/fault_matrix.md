# hpclab fault-injection matrix

Generated 2026-09-28T07:24:39Z · hpclab 0.1.0 · Python 3.12.4 · Darwin 24.1.0 (arm64) · mode **full** · 5 rep(s) per scenario · 6 parallel workers · wall time 250 s

**70/70 repetitions met their expected outcome.** Every repetition ran in a fresh hpclab home with a real controller subprocess, the real services and `SlurmScheduler` against the Slurm simulator (ground truth). Methodology and limits: [README.md](../README.md). Site-policy change vs the bundled sim profile: `{"reconcile_window_s": 20}`.

## Summary (sums over all repetitions; nothing is omitted)

| Scenario | Injected fault | Pass | sbatch calls by hpclab | Jobs accepted by Slurm | Duplicates | Cancels (wrong job) | Unresolved (flagged for a human) | Wrong final states | Decided, s (median / max) | Settled, s (median / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lost_reply` | sbatch_lose_reply=1 (job accepted, reply lost) | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 3.4 / 3.6 | 5.2 / 5.8 |
| `ambiguous_error_queued` | sbatch_ambiguous_error=1, job queued | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 3.4 / 3.4 | 5.7 / 5.9 |
| `ambiguous_error_not_queued` | sbatch_ambiguous_error=1, nothing queued | 5/5 | 5 / 5 attempts | 0 | 0 | 0 (0) | 5 (5) | 0 | 26.6 / 27.0 | — |
| `crash_before_inflight` | controller SIGKILL before the in-flight commit | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 0.5 / 0.5 | 7.0 / 7.2 |
| `crash_after_inflight` | controller SIGKILL after the in-flight commit, before sbatch | 5/5 | 5 / 5 attempts | 0 | 0 | 0 (0) | 5 (5) | 0 | 27.0 / 27.2 | — |
| `crash_after_sbatch` | controller SIGKILL after sbatch accepted, reply unread | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 3.6 / 3.9 | 6.1 / 6.3 |
| `crash_before_identity` | controller SIGKILL before the job identity is committed | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 3.8 / 3.9 | 5.9 / 6.3 |
| `accounting_gap` | accounting_lag_s=10 + min_job_age_s=0.2 | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 0.2 / 0.2 | 11.9 / 13.1 |
| `squeue_timeouts` | squeue transport timeouts for 12 s | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 4.4 / 9.3 | 4.4 / 9.3 |
| `auth_required_window` | auth_required for 6 s from submit | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 119.6 / 121.1 | 119.6 / 121.1 |
| `duplicate_same_token` | lost reply + external resubmission (same token) | 5/5 | 5 / 5 attempts | 5 (+5 external) | 0 | 0 (0) | 5 (5) | 0 | 3.3 / 3.8 | — |
| `cancel_genuine` | run_cancel on a running job (control) | 5/5 | 5 / 5 attempts | 5 | 0 | 5 (0) | 0 (0) | 0 | 5.6 / 6.0 | — |
| `cancel_foreign_owner` | run_cancel after the job id names another user's job | 5/5 | 5 / 5 attempts | 5 | 0 | 0 (0) | 0 (0) | 0 | 0.2 / 0.2 | — |
| `kill9_random_workflow` | kill -9 at a random moment of a 2x2 workflow | 5/5 | 20 / 20 attempts | 20 | 0 | 0 (0) | 0 (0) | 0 | 6.5 / 13.1 | 6.5 / 13.1 |

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
| `lost_reply` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.35 | 5.22 |  |
| `lost_reply` | 2 | yes | terminal/COMPLETED/valid job 1000 | 3.34 | 5.01 |  |
| `lost_reply` | 3 | yes | terminal/COMPLETED/valid job 1000 | 3.55 | 5.66 |  |
| `lost_reply` | 4 | yes | terminal/COMPLETED/valid job 1000 | 3.34 | 5.02 |  |
| `lost_reply` | 5 | yes | terminal/COMPLETED/valid job 1000 | 3.56 | 5.85 |  |
| `ambiguous_error_queued` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.35 | 5.03 |  |
| `ambiguous_error_queued` | 2 | yes | terminal/COMPLETED/valid job 1000 | 3.37 | 5.67 |  |
| `ambiguous_error_queued` | 3 | yes | terminal/COMPLETED/valid job 1000 | 3.36 | 5.66 |  |
| `ambiguous_error_queued` | 4 | yes | terminal/COMPLETED/valid job 1000 | 3.32 | 5.0 |  |
| `ambiguous_error_queued` | 5 | yes | terminal/COMPLETED/valid job 1000 | 3.35 | 5.86 |  |
| `ambiguous_error_not_queued` | 1 | yes | submission_uncertain (op unresolved) | 26.46 | — |  |
| `ambiguous_error_not_queued` | 2 | yes | submission_uncertain (op unresolved) | 26.58 | — |  |
| `ambiguous_error_not_queued` | 3 | yes | submission_uncertain (op unresolved) | 26.59 | — |  |
| `ambiguous_error_not_queued` | 4 | yes | submission_uncertain (op unresolved) | 26.37 | — |  |
| `ambiguous_error_not_queued` | 5 | yes | submission_uncertain (op unresolved) | 26.98 | — |  |
| `crash_before_inflight` | 1 | yes | terminal/COMPLETED/valid job 1000 | 0.51 | 7.01 | phase at crash: staging |
| `crash_before_inflight` | 2 | yes | terminal/COMPLETED/valid job 1000 | 0.49 | 6.55 | phase at crash: staging |
| `crash_before_inflight` | 3 | yes | terminal/COMPLETED/valid job 1000 | 0.48 | 6.77 | phase at crash: staging |
| `crash_before_inflight` | 4 | yes | terminal/COMPLETED/valid job 1000 | 0.5 | 7.18 | phase at crash: staging |
| `crash_before_inflight` | 5 | yes | terminal/COMPLETED/valid job 1000 | 0.5 | 7.16 | phase at crash: staging |
| `crash_after_inflight` | 1 | yes | submission_uncertain (op unresolved) | 26.89 | — | phase at crash: submission_in_flight |
| `crash_after_inflight` | 2 | yes | submission_uncertain (op unresolved) | 27.04 | — | phase at crash: submission_in_flight |
| `crash_after_inflight` | 3 | yes | submission_uncertain (op unresolved) | 26.52 | — | phase at crash: submission_in_flight |
| `crash_after_inflight` | 4 | yes | submission_uncertain (op unresolved) | 27.02 | — | phase at crash: submission_in_flight |
| `crash_after_inflight` | 5 | yes | submission_uncertain (op unresolved) | 27.17 | — | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.85 | 6.13 | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 2 | yes | terminal/COMPLETED/valid job 1000 | 3.44 | 5.74 | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 3 | yes | terminal/COMPLETED/valid job 1000 | 3.63 | 6.35 | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 4 | yes | terminal/COMPLETED/valid job 1000 | 3.84 | 5.71 | phase at crash: submission_in_flight |
| `crash_after_sbatch` | 5 | yes | terminal/COMPLETED/valid job 1000 | 3.43 | 6.1 | phase at crash: submission_in_flight |
| `crash_before_identity` | 1 | yes | terminal/COMPLETED/valid job 1000 | 3.83 | 5.93 | phase at crash: submission_in_flight |
| `crash_before_identity` | 2 | yes | terminal/COMPLETED/valid job 1000 | 3.43 | 6.35 | phase at crash: submission_in_flight |
| `crash_before_identity` | 3 | yes | terminal/COMPLETED/valid job 1000 | 3.85 | 5.53 | phase at crash: submission_in_flight |
| `crash_before_identity` | 4 | yes | terminal/COMPLETED/valid job 1000 | 3.85 | 6.34 | phase at crash: submission_in_flight |
| `crash_before_identity` | 5 | yes | terminal/COMPLETED/valid job 1000 | 3.4 | 5.49 | phase at crash: submission_in_flight |
| `accounting_gap` | 1 | yes | terminal/COMPLETED/valid job 1000 | 0.2 | 13.12 | observation_gap flagged: True; terminal state from: sacct |
| `accounting_gap` | 2 | yes | terminal/COMPLETED/valid job 1000 | 0.21 | 11.47 | observation_gap flagged: True; terminal state from: sacct |
| `accounting_gap` | 3 | yes | terminal/COMPLETED/valid job 1000 | 0.21 | 11.72 | observation_gap flagged: True; terminal state from: sacct |
| `accounting_gap` | 4 | yes | terminal/COMPLETED/valid job 1000 | 0.21 | 11.86 | observation_gap flagged: True; terminal state from: sacct |
| `accounting_gap` | 5 | yes | terminal/COMPLETED/valid job 1000 | 0.21 | 12.3 | observation_gap flagged: True; terminal state from: sacct |
| `squeue_timeouts` | 1 | yes | terminal/COMPLETED/valid job 1000 | 4.8 | 4.8 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `squeue_timeouts` | 2 | yes | terminal/COMPLETED/valid job 1000 | 4.4 | 4.4 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `squeue_timeouts` | 3 | yes | terminal/COMPLETED/valid job 1000 | 3.98 | 3.98 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `squeue_timeouts` | 4 | yes | terminal/COMPLETED/valid job 1000 | 9.32 | 9.32 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `squeue_timeouts` | 5 | yes | terminal/COMPLETED/valid job 1000 | 4.38 | 4.38 | ledger states during the outage: ['RUNNING']; times measured from recovery |
| `auth_required_window` | 1 | yes | terminal/COMPLETED/valid job 1000 | 121.08 | 121.08 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `auth_required_window` | 2 | yes | terminal/COMPLETED/valid job 1000 | 116.36 | 116.36 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `auth_required_window` | 3 | yes | terminal/COMPLETED/valid job 1000 | 119.63 | 119.63 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `auth_required_window` | 4 | yes | terminal/COMPLETED/valid job 1000 | 116.91 | 116.91 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `auth_required_window` | 5 | yes | terminal/COMPLETED/valid job 1000 | 120.57 | 120.57 | blocked with remediation after 0.2s; times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s) |
| `duplicate_same_token` | 1 | yes | duplicate_detected (duplicate_detected) | 3.75 | — |  |
| `duplicate_same_token` | 2 | yes | duplicate_detected (duplicate_detected) | 3.36 | — |  |
| `duplicate_same_token` | 3 | yes | duplicate_detected (duplicate_detected) | 3.33 | — |  |
| `duplicate_same_token` | 4 | yes | duplicate_detected (duplicate_detected) | 3.33 | — |  |
| `duplicate_same_token` | 5 | yes | duplicate_detected (duplicate_detected) | 3.31 | — |  |
| `cancel_genuine` | 1 | yes | cancel confirmed, CANCELLED | 5.85 | — |  |
| `cancel_genuine` | 2 | yes | cancel confirmed, CANCELLED | 5.2 | — |  |
| `cancel_genuine` | 3 | yes | cancel confirmed, CANCELLED | 6.03 | — |  |
| `cancel_genuine` | 4 | yes | cancel confirmed, CANCELLED | 5.62 | — |  |
| `cancel_genuine` | 5 | yes | cancel confirmed, CANCELLED | 5.4 | — |  |
| `cancel_foreign_owner` | 1 | yes | cancel refused | 0.21 | — |  |
| `cancel_foreign_owner` | 2 | yes | cancel refused | 0.22 | — |  |
| `cancel_foreign_owner` | 3 | yes | cancel refused | 0.21 | — |  |
| `cancel_foreign_owner` | 4 | yes | cancel refused | 0.21 | — |  |
| `cancel_foreign_owner` | 5 | yes | cancel refused | 0.22 | — |  |
| `kill9_random_workflow` | 1 | yes | workflow completed {'succeeded': 4} | 0.81 | 0.81 | killed 9.6s after submit; attempt phases then: {'observing': 2, 'terminal': 2} |
| `kill9_random_workflow` | 2 | yes | workflow completed {'succeeded': 4} | 6.94 | 6.94 | killed 6.3s after submit; attempt phases then: {'observing': 2} |
| `kill9_random_workflow` | 3 | yes | workflow completed {'succeeded': 4} | 0.82 | 0.82 | killed 11.2s after submit; attempt phases then: {'observing': 2, 'terminal': 2} |
| `kill9_random_workflow` | 4 | yes | workflow completed {'succeeded': 4} | 13.12 | 13.12 | killed 1.5s after submit; attempt phases then: {'observing': 2} |
| `kill9_random_workflow` | 5 | yes | workflow completed {'succeeded': 4} | 6.5 | 6.5 | killed 7.8s after submit; attempt phases then: {'observing': 2, 'terminal': 2} |

## Limits

The simulator reproduces Slurm's command-line behaviour and output formats, not its internals: real slurmctld/slurmdbd timing, multi-cluster federation, SSH ControlMaster failures and site-specific wrappers are not exercised. Timings depend on the controller's 5 s observation tick and this machine's load (repetitions run in parallel). See README.md.
