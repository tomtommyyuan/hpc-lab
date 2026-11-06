# CIFAR-10: mixup vs ERM

Table 2 of the workshop paper (draft).

- **Generated:** 2026-09-27T12:00:00Z by hpclab 0.1.0 (`hpclab.evidence/v1`)
- **Experiment:** cifar-mixup (`exp_0001`)
- **Question:** Does mixup improve val accuracy?
- **Treatment (may differ):** method, lr, wd
- **Invariants (must match):** split, evaluator, dataset
- **Baseline:** method=erm

> **Coverage: complete.** All 4 planned runs are included.

## Results

Each cell is mean ± sample standard deviation over seeds, followed by (included/expected runs). ↑ higher is better, ↓ lower is better; — means undefined (a standard deviation needs at least 2 seeds).

| Group | acc ↑ |
| --- | --- |
| erm (baseline) | 0.9350 ± 0.0071 (2/2) |
| mixup | 0.9550 ± 0.0071 (2/2) |

## Missing, failed and excluded runs

Planned runs: 4; included 4, missing 0, failed 0, excluded 0. Nothing is dropped from a denominator silently: every expected seed is counted as included, missing, failed or excluded.

No gaps: every planned run contributed a value to every reported metric.

## Per-seed values

### acc ↑

| Group | seed 1 | seed 2 |
| --- | --- | --- |
| erm | 0.9300 | 0.9400 |
| mixup | 0.9500 | 0.9600 |

## Summary statistics

| Group | Metric | n | Mean | Std | SEM | 95% CI | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erm | acc | 2/2 | 0.9350 | 0.0071 | 0.0050 | [0.8715, 0.9985] | 0.9300 | 0.9400 |
| mixup | acc | 2/2 | 0.9550 | 0.0071 | 0.0050 | [0.8915, 1.0185] | 0.9500 | 0.9600 |

## Differences vs baseline (descriptive)

Group − baseline, in the metric's units; read the sign together with the metric's direction. Paired rows use only seeds present in both groups; unpaired rows compare means (Welch). The 95% interval describes the uncertainty of the estimate. These are descriptive summaries, not hypothesis tests, and they do not by themselves establish that one method is better.

No differences were computed for this bundle.

## Incomparable results

None flagged by the comparison contract checks.

## Definitions

- **n / expected:** seeds with an included value / seeds planned for the group.
- **Mean:** arithmetic mean of the included per-seed values.
- **Std:** sample standard deviation with denominator n − 1; undefined (—) for n < 2, never reported as 0.
- **SEM:** Std / √n.
- **95% CI:** mean ± t(q=0.975, df=n − 1) × SEM (two-sided Student-t interval).
- **Paired difference:** d_s = group_s − baseline_s for each seed s present in both groups; summarized like a group (n = number of pairs). Seeds present in only one group are not paired.
- **Unpaired difference:** mean(group) − mean(baseline); SE = √(s²_g/n_g + s²_b/n_b); Welch–Satterthwaite df; 95% t interval.
- **Missing:** planned but no result. **Failed:** the run failed or its value was unusable (e.g. non-finite). **Excluded:** removed explicitly, with the stated reason.

## Provenance

Each value traces to evaluation → artifact digest → attempt → scheduler job → run (spec hash) → code.

| Group | Seed | Metric | Value | Evaluation | Artifact digest | Attempt | Job | Run | Code | Environment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erm | 1 | acc | 0.93 | `ev_d3dfde57ec` | `sha256:9e8229199bc6` | `att_9e9e368fbd` | `haic:4101` | `run_e9df61afbd` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 2 | acc | 0.9400000000000001 | `ev_bbebbff41c` | `sha256:9fbac16708fc` | `att_37cd3b2214` | `haic:4102` | `run_031a5063d9` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 1 | acc | 0.9500000000000001 | `ev_0ef4d2f1fa` | `sha256:d813c7608d08` | `att_50a02b860b` | `haic:4121` | `run_85d9eaad54` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 2 | acc | 0.9600000000000001 | `ev_61396a8174` | `sha256:eea9d3839893` | `att_bfae0b296e` | `haic:4122` | `run_5e0f4cf352` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |

## Limitations

- Record consistency is not proof of scientific validity: this bundle shows that each number traces to recorded code, configuration, jobs and artifacts, not that the experimental design, data or evaluation code are correct.
