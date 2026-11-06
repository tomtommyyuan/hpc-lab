# CIFAR-10: mixup vs ERM

Table 2 of the workshop paper (draft).

- **Generated:** 2026-09-27T12:00:00Z by hpclab 0.1.0 (`hpclab.evidence/v1`)
- **Experiment:** cifar-mixup (`exp_0001`)
- **Question:** Does mixup improve val accuracy?
- **Treatment (may differ):** method, lr, wd
- **Invariants (must match):** split, evaluator, dataset
- **Baseline:** method=erm

> **Coverage: incomplete.** 6 of 9 planned runs included (1 missing, 1 failed, 1 excluded). Every gap is listed under *Missing, failed and excluded runs*.
>
> **Warning:** 1 incomparability issue(s) flagged; see *Incomparable results* before reading the tables.

## Results

Each cell is mean ± sample standard deviation over seeds, followed by (included/expected runs). ↑ higher is better, ↓ lower is better; — means undefined (a standard deviation needs at least 2 seeds).

| Group | acc (fraction) ↑ | loss (nats) ↓ |
| --- | --- | --- |
| erm (baseline) | 0.9117 ± 0.0065 (3/3) | 0.3060 ± 0.0142 (3/3) |
| mixup | 0.9190 ± 0.0028 (2/3) | 0.2925 ± 0.0064 (2/3) |
| lr=1e-3 & wd\_0.1 (50%) | 0.8740 ± — (1/3) | 0.4100 ± — (1/3) |

## Missing, failed and excluded runs

Planned runs: 9; included 6, missing 1, failed 1, excluded 1. Nothing is dropped from a denominator silently: every expected seed is counted as included, missing, failed or excluded.

| Group | Metric | Included | Missing seeds | Failed seeds | Excluded seeds (reason) |
| --- | --- | --- | --- | --- | --- |
| mixup | acc | 2/3 | — | 3 | — |
| mixup | loss | 2/3 | — | 3 | — |
| lr=1e-3 & wd\_0.1 (50%) | acc | 1/3 | 2 | — | 3 (diverged: loss=nan at step 400) |
| lr=1e-3 & wd\_0.1 (50%) | loss | 1/3 | 2 | — | 3 (diverged: loss=nan at step 400) |

Notes:
- mixup seed 3: OUT\_OF\_MEMORY on gpu-07 (attempt att\_oom)

## Per-seed values

### acc (fraction) ↑

| Group | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- |
| erm | 0.9120 | 0.9050 | 0.9180 |
| mixup | 0.9210 | 0.9170 | failed |
| lr=1e-3 & wd\_0.1 (50%) | 0.8740 | missing | excluded |

### loss (nats) ↓

| Group | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- |
| erm | 0.3010 | 0.3220 | 0.2950 |
| mixup | 0.2880 | 0.2970 | failed |
| lr=1e-3 & wd\_0.1 (50%) | 0.4100 | missing | excluded |

## Summary statistics

| Group | Metric | n | Mean | Std | SEM | 95% CI | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erm | acc | 3/3 | 0.9117 | 0.0065 | 0.0038 | [0.8955, 0.9278] | 0.9050 | 0.9180 |
| erm | loss | 3/3 | 0.3060 | 0.0142 | 0.0082 | [0.2708, 0.3412] | 0.2950 | 0.3220 |
| mixup | acc | 2/3 | 0.9190 | 0.0028 | 0.0020 | [0.8936, 0.9444] | 0.9170 | 0.9210 |
| mixup | loss | 2/3 | 0.2925 | 0.0064 | 0.0045 | [0.2353, 0.3497] | 0.2880 | 0.2970 |
| lr=1e-3 & wd\_0.1 (50%) | acc | 1/3 | 0.8740 | — | — | — | 0.8740 | 0.8740 |
| lr=1e-3 & wd\_0.1 (50%) | loss | 1/3 | 0.4100 | — | — | — | 0.4100 | 0.4100 |

## Differences vs baseline (descriptive)

Group − baseline, in the metric's units; read the sign together with the metric's direction. Paired rows use only seeds present in both groups; unpaired rows compare means (Welch). The 95% interval describes the uncertainty of the estimate. These are descriptive summaries, not hypothesis tests, and they do not by themselves establish that one method is better.

| Group | Baseline | Metric | Kind | n | Mean diff | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| mixup | erm | acc ↑ | paired | 2 pairs | +0.0105 | [-0.0086, 0.0296] |
| mixup | erm | loss ↓ | paired | 2 pairs | -0.0190 | [-0.0952, 0.0572] |
| lr=1e-3 & wd\_0.1 (50%) | erm | acc ↑ | unpaired | 1 vs 3 | -0.0377 | — |

## Incomparable results

**Warning:** the comparison contract flagged the following; numbers affected by them should not be compared directly.

- **environment differs**: env\_torch=\["2.7.1", "2.8.0+cu128"\]; group=mixup

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
| erm | 1 | acc | 0.912 | `ev_d3dfde57ec` | `sha256:9e8229199bc6` | `att_9e9e368fbd` | `haic:4101` | `run_e9df61afbd` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 2 | acc | 0.905 | `ev_bbebbff41c` | `sha256:9fbac16708fc` | `att_37cd3b2214` | `haic:4102` | `run_031a5063d9` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 3 | acc | 0.918 | `ev_21c0a8afed` | `sha256:959c5f4050e1` | `att_465cd9030c` | `haic:4103` | `run_caaa23ac1b` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 1 | loss | 0.301 | `ev_ed77605c05` | `sha256:9e8229199bc6` | `att_9e9e368fbd` | `haic:4101` | `run_e9df61afbd` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 2 | loss | 0.322 | `ev_199a966351` | `sha256:9fbac16708fc` | `att_37cd3b2214` | `haic:4102` | `run_031a5063d9` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| erm | 3 | loss | 0.295 | `ev_49edcaeb35` | `sha256:959c5f4050e1` | `att_465cd9030c` | `haic:4103` | `run_caaa23ac1b` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 1 | acc | 0.921 | `ev_0ef4d2f1fa` | `sha256:d813c7608d08` | `att_50a02b860b` | `haic:4121` | `run_85d9eaad54` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 2 | acc | 0.917 | `ev_61396a8174` | `sha256:eea9d3839893` | `att_bfae0b296e` | `haic:4122` | `run_5e0f4cf352` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 1 | loss | 0.288 | `ev_5f022e2694` | `sha256:d813c7608d08` | `att_50a02b860b` | `haic:4121` | `run_85d9eaad54` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| mixup | 2 | loss | 0.297 | `ev_eaaea5bd8b` | `sha256:eea9d3839893` | `att_bfae0b296e` | `haic:4122` | `run_5e0f4cf352` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| lr=1e-3 & wd\_0.1 (50%) | 1 | acc | 0.874 | `ev_98c8d58766` | `sha256:fc57ef5a27ba` | `att_bab427c05d` | `haic:4111` | `run_21633cab0f` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| lr=1e-3 & wd\_0.1 (50%) | 1 | loss | 0.41 | `ev_b9e3c61e63` | `sha256:fc57ef5a27ba` | `att_bab427c05d` | `haic:4111` | `run_21633cab0f` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |
| lr=1e-3 & wd\_0.1 (50%) | 3 | loss | — | `ev_210ac0d940` | `sha256:0479d2575180` | `att_374fbd4da7` | `haic:4113` | `run_0656a63e41` | `0123456789ab` | NVIDIA A100-SXM4-80GB; torch 2.8.0+cu128 |

## Limitations

- Record consistency is not proof of scientific validity: this bundle shows that each number traces to recorded code, configuration, jobs and artifacts, not that the experimental design, data or evaluation code are correct.
- Three seeds per method; intervals are wide.
