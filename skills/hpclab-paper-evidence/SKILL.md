---
name: hpclab-paper-evidence
description: Produce paper-ready, traceable result tables and results paragraphs from HPC Lab experiments — collect evaluations, compare with the full planned denominator (missing seeds stay visible), export LaTeX/CSV/Markdown evidence bundles, and verify a paper's numbers against the bundle before submission. Use when the user wants a results table, mean ± std over seeds, an Experiments section draft, or to check that numbers in a .tex file are current.
---

# From runs to a paper table you can defend

Every number you write must come from an **evidence bundle**, never from reading logs or doing
arithmetic in the chat. The runtime computes; you explain what it computed and what it could not.

## 0. Find the experiment

In a new session, the experiments are listed by resource `hpclab://experiments` (newest first, with
`experiment_id`). A manuscript that already contains an hpclab table names its bundle in every tag:
`\hpcval{<bundle_id>:<cell>}{...}`.

## 1. Collect

Runs are evaluated automatically when they finish with valid outputs.
`evaluation_collect(run_set_ref=<experiment_id>, idempotency_key=...)` makes sure every finished run has
an evaluation and lists the runs it skipped with reasons (`already_evaluated` is fine; anything else is
part of the story — mention it).

## 2. Compare against the plan, not against what happened to finish

`results_compare(planned_run_set_ref=<experiment_id>, metric_names=["acc"], expected_seeds=[1, 2, 3])`

* `runs` lists every planned main run as included / missing / failed / excluded and `counts` sums
  them. The denominator is the plan. Never drop a failed seed to "clean up" the table.
* Exclusions need a concrete reason (`exclusions=[{"run_id": ..., "reason": ...}]`) and are disclosed;
  exclusions decided *after* seeing the metric are flagged as such.
* `incomparable` (different split/evaluator/dataset across cells) and `selection_leak` are errors in the
  experiment, not formatting problems — stop and tell the user.
* `integrity` discloses post-hoc seeds, mixed GPU types or code versions, and multiple valid attempts.
  Carry these into the paper's footnotes or appendix.
* A failed or missing seed can often still be recovered: hand it to **hpclab-run-doctor** first if the
  user wants the complete table.

## 3. Export

`evidence_export(comparison_ref=<comparison_id>, purpose="Table 2, NeurIPS draft", idempotency_key=...)`
(optionally `formats=["latex", "md", "csv"]`, `table_spec={"scale": 100, "decimals": 1}`).

* `table.tex` (resource `hpclab://evidence/{bundle_id}/table.tex`) uses `booktabs`; cells are
  `\hpcval{<bundle>:<cell>}{73.4}`, which renders as the plain number (preamble included) and lets
  `evidence_verify` / `hpclab paper check` find stale numbers later. `hpclab paper strip` removes the
  macros for arXiv.
* A dagger footnote marks cells with n < expected and names the missing seeds and why.
* Nothing is bolded unless the comparison contract declares a bold rule; no significance stars.
* Paste `table.tex` unchanged. Never hand-edit a tagged number.

## 4. Write the results paragraph

* State n per cell ("mean ± sample std over 3 seeds; one mixup seed timed out and is reported as
  missing"). Define ± once.
* Differences vs baseline are **descriptive** (mean difference and its interval). Do not write
  "significant", "p <", or "outperforms" unless the user ran a pre-specified test outside HPC Lab.
* Every number in your prose must appear in the bundle; cite the cell if the user wants traceability
  (resource `hpclab://trace/{bundle_id}:{cell_id}` resolves value → evaluation → artifact digest →
  attempt → Slurm job → run spec → code commit → environment). The verifier checks tagged numbers;
  numbers you retype in prose are only as good as your copying — tag them or quote them exactly.

## 5. Before submission (and after every rerun)

`evidence_verify(bundle_ref=<bundle_id>, tex_paths=["paper/main.tex"])` (or `tex_content=...`;
CLI: `hpclab paper check paper/*.tex`) reports `STALE_TEX_NUMBER` (newer results differ: re-compare,
re-export, replace the table), `VALUE_MISMATCH` (a hand-edited number), `N_BELOW_EXPECTED_UNFOOTNOTED`,
`MIXED_CODE_DIGEST`, … Treat any `error` finding as a blocker and explain it.
Files pulled in with `\input`/`\include`/`\subfile`/`\import` are scanned too; `files_scanned` lists
everything that was read. `NO_TAGS_FOUND` (error, "manuscript NOT checked": no `\hpcval` tag in any scanned
file) and `UNRESOLVED_INPUT` (warning: an included file could not be read, so its numbers were not checked)
are not a pass — fix the paths and verify again; so is "INCOMPLETE". CLI: `hpclab paper check paper/main.tex`
(add `--bundle <bundle_id>` when `main.tex` has no tag itself and the tables come in through `\input`). The
verifier checks that records are consistent; it does not certify that the science is right — say that too.
