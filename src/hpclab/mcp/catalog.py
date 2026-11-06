"""MCP tool catalog: names, LLM-facing descriptions, annotations, parameter docs.

Input schemas are derived from the service function signatures (via the registry) and
enriched here: ``ToolSpec.params`` documents every top-level parameter and ``DEF_DOCS``
documents the fields of nested objects (keyed by the pydantic model name that appears in the
schema's ``$defs``). Output schemas are ``Envelope[<Data model>]``. Keeping the prose here (not
in service docstrings) lets us tune what the model reads without touching service code; the
tests in ``tests/mcp`` fail when a parameter or nested field is left undocumented.

Annotations are UI hints, never authorization (design §7): every mutation is still checked
by the runtime (idempotency, execution claim, identity, budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    name: str
    method: str
    title: str
    description: str
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = True
    open_world: bool = True
    params: dict[str, str] = field(default_factory=dict)


_IDEM = (
    "Your key for this request. Retrying with the same key and arguments (e.g. after a lost reply) returns "
    "the original result; reusing it with other arguments is rejected. Use a new key for a new request."
)
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="cluster_inspect",
        method="cluster_inspect",
        title="Inspect a Slurm cluster",
        description=(
            "Check a registered cluster before using it. One entry per check with status "
            "ok|warn|fail|unavailable|unsupported|skipped, findings and remediation: auth (SSH/MFA state and "
            "the exact command the user must run), identity, scheduler (Slurm version, accounting, PrivateData), "
            "storage/quota, runtime tools on the login node, my queue (hpclab-managed vs other jobs), GPU "
            "capacity per partition and type, limits, drained or locally suspect nodes, profile facts. "
            "Read-only; only bounded metadata commands run on the login node. Unknown data is reported as "
            "unavailable, never as zero. candidate_shapes adds low-confidence `sbatch --test-only` start "
            "estimates for up to 3 placements."
        ),
        read_only=True,
        params={
            "cluster_id": "Registered cluster id, e.g. 'sim' or 'hai' (the user registers clusters with `hpclab cluster add`).",
            "checks": "Subset of checks to run. Default: all of auth, identity, scheduler, storage, runtime, queue, capacity, limits, nodes, profile_facts.",
            "candidate_shapes": 'Up to 3 placements to compare expected start times for, e.g. [{"partition": "gpu", "gpu_type": "a100", "gpus": 1, "time": "04:00:00"}].',
        },
    ),
    ToolSpec(
        name="experiment_prepare",
        method="experiment_prepare",
        title="Freeze an experiment plan",
        description=(
            "Freeze an ExperimentSpec into an immutable plan: snapshots the project code (content-addressed, "
            "git-aware, secrets excluded), creates one Run per matrix cell (plus the smoke run) with its "
            "execution plan, and validates everything. Nothing is submitted. Returns experiment_id, plan_id, "
            "status ready|preparing|invalid, plan_hash (only when ready), runs[] (run_id, label, "
            "exec_plan_hash), GPU-hours vs budget, resource advice from past runs and validation findings: "
            "severity 'block' must be fixed (e.g. torch CUDA build newer than the driver, editable install "
            "bypassing the snapshot, time above the partition max); 'unchecked' means it could not be "
            "verified. status 'preparing' (large project) returns operation_ref to poll. Give spec_path (YAML) "
            "or spec. Each new idempotency_key prepares a NEW experiment; to add seeds or cells to an existing "
            "one use extends + add_seeds/add_matrix (finished cells are reused, not duplicated); to change one "
            "run use derive_from. Next step: workflow_submit(plan_id, plan_hash) for the whole matrix."
        ),
        idempotent=True,
        params={
            "idempotency_key": _IDEM,
            "spec_path": "Path to an ExperimentSpec YAML, e.g. 'hpclab.yaml'. Relative paths resolve against the agent's working directory; the spec's code.project_root resolves against the YAML's directory.",
            "spec": "ExperimentSpec as an object (alternative to spec_path): name, cluster, entrypoint.argv, matrix, seeds, resources, outputs, smoke, budget, evaluation, comparison. code.project_root defaults to the working directory.",
            "cluster_id": "Override spec.cluster (a registered cluster id).",
            "experiment": "Override spec.name (the experiment name).",
            "matrix": 'Override spec.matrix, e.g. {"method": ["baseline", "mixup"]}. Seeds stay in spec.seeds.',
            "extends": "experiment_id to extend with new cells (use with add_seeds and/or add_matrix). Only the new cells get runs.",
            "add_seeds": "Seeds to add to every cell when extending, e.g. [4, 5].",
            "add_matrix": 'Extra values per matrix axis when extending, e.g. {"method": ["cutmix"]}.',
            "derive_from": 'Derive a new run from an existing one with changed params/resources/argv, e.g. {"run_id": "run_...", "overrides": {"params": {"lr": 0.05}}}. The original run is untouched.',
            "overrides": "[{check, reason}] to downgrade a blocking finding to a warning. Only with the user's explicit agreement; recorded in the ledger.",
            "validate_remote": "Run the bounded remote checks (venv, torch CUDA build vs driver, sbatch --test-only). Default true; false skips them (those findings become 'unchecked').",
        },
    ),
    ToolSpec(
        name="run_submit",
        method="run_submit",
        title="Submit one run",
        description=(
            "Submit ONE prepared run to Slurm (for a whole matrix use workflow_submit). Returns a receipt at "
            "once (attempt_id, operation_ref, poll_after_ms) — the receipt is NOT proof that Slurm accepted "
            "the job; check run_inspect(run_id) or the operation after poll_after_ms. A run has at most one "
            "active execution: a second submit (even with another key) is refused while an attempt is active "
            "or its submission is uncertain. The smoke gate and budget are enforced. Spends GPU-hours: "
            "confirm with the user first."
        ),
        idempotent=True,
        params={
            "run_id": "Run to execute (runs[].run_id from experiment_prepare, or run_id from run_recover).",
            "execution_plan_hash": "The exec_plan_hash to execute: runs[].exec_plan_hash from experiment_prepare, or exec_plan_hash from run_recover. Each execution plan can be used once.",
            "idempotency_key": _IDEM,
            "force_without_gate": "Submit a main run before its smoke gate passed. Requires reason; avoid unless the user insists.",
            "reason": "Why the smoke gate is bypassed (recorded in the ledger). Required with force_without_gate.",
        },
    ),
    ToolSpec(
        name="workflow_submit",
        method="workflow_submit",
        title="Dispatch an experiment matrix",
        description=(
            "Dispatch every run of a prepared plan, at most max_parallel active at once; the controller keeps "
            "dispatching after the chat ends and survives restarts. The smoke run goes first; main cells start "
            "only after it COMPLETED with valid outputs (gate). A circuit breaker pauses dispatch after 2 cells "
            "fail with the same environment error. One workflow per plan: calling it again for the same plan "
            "returns admission_denied with the existing workflow_id. Returns workflow_id, state, gate and "
            "cells; watch progress with runs_list(experiment_id, view='grid'). To resume a paused (circuit breaker) or "
            "gate_failed workflow after the fix, pass workflow_id + resume_dispatch=true + expected_version (the current version from "
            "hpclab://workflows/<workflow_id>). Spends GPU-hours: confirm with the user first."
        ),
        idempotent=True,
        params={
            "idempotency_key": _IDEM,
            "plan_id": "Plan to dispatch: plan_id from experiment_prepare, or the recovery plan_id from run_recover(selector=...).",
            "prepared_experiment_id": "Alternative to plan_id: dispatch the experiment's latest ready plan.",
            "plan_hash": "Optional guard: the plan_hash returned by experiment_prepare. Dispatch is refused if the plan differs from the one reviewed.",
            "max_parallel": "Max simultaneously active cells, 1-256 (default: spec budget.max_parallel, else site policy).",
            "failure_policy": 'Circuit-breaker settings, e.g. {"circuit_breaker": {"enabled": true, "threshold": 2}}. Default: enabled, threshold 2, environment failures only.',
            "workflow_id": "Existing workflow to resume (only with resume_dispatch=true).",
            "resume_dispatch": "Re-open dispatch of a paused workflow (breaker tripped) or a gate_failed one (the smoke run passed after a recovery), once the cause was fixed. Needs workflow_id and expected_version; takes no other options.",
            "expected_version": "The workflow's current version (from hpclab://workflows/<workflow_id>); required with resume_dispatch. A stale version gets version_conflict with details.current_version.",
        },
    ),
    ToolSpec(
        name="runs_list",
        method="runs_list",
        title="List runs / experiment grid",
        description=(
            "Runs of an experiment or workflow from the ledger (fast; no cluster calls beyond the controller's "
            "shared poll). view='grid' returns a rows × cols matrix of the main runs (default: treatment axes × "
            "seed) with one code per cell: V valid result, R running, Q queued, S submitting, P not dispatched, "
            "E completed, evaluation pending (not a failure; evaluation_collect if it persists), F failed (with "
            "the diagnosis), C! completed but outputs or evaluation invalid, U submission uncertain (do not "
            "resubmit), X cancelled/rejected (incl. cells of a cancelled workflow), ? stale observation. "
            "metric='acc' puts the validated value in V cells. view='list' (default) returns one row per run "
            "incl. the smoke run, with the "
            "latest attempt (attempt_id, job_id, phase, pending reason, progress). The result carries "
            "changes_cursor: pass it back as changed_since to get only runs that changed since."
        ),
        read_only=True,
        open_world=False,
        params={
            "experiment_id": "Experiment to list (experiment_id from experiment_prepare; resource hpclab://experiments lists them, newest first). The grid needs this or workflow_id.",
            "workflow_id": "Workflow to list (alternative to experiment_id).",
            "states": "Only runs in these statuses (list view), e.g. ['failed', 'completed_unverified'].",
            "cursor": "Page cursor: next_cursor from the previous call (same other arguments).",
            "limit": "Page size for the list view (default 20, max 100).",
            "view": "'list' (default): one row per run. 'grid': rows × cols code matrix of the main runs.",
            "rows": "Grid row axes (default: the treatment axes, e.g. ['method']).",
            "cols": "Grid column axes (default: ['seed']). At most 64 columns.",
            "metric": "Metric name to show in V grid cells, e.g. 'acc' (validated evaluations only).",
            "changed_since": "changes_cursor from an earlier call: return only runs with events after it.",
        },
    ),
    ToolSpec(
        name="run_inspect",
        method="run_inspect",
        title="Inspect a run",
        description=(
            "Everything known about one run: spec (argv, resources) and code/commit refs; every attempt with its "
            "Slurm job identity and the three separate state axes (scheduler_state, app_state, result_state); "
            "requeue segments; decoded pending reason; progress/ETA vs remaining walltime; GPU health flags "
            "(advisory); a deterministic diagnosis of failures with evidence lines; environment drift; "
            "artifacts and evaluations; and recommended_next (exact follow-up calls). Each attempt carries the "
            "`version` that run_cancel needs as expected_version. refresh=true asks for a bounded fresh "
            "observation; probe='gpu' samples nvidia-smi inside the running job (rate-limited). "
            "job_ref={cluster_id, job_id} finds the run behind a Slurm job id. Give exactly one of run_id, "
            "attempt_id, job_ref."
        ),
        read_only=True,
        params={
            "run_id": "Run to inspect (shows its latest attempt in focus).",
            "attempt_id": "Inspect the run of this attempt, focused on this attempt.",
            "job_ref": 'Reverse lookup by Slurm job, e.g. {"cluster_id": "hai", "job_id": "123456"}; add submit_time when job ids were reused.',
            "refresh": "Ask the controller for a fresh bounded observation first (returns cached data with stale=true if it times out).",
            "probe": "'gpu' to sample GPU utilization/memory inside the running job's own allocation (rate-limited to 1 per 10 min per attempt).",
        },
    ),
    ToolSpec(
        name="run_logs",
        method="run_logs",
        title="Read run logs",
        description=(
            "Read an attempt's stdout or stderr, bounded (default 32 KiB, max 64 KiB per call). tail_lines=N "
            "returns the last N lines (best for errors); otherwise read forward from cursor and continue with "
            "next_cursor. rotated=true means the file was replaced and reading restarted at 0. A missing file "
            "usually means the job has not started (a note, not an error). Does not wait for new output. For "
            "failures prefer run_inspect first: its diagnosis already quotes the decisive lines."
        ),
        read_only=True,
        params={
            "attempt_id": "Attempt whose logs to read (attempts[].attempt_id from run_inspect, or latest_attempt.attempt_id from runs_list).",
            "stream": "'stdout' (default) or 'stderr'. Python tracebacks and Slurm errors are on stderr.",
            "cursor": "next_cursor from the previous call, to continue reading forward (not with tail_lines).",
            "max_bytes": "Max bytes to return (default 32768, max 65536).",
            "tail_lines": "Return only the last N lines (max 10000), e.g. 80. Do not combine with cursor.",
        },
    ),
    ToolSpec(
        name="run_cancel",
        method="run_cancel",
        title="Cancel an attempt",
        description=(
            "Cancel one attempt after re-verifying its live Slurm identity (job id, owner, submit token, submit "
            "time) right before scancel — any mismatch cancels nothing (cancel_state refused). Not yet "
            "submitted: cancelled immediately. Submission in flight/uncertain: the intent is recorded and applied "
            "after reconciliation. cancel_state goes 'requested' (intent recorded) -> 'sent' (identity-checked "
            "scancel issued) -> 'confirmed' (a later observation shows the job terminal; a job found already "
            "ended is confirmed with nothing sent). 'failed' means nothing was cancelled: the scheduler "
            "refused the scancel, the job never became visible in squeue/sacct, the adapter cannot cancel, "
            "or an abandon dropped the pending request. "
            "mode='request_checkpoint' sends the declared checkpoint signal instead (only for runs declaring "
            "resilience.stop_behavior=checkpoint_and_exit). Never use raw scancel. Destroys running work: "
            "confirm with the user."
        ),
        destructive=True,
        idempotent=True,
        params={
            "attempt_id": "Attempt to cancel (attempts[].attempt_id from run_inspect).",
            "expected_version": "The attempt's current version (attempts[].version from run_inspect). A stale version gets version_conflict with details.current_version.",
            "idempotency_key": _IDEM,
            "reason": "Why it is cancelled (recorded in the ledger).",
            "mode": "'cancel' (default) or 'request_checkpoint' (ask the job to checkpoint and exit).",
        },
    ),
    ToolSpec(
        name="run_recover",
        method="run_recover",
        title="Plan a recovery",
        description=(
            "Diagnose a stopped attempt and build a recovery execution plan for the SAME run — never submits. "
            "strategy: retry (same resources; excludes nodes implicated by 2+ attempts), resume (from a "
            "validated checkpoint; may first create a checkpoint probe run to submit), adjust_resources (e.g. "
            "more mem after host_oom, more time after timeout; resource_patch or the diagnosis hint). Default: "
            "the strategy the diagnosis suggests. Returns eligible, blocking reasons, diagnosis, the new "
            "resources and exec_plan_hash → submit with run_submit(run_id, execution_plan_hash) after the user "
            "agrees. Active or submission-uncertain attempts are never eligible. Scientific changes (smaller batch "
            "after CUDA OOM, lower lr after NaN) are not recoveries: derive a new run with "
            "experiment_prepare(derive_from=...). selector={workflow_id, cells: 'missing_valid'} plans every "
            "broken cell of a workflow at once → dispatch with workflow_submit(plan_id=<returned plan_id>)."
        ),
        idempotent=True,
        params={
            "idempotency_key": _IDEM,
            "attempt_id": "Stopped attempt to recover (the failed attempt_id from run_inspect or runs_list). Give this or selector.",
            "selector": 'Batch recovery over a workflow, e.g. {"workflow_id": "wf_...", "cells": "missing_valid"}; cells may also be {"diagnosis_in": ["node_fail"]} or {"run_ids": ["run_..."]}.',
            "strategy": "'retry', 'resume' or 'adjust_resources' (default: suggested by the diagnosis).",
            "checkpoint_ref": "Checkpoint artifact (id or hpclab://artifacts/<id>, role checkpoint) to resume from (default: the newest valid one of this run).",
            "resource_patch": 'Resource changes for adjust_resources (or resume), e.g. {"mem": "16G"} or {"time": "12:00:00"}. Validated against partition limits.',
        },
    ),
    ToolSpec(
        name="artifacts_list",
        method="artifacts_list",
        title="List artifacts",
        description=(
            "Declared outputs of an attempt (from the spec's outputs) with role, size, digest and integrity: "
            "mutable (attempt still active), finalized, incomplete, missing, unknown. result_state says whether "
            "all required outputs are finalized (valid). Only declared outputs are listed — this is not a file "
            "browser; directory outputs list their files with include_members=true. Read one with "
            "artifact_read(artifact_ref=<artifact_id>)."
        ),
        read_only=True,
        params={
            "attempt_id": "Attempt whose outputs to list.",
            "kind": "Filter: 'file' or 'dir', or a role: metrics, checkpoint, log, prediction, figure, other.",
            "include_members": "Also list the indexed files inside declared directory outputs (e.g. checkpoint files).",
            "refresh": "Re-index the outputs on the cluster first (bounded stat/listing); useful while the attempt runs.",
            "cursor": "next_cursor from the previous call.",
            "limit": "Page size (default 20, max 100).",
        },
    ),
    ToolSpec(
        name="artifact_read",
        method="artifact_read",
        title="Read an artifact",
        description=(
            "Bounded, integrity-checked read of one artifact. Default view by type: .jsonl and metrics logs → "
            "jsonl_tail (last records, parsed), .json → json (parsed), others → head. A directory artifact "
            "returns its member files instead (read a member by its artifact_id). Refuses paths escaping the "
            "attempt directory; returns 'stale' if the file changed since it was indexed (never silently reads "
            "new content); binary files return metadata plus a download operation. Numbers for papers come "
            "from evaluations/evidence bundles, not from reading artifacts."
        ),
        read_only=True,
        params={
            "artifact_ref": "artifact_id from artifacts_list, or hpclab://artifacts/<id>; append #<member path> for a file inside a directory artifact.",
            "range": "Byte window {offset, length} (only with view 'text' or 'head').",
            "view": "head | tail | text | json | jsonl_tail (default chosen from the file type).",
            "max_bytes": "Max bytes to read (default 32768, max 65536).",
        },
    ),
    ToolSpec(
        name="evaluation_collect",
        method="evaluation_collect",
        title="Collect metrics",
        description=(
            "Extract metrics from digest-verified outputs with a registered evaluator (hpclab_metrics, json_path, "
            "jsonl_record) and archive an Evaluation with provenance (artifact digests, attempt, job, evaluator "
            "version, config hash). Runs are normally evaluated automatically when they finish with valid "
            "outputs, so this mostly confirms coverage: run_set_ref=<experiment_id|plan_id|workflow_id> "
            "evaluates every finished run still lacking an evaluation and lists skipped runs with reasons "
            "(already_evaluated is fine). attempt_id (+ evaluator_id/config) evaluates one attempt, e.g. with a "
            "different split. Idempotent on identical inputs; missing fields or parse errors are reported, "
            "never marked valid. Next: results_compare."
        ),
        idempotent=True,
        params={
            "idempotency_key": _IDEM,
            "attempt_id": "Evaluate this one attempt. Give this or run_set_ref.",
            "run_set_ref": "experiment_id, plan_id or workflow_id: evaluate every finished run that lacks an evaluation.",
            "input_artifact_refs": "Artifact ids to feed the evaluator (default: the spec's evaluation.output).",
            "evaluator_id": "Evaluator: hpclab_metrics, json_path or jsonl_record (default: the spec's evaluation.evaluator_id).",
            "evaluator_version": "Evaluator version (default: the spec's).",
            "config": 'Evaluator config (default: the spec\'s), e.g. {"split": "val"} for hpclab_metrics.',
        },
    ),
    ToolSpec(
        name="results_compare",
        method="results_compare",
        title="Compare results",
        description=(
            "Compare methods over the PLANNED run set: every planned main run appears as included, missing, "
            "failed or excluded (exclusions need reasons and are disclosed) — the denominator is the plan, not "
            "what happened to finish. Per group and metric: n/expected_n, mean, sample std, SEM, t-CI, min/max "
            "and per-seed values; descriptive differences vs the baseline (paired by seed); comparability "
            "problems (different split/evaluator/dataset); selection leaks; and an integrity section (post-hoc "
            "seeds, mixed GPU types or code versions). No significance claims. Archived as an immutable "
            "comparison (comparison_id). Next: evidence_export(comparison_ref=<comparison_id>)."
        ),
        read_only=False,
        idempotent=True,
        open_world=False,
        params={
            "planned_run_set_ref": "experiment_id, plan_id or workflow_id defining the planned denominator (usually the experiment_id).",
            "metric_names": "Metrics to compare, e.g. ['acc', 'loss'] (names from the evaluations / spec.comparison.metrics).",
            "expected_seeds": "Seeds the plan must contain, e.g. [1, 2, 3]; a mismatch with the plan is an error (guards against a wrong run set).",
            "comparison_contract": "Override the spec's comparison contract: {treatment, invariants, baseline, metrics, pair_by_seed, ci_level, selection, attempt_selection, min_n, bold_rule}.",
            "evaluation_refs": "Use exactly these evaluation ids (e.g. from a custom evaluator) instead of the default selection.",
            "exclusions": "[{run_id, reason}] planned runs to exclude from the statistics; each is disclosed with its reason.",
            "idempotency_key": "Optional. " + _IDEM,
        },
    ),
    ToolSpec(
        name="evidence_export",
        method="evidence_export",
        title="Export an evidence bundle",
        description=(
            "Build a paper-ready evidence bundle from a comparison: Markdown summary, LaTeX booktabs table "
            "(cells are \\hpcval{<bundle>:<cell>}{value} so numbers stay verifiable; the preamble makes the macro "
            "print just the value; cells with missing seeds are footnoted), long-format CSV and a JSON manifest "
            "tracing every value to evaluation → artifact digest → attempt → Slurm job → run spec → code commit "
            "→ environment. Files are resources: hpclab://evidence/<bundle_id>/table.tex, summary.md, table.csv, "
            "manifest.json. Incomplete coverage still exports, with warnings. Give comparison_ref, or "
            "planned_run_set_ref (+ optional metric_names) to compare and export in one call. Next: paste table.tex into "
            "the paper, later evidence_verify(bundle_ref, tex_paths) to catch stale numbers."
        ),
        idempotent=True,
        open_world=False,
        params={
            "purpose": "What the bundle is for, shown in the summary, e.g. 'Table 2, NeurIPS draft'.",
            "idempotency_key": _IDEM,
            "comparison_ref": "comparison_id from results_compare, or hpclab://comparisons/<id>.",
            "planned_run_set_ref": "Alternative to comparison_ref: experiment/plan/workflow id to compare and export in one step (needs metric_names).",
            "evaluation_refs": "With planned_run_set_ref: use exactly these evaluation ids.",
            "metric_names": "With planned_run_set_ref: metrics to compare, e.g. ['acc'] (default: the spec's comparison.metrics).",
            "formats": "Subset of md, json, csv, latex (default: all).",
            "table_spec": 'LaTeX table layout, e.g. {"cols": ["acc"], "decimals": 1, "scale": 100, "stat": "mean_std"}.',
        },
    ),
    ToolSpec(
        name="evidence_verify",
        method="evidence_verify",
        title="Verify evidence and paper numbers",
        description=(
            "Check an evidence bundle and a manuscript before submission: recomputes the bundle's aggregates "
            "from the per-seed evaluations and checks every \\hpcval-tagged number in the .tex against the "
            "bundle and the current ledger. Findings (severity error|warn|info): STALE_TEX_NUMBER (newer results "
            "differ → re-compare and re-export), VALUE_MISMATCH (hand-edited number), AGGREGATE_MISMATCH, "
            "N_BELOW_EXPECTED_UNFOOTNOTED, SEED_SELECTION, MIXED_CODE_DIGEST, MIXED_GPU_TYPE, "
            "INVARIANT_MISMATCH, SELECTION_LEAK, ARTIFACT_DIGEST_CHANGED, UNRESOLVED_REF, NO_TAGS_FOUND (no tag: "
            "NOT checked), UNRESOLVED_INPUT (an \\input that could not be read), UNTAGGED_TEX_NUMBER. "
            "Read-only. Verifies record consistency, not scientific correctness. In a new session the "
            "bundle id can be read from the .tex itself: \\hpcval{<bundle_id>:<cell>}{...}."
        ),
        read_only=True,
        open_world=False,
        params={
            "bundle_ref": "bundle_id from evidence_export, or hpclab://evidence/<id>.",
            "tex_paths": "LaTeX files to check, e.g. ['paper/main.tex'] (relative paths resolve against the agent's working directory). Files they \\input/\\include/\\subfile/\\import are scanned too (files_scanned in the result).",
            "tex_content": "LaTeX source as a string (alternative to tex_paths).",
            "recheck_remote": "Also re-hash small output files on the cluster (bounded; network I/O). Default false.",
        },
    ),
    ToolSpec(
        name="workflow_cancel",
        method="workflow_cancel",
        title="Cancel a workflow",
        description=(
            "Stop a matrix: closes dispatch atomically and cancels never-dispatched cells. scope='queued_only' "
            "(default) leaves submitted jobs running; scope='all_managed' also cancels every active cell, each "
            "after a live identity check. Submissions in flight or uncertain are reconciled first, never assumed "
            "cancelled. expected_version must be the workflow's current version (it changes as dispatch "
            "proceeds; read hpclab://workflows/<workflow_id> right before). Returns per-cell cancel states. "
            "Confirm with the user."
        ),
        destructive=True,
        idempotent=True,
        params={
            "workflow_id": "Workflow to cancel (workflow_id from workflow_submit).",
            "expected_version": "The workflow's current version (from hpclab://workflows/<workflow_id>). A stale version gets version_conflict with details.current_version.",
            "scope": "'queued_only' (default): stop dispatching, cancel undispatched cells. 'all_managed': also cancel running/queued jobs.",
            "idempotency_key": _IDEM,
        },
    ),
    ToolSpec(
        name="usage_report",
        method="usage_report",
        title="GPU usage report",
        description=(
            "Allocated GPU-hours (GPUs × wall time, not utilization) from Slurm accounting for me (scope='self') "
            "or a lab account (scope='account') over a window (default: last 7 days, max 93), grouped by "
            "user/partition/gpu_type/state/..., with waste by cause (diagnosis class for hpclab runs), median "
            "queue wait and walltime accuracy. `visibility` says whether other users' jobs were visible: when it "
            "is 'self_only' (PrivateData hides them) there are no lab totals — say so instead of presenting a "
            "partial sum. Long windows may return partial results (coverage incomplete; call again to continue). "
            "Not a leaderboard."
        ),
        read_only=True,
        params={
            "cluster_id": "Registered cluster id.",
            "scope": "'self' (default): my jobs. 'account': all visible jobs of a Slurm account.",
            "account": "Slurm account for scope='account' (default: the overlay's account).",
            "since": "Window start: ISO date/time or relative like '7d', '24h', '30m' (default 7d ago).",
            "until": "Window end: ISO date/time or relative (default now).",
            "group_by": "Grouping dimensions (default ['partition'] for self, ['user'] for account).",
            "include": "Extra sections: now (current queue), waste, queue_wait, walltime_accuracy (default: all but now).",
            "limit": "Max rows (default 20, max 100).",
        },
    ),
]

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}

# Descriptions for fields of nested input objects, keyed by the model name in the schema's $defs.
DEF_DOCS: dict[str, dict[str, str]] = {
    "CandidateShape": {
        "partition": "Partition name.",
        "gpu_type": "GPU type, e.g. 'a100' (default: any).",
        "gpus": "Total GPUs of the job (spread over nodes).",
        "time": "Time limit, Slurm format, e.g. '04:00:00' or '1-00:00:00'.",
        "nodes": "Number of nodes (default 1).",
    },
    "DeriveFrom": {
        "run_id": "Run to derive from.",
        "overrides": "What changes: {params?, resources?, argv_append?}.",
        "recapture_code": "Snapshot the project code again instead of reusing the original run's snapshot.",
    },
    "DeriveOverrides": {
        "params": 'Parameter changes, e.g. {"lr": 0.05} (placeholders in argv are re-rendered).',
        "resources": 'Resource changes, e.g. {"mem": "32G", "time": "08:00:00"}.',
        "argv_append": "Extra arguments appended to the entrypoint argv.",
    },
    "ValidationOverride": {
        "check": "Name of the blocking check to downgrade (validation.findings[].check).",
        "reason": "The user's reason (recorded).",
    },
    "FailurePolicy": {
        "circuit_breaker": "Circuit breaker settings.",
    },
    "CircuitBreakerPolicy": {
        "enabled": "Pause dispatch when cells keep failing the same way (default true).",
        "threshold": "Distinct failing cells with the same fingerprint before pausing (default 2).",
        "categories": "Diagnosis categories that count: environment (default), program, numerics, results.",
        "include_rejections": "Also count identical sbatch refusals (bad account, bad gres, ...). Default true.",
    },
    "JobRef": {
        "cluster_id": "Cluster the job ran on.",
        "job_id": "Slurm job id.",
        "submit_time": "Submit time (ISO) to disambiguate reused job ids.",
    },
    "RecoverySelector": {
        "workflow_id": "Workflow whose cells to recover.",
        "cells": "'missing_valid' (default: every cell without a valid result), {diagnosis_in: [codes]} or {run_ids: [ids]}.",
    },
    "DiagnosisSelector": {
        "diagnosis_in": "Diagnosis codes to select, e.g. ['node_fail', 'timeout'].",
    },
    "RunIdsSelector": {
        "run_ids": "Run ids to select.",
    },
    "ResourcePatch": {
        "time": "New time limit, e.g. '12:00:00'.",
        "mem": "New memory per node, e.g. '64G'.",
        "gpus": "New GPU count.",
        "gpu_type": "New GPU type.",
        "partition": "New partition.",
        "account": "New Slurm account.",
        "qos": "New QoS.",
        "cpus": "New CPUs per task.",
        "nodes": "New node count.",
        "ntasks_per_node": "New tasks per node.",
        "exclude_nodes": "Nodes to avoid, e.g. ['node-12'].",
        "constraint": "Slurm --constraint expression.",
        "extra_sbatch": "Extra raw #SBATCH options.",
    },
    "ByteRange": {
        "offset": "Start byte (default 0).",
        "length": "Number of bytes (capped by max_bytes).",
    },
    "ComparisonContract": {
        "treatment": "Axes that may differ between groups, e.g. ['method'].",
        "invariants": "Properties that must match across compared results (default split, evaluator, dataset).",
        "baseline": 'Treatment values of the baseline group, e.g. {"method": "baseline"}.',
        "metrics": "Metric declarations: [{name, higher_is_better, unit}].",
        "pair_by_seed": "Pair differences by seed (default true).",
        "ci_level": "Confidence level of intervals (default 0.95).",
        "selection": "Which record a reported number comes from.",
        "attempt_selection": "Which valid attempt of a run counts: latest_valid (default) or first_valid.",
        "min_n": "Minimum included seeds per cell before a value is reported (default 1).",
        "bold_rule": "'none' (default) or 'best_mean'.",
    },
    "MetricDecl": {
        "name": "Metric name, e.g. 'acc'.",
        "higher_is_better": "Direction of improvement.",
        "unit": "Unit, e.g. 'fraction' or 'nats'.",
    },
    "SelectionRule": {
        "rule": "'final' (the record marked final, default), 'last', 'best_by' (needs select_metric/select_split/mode) or 'at_step'.",
        "select_metric": "Metric used to pick the record for best_by.",
        "select_split": "Split used to pick the record for best_by (must differ from report_split).",
        "mode": "'max' or 'min' for best_by.",
        "step": "Step for at_step.",
        "report_split": "Split whose values are reported.",
    },
    "ExclusionSpec": {
        "run_id": "Planned run to exclude.",
        "reason": "Concrete reason (disclosed in the bundle).",
    },
    "TableSpec": {
        "stat": "'mean_std' (default) or 'mean_sem'.",
        "decimals": "Decimals shown (default 3).",
        "scale": "Multiply values, e.g. 100 to show fractions as percent.",
        "min_n": "Show a value only when at least this many seeds are included.",
        "caption": "Table caption.",
        "label": "LaTeX \\label.",
        "bold_best": "Bold the best mean per metric (default false; not a significance claim).",
    },
}

SERVER_INSTRUCTIONS = """\
HPC Lab runs ML experiments on Slurm clusters reliably and keeps a ledger linking every reported number to \
code, config, Slurm job and output digests. Typical flow: cluster_inspect → experiment_prepare (read every \
validation finding) → workflow_submit (whole matrix; smoke gate first) or run_submit (one run) — confirm GPU \
spend with the user → runs_list(view='grid') / run_inspect to watch → run_recover + run_submit for failures → \
evaluation_collect → results_compare → evidence_export → evidence_verify(tex_paths) before submission.
Rules: never resubmit a run whose submission is 'submission_uncertain' (U) — the controller reconciles it; \
never cancel by guessing a job id — use run_cancel; 'unchecked' / 'unavailable' / 'stale' are not success; \
report missing or failed seeds, never drop them; numbers for papers come from evidence bundles, not from logs. \
Mutations take an idempotency_key: reuse it when retrying the same request. Jobs take minutes to days: after \
a submit, check back after poll_after_ms (or when the user asks) instead of polling in a tight loop. Each \
result's text ends with 'Next:' suggestions (exact calls with the ids to use); the full data is in \
structuredContent. Resources: hpclab://experiments (all experiments, newest first — start here in a new session), \
experiments/<id>, operations/<id>, plans/<id> (+/script), workflows/<id> (current version), runs/<id>, \
artifacts/<id>, evaluations/<id>, comparisons/<id>, evidence/<id> (+/table.tex, /summary.md, /table.csv, \
/manifest.json), trace/<bundle_id>:<cell_id>."""
