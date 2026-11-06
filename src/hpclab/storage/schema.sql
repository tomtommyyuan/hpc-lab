-- HPC Lab ledger schema, migration 1.
-- Conventions:
--   * ids are TEXT with type prefixes (see hpclab.domain.ids)
--   * timestamps are TEXT ISO-8601 UTC with 'Z'
--   * *_json columns hold canonical JSON (hpclab.domain.hashing.canonical_json)
--   * every mutable entity has an integer `version` bumped on each update (optimistic concurrency)
--   * current-state rows, their append-only `events`, `operations` and `outbox` rows are
--     written in ONE transaction (BEGIN IMMEDIATE); network I/O never happens inside a transaction.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Registered clusters: the resolved profile (public profile + private overlay, secrets excluded).
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id        TEXT PRIMARY KEY,
    profile_json      TEXT NOT NULL,
    overlay_json      TEXT NOT NULL,
    profile_revision  TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    cluster_id        TEXT NOT NULL,
    question          TEXT,
    spec_json         TEXT NOT NULL,          -- ExperimentSpec as submitted (normalized)
    spec_hash         TEXT NOT NULL,
    project_root      TEXT,
    budget_gpu_hours  REAL,                   -- NULL = no hpclab budget (site quota still applies)
    max_parallel      INTEGER,
    status            TEXT NOT NULL,          -- preparing | ready | invalid
    tags_json         TEXT NOT NULL DEFAULT '[]',
    version           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_experiments_name ON experiments(name);

-- A prepared plan: the frozen set of Runs + execution plans for one experiment_prepare call
-- (or a recovery / probe plan). Only status='ready' plans are submittable.
CREATE TABLE IF NOT EXISTS plans (
    plan_id           TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL REFERENCES experiments(experiment_id),
    kind              TEXT NOT NULL,          -- matrix | recovery | probe | derived
    status            TEXT NOT NULL,          -- preparing | ready | invalid
    plan_hash         TEXT,                   -- digest over run spec hashes + execution plan hashes
    snapshot_digest   TEXT,
    validation_json   TEXT NOT NULL DEFAULT '{}',
    smoke_run_id      TEXT,
    operation_id      TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_plans_experiment ON plans(experiment_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL REFERENCES experiments(experiment_id),
    plan_id           TEXT NOT NULL REFERENCES plans(plan_id),
    kind              TEXT NOT NULL,          -- main | smoke | probe
    label             TEXT NOT NULL,
    cell_key          TEXT NOT NULL,          -- canonical JSON of matrix coordinates (incl. seed)
    seed              INTEGER,
    spec_json         TEXT NOT NULL,          -- RunSpec
    spec_hash         TEXT NOT NULL,
    derived_from      TEXT,
    current_exec_plan_hash TEXT,              -- latest ready execution plan for this run
    claim_attempt_id  TEXT,                   -- attempt holding the single execution claim, or NULL
    latest_attempt_id TEXT,
    status            TEXT NOT NULL,          -- RunStatus summary (derived; refreshed on attempt change)
    version           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    UNIQUE (plan_id, cell_key, kind)
);
CREATE INDEX IF NOT EXISTS ix_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);

-- Execution plans are content addressed and consumed at most once.
CREATE TABLE IF NOT EXISTS execution_plans (
    exec_plan_hash    TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    plan_json         TEXT NOT NULL,          -- ExecutionPlan
    status            TEXT NOT NULL,          -- ready | not_ready | invalid | consumed
    validation_json   TEXT NOT NULL DEFAULT '{}',
    consumed_by_attempt_id TEXT UNIQUE,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exec_plans_run ON execution_plans(run_id);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id        TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    exec_plan_hash    TEXT NOT NULL REFERENCES execution_plans(exec_plan_hash),
    seq               INTEGER NOT NULL,       -- 1,2,3 per run
    cluster_id        TEXT NOT NULL,
    submit_token      TEXT NOT NULL UNIQUE,
    job_name          TEXT NOT NULL,
    remote_dir        TEXT NOT NULL,
    phase             TEXT NOT NULL,          -- AttemptPhase
    scheduler_state   TEXT NOT NULL DEFAULT 'UNKNOWN',
    raw_scheduler_state TEXT,
    app_state         TEXT NOT NULL DEFAULT 'unknown',
    result_state      TEXT NOT NULL DEFAULT 'unchecked',
    cancel_state      TEXT NOT NULL DEFAULT 'none',
    cancel_reason     TEXT,
    job_id            TEXT,
    job_owner         TEXT,
    job_submit_time   TEXT,
    array_task_id     TEXT,
    restart_count     INTEGER NOT NULL DEFAULT 0,
    exit_code         INTEGER,
    exit_signal       INTEGER,
    reason            TEXT,
    node_list         TEXT,
    start_time        TEXT,
    end_time          TEXT,
    elapsed_s         INTEGER,
    time_limit_s      INTEGER,
    gpus              INTEGER NOT NULL DEFAULT 0,
    dispatch_at       TEXT,                   -- when submission_in_flight was committed
    identified_at     TEXT,
    terminal_at       TEXT,
    last_observed_at  TEXT,
    last_observation_json TEXT,
    observation_gap   INTEGER NOT NULL DEFAULT 0,  -- 1 if job vanished from squeue and sacct has no row yet
    uncertain_since   TEXT,
    workflow_id       TEXT,
    operation_id      TEXT,                   -- the submit operation
    version           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_attempts_run ON attempts(run_id);
CREATE INDEX IF NOT EXISTS ix_attempts_phase ON attempts(phase);
CREATE INDEX IF NOT EXISTS ix_attempts_job ON attempts(cluster_id, job_id);

-- Scheduler requeue / restart history as explicit execution segments.
CREATE TABLE IF NOT EXISTS attempt_segments (
    attempt_id        TEXT NOT NULL REFERENCES attempts(attempt_id),
    segment           INTEGER NOT NULL,
    job_id            TEXT,
    start_time        TEXT,
    end_time          TEXT,
    state             TEXT,
    node_list         TEXT,
    observed_at       TEXT NOT NULL,
    PRIMARY KEY (attempt_id, segment)
);

CREATE TABLE IF NOT EXISTS operations (
    operation_id      TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,          -- prepare | submit | reconcile | cancel | workflow_dispatch | workflow_cancel | collect | download | refresh
    state             TEXT NOT NULL,          -- queued | running | blocked | succeeded | failed
    stage             TEXT,
    target_type       TEXT,
    target_id         TEXT,
    progress_json     TEXT NOT NULL DEFAULT '{}',   -- {"done": n, "total": m} or {} when unknown; never faked
    result_json       TEXT,
    error_json        TEXT,
    poll_after_ms     INTEGER NOT NULL DEFAULT 2000,
    version           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS ix_operations_state ON operations(state);
CREATE INDEX IF NOT EXISTS ix_operations_target ON operations(target_type, target_id);

-- Durable work queue. The worker leases rows; leases expire so a crashed worker's work resumes.
-- Steps that performed an external side effect record it before completing, and are designed
-- so a re-run after crash never repeats a non-idempotent side effect (sbatch, scancel).
CREATE TABLE IF NOT EXISTS outbox (
    outbox_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL,
    operation_id      TEXT,
    target_id         TEXT,
    payload_json      TEXT NOT NULL DEFAULT '{}',
    state             TEXT NOT NULL DEFAULT 'pending',  -- pending | leased | done | dead
    not_before        TEXT NOT NULL,
    lease_owner       TEXT,
    lease_until       TEXT,
    tries             INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    dedupe_key        TEXT UNIQUE,            -- prevents duplicate enqueue of the same logical step
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_outbox_due ON outbox(state, not_before);

-- Idempotency records: (principal, tool, key) -> payload hash + stored response.
CREATE TABLE IF NOT EXISTS idempotency (
    principal         TEXT NOT NULL,
    tool              TEXT NOT NULL,
    idem_key          TEXT NOT NULL,
    payload_hash      TEXT NOT NULL,
    response_json     TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (principal, tool, idem_key)
);

-- Append-only event log. Projections above can be rebuilt/audited from here.
CREATE TABLE IF NOT EXISTS events (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id          TEXT NOT NULL UNIQUE,
    entity_type       TEXT NOT NULL,
    entity_id         TEXT NOT NULL,
    entity_version    INTEGER,
    event_type        TEXT NOT NULL,
    payload_json      TEXT NOT NULL DEFAULT '{}',
    command_id        TEXT,
    causation_id      TEXT,
    source            TEXT NOT NULL,          -- api | worker | reconcile | observer | human | migration
    observed_at       TEXT,
    recorded_at       TEXT NOT NULL,
    schema_version    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_entity ON events(entity_type, entity_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(event_type, seq);

-- GPU-hour budget ledger (design §6.5). Amounts in GPU-hours.
CREATE TABLE IF NOT EXISTS budget_reservations (
    reservation_id    TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    attempt_id        TEXT NOT NULL UNIQUE,
    reserved          REAL NOT NULL,          -- gpus x time limit at submission
    estimated_consumed REAL,                  -- running estimate (elapsed so far)
    settled_consumed  REAL,                   -- from accounting after terminal
    state             TEXT NOT NULL,          -- reserved | settled | released | unknown
    note              TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_budget_experiment ON budget_reservations(experiment_id);

CREATE TABLE IF NOT EXISTS workflows (
    workflow_id       TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL REFERENCES experiments(experiment_id),
    plan_id           TEXT NOT NULL UNIQUE REFERENCES plans(plan_id),  -- one execution group per frozen plan
    plan_hash         TEXT NOT NULL,
    max_parallel      INTEGER NOT NULL,
    dispatch_open     INTEGER NOT NULL DEFAULT 1,
    state             TEXT NOT NULL,          -- gating | dispatching | draining | completed | cancelling | cancelled | gate_failed
    gate_run_id       TEXT,
    gate_state        TEXT,                   -- none | pending | passed | failed
    cancel_scope      TEXT,
    cancel_operation_id TEXT,
    operation_id      TEXT,
    version           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_cells (
    workflow_id       TEXT NOT NULL REFERENCES workflows(workflow_id),
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    ordinal           INTEGER NOT NULL,
    cell_state        TEXT NOT NULL,          -- planned | dispatched | active | succeeded | failed | cancelled | uncertain | skipped
    attempt_id        TEXT,
    dispatch_key      TEXT NOT NULL,          -- per-cell idempotency key used for its submission
    note              TEXT,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (workflow_id, run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id       TEXT PRIMARY KEY,
    attempt_id        TEXT NOT NULL REFERENCES attempts(attempt_id),
    run_id            TEXT NOT NULL,
    name              TEXT NOT NULL,          -- declared output name (or name/relpath for dir members)
    role              TEXT NOT NULL,
    kind              TEXT NOT NULL,          -- file | dir
    format            TEXT,
    remote_path       TEXT NOT NULL,
    size_bytes        INTEGER,
    mtime             TEXT,
    digest            TEXT,                   -- sha256:<hex>
    digest_state      TEXT NOT NULL,          -- pending | computed | skipped_too_large | mismatch | unknown
    digest_source     TEXT,                   -- compute_node | login_node | controller
    integrity         TEXT NOT NULL,          -- mutable | incomplete | finalized | missing | unknown
    required          INTEGER NOT NULL DEFAULT 1,
    cache_path        TEXT,
    observed_at       TEXT NOT NULL,
    version           INTEGER NOT NULL DEFAULT 1,
    UNIQUE (attempt_id, name)
);
CREATE INDEX IF NOT EXISTS ix_artifacts_run ON artifacts(run_id);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id     TEXT PRIMARY KEY,
    attempt_id        TEXT NOT NULL REFERENCES attempts(attempt_id),
    run_id            TEXT NOT NULL,
    evaluator_id      TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    config_hash       TEXT NOT NULL,
    input_key         TEXT NOT NULL,          -- digest over (sorted input artifact digests, evaluator, version, config_hash)
    input_refs_json   TEXT NOT NULL,
    metrics_json      TEXT NOT NULL,          -- [{name, value, unit, split, step, higher_is_better, ...}]
    provenance_json   TEXT NOT NULL,
    validation_json   TEXT NOT NULL,          -- {status: valid|partial|invalid, issues: [...]}
    status            TEXT NOT NULL,          -- valid | partial | invalid
    created_at        TEXT NOT NULL,
    UNIQUE (input_key)
);
CREATE INDEX IF NOT EXISTS ix_evaluations_run ON evaluations(run_id);

CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id     TEXT PRIMARY KEY,
    run_set_ref       TEXT NOT NULL,
    contract_json     TEXT NOT NULL,
    request_json      TEXT NOT NULL,
    result_json       TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_bundles (
    bundle_id         TEXT PRIMARY KEY,
    comparison_id     TEXT,
    run_set_ref       TEXT,
    purpose           TEXT NOT NULL,
    manifest_json     TEXT NOT NULL,
    files_json        TEXT NOT NULL,          -- {"summary.md": "...", "table.tex": "...", "table.csv": "..."}
    coverage_json     TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

-- Time-limited observations about the site (flaky node, NFS issue, driver version) with provenance.
CREATE TABLE IF NOT EXISTS observations (
    cluster_id        TEXT NOT NULL,
    obs_key           TEXT NOT NULL,
    value_json        TEXT NOT NULL,
    source            TEXT NOT NULL,
    observed_at       TEXT NOT NULL,
    expires_at        TEXT,
    PRIMARY KEY (cluster_id, obs_key)
);

-- Scheduler poll bookkeeping (rate limits are per cluster and shared by all clients).
CREATE TABLE IF NOT EXISTS poll_state (
    cluster_id        TEXT NOT NULL,
    source            TEXT NOT NULL,          -- squeue | sacct | sinfo
    last_polled_at    TEXT,
    last_ok_at        TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    backoff_until     TEXT,
    last_error        TEXT,
    PRIMARY KEY (cluster_id, source)
);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id   TEXT PRIMARY KEY,
    event_type        TEXT NOT NULL,
    entity_id         TEXT NOT NULL,
    channel           TEXT NOT NULL,
    title             TEXT NOT NULL,
    body              TEXT NOT NULL,
    state             TEXT NOT NULL,          -- pending | sent | failed
    tries             INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    dedupe_key        TEXT UNIQUE,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
