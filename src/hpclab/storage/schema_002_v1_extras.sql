-- Migration 2: v1 extras (grid, circuit breaker, health/progress, vault, report cache).

-- Workflow dispatch can be paused (circuit breaker) and resumed; distinct from cancelled.
ALTER TABLE workflows ADD COLUMN dispatch_state TEXT NOT NULL DEFAULT 'open';   -- open | paused | closed
ALTER TABLE workflows ADD COLUMN breaker_json TEXT;                             -- evidence for a tripped breaker
ALTER TABLE workflows ADD COLUMN failure_policy_json TEXT NOT NULL DEFAULT '{}';

-- Experiments can extend a previous experiment (e.g. add seeds) keeping lineage.
ALTER TABLE experiments ADD COLUMN parent_experiment_id TEXT;

-- Runs created after the experiment already had evaluations are disclosed as post-hoc.
ALTER TABLE runs ADD COLUMN added_post_hoc INTEGER NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN reproduction_of TEXT;

-- Cached per-attempt derived views (recomputable from events/files; never the source of truth).
ALTER TABLE attempts ADD COLUMN pending_reason TEXT;
ALTER TABLE attempts ADD COLUMN time_left_s INTEGER;
ALTER TABLE attempts ADD COLUMN status_json TEXT;        -- last read wrapper status.json
ALTER TABLE attempts ADD COLUMN env_json TEXT;           -- last read wrapper env.json
ALTER TABLE attempts ADD COLUMN progress_json TEXT;      -- services.progress.Progress
ALTER TABLE attempts ADD COLUMN health_json TEXT;        -- GPU telemetry / heartbeat flags
ALTER TABLE attempts ADD COLUMN diagnosis_json TEXT;     -- services.diagnosis.Diagnosis
ALTER TABLE attempts ADD COLUMN cancel_mode TEXT;        -- cancel | request_checkpoint
ALTER TABLE attempts ADD COLUMN signal_sent_at TEXT;
ALTER TABLE attempts ADD COLUMN gpu_type TEXT;
ALTER TABLE attempts ADD COLUMN partition TEXT;
ALTER TABLE attempts ADD COLUMN resolution TEXT;         -- human resolution note for abandoned/duplicate attempts

-- Local provenance vault: small, durable copies captured at terminal time so evidence
-- survives scratch purges and slurmdbd MinJobAge/purge (sacct is not an archive).
CREATE TABLE IF NOT EXISTS vault_items (
    item_id           TEXT PRIMARY KEY,
    attempt_id        TEXT NOT NULL REFERENCES attempts(attempt_id),
    kind              TEXT NOT NULL,          -- sacct_raw | script | plan | status | env | outputs_manifest | log_head | log_tail | metrics_file
    source_path       TEXT,
    local_path        TEXT NOT NULL,
    digest            TEXT,
    size_bytes        INTEGER,
    truncated         INTEGER NOT NULL DEFAULT 0,
    captured_at       TEXT NOT NULL,
    UNIQUE (attempt_id, kind, source_path)
);
CREATE INDEX IF NOT EXISTS ix_vault_attempt ON vault_items(attempt_id);

-- Short-lived cache for expensive read-only cluster queries (usage windows, --test-only shapes).
CREATE TABLE IF NOT EXISTS report_cache (
    cache_key         TEXT PRIMARY KEY,
    payload_json      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    expires_at        TEXT
);
