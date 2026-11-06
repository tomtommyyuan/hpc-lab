# Contributing to HPC Lab

HPC Lab sits between a coding agent and a shared GPU cluster, so its bugs cost other people's
GPU-hours and can put wrong numbers into papers. Contributions are welcome. The bar is that
every rule below still holds afterwards, with a test that would fail if it did not.

Start with [docs/architecture.md](docs/architecture.md) (components, state machine, outbox)
and [docs/guarantees.md](docs/guarantees.md) (what is promised).

## Development setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # runtime + dev dependencies into .venv
uv run hpclab --version                   # hpclab 0.1.0
export HPCLAB_HOME=/tmp/hpclab-dev        # never experiment in your real ~/.hpclab
uv run hpclab init --sim                  # local Slurm simulator as cluster "sim"
uv run hpclab demo --fast                 # end-to-end smoke test of your checkout
```

The CLI autostarts one controller per `HPCLAB_HOME`. After changing service or controller code,
run `uv run hpclab controller restart` so the running controller picks it up. Code snapshots
are read-only on purpose, so clean up a scratch home with
`chmod -R u+w "$HPCLAB_HOME" && rm -rf "$HPCLAB_HOME"`.

Runtime dependencies are deliberately few (`mcp`, `pydantic`, `typer`, `rich`, `pyyaml`); no
numpy, scipy or pandas at runtime (the statistics are hand-written and tested). Adding a
runtime dependency needs a strong reason.

## Repository layout

| Path | What |
|---|---|
| `src/hpclab/domain/` | Pure models, state enums and transitions, ids, hashing, clock, typed errors. Imports neither MCP, SSH nor SQLite. |
| `src/hpclab/storage/` | `db.py` (SQLite WAL, `BEGIN IMMEDIATE` transactions, events, outbox, idempotency) and the schema migrations. |
| `src/hpclab/services/` | The product: one module per area, plain functions `fn(ctx, *, ...) -> XData`. `app.py` is the method table. |
| `src/hpclab/adapters/` | `Transport` (OpenSSH, local) and `SlurmScheduler`, the only code that talks to a cluster. |
| `src/hpclab/controller/` | The daemon (single writer), its JSON-RPC protocol and the client. |
| `src/hpclab/mcp/` | The stdio MCP server; tool names and prose live in `catalog.py`. |
| `src/hpclab/cli/` | The Typer CLI and its rich views. |
| `src/hpclab/sim/` | The Slurm simulator (fake `sbatch`/`squeue`/`sacct`/... executables). |
| `profiles/`, `skills/`, `examples/` | Bundled into the wheel: site profiles and overlay templates, Claude Code skills, the `tiny-classifier` example. |
| `benchmarks/` | The fault-injection matrix and its latest results. |
| `scripts/gen_docs.py` | Generates `docs/tools.md` and `docs/cli.md` from the code. |
| `tests/<area>/` | Tests per area; `tests/conftest.py` has the `hpclab_home`, `paths`, `fake_clock` and `db` fixtures. |

## Tests

3,861 tests (`uv run pytest --co | tail -1`; add `-m "not slow"` etc. to count a subset).
Markers are declared in `pyproject.toml`:

| Command | What runs | Time (M-series laptop) |
|---|---|---|
| `uv run pytest -m "not sim and not slow"` | The fast suite (3,745 tests): units, services against a fake clock and fake adapters, parsers against recorded real Slurm output, MCP contracts, CLI, docs | under 3 min |
| `uv run pytest -m sim` | End-to-end tests that run real local processes through the simulator (116) | about 9 min |
| `uv run pytest -m "not slow"` | Everything except the slow simulator tests (3,842 tests); the pre-merge check | |
| `uv run pytest` | Everything | |
| `uv run pytest tests/<area> -q` | One area while you work | |

* Mark a test `@pytest.mark.sim` when it spawns simulator jobs or a real controller subprocess;
  keep each under 30 s with short jobs. Mark it `@pytest.mark.slow` when it takes more than a
  few seconds for another reason. Every test has a 120 s timeout (`pytest-timeout`).
* Services take their time from `ctx.clock`; use the `fake_clock` fixture instead of `sleep`.
* Parsers are tested against real Slurm output in `tests/fixtures/slurm/`. When you touch a
  parser, add the real output that motivated the change (with usernames and accounts replaced).
* Crash safety is tested by crashing: `tests/controller`, `tests/e2e/test_controller_crash.py`
  and `benchmarks/crash_controller.py` kill the controller at precise points.

Before you open a pull request:

```bash
uv run ruff check src tests scripts benchmarks
uv run ruff format --check src tests scripts
uv run mypy
uv run pytest -m "not sim and not slow"
uv run pytest -m sim                      # when you touched submission, observation, the controller or the sim
```

If you changed submission, reconciliation, observation, cancel or the controller, also run the
fault matrix and include its summary line in the pull request:

```bash
uv run python benchmarks/fault_matrix.py --quick --out /tmp/fm   # 14 scenarios x 1 rep, about 2 min
```

### Generated and checked docs

`docs/tools.md` and `docs/cli.md` are generated from the MCP catalog, the service schemas and
the Typer command tree. After changing a tool, a parameter, a resource, a prompt or a CLI
command or option:

```bash
uv run python scripts/gen_docs.py            # rewrite both files
uv run python scripts/gen_docs.py --check    # exit 1 if either is stale
```

`tests/docs/test_docs_fresh.py` fails while they are stale. `tests/docs/test_docs_consistency.py`
checks the hand-written docs, the bundled skills (`skills/*/SKILL.md`) and the example READMEs
(`examples/*/README.md`): every relative link and anchor resolves, and every `hpclab ...`
command they show exists with the options they use. `tests/docs/test_docs_test_counts.py` keeps
the test counts quoted here, in the README, the changelog and `docs/guarantees.md` within 5% of
the suite. Output snippets in the guides come from
real runs on the simulator. Re-run the commands when you change what they print; do not edit
output by hand.

## Architecture rules

These are the rules reviewers check. Most have a test that fails when they are broken.

1. **Layers.** Frontends (`mcp/`, `cli/`, the Python client) call controller methods; they hold
   no scheduler logic. Services depend on `domain/` and on the adapter interfaces in
   `adapters/base.py`; concrete adapters are injected through `ServiceContext`. `domain/` imports
   neither MCP, SSH nor SQLite.
2. **The controller is the only writer of the ledger.** Nothing else opens `hpclab.db` for
   writing. Long or remote work is an Operation plus an outbox item, and the call returns a
   receipt (`operation_ref`, `poll_after_ms`) instead of blocking.
3. **Services never print.** stdout is the MCP protocol channel. Use
   `logging.getLogger("hpclab.<area>")`; results travel in the returned `...Data` model.
   `tests/e2e/test_mcp_stdio.py::test_stdout_is_protocol_only` and
   `tests/mcp/test_agent_scenarios.py` fail on stray output. (Job-side scripts embedded in
   `jobscript.py` and `probe.py` print on the compute node; that is their job.)
4. **No network I/O inside a transaction.** The pattern is always *tx (record intent) → network
   call → tx (re-read the row, record the outcome)*. `db.tx()` is `BEGIN IMMEDIATE`, and holding
   it across an `ssh` call would stall every other writer. See
   `tests/experiments/test_review.py::test_no_network_io_inside_a_transaction` and
   `tests/artifacts/test_review_regressions.py::test_no_remote_call_runs_inside_a_transaction`.
5. **Service shape.** `def fn(ctx: ServiceContext, *, ...) -> NameData`, keyword-only arguments,
   Pydantic output models with `extra="forbid"`, typed `HpcLabError` subclasses for failures
   (never a bare exception for an expected condition), time from `ctx.clock.now()` (never
   `datetime.now()`). A service that reports coverage returns it in a field named `coverage`.
6. **Mutations are retry-safe.** Every externally visible mutation takes an `idempotency_key`
   and stores its response in the same transaction; mutations of versioned state (attempts,
   workflows) take `expected_version`.
7. **Handlers are crash-safe and re-entrant.** An outbox handler commits an intent state before
   any non-idempotent side effect (`sbatch`, `scancel`) and, on re-entry, inspects that state
   instead of repeating the effect.
8. **One source of truth per question.** Cell codes and "which evaluation counts" come only from
   `services/coverage.py`; admission only from `submission.admit`; state transitions only
   through `domain/states.py`. Grid, comparison, evidence and digest must not re-derive them.
9. **Unknown is not zero.** A source that could not be read makes `coverage.complete` false or
   raises a typed error; a check that could not run is `unchecked`; unknown GPU-hours are
   `unknown`. Never return an empty success for a failed read.
10. **The login node gets bounded metadata commands only** (`squeue`, `sacct`, `sinfo`,
    `sacctmgr`, `scontrol show`, `stat`, small range reads, sha256 up to
    `max_login_hash_bytes`). Anything heavier runs as a Slurm job.
11. **Schema changes are migrations.** Never edit `storage/schema.sql`; add
    `storage/schema_00N_<area>.sql` and append it to `MIGRATIONS` in `storage/db.py`.
12. **Public profiles hold no personal data.** Usernames, accounts, QoS names and SSH aliases
    belong in the 0600 overlay. `tests/profiles/test_bundled_profiles.py` scans the shipped
    profiles, overlay templates and examples, comments included.
13. **The MCP surface is documented in the code.** Tool descriptions and every parameter's
    prose live in `mcp/catalog.py`; `tests/mcp/test_catalog.py` fails on an undocumented
    parameter.

## Invariants

The state machine and services keep these at all times. Each is covered by tests in the named
areas; a change that weakens one needs a design discussion first.

1. **One execution claim per Run.** `runs.claim_attempt_id` is set in the transaction that
   creates the Attempt and cleared only when the attempt is `terminal`, `submission_rejected`,
   `cancelled_before_submit` or `abandoned`. An execution plan is consumed once
   (`UNIQUE consumed_by_attempt_id`). (`tests/submission`, `tests/workflows`)
2. **`sbatch` at most once per attempt.** `submit.dispatch` commits `submission_in_flight`
   before calling `sbatch`; a handler that finds the attempt already in flight moves it to
   `submission_uncertain` and enqueues reconcile. It never re-sends. (`tests/submission`,
   `tests/e2e/test_lost_reply.py`, `tests/e2e/test_controller_crash.py`)
3. **Uncertain submissions bind only to a unique, provable match**: owner is the expected
   remote user, the attempt's token is in the job name or comment, and the submit time is at or
   after `dispatch_at − skew`. More than one match → `duplicate_detected` (no automatic cancel).
   None → keep looking (receipt, `squeue`, `sacct`) until the reconcile window closes, then block
   for a human. Never resubmit automatically.
4. **Cancel re-verifies live identity** (job id, owner, job-name token, submit time, restart
   count) right before `scancel`. Mismatch → `cancel_state: refused` and zero cancellations.
   (`tests/e2e/test_cancel_identity.py`)
5. **Terminal is sticky.** Observations never move a terminal attempt back to running unless the
   restart count increased (a requeue, recorded as a new `attempt_segments` row). Older,
   out-of-order observations are ignored. (`tests/observe`)
6. **Not in `squeue` is not finished.** Ask `sacct`; no row yet → `observation_gap`, state
   unchanged, never `COMPLETED`.
7. **Budgets reserve GPUs × time limit at admission.** Staging, in-flight and uncertain attempts
   keep their reservation; settlement comes from accounting elapsed × GPUs; no accounting →
   `unknown` (kept at the reservation), never 0. (`tests/budget`)
8. **Idempotency is `(principal, tool, key)`.** Same payload → the stored response; different
   payload → `idempotency_conflict`; the response is stored in the same transaction as the
   mutation.
9. **Every state change writes an `events` row in the same transaction.** Operations never
   fake progress.
10. **No empty success when a source was unavailable.** Report it in `coverage`
    (`unavailable`/`stale`/`unsupported`, `complete: false`) or raise the typed error.
    (`tests/coverage`)

## Common changes

* **A new MCP tool or parameter**: the service function and its `...Data` model, an entry in
  `services/app.py` `METHODS`, the tool in `mcp/catalog.py` (description, parameter prose,
  annotations), a CLI command if a human needs it, tests in `tests/mcp` and the area, then
  `scripts/gen_docs.py`.
* **A new site**: `profiles/<site>.yaml` (site rules only; time-varying facts and known issues
  with `observed_at`, `expires_at` and `source`), `profiles/overlays/<site>.example.yaml`,
  tests in `tests/profiles`, and `docs/sites/<site>.md`. Start from
  `profiles/generic-slurm.yaml`.
* **A simulator fault**: `sim/config.py` (`SimFaults`) plus the command that honours it, a
  test in `tests/sim`, and, if it models a real failure mode, a scenario in
  `benchmarks/fault_matrix.py`.
* **A new diagnosis signature**: `services/diagnosis.py`, with the real log excerpt as a test
  fixture (identifiers removed).

## Pull requests

Keep a pull request to one concern, explain the failure mode it addresses and how the tests
show it, and update `CHANGELOG.md` under an "Unreleased" heading. Security problems go through
[SECURITY.md](SECURITY.md), not public issues.

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).
