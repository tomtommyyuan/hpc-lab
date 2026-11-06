# ExperimentSpec reference

An `ExperimentSpec` (usually `hpclab.yaml` next to your code) is what `experiment_prepare`
(`hpclab prepare`) freezes into Runs and execution plans. Everything that defines the *science*
of a run (code, argv, params, seed, inputs, runtime, outputs) becomes part of its immutable
`RunSpec`; changing any of it later creates a new Run (`hpclab prepare --derive-from`). Resources
can change between attempts of the same Run (`run_recover`).

The schema is `ExperimentSpec` in `src/hpclab/domain/models.py`. Unknown keys are rejected at
every level. A complete, runnable example:
[`examples/tiny-classifier/hpclab.yaml`](../examples/tiny-classifier/hpclab.yaml).

## Top level

| Field | Type | Meaning |
|---|---|---|
| `name` | string `[A-Za-z0-9][A-Za-z0-9_.-]{0,79}` | Experiment name (required; not unique, ids are). The CLI accepts a name where it takes an experiment and uses the newest one of that name. |
| `question` | string | The research question. Shown in reports and evidence bundles. |
| `cluster` | string | Registered cluster id (required). |
| `code` | [CodeOptions](#code) | What to snapshot. |
| `entrypoint` | [Entrypoint](#entrypoint) | The program and its argv template (required). |
| `runtime` | [Runtime](#runtime) | Environment on the compute node. |
| `inputs` | list of [InputRef](#inputs) | Datasets / pretrained models, with pin level. |
| `parameters` | map name → scalar | Fixed parameters for every cell (usable as `{name}` in argv). |
| `matrix` | map axis → list of scalars | Treatment axes; the cartesian product defines the cells. Values must be non-empty and unique per axis. |
| `seeds` | list of int | Seeds per cell, at most 64. Alternatively a `seed` axis in `matrix`, not both. |
| `resources` | [Resources](#resources) | Slurm resources per run. |
| `outputs` | list of [OutputDecl](#outputs) | Declared outputs; only these become artifacts. Names must be unique. |
| `smoke` | [Smoke](#smoke) | Optional gate run before the matrix is dispatched. |
| `budget` | `{gpu_hours, max_parallel}` | `gpu_hours` (≥ 0): cap for everything this experiment submits, reserved as GPUs × time limit per attempt. `max_parallel` (1–256): default concurrency (else the site policy's `max_parallel_default`). |
| `evaluation` | [EvaluationDefaults](#evaluation) | How metrics are extracted. When set, every attempt with valid outputs is evaluated automatically. |
| `comparison` | [ComparisonContract](#comparison) | How `results_compare` groups, pairs and checks runs. |
| `resilience` | [Resilience](#resilience) | Checkpoint-on-signal / resumability declaration. |
| `tags` | list of string | Free-form labels. |
| `notes` | string | Free-form notes. |

The matrix expands to at most **256 cells** (cells = product of non-seed axes × seeds). Labels
look like `method-mixup.seed-1`; the smoke run is `smoke.<label of its cell>`.

## code

| Field | Default | |
|---|---|---|
| `project_root` | `.` | Relative to the spec file. For an inline spec sent through MCP, relative to the MCP server's working directory; over the raw controller API an inline spec needs an absolute path. |
| `include_untracked` | `[]` | Untracked files to include explicitly (git projects include tracked + modified files only). |
| `exclude` | `[]` | Extra gitignore-like patterns to exclude. Always excluded, even when tracked or listed in `include_untracked`: likely credentials (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, `*.p12`, `credentials*`, `kaggle.json`, `.aws/`, `.ssh/`, `.config/gh/`, ..., and files whose first 64 KiB hold a PEM private key or a token in a well-known format such as GitHub `gh?_`, Hugging Face `hf_` or AWS keys), virtualenvs and caches. Data/output/checkpoint directories (`data/`, `outputs/`, `wandb/`, `checkpoints/`, ...) are excluded unless a file is tracked, explicitly included, or the directory is a Python package; weight/array files (`*.pt`, `*.ckpt`, `*.safetensors`, ...) over 5 MB unless explicitly included. |
| `max_bytes` / `max_files` | 256 MiB / 50 000 | Hard limits; exceeding them lists the largest offenders. |
| `allow_dirty` | `true` | Uncommitted changes are captured and their diff digest recorded. `false` refuses to snapshot a dirty working tree. |

Non-git projects are snapshotted with the same excludes and a warning (no commit or dirty
tracking). The snapshot is content-addressed: identical trees share a digest.

## entrypoint

```yaml
entrypoint:
  argv: [python, train.py, --method, "{method}", --lr, "{lr}", --seed, "{seed}", --out, "{output_dir}"]
  workdir: "."        # relative to the snapshot root; no '..'
```

`argv` must not be empty. Placeholders:

* `{<parameter or matrix axis>}` and `{seed}`: rendered at prepare time (part of the RunSpec).
* Runtime placeholders, resolved on the compute node: `{output_dir}`, `{checkpoint_dir}`,
  `{attempt_dir}`, `{code_dir}`, `{run_id}`, `{attempt_id}`, `{job_id}`, `{resume_from}` (empty
  string when not resuming).

Unknown placeholders are rejected. The argv reaches your program byte-exact (spaces, quotes, `$`,
unicode). The program also sees these environment variables:

| Variable | Value |
|---|---|
| `HPCLAB_RUN_ID`, `HPCLAB_ATTEMPT_ID`, `HPCLAB_TOKEN` | Identity of this run and attempt |
| `HPCLAB_OUTPUT_DIR`, `HPCLAB_CHECKPOINT_DIR`, `HPCLAB_ATTEMPT_DIR` | This attempt's own directories |
| `HPCLAB_CODE_DIR` | The work directory (a hardlink farm of the read-only snapshot) |
| `HPCLAB_METRICS_PATH` | `$HPCLAB_OUTPUT_DIR/metrics.jsonl` |
| `HPCLAB_RESUME_FROM` | Checkpoint to resume from, or empty |
| `HPCLAB_SEED` | The seed, or empty |
| `HPCLAB_PARAMS_JSON` | Canonical JSON of the run's params |
| `HPCLAB_RUN_SPEC_HASH` | The RunSpec digest (write it into checkpoints) |
| `HPCLAB_RESTART_COUNT` | Slurm requeue count of this attempt |
| `HPCLAB_STOP_FILE` | `<attempt dir>/STOP`; written (with the signal name) when `TERM`, `INT` or the checkpoint signal arrives, before the signal is forwarded to your program (see [resilience](#resilience)) |

plus `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `MASTER_PORT` (derived from the Slurm
job id, only when it is not already set) and `runtime.env`.

## runtime

| Field | Default | |
|---|---|---|
| `venv` | none | Cluster path of a virtualenv (may use `{scratch}`, `{user}`, `{home}`). Activated by the wrapper and inspected at prepare time: torch CUDA build vs the driver, GPU architecture, editable installs that would bypass the snapshot (see [sites/hai.md](sites/hai.md#the-prepare-time-driver-check)). |
| `setup` | `[]` | Bash lines run before the program, in the code directory (`module load cuda/12.8`, ...). Each is executed verbatim: trusted text from your own spec. |
| `env` | `{}` | Extra environment variables. Keys must be valid names; `HPCLAB_*` is reserved except the wrapper knobs `HPCLAB_PROBE_TIMEOUT_S`, `HPCLAB_TELEMETRY_INTERVAL_S`, `HPCLAB_OUTPUT_HASH_BUDGET_S`, `HPCLAB_HELPER_PYTHON`. Values may use `{scratch}`, `{user}`, `{home}`, `{remote_root}`; `$USER`, `$HOME` and `~/` are expanded on the node. Keep caches (`HF_HOME`, `UV_CACHE_DIR`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`) on scratch. |
| `lockfiles` | `[]` | Relative paths inside the snapshot (e.g. `uv.lock`) whose digests make the environment *fingerprinted* rather than *unpinned*. |
| `python` | `python` | Interpreter name used for helper probes. |
| `guard` | `true` | For jobs that request GPUs: before the program starts, check that `nvidia-smi` works and (if torch is importable) that torch sees the GPUs and supports their architecture; otherwise exit 3 with `HPCLAB_GUARD_FAILED:<check>` on stderr. (A `venv` that does not exist or fails to activate stops any job the same way.) |
| `container` | none | Not supported by wrapper v1: a spec that sets it cannot be rendered and the plan is invalid. Put `apptainer exec --nv <image> ...` at the start of `entrypoint.argv` instead. |

## inputs

`{name, uri, version?, digest?, pin?, notes?}`. `uri` is a cluster path (may use `{user}` /
`{scratch}`) or an external URL. `pin` is `pinned` (digest verified), `fingerprinted` (version
label / cheap identity), `unpinned` (mutable path) or `unresolved`; when omitted, prepare
computes it. hpclab does not promise to hash arbitrarily large datasets; it records what it could
verify, and the wrapper records each input's existence, size and mtime at job start.

## resources

| Field | Default | |
|---|---|---|
| `partition`, `account`, `qos` | from overlay/profile | |
| `gpus` | 0 | GPUs per node, 0–64 (`--gres=gpu[:type]:N`). |
| `gpu_type` | any | e.g. `a100`, `h100`. |
| `cpus` | 1 | `--cpus-per-task`, 1–512. |
| `mem` | site default | `--mem`. Normalized to what sbatch accepts: `64GB`/`64gib` → `64G`, `1.5G` → `1536M`; a plain number means MB. |
| `time` | `01:00:00` | Walltime, `HH:MM:SS` or `D-HH:MM:SS`. Rule of thumb: estimate × 1.15 + eval/checkpoint overhead. |
| `nodes`, `ntasks_per_node` | 1, 1 | 1–64 each. v0.1 targets single-node (multi-GPU) jobs. |
| `exclude_nodes` | `[]` | e.g. a node with an active known issue. |
| `constraint` | none | `--constraint`. |
| `extra_sbatch` | `[]` | Allow-listed flags only: `--signal=`, `--constraint=`, `--gpu-bind=`, `--mem-per-gpu=`, `--mem-per-cpu=`, `--cpus-per-gpu=`, `--requeue`, `--no-requeue`, `--hint=`, `--mail-type=`, `--mail-user=`, `--licenses=`, `--reservation=`, `--nice=`, `--tmp=`, `--switches=`. No whitespace or shell metacharacters. Output paths, job name, comment and dependencies are owned by hpclab. Use `resilience` instead of a raw `--signal`. Flags that allocate GPUs (`--gres`, `--gpus`, `--gpus-per-node`, `--gpus-per-socket`, `--gpus-per-task`, `--exclusive`, which takes every GPU on the node) are rejected: request GPUs with `gpus`, so that the budget, the GPU-hour estimate, the limit checks and the compute guard all use the same count. |

Budgets reserve `gpus × nodes × time`, never an optimistic runtime.

## outputs

```yaml
outputs:
  - {name: metrics, path: "{output_dir}/metrics.jsonl", role: metrics, format: hpclab.metrics/v1}
  - {name: summary, path: "{output_dir}/metrics.json", role: metrics, required: false}
  - {name: checkpoints, path: "{checkpoint_dir}", role: checkpoint, kind: dir, required: false}
```

| Field | Default | |
|---|---|---|
| `name` | | `[A-Za-z0-9_.-]{1,64}`, unique. |
| `path` | | Relative to the attempt work dir, or starting with `{output_dir}` / `{checkpoint_dir}`. No `..`, no absolute paths, no globs (declare a `dir`). |
| `role` | `other` | `metrics`, `checkpoint`, `log`, `prediction`, `figure` or `other`. |
| `kind` | `file` | `file` or `dir` (a dir's members are listed and hashed, up to 10 000). |
| `format` | none | A label such as `hpclab.metrics/v1` or `hpclab.checkpoint/v1`. |
| `required` | `true` | A run's results are `valid` only when all required outputs are finalized. |

Digests are computed on the compute node at job end (`outputs.json`), then checked against a
`stat` of each file.

## smoke

```yaml
smoke:
  enabled: true
  argv_append: [--max-steps, "20"]
  params_override: {}
  time: "00:15:00"
  gpus: null               # default: the main runs' GPU count
  partition: null          # e.g. a debug partition if the site allows batch jobs there
  require_outputs: [metrics]
  cell: null               # which matrix cell to smoke, e.g. {method: baseline}; default: the first
```

The smoke run uses the same snapshot and environment. `workflow_submit` dispatches main cells only
after the smoke attempt reached scheduler `COMPLETED` **and** its results validated
(`result_state: valid`, i.e. every output declared `required` is finalized); `require_outputs`
additionally names outputs that must be finalized, which matters for outputs declared
`required: false`. A failed gate leaves the workflow `gate_failed` (resume it with
`hpclab workflow resume` after a successful recovery).
`run_submit` of a main run before the gate passed needs `force_without_gate` plus a recorded
reason.

## evaluation

```yaml
evaluation:
  evaluator_id: hpclab_metrics   # default
  evaluator_version: "1"         # default
  output: metrics                # default: the declared output to read
  config: {split: val}           # default {}
```

When `evaluation` is declared, each attempt is evaluated automatically once its outputs validate;
`evaluation_collect` (`hpclab collect`) re-runs or adds evaluations. Without it, nothing is
evaluated automatically and `evaluation_collect` falls back to `hpclab_metrics@1` on the output
named `metrics`.

Built-in evaluators:

* `hpclab_metrics@1` reads the `hpclab.metrics/v1` format. Config: `split` (default `val`; `null`
  accepts any), `select` (`final`, the default: the last `final: true` record of the split;
  `last`; or `best:<metric>[:max|min]`), `select_split` (only with `best:`), `metrics` (only
  these), `require` (these must be finite or the evaluation is invalid).
  **Early stopping** is declared, not guessed: `select: "best:loss:min"`, `select_split: val`,
  `split: test` picks the step with the best *val* loss and reports the *test* metrics at that
  step (provenance records select split, report split and step). Selecting on the same split that
  is reported (no `select_split`, or `select_split == split`) is flagged `selection_on_split` and
  `results_compare` refuses it as a `selection_leak`.
* `json_path@1` picks fields from any JSON document. Config: `fields` (metric → path), optional
  `split`/`split_path`, `step_path`, `seed_path`, `dataset`/`dataset_path`, `units`,
  `higher_is_better`.
* `jsonl_record@1` selects one record from any JSONL file (`select`: `last`, `step:<n>`,
  `best:<metric>:max|min` or `where:<path>=<value>`), then extracts fields like `json_path`. With
  `split_path` + `step_path`, `best:` accepts `select_split` too (choose the step on one split,
  report another split's record at that step).

A run killed mid-training has no `final: true` record; `hpclab_metrics` then reports an error
(`no_final_record`) instead of the last number it saw, unless `step` reached `total_steps`.

## comparison

```yaml
comparison:
  treatment: [method]                 # what may differ; default: the matrix axes
  invariants: [split, evaluator, dataset]   # default
  baseline: {method: baseline}
  metrics: [{name: acc, higher_is_better: true, unit: fraction}]
  pair_by_seed: true
  ci_level: 0.95
  selection:
    rule: final                       # final | last | best_by | at_step
    select_metric: null               # best_by: metric used to pick the record
    select_split: null                # best_by: split used to pick it (must differ from report_split)
    mode: null                        # best_by: max | min
    step: null                        # at_step
    report_split: null                # split whose values are reported
  attempt_selection: latest_valid     # or first_valid, when a run has several valid attempts
  min_n: 1                            # included seeds needed before a value is reported
  bold_rule: none                     # or best_mean (footnoted in the table; not a significance claim)
```

`results_compare` refuses `best_by` selection on the split that is reported (`selection_leak`)
and flags results that differ in an invariant as incomparable.

## resilience

```yaml
resilience:
  stop_behavior: checkpoint_and_exit  # default none
  signal: USR1                        # USR1 | USR2 | TERM
  lead_s: 900                         # default 600; 60–65535 (sbatch --signal limit); ≥ checkpoint time + 60 s, < time limit
  resumable: true                     # default false; the program resumes from $HPCLAB_RESUME_FROM
  requeue_on_preemption: null         # default null; false opts out of --requeue
  batch_invariant_to_gpus: false      # does the global batch stay fixed if the GPU count changes?
  max_checkpoint_interval_min: null   # recorded with the spec; not used by v0.1
```

What it unlocks:

* `stop_behavior: checkpoint_and_exit` adds `--signal=B:<signal>@<lead_s>`: the wrapper touches
  `$HPCLAB_STOP_FILE` and forwards the signal so the program can checkpoint and exit before the
  time limit; `run_cancel(mode="request_checkpoint")` sends it on demand. Slurm may deliver the
  signal up to ~60 s late. The signal goes to the topmost processes of the program's tree that
  handle it, so behind a launcher that does not (`torchrun`, `accelerate launch`, `deepspeed`, a
  `bash train.sh` without `exec`) the training processes get it instead of the launcher dying of
  the default action; if nothing handles it, the main process gets it and is killed.
* `resumable: true` enables `resume` recovery and **preemptible partitions**, which are blocked
  for non-resumable programs (`preemptible_gate`). A program that is not resumable always gets an
  explicit `--no-requeue`, on every partition: with the default `JobRequeue=1` Slurm also requeues
  a batch job after a node failure, and a restart from scratch would overwrite the attempt's
  outputs. On a preemptible partition a resumable program is submitted with `--requeue` (unless
  `requeue_on_preemption: false`, then `--no-requeue`); elsewhere it keeps the site default and
  resumes from the attempt's own checkpoint after a requeue.
* `batch_invariant_to_gpus: false` (the default) means a GPU-count change would change the
  global batch, i.e. the science: `run_recover` refuses such a resource change and points to a
  derived Run instead.

A resumable program without `stop_behavior` gets a warning: a timeout loses the work since the
last periodic checkpoint.

## File formats your program can write

* Metrics: `hpclab.metrics/v1` JSONL at `$HPCLAB_METRICS_PATH`, one object per line:
  `{"schema":"hpclab.metrics/v1","step":…,"total_steps":…,"split":"val","time":…,"metrics":{…},"final":true}`.
  `total_steps` enables progress/ETA; `final: true` marks the record the default evaluator
  selects; `dataset`, `seed`, `units` and `higher_is_better` let comparisons refuse to mix
  incompatible numbers. Unknown keys are kept.
* Checkpoints: a directory containing `hpclab-ckpt.json` (format `hpclab.checkpoint/v1`, written
  last, by atomic rename) that lists files, sizes and sha256 digests, step, config, `semantic_keys`
  and `run_spec_hash`. Its metadata is validated deterministically; whether it *loads* is checked
  by a probe run on a compute node.

`examples/tiny-classifier/hpclab_io.py` is a dependency-free single file implementing both; copy
it into your project.
