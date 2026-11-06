# tiny-classifier: a complete hpclab research project

A deliberately small, **stdlib-only** training script that behaves like a real one: it logs
metrics while it trains, checkpoints, resumes exactly, stops cleanly when Slurm warns it, and
can fail on purpose. Use it to learn hpclab on the local simulator in a few minutes, as the
first job on a real cluster, and as a template for wiring your own project.

The research question (see `hpclab.yaml`):

> Does mixup improve clean validation accuracy over plain cross-entropy for a small MLP
> trained on labels with 10% symmetric noise (3 seeds, paired by seed)?

At this scale the honest answer is "no measurable gain": with the default settings mixup is
1.4 points *below* baseline on validation accuracy (paired by seed, n=3, 95% CI about
[-3.6, +0.8] points). That is the point of the demo: hpclab reports paired differences, n and
intervals, and does not bold the best mean.

Results are bit-for-bit reproducible: the same seed gives the same numbers on every run and on
every Python from 3.8 to 3.14 (the script avoids `sum()` over floats, whose rounding changed
in Python 3.12).

## Files and the hpclab concepts they map to

| File | hpclab concept |
|---|---|
| `train.py` | The **entrypoint** of every Run. Reads `HPCLAB_*` defaults, writes declared outputs. |
| `hpclab_io.py` | Vendorable helper (one file, Python 3.8+, no dependencies) implementing the **`hpclab.metrics/v1`** and **`hpclab.checkpoint/v1`** formats, the `HPCLAB_*` environment and graceful stop. Copy it into your project. |
| `hpclab.yaml` | The **ExperimentSpec** for the local simulator (`profiles/sim.yaml`): question, code snapshot, matrix x seeds, resources, declared outputs, smoke gate, budget, evaluator and **comparison contract**. |
| `hpclab.hai.yaml` | The same experiment for the Stanford HAI cluster (`profiles/hai.yaml`); CPU-only, a safe first real job. |

How the spec expands:

* `matrix: {method: [baseline, mixup]}` x `seeds: [1, 2, 3]` = **6 Runs**. Each Run has an
  immutable RunSpec (code digest + argv + params); changing anything creates a new Run.
* `entrypoint.argv` placeholders: `{method}` and `{seed}` are filled per cell at prepare
  time; `{output_dir}` and `{checkpoint_dir}` are resolved on the compute node to the
  Attempt's directories (`<remote_root>/attempts/<attempt_id>/outputs|checkpoints`).
* `smoke` runs one 20-step cell first; the 6 main runs are only dispatched after it
  COMPLETED **and** its outputs validated (every `required` output, here `metrics` and
  `summary`, finalized; `require_outputs: [metrics]` names `metrics` explicitly).
* `budget.gpu_hours` is checked against *requested* GPUs x time limit (6 x 1 GPU x 10 min +
  smoke), never against optimistic runtime; `max_parallel` caps concurrent jobs.
* `outputs` are the only files hpclab indexes as artifacts (with sha256 computed on the
  compute node). `evaluation` picks the `final: true` record of the `val` split from the
  `metrics` output. `comparison` says what may differ (`treatment: [method]`) and what must
  match (`split`, `evaluator`, `dataset`) for two runs to be compared.

## What `train.py` writes

`<out>/metrics.jsonl` — one JSON object per line (`hpclab.metrics/v1`):

```json
{"schema": "hpclab.metrics/v1", "step": 10, "total_steps": 300, "epoch": 0, "split": "train", "time": 1790000000.1, "metrics": {"loss": 0.67, "lr": 0.1}}
{"schema": "hpclab.metrics/v1", "step": 50, "total_steps": 300, "epoch": 0, "split": "val", "time": 1790000000.2, "metrics": {"acc": 0.828, "loss": 0.414}, "dataset": "synthetic-v1", "seed": 1, "method": "baseline"}
{"schema": "hpclab.metrics/v1", "step": 300, "total_steps": 300, "epoch": 4, "split": "val", "final": true, "time": 1790000000.9, "metrics": {"acc": 0.844, "loss": 0.359}, "dataset": "synthetic-v1", "seed": 1, "method": "baseline", "units": {"acc": "fraction", "loss": "nats"}, "higher_is_better": {"acc": true, "loss": false}}
```

* `total_steps` lets hpclab estimate progress and ETA (and warn before a time limit hits).
* The `final: true` record is what the default evaluator selects; `dataset`, `units` and
  `higher_is_better` let the comparison refuse to mix incompatible numbers.
* Non-finite values are written as `null` with `"nonfinite": {"loss": "nan"}` (NaN is not
  valid JSON).

`<out>/metrics.json` — the held-out **test** summary (same schema, `split: "test"`).

`<ckpt-dir>/ckpt-000000100/` — a checkpoint directory: `model.json`, `optimizer.json` and
the manifest `hpclab-ckpt.json` (step, config, `semantic_keys`, sha256 + size of every
file), written **last** after all files are flushed, then the directory is renamed into
place. A checkpoint without a complete manifest is ignored, so a job killed mid-write never
leaves a "latest checkpoint" that cannot be loaded.

## Run it by hand

```bash
python3 train.py --method mixup --seed 1 --out /tmp/tc/full                          # ~0.2 s
python3 train.py --method mixup --seed 1 --out /tmp/tc/out --ckpt-dir /tmp/tc/ckpt --fail-at-step 250
python3 train.py --method mixup --seed 1 --out /tmp/tc/out2 --ckpt-dir /tmp/tc/ckpt --resume /tmp/tc/ckpt
# /tmp/tc/out2/metrics.json == /tmp/tc/full/metrics.json (except "time")
```

Under hpclab every Attempt gets fresh `outputs/` and `checkpoints/` directories; a resume
Attempt receives the chosen checkpoint in `HPCLAB_RESUME_FROM`.

Reusing directories by hand is safe by construction:

* A **fresh** run starts a new `metrics.jsonl` and deletes a stale `metrics.json`, so a rerun
  that crashes can never be credited with the previous run's final numbers.
* A fresh run **refuses** (exit 2) a `--ckpt-dir` that already holds checkpoints: mixing two
  runs' checkpoints would let a later resume continue the wrong one. Continue it with
  `--resume auto` or use an empty directory.
* A **resumed** run appends to `metrics.jsonl`; records for the steps after the checkpoint
  then appear twice (the later ones are the resumed run's).

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `baseline` | `baseline`, `mixup` or `label_smoothing` |
| `--seed` | `$HPCLAB_SEED` or 0 | model init, batch order and mixup draws |
| `--data-seed` | 0 | the dataset; keep it fixed across compared runs |
| `--steps` / `--lr` / `--batch-size` | 300 / 0.1 / 32 | training length and optimizer |
| `--eval-every` / `--log-every` | 50 / 10 | val evaluation and train-loss logging period |
| `--out` | `$HPCLAB_OUTPUT_DIR` or `./outputs` | metrics directory |
| `--ckpt-dir` / `--ckpt-every` / `--keep-last` | `$HPCLAB_CHECKPOINT_DIR` / 100 / 3 | checkpointing (disabled without a dir) |
| `--resume` | `$HPCLAB_RESUME_FROM` | a checkpoint, a directory of checkpoints (latest wins) or `auto`; a requeued job always continues from its own `--ckpt-dir` |
| `--sleep-per-step` | 0 | slow the job down (to watch progress, cancel or preempt it) |

**Resume is exact.** Batches and mixup coefficients are a pure function of `(seed, step)` and
the checkpoint holds parameters *and* optimizer momentum, so a run interrupted at any
checkpoint and resumed ends with bit-identical metrics. Resuming refuses a checkpoint whose
`method`, `lr`, `batch_size`, `seed` or `data_seed` differ (exit 2), or whose files do not
match their recorded sha256.

**Graceful stop.** On SIGUSR1 (Slurm `--signal=USR1@<lead>` before the time limit) or SIGTERM
(scancel, preemption) the script finishes the current step, writes a checkpoint and exits with
128 + the signal number, ready to be resumed. The hpclab wrapper also touches
`$HPCLAB_STOP_FILE` when it receives the warning signal; the script treats that file exactly
like the signal (same checkpoint, same 128 + N exit code, never the failure code 1).

**Requeue.** When Slurm requeues the job (preemption with `--requeue`, node failure), it reruns
the same script in the same attempt directory with `SLURM_RESTART_COUNT` > 0. The script then
continues from its *own* latest checkpoint (newer than anything it originally resumed from)
instead of starting over.

Exit codes: 0 success; 1 training failure (exception, OOM, NaN); 2 bad arguments, an unusable
or incompatible checkpoint, or a `--ckpt-dir` holding another run's checkpoints; 128 + N
stopped by signal N after checkpointing.

## Failure injection

Each flag reproduces a failure hpclab must diagnose correctly instead of blindly retrying:

| Flag | What happens | Exit |
|---|---|---|
| `--fail-at-step N` | `RuntimeError: injected failure at step N` with a Python traceback | 1 |
| `--oom-at-step N` | a realistic `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB ...` traceback on stderr | 1 |
| `--nan-at-step N` | the step's loss becomes NaN, is logged as `null` + `nonfinite`, then `Loss is NaN at step N ...` on stderr | 1 |

Combined with the simulator's scheduler faults (lost sbatch replies, preemption, node
failure, accounting lag) this covers the recovery paths: OOM -> adjust resources, NaN -> fix
the config (never a plain retry), crash after a checkpoint -> resume.

## Adapting it to your project

1. Copy `hpclab_io.py` next to your training script.
2. Log with `MetricsWriter(os.path.join(out, "metrics.jsonl"))`; give the final evaluation
   record `final=True`, `dataset=`, `seed=`, `units=` and `higher_is_better=`.
3. Save with `write_checkpoint(ckpt_dir, {"model.pt": bytes, ...}, step, config, semantic_keys)`
   and resume from `latest_checkpoint(os.environ.get("HPCLAB_RESUME_FROM") or ckpt_dir)`
   (on a requeue, `SLURM_RESTART_COUNT` > 0, prefer `latest_checkpoint(ckpt_dir)`; see
   `find_resume_checkpoint` in `train.py`). Open the metrics log with
   `MetricsWriter(path, overwrite=<not resuming>)`.
4. Poll `GracefulStop().requested` once per step and checkpoint + exit when it is set.
5. Write an `hpclab.yaml` like this one: declare outputs, a smoke gate and the comparison
   contract up front, so the results table can later be traced back to exact code and jobs.
