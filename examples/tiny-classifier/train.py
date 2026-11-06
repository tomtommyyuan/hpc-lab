#!/usr/bin/env python3
"""Tiny classifier — a stdlib-only stand-in for a real training script.

It trains a one-hidden-layer MLP with momentum SGD on a synthetic binary task whose *training*
labels carry 10% symmetric noise (validation and test labels are clean), comparing three
methods: plain cross-entropy (``baseline``), ``mixup`` and ``label_smoothing``. Small enough to
finish in about a second on a laptop, yet it behaves like a real job:

* writes ``hpclab.metrics/v1`` JSONL (train loss with ``total_steps`` for ETA, periodic val,
  one ``final`` val record) and a test-split summary ``metrics.json``;
* writes ``hpclab.checkpoint/v1`` checkpoints and resumes from them *exactly*: a resumed run
  ends with bit-identical metrics to an uninterrupted one (batches and mixup coefficients are
  derived from ``(seed, step)``, so no RNG state has to be carried);
* checkpoints and exits cleanly on SIGUSR1/SIGTERM (Slurm time-limit warning, preemption)
  and, when Slurm requeues the job, continues from its own latest checkpoint;
* can fail on purpose (``--fail-at-step``, ``--oom-at-step``, ``--nan-at-step``) so the
  diagnosis and recovery paths of hpclab can be exercised without a GPU.

Exit codes: 0 success; 1 training failure (exception, OOM, NaN); 2 bad arguments, an
unusable checkpoint or a --ckpt-dir holding another run's checkpoints; 128+N stopped by
signal N (or the wrapper's stop file) after checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import TYPE_CHECKING, Any

from hpclab_io import (
    METRICS_SCHEMA,
    CheckpointError,
    GracefulStop,
    MetricsWriter,
    atomic_write_json,
    check_compatible,
    latest_checkpoint,
    read_checkpoint,
    write_checkpoint,
)

DATASET = "synthetic-v1"
N_FEATURES = 8
HIDDEN = 8
SPLIT_SIZES = {"train": 2000, "val": 500, "test": 500}
TRAIN_LABEL_NOISE = 0.10
MIXUP_ALPHA = 0.4
SMOOTHING = 0.1
MOMENTUM = 0.9
METHODS = ("baseline", "mixup", "label_smoothing")
SEMANTIC_KEYS = ("method", "lr", "batch_size", "seed", "data_seed")
UNITS = {"acc": "fraction", "loss": "nats"}
HIGHER_IS_BETTER = {"acc": True, "loss": False}

EXIT_FAILED = 1
EXIT_USAGE = 2

if TYPE_CHECKING:  # aliases for annotations only, so the script still runs on Python 3.8
    Params = dict[str, Any]
    Split = tuple[list[list[float]], list[float]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def dot(a: list[float], b: list[float]) -> float:
    """Plain left-to-right accumulation.

    Deliberately not ``sum()``: Python 3.12 switched float ``sum`` to compensated summation, so
    results would differ in the last bits between interpreter versions. This keeps the
    example's numbers identical on every Python from 3.8 on.
    """
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def make_dataset(data_seed: int) -> dict[str, Split]:
    """Deterministic synthetic task: a linear score plus a quadratic term, thresholded at 0."""
    rng = random.Random(f"{DATASET}:{data_seed}")
    direction = [rng.gauss(0.0, 1.0) for _ in range(N_FEATURES)]
    norm = math.sqrt(dot(direction, direction))
    direction = [d / norm for d in direction]
    splits: dict[str, Split] = {}
    for split, size in SPLIT_SIZES.items():
        xs: list[list[float]] = []
        ys: list[float] = []
        for _ in range(size):
            x = [rng.gauss(0.0, 1.0) for _ in range(N_FEATURES)]
            score = dot(direction, x) + 0.5 * (x[0] * x[1]) + 0.3 * rng.gauss(0.0, 1.0)
            label = 1.0 if score > 0 else 0.0
            if split == "train" and rng.random() < TRAIN_LABEL_NOISE:
                label = 1.0 - label
            xs.append(x)
            ys.append(label)
        splits[split] = (xs, ys)
    return splits


# ---------------------------------------------------------------------------
# Model: x -> tanh(W1 x + b1) -> w2 . h + b2 (a logit)
# ---------------------------------------------------------------------------


def init_params(seed: int) -> Params:
    rng = random.Random(f"init:{seed}")
    return {
        "w1": [
            [rng.gauss(0.0, 1.0 / math.sqrt(N_FEATURES)) for _ in range(N_FEATURES)] for _ in range(HIDDEN)
        ],
        "b1": [0.0] * HIDDEN,
        "w2": [rng.gauss(0.0, 1.0 / math.sqrt(HIDDEN)) for _ in range(HIDDEN)],
        "b2": 0.0,
    }


def zeros() -> Params:
    return {
        "w1": [[0.0] * N_FEATURES for _ in range(HIDDEN)],
        "b1": [0.0] * HIDDEN,
        "w2": [0.0] * HIDDEN,
        "b2": 0.0,
    }


def forward(params: Params, x: list[float]) -> tuple[list[float], float]:
    hidden = [math.tanh(b + dot(row, x)) for row, b in zip(params["w1"], params["b1"])]
    logit = params["b2"] + dot(params["w2"], hidden)
    return hidden, logit


def bce_with_logits(logit: float, target: float) -> float:
    """Numerically stable binary cross-entropy in nats; ``target`` may be soft."""
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def sgd_step(params: Params, velocity: Params, xs: list[list[float]], ys: list[float], lr: float) -> float:
    """One momentum-SGD step on a batch; returns the mean batch loss (before the update)."""
    grads = zeros()
    total = 0.0
    for x, y in zip(xs, ys):
        hidden, logit = forward(params, x)
        total += bce_with_logits(logit, y)
        dz = sigmoid(logit) - y
        grads["b2"] += dz
        for j, h in enumerate(hidden):
            grads["w2"][j] += dz * h
            dh = dz * params["w2"][j] * (1.0 - h * h)
            grads["b1"][j] += dh
            row = grads["w1"][j]
            for k, v in enumerate(x):
                row[k] += dh * v
    scale = 1.0 / len(xs)
    velocity["b2"] = MOMENTUM * velocity["b2"] + grads["b2"] * scale
    params["b2"] -= lr * velocity["b2"]
    for j in range(HIDDEN):
        velocity["w2"][j] = MOMENTUM * velocity["w2"][j] + grads["w2"][j] * scale
        params["w2"][j] -= lr * velocity["w2"][j]
        velocity["b1"][j] = MOMENTUM * velocity["b1"][j] + grads["b1"][j] * scale
        params["b1"][j] -= lr * velocity["b1"][j]
        vrow, prow, grow = velocity["w1"][j], params["w1"][j], grads["w1"][j]
        for k in range(N_FEATURES):
            vrow[k] = MOMENTUM * vrow[k] + grow[k] * scale
            prow[k] -= lr * vrow[k]
    return total * scale


def evaluate(params: Params, data: Split) -> dict[str, float]:
    xs, ys = data
    loss = 0.0
    correct = 0
    for x, y in zip(xs, ys):
        _, logit = forward(params, x)
        loss += bce_with_logits(logit, y)
        correct += int((logit > 0) == (y > 0.5))
    return {"acc": correct / len(xs), "loss": loss / len(xs)}


def make_batch(
    train: Split, method: str, seed: int, step: int, batch_size: int
) -> tuple[list[list[float]], list[float]]:
    """Batch for ``step``: a pure function of (seed, step), which makes resume exact."""
    rng = random.Random(f"batch:{seed}:{step}")
    xs_all, ys_all = train
    idx = [rng.randrange(len(xs_all)) for _ in range(batch_size)]
    xs = [xs_all[i] for i in idx]
    ys = [ys_all[i] for i in idx]
    if method == "label_smoothing":
        ys = [y * (1.0 - SMOOTHING) + 0.5 * SMOOTHING for y in ys]
    elif method == "mixup":
        lam = rng.betavariate(MIXUP_ALPHA, MIXUP_ALPHA)
        perm = list(range(batch_size))
        rng.shuffle(perm)
        xs = [[lam * a + (1.0 - lam) * b for a, b in zip(xs[i], xs[p])] for i, p in enumerate(perm)]
        ys = [lam * ys[i] + (1.0 - lam) * ys[p] for i, p in enumerate(perm)]
    return xs, ys


# ---------------------------------------------------------------------------
# Failure injection (for exercising hpclab diagnosis/recovery without a GPU)
# ---------------------------------------------------------------------------

_OOM_TRACEBACK = """Traceback (most recent call last):
  File "{script}", line 412, in <module>
    sys.exit(main())
  File "{script}", line 371, in main
    loss = model(batch).loss
  File "/opt/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "{script}", line 188, in forward
    hidden = self.mlp(torch.cat([x, x @ self.proj], dim=-1))
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 1.12 GiB is free. Including non-PyTorch memory, this process has 78.06 GiB memory in use. Of the allocated memory 75.31 GiB is allocated by PyTorch, and 1.24 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
"""


def inject_failures(args: argparse.Namespace, step: int) -> None:
    if args.fail_at_step == step:
        raise RuntimeError(f"injected failure at step {step} (--fail-at-step)")
    if args.oom_at_step == step:
        sys.stderr.write(_OOM_TRACEBACK.format(script=os.path.abspath(__file__)))
        sys.stderr.flush()
        sys.exit(EXIT_FAILED)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def fail(message: str, code: int) -> int:
    sys.stderr.write(f"error: {message}\n")
    sys.stderr.flush()
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--method", choices=METHODS, default="baseline")
    # A *string* default goes through type=int like a command-line value, so a malformed
    # $HPCLAB_SEED is a usage error (exit 2), never a traceback or a silent seed 0.
    p.add_argument("--seed", type=int, default=env.get("HPCLAB_SEED") or "0")
    p.add_argument(
        "--data-seed", type=int, default=0, help="fixes the dataset; keep it constant across compared runs"
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default=env.get("HPCLAB_OUTPUT_DIR") or "outputs")
    p.add_argument("--ckpt-dir", default=env.get("HPCLAB_CHECKPOINT_DIR") or "")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--keep-last", type=int, default=3, help="complete checkpoints to keep (0 = all)")
    p.add_argument(
        "--resume",
        default=env.get("HPCLAB_RESUME_FROM") or "",
        help="checkpoint dir, a dir of checkpoints (latest wins), or 'auto' (latest in --ckpt-dir, else "
        "fresh); a requeued job (SLURM_RESTART_COUNT > 0) always continues from its own --ckpt-dir",
    )
    p.add_argument(
        "--sleep-per-step", type=float, default=0.0, help="seconds to sleep per step (simulate slow jobs)"
    )
    p.add_argument("--fail-at-step", type=int, default=None, help="raise RuntimeError at this step")
    p.add_argument(
        "--oom-at-step", type=int, default=None, help="print a CUDA OOM traceback and exit 1 at this step"
    )
    p.add_argument("--nan-at-step", type=int, default=None, help="loss becomes NaN at this step (exit 1)")
    args = p.parse_args(argv)
    positive = {
        "--steps": args.steps,
        "--batch-size": args.batch_size,
        "--eval-every": args.eval_every,
        "--log-every": args.log_every,
        "--ckpt-every": args.ckpt_every,
    }
    for flag, value in positive.items():
        if value < 1:
            p.error(f"{flag} must be >= 1")
    if not (args.lr > 0 and math.isfinite(args.lr)):
        p.error("--lr must be a positive finite number")
    if args.sleep_per_step < 0 or args.keep_last < 0:
        p.error("--sleep-per-step and --keep-last must be >= 0")
    return args


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.method,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "steps": args.steps,
        "hidden": HIDDEN,
        "momentum": MOMENTUM,
        "world_size": 1,
    }


def restart_count(environ: Any = None) -> int:
    """How often Slurm requeued this job (``SLURM_RESTART_COUNT``); 0 on a first start."""
    env = os.environ if environ is None else environ
    try:
        return max(0, int(env.get("SLURM_RESTART_COUNT") or 0))
    except ValueError:
        return 0


def find_resume_checkpoint(args: argparse.Namespace) -> str | None:
    """The checkpoint to continue from, or None for a fresh start.

    * A requeued job (preemption, node failure: Slurm reruns the same script in the same
      directories) continues from its *own* latest checkpoint, which is newer than whatever
      it originally resumed from.
    * Explicit resume paths must yield a checkpoint; 'auto' starts fresh if there is none.
    * A fresh start refuses a --ckpt-dir that already holds checkpoints: interleaving two runs'
      checkpoints lets a later resume pick up the wrong one.
    """
    own = latest_checkpoint(args.ckpt_dir) if args.ckpt_dir else None
    if own is not None and restart_count() > 0:
        return own
    if args.resume == "auto":
        return own
    if args.resume:
        found = latest_checkpoint(args.resume)
        if found is None:
            raise CheckpointError(f"--resume {args.resume}: no complete checkpoint found")
        return found
    if own is not None:
        raise CheckpointError(
            f"--ckpt-dir {args.ckpt_dir} already holds checkpoints of another run (latest: {own}); "
            "continue it with --resume auto, or use an empty directory"
        )
    return None


def restore(path: str, config: dict[str, Any]) -> tuple[int, Params, Params]:
    manifest, blobs = read_checkpoint(path)
    mismatched = check_compatible(manifest, config)
    if mismatched:
        saved = manifest.get("config") or {}
        detail = ", ".join(f"{k}: checkpoint={saved.get(k)!r} now={config.get(k)!r}" for k in mismatched)
        raise CheckpointError(f"checkpoint {path} is incompatible ({detail})")
    step = int(manifest["step"])
    if step > config["steps"]:
        raise CheckpointError(
            f"checkpoint {path} is at step {step}, beyond --steps {config['steps']}; "
            "its weights would be reported as a shorter run's result"
        )
    try:
        params = json.loads(blobs["model.json"])
        velocity = json.loads(blobs["optimizer.json"])["velocity"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {path} lacks model/optimizer state: {exc}") from exc
    return step, params, velocity


def save(
    args: argparse.Namespace, config: dict[str, Any], step: int, params: Params, velocity: Params
) -> str:
    files = {
        "model.json": json.dumps(params, sort_keys=True).encode("utf-8"),
        "optimizer.json": json.dumps({"momentum": MOMENTUM, "velocity": velocity}, sort_keys=True).encode(
            "utf-8"
        ),
    }
    return write_checkpoint(
        args.ckpt_dir,
        files,
        step=step,
        config=config,
        semantic_keys=SEMANTIC_KEYS,
        epoch=epoch_of(step, args.batch_size),
        framework={"name": "python-stdlib", "version": sys.version.split()[0]},
        keep_last=args.keep_last or None,
    )


def epoch_of(step: int, batch_size: int) -> int:
    return step * batch_size // SPLIT_SIZES["train"]


def dataset_name(data_seed: int) -> str:
    return DATASET if data_seed == 0 else f"{DATASET}-ds{data_seed}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = run_config(args)
    stop = GracefulStop()
    data = make_dataset(args.data_seed)
    dataset = dataset_name(args.data_seed)

    try:
        resume_path = find_resume_checkpoint(args)
        if resume_path is not None:
            start, params, velocity = restore(resume_path, config)
            log(f"resumed from {resume_path} at step {start}")
        else:
            start, params, velocity = 0, init_params(args.seed), zeros()
    except CheckpointError as exc:
        return fail(str(exc), EXIT_USAGE)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "metrics.json")
    fresh = resume_path is None
    if fresh and os.path.exists(summary_path):
        os.remove(summary_path)  # a previous run's summary must not survive a crash of this one
    # A resumed run continues its metrics log; a fresh one starts a new file.
    writer = MetricsWriter(os.path.join(args.out, "metrics.jsonl"), overwrite=fresh)
    tag = {"dataset": dataset, "seed": args.seed, "method": args.method}
    last_finite = None
    for step in range(start + 1, args.steps + 1):
        inject_failures(args, step)
        loss = sgd_step(
            params,
            velocity,
            *make_batch(data["train"], args.method, args.seed, step, args.batch_size),
            args.lr,
        )
        if args.nan_at_step == step:
            loss = float("nan")
        epoch = epoch_of(step, args.batch_size)
        if not math.isfinite(loss):
            writer.log(step, "train", {"loss": loss}, total_steps=args.steps, epoch=epoch)
            return fail(
                f"Loss is NaN at step {step} (last finite loss: {last_finite}). Stopping; lower the learning rate or check "
                "the input data.",
                EXIT_FAILED,
            )
        last_finite = loss
        if step % args.log_every == 0:
            writer.log(step, "train", {"loss": loss, "lr": args.lr}, total_steps=args.steps, epoch=epoch)
        if step % args.eval_every == 0 and step < args.steps:
            val = evaluate(params, data["val"])
            writer.log(step, "val", val, total_steps=args.steps, epoch=epoch, **tag)
            log(
                "step {}/{} loss={:.4f} val_acc={:.4f} val_loss={:.4f}".format(
                    step, args.steps, loss, val["acc"], val["loss"]
                )
            )
        saved = None
        if args.ckpt_dir and (step % args.ckpt_every == 0 or step == args.steps):
            saved = save(args, config, step, params, velocity)
        if stop.requested:
            if args.ckpt_dir:
                where = saved or save(args, config, step, params, velocity)
            else:
                where = "none (no --ckpt-dir)"
            log(f"received {stop.signal_name} at step {step}; checkpoint: {where}")
            writer.close()
            return stop.exit_code
        if args.sleep_per_step:
            time.sleep(args.sleep_per_step)

    val = evaluate(params, data["val"])
    extras = dict(tag, units=UNITS, higher_is_better=HIGHER_IS_BETTER)
    writer.log(
        args.steps,
        "val",
        val,
        total_steps=args.steps,
        final=True,
        epoch=epoch_of(args.steps, args.batch_size),
        **extras,
    )
    writer.close()
    test = evaluate(params, data["test"])
    summary = {
        "schema": METRICS_SCHEMA,
        "step": args.steps,
        "total_steps": args.steps,
        "split": "test",
        "final": True,
        "time": time.time(),
        "metrics": test,
        **extras,
    }
    atomic_write_json(summary_path, summary)
    log(
        "done: method={} seed={} val_acc={:.4f} test_acc={:.4f}".format(
            args.method, args.seed, val["acc"], test["acc"]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
