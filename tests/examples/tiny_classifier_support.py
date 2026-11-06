"""Helpers shared by the tiny-classifier example tests (imported by conftest and test modules)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"
TRAIN = EXAMPLE_DIR / "train.py"


@dataclass
class TrainResult:
    returncode: int
    stdout: str
    stderr: str
    out: Path
    ckpt: Path

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.out / "metrics.jsonl")

    def summary(self) -> dict[str, Any]:
        return json.loads((self.out / "metrics.json").read_text())

    def final(self) -> dict[str, Any]:
        finals = [r for r in self.records() if r.get("final")]
        assert len(finals) == 1, finals
        return finals[0]


def _strict_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse strictly: NaN/Infinity tokens are not JSON and must never be written."""
    lines = path.read_text().splitlines()
    return [json.loads(line, parse_constant=_strict_constant) for line in lines if line.strip()]


def strip_times(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "time"} for r in records]


def train_env(tmp: Path, **extra: str) -> dict[str, str]:
    """A job-like environment: HPCLAB_* point into ``tmp``; no bytecode in the example dir."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HPCLAB_", "SLURM_"))}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HPCLAB_RUN_ID="run_test",
        HPCLAB_ATTEMPT_ID="att_test",
        HPCLAB_OUTPUT_DIR=str(tmp / "outputs"),
        HPCLAB_CHECKPOINT_DIR=str(tmp / "checkpoints"),
        HPCLAB_RESUME_FROM="",
        HPCLAB_SEED="",
    )
    env.update(extra)
    return env


RunTrain = Callable[..., TrainResult]
