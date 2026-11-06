"""Fixtures for the tiny-classifier example: run train.py as a subprocess, import hpclab_io."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tiny_classifier_support import EXAMPLE_DIR, TRAIN, RunTrain, TrainResult, train_env


@pytest.fixture
def run_train(tmp_path: Path) -> RunTrain:
    """``run_train(*args, name="a", env=None, timeout=60, python=None)`` runs train.py in its own dir.

    Runs sharing a ``name`` share a working directory (like a requeued job sharing its attempt
    dir); ``python`` selects another interpreter (default: the one running the tests).
    """

    def _run(
        *args: str,
        name: str = "run",
        env: dict[str, str] | None = None,
        timeout: float = 60,
        python: str | None = None,
    ) -> TrainResult:
        work = tmp_path / name
        work.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [python or sys.executable, str(TRAIN), *args],
            cwd=work,
            env=env if env is not None else train_env(work),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return TrainResult(proc.returncode, proc.stdout, proc.stderr, work / "outputs", work / "checkpoints")

    return _run


@pytest.fixture
def hpclab_io(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the example's vendorable helper as a fresh module (without writing .pyc)."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("hpclab_io_under_test", EXAMPLE_DIR / "hpclab_io.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
