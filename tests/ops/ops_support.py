"""Helpers shared by the ops tests (imported by conftest and test modules)."""

from __future__ import annotations

from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "logs"
EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"


def read_log(name: str) -> str:
    return (LOG_DIR / name).read_text(encoding="utf-8")
