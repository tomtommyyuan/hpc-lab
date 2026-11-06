"""Checkpoint fixtures for the recovery suite: hpclab.checkpoint/v1 dirs, artifact rows, and
running the built-in probe program locally the way the job wrapper would."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id
from hpclab.services.context import ServiceContext
from hpclab.services.probe import PROBE_SOURCE


def write_checkpoint(
    ckpt_dir: Path,
    files: dict[str, bytes],
    *,
    step: int,
    run_id: str,
    attempt_id: str,
    run_spec_hash: str,
    config: dict[str, Any],
    semantic_keys: list[str],
    resumable: bool = True,
) -> dict[str, Any]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in files.items():
        (ckpt_dir / name).write_bytes(data)
        entries.append(
            {"path": name, "size": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
        )
    manifest = {
        "schema": "hpclab.checkpoint/v1",
        "step": step,
        "epoch": 1,
        "complete": True,
        "created_at": 1727400300.0,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_spec_hash": run_spec_hash,
        "config": config,
        "semantic_keys": semantic_keys,
        "files": entries,
        "framework": {"name": "pytorch", "version": "2.8.0"},
        "resumable": resumable,
    }
    (ckpt_dir / "hpclab-ckpt.json").write_text(json.dumps(manifest))
    return manifest


def manifest_digest(manifest: dict[str, Any]) -> str:
    return digest(manifest)


def add_artifact(
    ctx: ServiceContext, *, attempt_id: str, run_id: str, remote_path: str, name: str = "checkpoints"
) -> str:
    art = new_id("artifact")
    with ctx.db.tx() as tx:
        tx.insert(
            "artifacts",
            {
                "artifact_id": art,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "name": name,
                "role": "checkpoint",
                "kind": "dir",
                "remote_path": remote_path,
                "digest_state": "pending",
                "integrity": "finalized",
                "required": 0,
                "observed_at": tx.now,
            },
        )
    return art


def run_probe(ckpt_dir: str, expected_digest: str, out_dir: Path, *, extra_path: Path | None = None) -> int:
    """Run the probe exactly as the job would: ``python -c <source> <dir> <digest>``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HPCLAB_", "PYTHON"))}
    env["HPCLAB_OUTPUT_DIR"] = str(out_dir)
    if extra_path is not None:
        env["PYTHONPATH"] = str(extra_path)
    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SOURCE, ckpt_dir, expected_digest],
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    return proc.returncode


FAKE_TORCH = """
__version__ = "2.8.0"
def load(path, map_location=None, **kw):
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(b"BROKEN"):
        raise RuntimeError("PytorchStreamReader failed reading zip archive")
    if kw.get("weights_only") and data.startswith(b"PICKLED"):
        raise RuntimeError("Weights only load failed")
    return {"ok": True}
"""


def fake_torch(base: Path) -> Path:
    pkg = base / "torch"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(FAKE_TORCH)
    return base
