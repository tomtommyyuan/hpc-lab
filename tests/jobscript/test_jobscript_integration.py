"""End to end on this machine: git project -> capture -> publish (LocalTransport) -> render -> bash."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from jobscript_support import ATTEMPT_ID, BASH, RUN_ID, TOKEN, Harness

from hpclab.adapters.local import LocalTransport
from hpclab.domain.models import (
    CodeOptions,
    EntrypointSpec,
    ExecutionPlan,
    OutputDecl,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.services.jobscript import attempt_layout, job_name_for, render_job_script
from hpclab.services.snapshot import SnapshotStore, capture_snapshot, publish_snapshot

TRAIN = """
import json, os, sys
out = os.environ["HPCLAB_OUTPUT_DIR"]
from lib.util import answer
with open(os.environ["HPCLAB_METRICS_PATH"], "a") as fh:
    fh.write(json.dumps({"schema": "hpclab.metrics/v1", "step": 1, "final": True,
                         "metrics": {"answer": answer(), "lr": float(sys.argv[2])}}) + "\\n")
try:
    open("train.py", "a").write("tamper")
except OSError:
    open(os.path.join(out, "snapshot-was-read-only"), "w").close()
"""


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def test_captured_snapshot_runs_through_the_wrapper(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "lib").mkdir(parents=True)
    (project / "train.py").write_text(TRAIN)
    (project / "lib" / "__init__.py").write_text("")
    (project / "lib" / "util.py").write_text("def answer():\n    return 42\n")
    (project / ".env").write_text("WANDB_API_KEY=secret\n")
    git(project, "init", "-q")
    git(project, "add", "-A")
    git(project, "commit", "-qm", "init")

    store = SnapshotStore(tmp_path / "store")
    captured = capture_snapshot(project, CodeOptions(), store)
    remote = tmp_path / "remote" / "hpclab"
    snapshot_path = publish_snapshot(LocalTransport("local"), store, captured.ref, str(remote))
    assert not (Path(snapshot_path) / ".env").exists()

    run_spec = RunSpec(
        experiment="e2e",
        code=captured.ref,
        entrypoint=EntrypointSpec(argv=["python", "train.py", "--lr", "{lr}"]),
        argv=[sys.executable, "train.py", "--lr", "0.1"],
        params={"lr": 0.1},
        runtime=RuntimeSpec(python=sys.executable),
        outputs=[OutputDecl(name="metrics", path="{output_dir}/metrics.jsonl", role="metrics")],
    )
    plan = ExecutionPlan(
        run_id=RUN_ID,
        run_spec_hash=run_spec.spec_hash(),
        cluster_id="sim",
        profile_revision="rev",
        resources=ResourceSpec(),
        remote_root=str(remote),
        snapshot_path=snapshot_path,
    )
    layout = attempt_layout(str(remote), ATTEMPT_ID)
    script = render_job_script(
        run_spec=run_spec,
        plan=plan,
        cluster=Harness.cluster(account="lab"),
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        token=TOKEN,
        job_name=job_name_for("e2e", TOKEN),
        layout=layout,
    )
    for d in layout.dirs_to_create():
        os.makedirs(d, exist_ok=True)
    Path(layout.script).write_text(script)

    h = Harness(tmp_path / "h")  # reuse the runner and readers against this layout
    h.layout = layout
    try:
        assert h.run() == 0, h.stderr()
        status = h.status()
        assert status["state"] == "exited" and status["exit_code"] == 0
        assert (Path(layout.output_dir) / "snapshot-was-read-only").exists()
        assert h.env_json()["hpclab"]["code_digest"] == captured.ref.digest
        metrics = h.outputs()["files"][0]
        assert metrics["exists"] is True and metrics["sha256"].startswith("sha256:")
        assert '"answer": 42' in Path(layout.output_dir, "metrics.jsonl").read_text()
        assert subprocess.run([BASH, "-n", layout.script]).returncode == 0
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(tmp_path / "remote")], check=False)
