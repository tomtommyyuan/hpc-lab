"""experiment_prepare hardening: bounded spec reads, no file content in YAML errors, when new
runs count as added after results existed, and requeue flags on non-preemptible partitions."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from prepare_support import Lab

from hpclab.domain.errors import InvalidArgument, LimitExceeded
from hpclab.domain.models import ExecutionPlan, ResilienceSpec
from hpclab.services import experiments as ex
from hpclab.storage.db import jload
from hpclab.testing.factories import make_attempt, make_evaluation


def prepare(lab: Lab, key: str = "k1", **kw: Any) -> ex.PrepareData:
    kw.setdefault("validate_remote", False)
    if "spec" not in kw and "spec_path" not in kw and "extends" not in kw and "derive_from" not in kw:
        kw["spec"] = lab.spec()
    return ex.experiment_prepare(lab.ctx, idempotency_key=key, **kw)


def findings(data: ex.PrepareData) -> dict[str, Any]:
    return {f.check: f for f in data.validation.findings}


def exec_plan(lab: Lab, run_id: str | None) -> ExecutionPlan:
    row = lab.ctx.db.get("runs", "run_id", run_id or "")
    ep = lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])
    return ExecutionPlan.model_validate(jload(ep["plan_json"]))


# ---------------------------------------------------------------------------
# spec_path: only a bounded regular file is read
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_spec_path_character_device_is_refused_without_reading() -> None:
    """``/dev/zero`` stats as 0 bytes and never ends: reading it would grow the controller
    without bound. Run in a child so a regression is killed instead of eating the test host."""
    code = (
        # a regression reads the device forever: exit after a few seconds, before it eats the host
        "import os, threading\n"
        "threading.Timer(5, os._exit, (3,)).start()\n"
        "from hpclab.domain.errors import InvalidArgument\n"
        "from hpclab.services.experiments import load_spec_file\n"
        "try:\n"
        "    load_spec_file('/dev/zero')\n"
        "except InvalidArgument as exc:\n"
        "    print('REFUSED', exc.message, flush=True)\n"
        "os._exit(0)\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=20, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail("load_spec_file('/dev/zero') did not return: it is reading the device")
    assert out.returncode != 3, "load_spec_file('/dev/zero') did not return: it is reading the device"
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("REFUSED") and "not a regular file" in out.stdout


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_spec_path_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "spec.yaml"
    os.mkfifo(fifo)
    result: dict[str, Exception | None] = {}

    def load() -> None:
        try:
            ex.load_spec_file(str(fifo))
            result["exc"] = None
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion below
            result["exc"] = exc

    t = threading.Thread(target=load, daemon=True)
    t.start()
    t.join(5)
    if t.is_alive():  # unblock the reader (open() waits for a writer) before failing
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        t.join(5)
        pytest.fail("load_spec_file blocked on a FIFO")
    exc = result["exc"]
    assert isinstance(exc, InvalidArgument) and "not a regular file" in exc.message


def test_spec_path_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument, match="not a regular file"):
        ex.load_spec_file(str(tmp_path))


def test_spec_path_oversized_file_is_refused(tmp_path: Path) -> None:
    big = tmp_path / "big.yaml"
    big.write_bytes(b"name: x\n" + b"#" * ex.MAX_SPEC_BYTES)
    with pytest.raises(LimitExceeded, match="larger than"):
        ex.load_spec_file(str(big))
    ok = tmp_path / "ok.yaml"
    ok.write_bytes(b"name: x\n")
    assert ex.load_spec_file(str(ok))[0] == {"name": "x"}


def test_spec_path_non_utf8_is_invalid_argument(tmp_path: Path) -> None:
    binary = tmp_path / "blob.yaml"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    with pytest.raises(InvalidArgument, match="UTF-8"):
        ex.load_spec_file(str(binary))


def test_spec_path_yaml_error_does_not_echo_file_content(tmp_path: Path) -> None:
    """A YAML error names where the problem is, never the offending line: spec_path may point
    at a credentials file outside the agent's sandbox."""
    creds = tmp_path / "credentials"
    creds.write_text(
        "[default]\naws_access_key_id = AKIAEXAMPLEKEY123456\naws_secret_access_key = s3cr3tValue\n"
    )
    with pytest.raises(InvalidArgument) as info:
        ex.load_spec_file(str(creds))
    err = info.value
    shown = err.message + repr(err.details)
    assert "AKIA" not in shown and "aws_access_key_id" not in shown and "s3cr3t" not in shown
    assert "not valid YAML" in err.message
    assert err.details["line"] == 2 and err.details["column"] == 1
    assert "line 2, column 1" in err.message

    netrc = tmp_path / "netrc"
    netrc.write_text("machine example.com\npassword ghp_ABCDEFsecret: x\n")
    with pytest.raises(InvalidArgument) as info:
        ex.load_spec_file(str(netrc))
    assert "ghp_" not in info.value.message + repr(info.value.details)

    alias = tmp_path / "alias.yaml"
    alias.write_text("token: *ghp_SECRETALIASNAME\n")
    with pytest.raises(InvalidArgument) as info:
        ex.load_spec_file(str(alias))
    assert "SECRET" not in info.value.message + repr(info.value.details)


def test_spec_path_yaml_error_still_reports_a_usable_problem(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nseeds: [1, 2\n")
    with pytest.raises(InvalidArgument) as info:
        ex.load_spec_file(str(bad))
    assert "expected" in info.value.message and info.value.details["line"] >= 2


# ---------------------------------------------------------------------------
# added_post_hoc: "results existed" = what compare calls "observed"
# ---------------------------------------------------------------------------


def _complete(lab: Lab, run_id: str | None, *, evaluate: bool = False, result_state: str = "valid") -> None:
    with lab.ctx.db.tx() as tx:
        ep = tx.one("SELECT current_exec_plan_hash AS h FROM runs WHERE run_id=?", (run_id,))
        att = make_attempt(
            tx,
            run_id or "",
            exec_plan_hash=ep["h"] if ep else None,
            phase="terminal",
            scheduler_state="COMPLETED",
            result_state=result_state,
            cluster_id="lab",
            terminal_at=tx.now,
        )
        if evaluate:
            make_evaluation(tx, att)


def test_extend_after_main_runs_completed_without_evaluations_is_post_hoc(lab: Lab) -> None:
    """A COMPLETED attempt with valid outputs has readable metrics before any evaluation is
    collected (no evaluator declared, or auto-evaluation still queued)."""
    base = prepare(lab, spec=lab.spec(seeds=[1, 2], smoke=None, evaluation=None))
    for r in base.runs:
        _complete(lab, r.run_id)
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[3])
    assert ext.runs and all(r.added_post_hoc for r in ext.runs)
    rows = lab.ctx.db.all("SELECT added_post_hoc FROM runs WHERE plan_id=?", (ext.plan_id,))
    assert rows and all(r["added_post_hoc"] == 1 for r in rows)
    assert findings(ext)["post_hoc"].status == "warn"


def test_extend_after_only_the_smoke_was_evaluated_is_not_post_hoc(lab: Lab) -> None:
    """The smoke gate (a few steps) is auto-evaluated as soon as it passes: that is not a result
    of the experiment, so seeds added before any main run finished are not post hoc."""
    base = prepare(lab, spec=lab.spec(seeds=[1, 2]))
    assert base.smoke_run_id
    _complete(lab, base.smoke_run_id, evaluate=True)
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[3])
    assert ext.runs and not any(r.added_post_hoc for r in ext.runs)
    assert "post_hoc" not in findings(ext)


def test_extend_while_main_outputs_are_unvalidated_is_not_post_hoc(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    for r in base.runs:
        _complete(lab, r.run_id, result_state="invalid")
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    assert not any(r.added_post_hoc for r in ext.runs)


def test_extend_after_a_main_evaluation_is_still_post_hoc(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1], smoke=None))
    main = base.runs[0]
    with lab.ctx.db.tx() as tx:
        ep = tx.one("SELECT current_exec_plan_hash AS h FROM runs WHERE run_id=?", (main.run_id,))
        att = make_attempt(
            tx, main.run_id or "", exec_plan_hash=ep["h"] if ep else None, phase="terminal", cluster_id="lab"
        )
        make_evaluation(tx, att)
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    assert all(r.added_post_hoc for r in ext.runs)


def test_derive_after_a_main_run_completed_is_post_hoc(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None, evaluation=None))
    src = base.runs[0]
    _complete(lab, src.run_id)
    der = prepare(lab, key="d1", derive_from={"run_id": src.run_id, "overrides": {"params": {"seed": 9}}})
    assert der.runs[0].added_post_hoc and findings(der)["post_hoc"].status == "warn"


def test_derive_from_the_smoke_after_its_evaluation_is_not_post_hoc(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1]))
    assert base.smoke_run_id
    _complete(lab, base.smoke_run_id, evaluate=True)
    der = prepare(lab, key="d1", derive_from={"run_id": base.smoke_run_id})
    assert not der.runs[0].added_post_hoc


# ---------------------------------------------------------------------------
# requeue: a program that cannot resume is never restarted from scratch by Slurm
# ---------------------------------------------------------------------------


def test_non_resumable_runs_get_no_requeue_on_every_partition(lab: Lab) -> None:
    """With the site default (JobRequeue=1) a node failure requeues the batch job into the same
    attempt dir; a non-resumable program would restart from scratch over its own outputs."""
    data = prepare(lab)  # default partition 'gpu', not preemptible, no resilience
    assert data.status == "ready"
    for r in data.runs:
        assert exec_plan(lab, r.run_id).requeue is False, r.label
    declared = prepare(lab, key="k2", spec=lab.spec(resilience={"resumable": False}))
    assert all(exec_plan(lab, r.run_id).requeue is False for r in declared.runs)


def test_resumable_runs_on_a_non_preemptible_partition_keep_the_site_default(lab: Lab) -> None:
    spec = lab.spec(smoke=None, resilience={"resumable": True})
    data = prepare(lab, spec=spec)
    assert data.status == "ready"
    assert all(exec_plan(lab, r.run_id).requeue is None for r in data.runs)


def test_requeue_for_table(lab: Lab) -> None:
    cluster = lab.ctx.clusters.get("lab")
    resumable = ResilienceSpec(resumable=True)
    declined = ResilienceSpec(resumable=True, requeue_on_preemption=False)
    assert ex.requeue_for(cluster, "gpu", None) is False
    assert ex.requeue_for(cluster, "gpu", ResilienceSpec()) is False
    assert ex.requeue_for(cluster, "gpu", resumable) is None
    assert ex.requeue_for(cluster, "gpu", declined) is None
    assert ex.requeue_for(cluster, "scavenge", None) is False
    assert ex.requeue_for(cluster, "scavenge", resumable) is True
    assert ex.requeue_for(cluster, "scavenge", declined) is False
    assert ex.requeue_for(cluster, "no-such-partition", None) is False
    assert ex.requeue_for(cluster, None, resumable) is None
    assert ex.requeue_for(None, "gpu", None) is False
    assert ex.requeue_for(None, "gpu", resumable) is None
