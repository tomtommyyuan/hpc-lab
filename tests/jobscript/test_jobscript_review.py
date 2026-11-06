"""Regression tests from the adversarial review of render_job_script and the bash wrapper."""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time
import types
from pathlib import Path

import pytest
from jobscript_support import ATTEMPT_ID, RUN_ID, TOKEN, Harness, fake_bin_dir

from hpclab.domain.errors import InvalidArgument
from hpclab.services.jobscript import (
    EXIT_DUPLICATE_JOB,
    HELPER_SOURCE,
    attempt_layout,
    job_name_for,
    render_job_script,
)

# ---------------------------------------------------------------------------
# render-time validation
# ---------------------------------------------------------------------------


def test_venv_may_use_runtime_placeholders(harness: Harness) -> None:
    """Used to crash with KeyError('code_dir') instead of rendering or raising a typed error."""
    harness.publish({"env/bin/activate": "export ACTIVATED_FROM=$PWD\n"})
    harness.write(runtime={"venv": "{code_dir}/env"})
    assert harness.run() == 0, harness.stderr()
    assert harness.dump()["env"]["ACTIVATED_FROM"] == harness.layout.work


@pytest.mark.parametrize("key", ["HL_WORK_DIR", "HL_PY", "IFS", "BASH_ENV", "SHELLOPTS", "BASHOPTS"])
def test_env_keys_that_would_clobber_wrapper_state_are_rejected(harness: Harness, key: str) -> None:
    """HL_* are the wrapper's own variables: HL_WORK_DIR=/tmp used to run the program in /tmp."""
    with pytest.raises(InvalidArgument, match=key):
        harness.render(runtime={"env": {key: "/tmp"}})


@pytest.mark.parametrize(
    "env",
    [
        {"PATH": "/opt/tools/bin:$PATH"},
        {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"},
        {"PYTHONPATH": "$PYTHONPATH:/extra"},
    ],
)
def test_search_path_values_referencing_variables_are_rejected_with_a_hint(
    harness: Harness, env: dict[str, str]
) -> None:
    """A literal '$PATH' in PATH breaks every command lookup (surfacing as a bogus guard failure)."""
    with pytest.raises(InvalidArgument) as err:
        harness.render(runtime={"env": env})
    assert "runtime.setup" in str(err.value) or "runtime.setup" in str(err.value.details)


def test_search_path_values_with_home_or_plain_paths_are_fine(harness: Harness) -> None:
    harness.render(runtime={"env": {"PATH": "$HOME/bin:/usr/bin:/bin", "PYTHONPATH": "/extra"}})


def test_mem_conflicting_with_per_gpu_or_per_cpu_memory_is_rejected(harness: Harness) -> None:
    for flag in ("--mem-per-gpu=10G", "--mem-per-cpu=4G"):
        with pytest.raises(InvalidArgument, match="mutually exclusive"):
            harness.render(resources={"mem": "64G", "extra_sbatch": [flag]})
    assert "#SBATCH --mem-per-gpu=10G" in harness.render(resources={"extra_sbatch": ["--mem-per-gpu=10G"]})


def test_plan_for_a_different_run_spec_is_rejected(harness: Harness) -> None:
    """HPCLAB_RUN_SPEC_HASH and checkpoint compatibility must describe the spec that runs."""
    run_spec, plan = harness.specs()
    stale = plan.model_copy(update={"run_spec_hash": "sha256:" + "0" * 64})
    with pytest.raises(InvalidArgument, match="run_spec_hash"):
        render_job_script(
            run_spec=run_spec,
            plan=stale,
            cluster=Harness.cluster(),
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            token=TOKEN,
            job_name=job_name_for("t", TOKEN),
            layout=harness.layout,
        )


@pytest.mark.parametrize("attempt", [".", "..", "..."])
def test_dot_only_ids_are_rejected(attempt: str) -> None:
    with pytest.raises(InvalidArgument):
        attempt_layout("/scratch/u/hpclab", attempt)


# ---------------------------------------------------------------------------
# wrapper runtime
# ---------------------------------------------------------------------------


def _inject_before_main(script_path: Path, code: str) -> None:
    text = script_path.read_text()
    marker = "\n_hl_main\nexit $?\n"
    assert text.endswith(marker)
    script_path.write_text(text[: -len(marker)] + "\n" + code + marker)


def test_stop_signal_arriving_while_the_program_is_being_launched_stops_the_attempt(
    harness: Harness,
) -> None:
    """A TERM that landed after the last pre-run check (argv build, the exec log line) used to be
    lost: the program started anyway, ran until Slurm's SIGKILL, and status.json stayed 'started'.
    (Forwarding it right after the fork does not work either: the child drops signals that
    arrive before it has reset bash's inherited trap.)"""
    script = harness.write()
    argv = " ".join(shlex.quote(a) for a in [sys.executable, "prog.py"])
    _inject_before_main(script, f"_hl_build_argv() {{ HL_ARGV=( {argv} ); kill -TERM $$; }}")
    t0 = time.monotonic()
    rc = harness.run(env={"PROG_MODE": "sleep", "PROG_SLEEP": "30"})

    assert time.monotonic() - t0 < 15, "the program was started despite the stop signal"
    assert rc == 128 + signal.SIGTERM
    status = harness.status()
    assert status["state"] == "killed" and status["signal"] == "TERM"
    assert "not started" in status["error"]
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()


def test_cpu_only_jobs_record_no_gpu_telemetry(harness: Harness, tmp_path: Path) -> None:
    """nvidia-smi on a shared GPU node shows other jobs' GPUs; a gpus=0 attempt owns none."""
    harness.write()  # gpus=0
    env = {
        "PATH": f"{fake_bin_dir(tmp_path / 'bin')}:{harness.base_env['PATH']}",
        "HPCLAB_TELEMETRY_INTERVAL_S": "1",
        "PROG_MODE": "sleep",
        "PROG_SLEEP": "2.5",
    }
    assert harness.run(env=env) == 0, harness.stderr()
    assert len(Path(harness.layout.heartbeat).read_text().split()) >= 2
    assert not Path(harness.layout.gpu_telemetry).exists()


SIGNALLING_SMI = r"""#!/bin/bash
# -L: poke the wrapper with USR1 (its pid is in status.json), then answer after a delay
case "$*" in
  -L)
    pid=$(sed -n 's/.*"pid":\([0-9][0-9]*\).*/\1/p' "$HPCLAB_ATTEMPT_DIR/status.json")
    [ -n "$pid" ] && kill -USR1 "$pid"
    sleep 1
    echo "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abc)"
    ;;
  *) exit 0 ;;
esac
"""


def test_bounded_command_survives_a_signal_to_the_wrapper(harness: Harness, tmp_path: Path) -> None:
    """Without coreutils `timeout` the watchdog path used to return 138 on USR1 while the
    command still ran, turning a healthy nvidia-smi into a guard failure."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(SIGNALLING_SMI)
    smi.chmod(0o755)
    harness.write(resources={"gpus": 1})
    env = {"PATH": f"{bindir}:{harness.base_env['PATH']}", "HPCLAB_HELPER_PYTHON": "none"}
    assert harness.run(env=env) == 0, harness.stderr()
    status = harness.status()
    assert status["exit_code"] == 0 and status["signals_received"] == ["USR1"]
    assert "HPCLAB_GUARD_FAILED" not in harness.stderr()


def test_started_status_reports_the_checkpoint_actually_resumed_from(harness: Harness) -> None:
    """While a requeued segment runs, status.json must not advertise the plan's stale path."""
    from hpclab.domain.models import ResilienceSpec

    harness.write(recovery_path="/ckpts/att_prev/step-100", resilience=ResilienceSpec(resumable=True))
    assert harness.run(env={"PROG_MODE": "requeue"}) == 0, harness.stderr()
    ready = Path(harness.layout.output_dir) / "ready"
    ready.unlink(missing_ok=True)

    proc = harness.start(env={"PROG_MODE": "sleep", "PROG_SLEEP": "30", "SLURM_RESTART_COUNT": "1"})
    try:
        harness.wait_for(ready)
        status = harness.status()
        assert status["state"] == "started" and status["restart_count"] == 1
        assert status["resume_from"] == f"{harness.layout.checkpoint_dir}/step-5"
        assert status["resume_source"] == "attempt_checkpoint"
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)


def test_requeued_segment_does_not_leave_the_previous_outputs_manifest_looking_current(
    harness: Harness,
) -> None:
    from hpclab.domain.models import OutputDecl

    outputs = [OutputDecl(name="metrics", path="{output_dir}/metrics.jsonl", role="metrics")]
    harness.write(outputs=outputs)
    assert harness.run(env={"PROG_MODE": "outputs"}) == 0, harness.stderr()
    assert harness.outputs()["hash_state"] == "complete"
    ready = Path(harness.layout.output_dir) / "ready"
    ready.unlink(missing_ok=True)

    proc = harness.start(env={"PROG_MODE": "sleep", "PROG_SLEEP": "30", "SLURM_RESTART_COUNT": "1"})
    try:
        harness.wait_for(ready)
        assert not Path(harness.layout.outputs_manifest).exists()
        kept = Path(harness.layout.internal_dir) / "outputs.before-restart-1.json"
        assert kept.exists()  # kept for forensics, out of the way of readers
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    assert harness.outputs()["schema"] == "hpclab.outputs/v1"  # the new segment wrote its own


# ---------------------------------------------------------------------------
# duplicate submissions (spec invariant 3: detected, never auto-cancelled)
# ---------------------------------------------------------------------------


def test_duplicate_job_for_the_same_attempt_leaves_without_touching_it(harness: Harness) -> None:
    """Two live jobs for one attempt used to share work/, outputs/, checkpoints/ and status.json
    (and the second deleted the first's half-built work dir)."""
    harness.write()
    first = harness.start(env={"PROG_MODE": "sleep", "PROG_SLEEP": "6"})
    harness.wait_for(Path(harness.layout.output_dir) / "ready")
    started = harness.status()

    t0 = time.monotonic()
    rc = harness.run(env={"SLURM_JOB_ID": "5555", "PROG_MODE": "exit", "PROG_EXIT": "0"})
    assert rc == EXIT_DUPLICATE_JOB
    assert time.monotonic() - t0 < 10
    assert "HPCLAB_DUPLICATE_JOB:4242" in harness.stderr()
    assert harness.status() == started  # the duplicate wrote nothing

    assert first.wait(timeout=30) == 0
    status = harness.status()
    assert status["job_id"] == "4242" and status["state"] == "exited" and status["exit_code"] == 0
    assert harness.dump()["env"]["SLURM_JOB_ID"] == "4242"


def test_requeue_with_the_same_job_id_is_not_a_duplicate(harness: Harness) -> None:
    harness.write()
    assert harness.run() == 0, harness.stderr()
    assert harness.run(env={"SLURM_RESTART_COUNT": "1"}) == 0, harness.stderr()
    assert harness.status()["restart_count"] == 1


def test_no_job_id_means_no_ownership_claim(harness: Harness) -> None:
    """Manual runs outside Slurm (no SLURM_JOB_ID) are not blocked by an earlier job's claim."""
    harness.write()
    assert harness.run() == 0
    env = {k: v for k, v in harness.base_env.items() if k != "SLURM_JOB_ID"}
    harness.base_env = env
    assert harness.run() == 0, harness.stderr()


# ---------------------------------------------------------------------------
# helper: env.json must not carry credentials
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def helper() -> types.ModuleType:
    module = types.ModuleType("hpclab_wrapper_helper_review")
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), module.__dict__)
    return module


def test_credentials_inside_url_values_are_redacted(
    helper: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_INDEX_URL", "https://alice:s3cr3t@pypi.corp.example/simple")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://__token__@upload.pypi.org/legacy/")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/api")
    recorded = helper.interesting_env()
    assert recorded["UV_INDEX_URL"] == "https://***@pypi.corp.example/simple"
    assert recorded["UV_EXTRA_INDEX_URL"] == "https://***@upload.pypi.org/legacy/"
    assert recorded["HF_ENDPOINT"] == "https://hf-mirror.example/api"
    assert "s3cr3t" not in repr(recorded)


def test_env_json_written_by_the_wrapper_has_no_url_credentials(harness: Harness) -> None:
    harness.write()
    assert harness.run(env={"UV_INDEX_URL": "https://bob:hunter2@pypi.corp.example/simple"}) == 0
    assert "hunter2" not in Path(harness.layout.env_json).read_text()
    assert os.path.exists(harness.layout.status_json)
