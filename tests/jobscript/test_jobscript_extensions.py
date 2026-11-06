"""Wave-3 wrapper extensions (spec §7.2 jobscript (a)-(f)), executed with the real /bin/bash:
(a) runtime.venv activation, (b) compute-side GPU guard, (c) telemetry sidecar,
(d) walltime signal + $HPCLAB_STOP_FILE, (e) --requeue/--no-requeue, (f) deterministic MASTER_PORT."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest
from jobscript_support import Harness, fake_bin_dir, fake_torch_dir

from hpclab.domain.errors import InvalidArgument
from hpclab.domain.models import ResilienceSpec
from hpclab.services.jobscript import EXIT_GUARD_FAILED, GUARD_MARKER, TELEMETRY_CAP_BYTES


def gpu_env(
    harness: Harness, tmp_path: Path, *, smi: bool = True, torch: dict | None = None
) -> dict[str, str]:
    env = {"PATH": f"{fake_bin_dir(tmp_path / 'bin', nvidia_smi=smi)}:{harness.base_env['PATH']}"}
    if torch is not None:
        env["PYTHONPATH"] = str(fake_torch_dir(tmp_path / "site", **torch))
    return env


def sbatch_lines(script: Path) -> list[str]:
    return [line for line in script.read_text().splitlines() if line.startswith("#SBATCH ")]


# ---------------------------------------------------------------------------
# (a) runtime.venv
# ---------------------------------------------------------------------------


def test_a_venv_is_activated_before_setup_and_the_program(harness: Harness, tmp_path: Path) -> None:
    venv = tmp_path / "venvs" / "proj"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'export VIRTUAL_ENV="{venv}"\nexport PATH="{venv}/bin:$PATH"\n')
    harness.write(runtime={"venv": str(venv), "setup": ['test "$VIRTUAL_ENV" = "' + str(venv) + '"']})
    assert harness.run() == 0, harness.stderr()
    env = harness.dump()["env"]
    assert env["VIRTUAL_ENV"] == str(venv) and env["PATH"].startswith(f"{venv}/bin:")
    assert "activated virtualenv" in harness.stderr()


def test_a_missing_venv_fails_fast_with_the_guard_contract(harness: Harness, tmp_path: Path) -> None:
    harness.write(runtime={"venv": str(tmp_path / "gone")})
    assert harness.run() == EXIT_GUARD_FAILED
    assert f"{GUARD_MARKER}venv\n" in harness.stderr()
    status = harness.status()
    assert status["guard_failed"] == "venv" and status["exit_code"] == EXIT_GUARD_FAILED
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()


# ---------------------------------------------------------------------------
# (b) compute-side guard (fake nvidia-smi + fake torch on PATH/PYTHONPATH)
# ---------------------------------------------------------------------------


def test_b_guard_failure_exits_3_with_marker_and_status(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    torch = {"version": "2.8.0+cu130", "cuda": "13.0", "available": False}
    t0 = time.monotonic()
    rc = harness.run(env=gpu_env(harness, tmp_path, torch=torch))
    assert rc == EXIT_GUARD_FAILED
    assert time.monotonic() - t0 < 30  # fails in seconds, not after burning the allocation
    stderr = harness.stderr()
    assert f"{GUARD_MARKER}torch_cuda_available\n" in stderr
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 3
    assert status["guard_failed"] == "torch_cuda_available" and status["failed_phase"] == "guard"
    guard = harness.env_json()["guard"]
    assert (
        guard["enabled"] is True and guard["status"] == "fail" and guard["failed"] == "torch_cuda_available"
    )
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()  # the program never ran


def test_b_guard_nvidia_smi_missing(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 2})
    assert harness.run(env=gpu_env(harness, tmp_path, smi=False)) == EXIT_GUARD_FAILED
    assert f"{GUARD_MARKER}nvidia_smi" in harness.stderr()


def test_b_guard_without_torch_records_unchecked_and_continues(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    assert harness.run(env=gpu_env(harness, tmp_path)) == 0, harness.stderr()
    guard = harness.env_json()["guard"]
    assert guard["status"] == "unchecked"
    assert {c["check"]: c["status"] for c in guard["checks"]}["torch_cuda_available"] == "unchecked"
    assert harness.status()["guard_failed"] is None


def test_b_guard_passes_and_cpu_jobs_skip_it(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    torch = {"cuda": "12.8", "available": True, "arch_list": ["sm_80", "sm_90"], "cap": (9, 0)}
    assert harness.run(env=gpu_env(harness, tmp_path, torch=torch)) == 0, harness.stderr()
    assert harness.env_json()["guard"]["status"] == "pass"
    harness.write(resources={"gpus": 0})
    assert harness.run(env=gpu_env(harness, tmp_path, smi=False)) == 0, harness.stderr()
    assert harness.env_json()["guard"]["enabled"] is False


# ---------------------------------------------------------------------------
# (c) telemetry sidecar
# ---------------------------------------------------------------------------


def test_c_sidecar_writes_rows_and_exits_with_the_job(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    env = {
        **gpu_env(harness, tmp_path),
        "HPCLAB_TELEMETRY_INTERVAL_S": "1",
        "PROG_MODE": "sleep",
        "PROG_SLEEP": "2.5",
    }
    assert harness.run(env=env) == 0, harness.stderr()
    heartbeat = Path(harness.layout.heartbeat)
    beats = heartbeat.read_text().split()
    assert len(beats) >= 2 and all(b.isdigit() for b in beats)
    rows = Path(harness.layout.gpu_telemetry).read_text().splitlines()
    assert len(rows) >= 2
    cols = [c.strip() for c in rows[0].split(",")]
    assert len(cols) == 7 and cols[1] == "0" and cols[3] == "97"  # timestamp,index,uuid,util,used,total,power
    # the sidecar was killed with the job: nothing is appended afterwards
    size = heartbeat.stat().st_size
    time.sleep(2.2)
    assert heartbeat.stat().st_size == size


def test_c_telemetry_stops_at_the_cap(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    csv = Path(harness.layout.gpu_telemetry)
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_bytes(b"x" * (2 * TELEMETRY_CAP_BYTES))
    env = {
        **gpu_env(harness, tmp_path),
        "HPCLAB_TELEMETRY_INTERVAL_S": "1",
        "PROG_MODE": "sleep",
        "PROG_SLEEP": "2.2",
    }
    assert harness.run(env=env) == 0, harness.stderr()
    assert csv.stat().st_size == 2 * TELEMETRY_CAP_BYTES  # capped: no more samples
    assert len(Path(harness.layout.heartbeat).read_text().split()) >= 2  # heartbeat continues


def test_c_cpu_only_job_has_heartbeat_but_no_gpu_samples(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 0})
    env = {
        **gpu_env(harness, tmp_path),
        "HPCLAB_TELEMETRY_INTERVAL_S": "1",
        "PROG_MODE": "sleep",
        "PROG_SLEEP": "1.5",
    }
    assert harness.run(env=env) == 0, harness.stderr()
    assert Path(harness.layout.heartbeat).exists()
    assert not Path(harness.layout.gpu_telemetry).exists()


# ---------------------------------------------------------------------------
# (d) walltime signal, forwarding and $HPCLAB_STOP_FILE
# ---------------------------------------------------------------------------


def test_d_usr1_is_announced_forwarded_and_the_stop_file_names_it(harness: Harness) -> None:
    res = ResilienceSpec(signal="USR1", lead_s=300, stop_behavior="checkpoint_and_exit", resumable=True)
    script = harness.write(resilience=res)
    assert "#SBATCH --signal=B:USR1@300" in sbatch_lines(script)
    proc = harness.start(env={"PROG_MODE": "usr1"})
    assert harness.signal_when_ready(proc, signal.SIGUSR1) == 0
    # the program saw the STOP file already present when the forwarded signal arrived
    assert (Path(harness.layout.output_dir) / "got_usr1").read_text() == "stop_file=True"
    assert Path(harness.layout.stop_file).read_text() == "USR1\n"
    assert harness.dump()["env"]["HPCLAB_STOP_FILE"] == harness.layout.stop_file
    status = harness.status()
    assert status["stop_requested"] is True and status["stop_signal"] == "USR1"
    assert "received SIGUSR1; forwarded to the program" in harness.stderr()


def test_d_term_names_term_in_the_stop_file(harness: Harness) -> None:
    harness.write()
    proc = harness.start(env={"PROG_MODE": "sleep"})
    assert harness.signal_when_ready(proc, signal.SIGTERM) == 128 + signal.SIGTERM
    assert Path(harness.layout.stop_file).read_text() == "TERM\n"
    assert harness.status()["state"] == "killed"


def test_d_no_signal_directive_without_checkpoint_and_exit(harness: Harness) -> None:
    script = harness.write(resilience=ResilienceSpec(stop_behavior="none"))
    assert not any(line.startswith("#SBATCH --signal") for line in sbatch_lines(script))


# ---------------------------------------------------------------------------
# (e) --requeue / --no-requeue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requeue", "flag"), [(True, "--requeue"), (False, "--no-requeue"), (None, None)])
def test_e_requeue_flag_follows_the_plan(harness: Harness, requeue: bool | None, flag: str | None) -> None:
    lines = sbatch_lines(harness.write(requeue=requeue))
    flags = [
        f for f in (line.removeprefix("#SBATCH ") for line in lines) if f in ("--requeue", "--no-requeue")
    ]
    assert flags == ([flag] if flag else [])


def test_e_contradicting_extra_sbatch_is_rejected(harness: Harness) -> None:
    with pytest.raises(InvalidArgument):
        harness.write(requeue=True, resources={"extra_sbatch": ["--no-requeue"]})
    lines = sbatch_lines(harness.write(requeue=True, resources={"extra_sbatch": ["--requeue"]}))
    assert lines.count("#SBATCH --requeue") == 1  # not duplicated


# ---------------------------------------------------------------------------
# (f) deterministic MASTER_PORT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("job_id", "preset", "expected"),
    [
        ("4242", None, "24242"),
        ("123456", None, str(20000 + 123456 % 20000)),
        ("20000", None, "20000"),
        ("0099", None, "20099"),  # leading zeros are not octal
        ("4242", "29500", "29500"),  # a user-provided port wins
        ("abc", None, None),  # non-numeric id: not set
    ],
)
def test_f_master_port_is_derived_from_the_job_id(
    harness: Harness, job_id: str, preset: str | None, expected: str | None
) -> None:
    harness.write()
    env = {"SLURM_JOB_ID": job_id}
    if preset is not None:
        env["MASTER_PORT"] = preset
    assert harness.run(env=env) == 0, harness.stderr()
    assert harness.dump()["env"].get("MASTER_PORT") == expected
    assert os.environ.get("MASTER_PORT") != "unexpected"
