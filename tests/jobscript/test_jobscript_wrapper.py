"""Execute rendered wrappers with the real /bin/bash (macOS: bash 3.2) and check the contract:
argv/env, status/env/outputs files, exit codes, signals, requeue, guard, degraded mode."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from jobscript_support import ATTEMPT_ID, RUN_ID, TOKEN, Harness, fake_bin_dir, fake_torch_dir

from hpclab.domain.models import InputRef, OutputDecl, ResilienceSpec
from hpclab.services.jobscript import EXIT_GUARD_FAILED, EXIT_PREPARE_FAILED, EXIT_SETUP_FAILED


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# argv + environment contract (§6.2)
# ---------------------------------------------------------------------------

HOSTILE_ARGV = [
    "a b",
    "",
    'it\'s "quoted"',
    "$HOME",
    "$(touch pwned)",
    "`id`",
    "back\\slash",
    "new\nline",
    "*",
    "~",
    "üñíçødé ✓",
    "{not_a_placeholder}",
    '{"json": [1, 2]}',
    "--",
]


def test_argv_reaches_the_program_byte_exact(harness: Harness) -> None:
    templates = [
        *HOSTILE_ARGV,
        "{output_dir}",
        "--ckpt={checkpoint_dir}/last",
        "{attempt_dir}|{code_dir}|{run_id}|{attempt_id}",
        "--job={job_id}",
        "{resume_from}",
    ]
    harness.write(argv=[sys.executable, "prog.py", *templates])
    assert harness.run() == 0, harness.stderr()

    lay = harness.layout
    assert harness.dump()["argv"] == [
        *HOSTILE_ARGV,
        lay.output_dir,
        f"--ckpt={lay.checkpoint_dir}/last",
        f"{lay.dir}|{lay.work}|{RUN_ID}|{ATTEMPT_ID}",
        "--job=4242",
        "",
    ]
    assert not (Path(lay.work) / "pwned").exists()


def test_environment_contract_and_working_directory(harness: Harness) -> None:
    harness.publish({"sub/.keep": ""})
    harness.write(
        params={"lr": 0.1, "method": "mixup"},
        seed=7,
        workdir="sub",
        argv=[sys.executable, "../prog.py"],
        runtime={
            "env": {
                "WANDB_DIR": "{output_dir}/wandb",
                "MY_VAR": "has space $NOT_EXPANDED ${HOME_NOT} $USERNAME",
                "HF_HOME": "{scratch}/hf",
                "TORCH_HOME": "~/torch",
                "RUN_TAG": "{run_id}-{job_id}",
            }
        },
    )
    assert harness.run() == 0, harness.stderr()

    dump = harness.dump()
    env = dump["env"]
    lay = harness.layout
    assert dump["cwd"] == f"{lay.work}/sub"
    assert env["HPCLAB_RUN_ID"] == RUN_ID
    assert env["HPCLAB_ATTEMPT_ID"] == ATTEMPT_ID
    assert env["HPCLAB_TOKEN"] == TOKEN
    assert env["HPCLAB_OUTPUT_DIR"] == lay.output_dir
    assert env["HPCLAB_CHECKPOINT_DIR"] == lay.checkpoint_dir
    assert env["HPCLAB_RESUME_FROM"] == ""
    assert env["HPCLAB_SEED"] == "7"
    assert env["HPCLAB_CODE_DIR"] == lay.work
    assert env["HPCLAB_ATTEMPT_DIR"] == lay.dir
    assert env["HPCLAB_METRICS_PATH"] == f"{lay.output_dir}/metrics.jsonl"
    assert json.loads(env["HPCLAB_PARAMS_JSON"]) == {"lr": 0.1, "method": "mixup"}
    assert env["HPCLAB_PARAMS_JSON"] == '{"lr":0.1,"method":"mixup"}'
    assert env["HPCLAB_STOP_FILE"] == lay.stop_file
    assert env["PYTHONUNBUFFERED"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["WANDB_DIR"] == f"{lay.output_dir}/wandb"
    assert env["MY_VAR"] == "has space $NOT_EXPANDED ${HOME_NOT} $USERNAME"
    # {scratch} -> /scratch/{user} -> /scratch/$USER, expanded on the node
    assert env["HF_HOME"] == f"/scratch/{os.environ.get('USER', '')}/hf"
    assert env["TORCH_HOME"] == f"{os.environ['HOME']}/torch"
    assert env["RUN_TAG"] == f"{RUN_ID}-4242"
    assert env["MASTER_PORT"] == str(20000 + 4242 % 20000)


def test_recovery_checkpoint_is_exported_as_resume_from(harness: Harness) -> None:
    harness.write(
        argv=[sys.executable, "prog.py", "--resume={resume_from}"], recovery_path="/ckpts/att_prev/step-100"
    )
    assert harness.run() == 0, harness.stderr()
    assert harness.dump()["env"]["HPCLAB_RESUME_FROM"] == "/ckpts/att_prev/step-100"
    assert harness.dump()["argv"] == ["--resume=/ckpts/att_prev/step-100"]
    assert harness.status()["resume_source"] == "plan"


# ---------------------------------------------------------------------------
# status.json / env.json / outputs.json (§6.4)
# ---------------------------------------------------------------------------


def test_status_env_and_outputs_files(harness: Harness, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.bin"
    dataset.write_bytes(b"data")
    harness.publish({"uv.lock": "lock-content\n", "requirements.txt": "torch\n"})
    harness.write(
        runtime={"lockfiles": ["uv.lock", "missing.lock"]},
        inputs=[
            InputRef(name="ds", uri=str(dataset), digest=sha(b"data")),
            InputRef(name="stale", uri=str(dataset), digest=sha(b"other")),
            InputRef(name="gone", uri=str(tmp_path / "missing")),
            InputRef(name="hub", uri="hf://org/model"),
        ],
        outputs=[
            OutputDecl(name="metrics", path="{output_dir}/metrics.jsonl", role="metrics"),
            OutputDecl(name="preds", path="{output_dir}/preds", kind="dir"),
            OutputDecl(name="results", path="results.txt"),
            OutputDecl(name="ckpt", path="{checkpoint_dir}/final.pt", role="checkpoint"),
            OutputDecl(name="optional", path="{output_dir}/nope.txt", required=False),
        ],
    )
    t0 = time.time()
    assert harness.run(env={"PROG_MODE": "outputs"}) == 0, harness.stderr()

    status = harness.status()
    assert status["schema"] == "hpclab.status/v1"
    assert status["state"] == "exited" and status["exit_code"] == 0 and status["signal"] is None
    assert status["attempt_id"] == ATTEMPT_ID and status["run_id"] == RUN_ID and status["token"] == TOKEN
    assert status["job_id"] == "4242" and status["host"]
    assert status["wrapper_version"] == "hpclab-wrapper/1"
    assert status["restart_count"] == 0 and status["failed_phase"] is None
    assert int(t0) - 1 <= status["start_time"] <= status["end_time"] <= time.time() + 1

    env = harness.env_json()
    assert env["schema"] == "hpclab.env/v1"
    assert (
        env["slurm"]["job_id"] == "4242" and env["slurm"]["cpus"] == 2 and env["slurm"]["partition"] == "gpu"
    )
    assert env["python"]["version"] == ".".join(map(str, sys.version_info[:3]))
    assert env["packages_digest"].startswith("sha256:") and env["packages_tool"] == "importlib.metadata"
    packages = Path(harness.layout.packages_file).read_text()
    assert env["packages_digest"] == sha(packages.encode()) and "pydantic==" in packages
    assert env["torch"] is None or env["torch"].get("installed") is not False
    assert env["lockfiles"]["uv.lock"] == sha(b"lock-content\n")
    assert env["lockfiles"]["requirements.txt"] == sha(b"torch\n")  # auto-detected
    assert env["lockfiles"]["missing.lock"] is None
    assert any("missing.lock" in e for e in env["errors"])
    inputs = {i["name"]: i for i in env["inputs"]}
    assert (
        inputs["ds"]["exists"] is True and inputs["ds"]["size"] == 4 and inputs["ds"]["digest_match"] is True
    )
    assert inputs["stale"]["digest_match"] is False
    assert inputs["gone"]["exists"] is False
    assert inputs["hub"]["kind"] == "url"
    assert any("stale" in w for w in env["warnings"])
    assert env["hpclab"]["run_id"] == RUN_ID and env["guard"]["status"] == "skipped"
    assert not any("TOKEN" in k or "KEY" in k for k in env["env_vars"])

    outputs = harness.outputs()
    assert outputs["schema"] == "hpclab.outputs/v1" and outputs["hash_state"] == "complete"
    files = {f["name"]: f for f in outputs["files"]}
    assert files["metrics"]["sha256"] == sha(b'{"step": 1}\n') and files["metrics"]["size"] == 12
    assert files["results"]["path"] == f"{harness.layout.work}/results.txt"
    assert files["results"]["sha256"] == sha(b"R")
    preds = files["preds"]
    assert preds["kind"] == "dir" and preds["size"] == 2 and preds["member_count"] == 2
    assert preds["members"] == [
        {"relpath": "a.txt", "size": 1, "sha256": sha(b"A")},
        {"relpath": "sub/b.txt", "size": 1, "sha256": sha(b"B")},
    ]
    assert preds["sha256"].startswith("sha256:")
    assert files["ckpt"]["exists"] is False and files["optional"]["exists"] is False
    assert outputs["missing_required"] == ["ckpt"]


def test_exit_code_is_propagated(harness: Harness) -> None:
    harness.write()
    assert harness.run(env={"PROG_MODE": "exit", "PROG_EXIT": "3"}) == 3
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 3
    assert status["failed_phase"] is None and status["guard_failed"] is None


def test_command_not_found_is_recorded(harness: Harness) -> None:
    harness.write(argv=["definitely-not-a-command-xyz"])
    assert harness.run() == 127
    assert harness.status()["exit_code"] == 127


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


def test_sigterm_mid_run_is_forwarded_and_recorded_as_killed(harness: Harness) -> None:
    harness.write()
    proc = harness.start(env={"PROG_MODE": "sleep"})
    rc = harness.signal_when_ready(proc, signal.SIGTERM)  # sent to the wrapper only

    assert rc == 128 + signal.SIGTERM
    status = harness.status()
    assert status["state"] == "killed" and status["signal"] == "TERM"
    assert status["exit_code"] == 128 + signal.SIGTERM
    assert status["signals_received"] == ["TERM"] and status["stop_requested"] is True
    assert Path(harness.layout.stop_file).exists()
    assert "forwarded to the program" in harness.stderr()
    assert harness.outputs()["schema"] == "hpclab.outputs/v1"


def test_checkpoint_signal_touches_stop_file_before_forwarding(harness: Harness) -> None:
    res = ResilienceSpec(signal="USR1", lead_s=300, stop_behavior="checkpoint_and_exit", resumable=True)
    script = harness.write(resilience=res)
    assert "#SBATCH --signal=B:USR1@300" in script.read_text()
    proc = harness.start(env={"PROG_MODE": "usr1"})
    rc = harness.signal_when_ready(proc, signal.SIGUSR1)

    assert rc == 0
    assert (Path(harness.layout.output_dir) / "got_usr1").read_text() == "stop_file=True"
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 0
    assert status["signals_received"] == ["USR1"] and status["stop_signal"] == "USR1"


def test_usr1_without_resilience_is_forwarded_but_is_no_stop_request(harness: Harness) -> None:
    harness.write()
    proc = harness.start(env={"PROG_MODE": "usr1"})
    assert harness.signal_when_ready(proc, signal.SIGUSR1) == 0
    assert (Path(harness.layout.output_dir) / "got_usr1").read_text() == "stop_file=False"
    assert harness.status()["stop_requested"] is False


def test_signal_that_breaks_a_setup_step_is_recorded_as_killed(harness: Harness) -> None:
    harness.write(runtime={"setup": ["kill -TERM $$; false", "touch not-reached"]})
    assert harness.run() == 128 + signal.SIGTERM
    status = harness.status()
    assert status["state"] == "killed" and status["signal"] == "TERM" and status["failed_phase"] == "setup"
    assert "received SIGTERM during setup" in status["error"]
    assert not (Path(harness.layout.work) / "not-reached").exists()


def test_signal_before_the_program_starts_stops_the_attempt(harness: Harness) -> None:
    harness.write(runtime={"setup": ["kill -TERM $$"]})
    assert harness.run() == 128 + signal.SIGTERM
    status = harness.status()
    assert status["state"] == "killed" and status["failed_phase"] == "setup"
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()


# ---------------------------------------------------------------------------
# requeue / restart
# ---------------------------------------------------------------------------


def test_requeue_reuses_work_dir_and_resumes_from_the_attempts_checkpoint(harness: Harness) -> None:
    harness.write(
        argv=[sys.executable, "prog.py", "--resume={resume_from}"], resilience=ResilienceSpec(resumable=True)
    )
    assert harness.run(env={"PROG_MODE": "requeue"}) == 0, harness.stderr()
    marker = Path(harness.layout.work) / "segment-marker"
    inode = marker.stat().st_ino
    assert harness.dump()["env"]["HPCLAB_RESUME_FROM"] == ""
    Path(harness.layout.stop_file).write_text("stale from the previous segment")

    assert harness.run(env={"PROG_MODE": "requeue", "SLURM_RESTART_COUNT": "1"}) == 0, harness.stderr()

    assert marker.stat().st_ino == inode
    ckpt = f"{harness.layout.checkpoint_dir}/step-5"
    assert harness.dump()["argv"] == [f"--resume={ckpt}"]
    status = harness.status()
    assert status["restart_count"] == 1
    assert status["resume_from"] == ckpt and status["resume_source"] == "attempt_checkpoint"
    assert not Path(harness.layout.stop_file).exists()
    assert "reusing work dir" in harness.stderr()
    assert harness.stderr().count("===== hpclab attempt") == 2
    # a resumable program continues in place: nothing of the attempt is set aside
    assert status["requeue_set_aside"] is None and status["resumable"] is True
    assert not list(Path(harness.layout.dir).glob("requeue-*"))
    assert (Path(harness.layout.checkpoint_dir) / "step-5" / "model.bin").exists()


def test_requeue_of_a_non_resumable_program_starts_from_scratch(harness: Harness) -> None:
    """A program that does not declare resilience.resumable is rerun from scratch on a requeue:
    the attempt's own checkpoint is never handed to it, and the previous segment's outputs/ and
    checkpoints/ move to requeue-<n>/ so the rerun's files cannot mix with them."""
    harness.write(argv=[sys.executable, "prog.py", "--resume={resume_from}"])
    assert harness.run(env={"PROG_MODE": "requeue"}) == 0, harness.stderr()
    assert harness.status()["requeue_set_aside"] is None and harness.status()["resumable"] is False
    lay = harness.layout

    assert harness.run(env={"PROG_MODE": "dump", "SLURM_RESTART_COUNT": "1"}) == 0, harness.stderr()

    dump = harness.dump()
    assert dump["argv"] == ["--resume="] and dump["env"]["HPCLAB_RESUME_FROM"] == ""
    aside = Path(lay.dir) / "requeue-1"
    assert dump["env"]["HPCLAB_REQUEUE_SET_ASIDE"] == str(aside)
    assert (aside / "checkpoints" / "step-5" / "model.bin").read_bytes() == b"weights"
    assert json.loads((aside / "outputs" / "dump.json").read_text())["argv"] == ["--resume="]
    assert os.listdir(lay.checkpoint_dir) == []  # the rerun starts with empty dirs
    assert sorted(os.listdir(lay.output_dir)) == ["dump.json"]
    status = harness.status()
    assert status["restart_count"] == 1 and status["resumable"] is False
    assert status["resume_from"] is None and status["resume_source"] is None
    assert status["requeue_set_aside"] == str(aside)
    assert harness.env_json()["hpclab"]["requeue_set_aside"] == str(aside)
    assert harness.env_json()["hpclab"]["resumable"] is False
    assert "set aside" in harness.stderr()

    # a second requeue keeps the first one's evidence and never overwrites it
    assert harness.run(env={"PROG_MODE": "dump", "SLURM_RESTART_COUNT": "2"}) == 0, harness.stderr()
    assert (aside / "checkpoints" / "step-5" / "model.bin").exists()
    assert (Path(lay.dir) / "requeue-2" / "outputs" / "dump.json").exists()
    assert not (Path(lay.dir) / "requeue-2" / "checkpoints").exists()  # it was empty: nothing to move


def test_requeue_of_a_non_resumable_program_gets_a_fresh_work_dir(harness: Harness) -> None:
    """Declared outputs may live in the work dir (a relative path): a from-scratch rerun gets a
    fresh work dir built from the snapshot, so the previous segment's files there can neither
    mix with the rerun's nor pass as its outputs; they move to requeue-<n>/work."""
    harness.write(outputs=[OutputDecl(name="results", path="results.txt")])
    assert harness.run(env={"PROG_MODE": "outputs"}) == 0, harness.stderr()
    lay = harness.layout
    assert (Path(lay.work) / "results.txt").read_text() == "R"
    assert harness.outputs()["missing_required"] == []

    assert harness.run(env={"PROG_MODE": "dump", "SLURM_RESTART_COUNT": "1"}) == 0, harness.stderr()

    aside = Path(lay.dir) / "requeue-1"
    assert (aside / "work" / "results.txt").read_text() == "R"  # the evidence is kept
    assert not (Path(lay.work) / "results.txt").exists()  # the rerun starts from the snapshot
    assert harness.outputs()["missing_required"] == ["results"]  # never the dead segment's file
    prog = Path(lay.work) / "prog.py"
    assert prog.stat().st_ino == (Path(harness.snapshot or "") / "prog.py").stat().st_ino
    assert harness.status()["requeue_set_aside"] == str(aside)
    assert "reusing work dir" not in harness.stderr()


def test_requeue_of_a_non_resumable_program_without_python_helper(harness: Harness) -> None:
    """The degraded (bash-only) status.json documents the set-aside too."""
    harness.write()
    assert harness.run(env={"PROG_MODE": "requeue"}) == 0, harness.stderr()
    env = {"SLURM_RESTART_COUNT": "1", "HPCLAB_HELPER_PYTHON": "none", "PROG_MODE": "dump"}
    assert harness.run(env=env) == 0, harness.stderr()
    status = harness.status()
    assert status["degraded"] is True and status["resumable"] is False
    assert status["requeue_set_aside"] == str(Path(harness.layout.dir) / "requeue-1")
    assert status["resume_from"] is None


def test_work_dir_is_a_read_only_hardlink_farm_with_writable_dirs(harness: Harness) -> None:
    snap = harness.publish({"pkg/mod.py": "x = 1\n"})
    harness.write()
    assert harness.run() == 0, harness.stderr()

    work = Path(harness.layout.work)
    assert (work / "pkg" / "mod.py").stat().st_ino == (snap / "pkg" / "mod.py").stat().st_ino
    assert not os.access(work / "pkg" / "mod.py", os.W_OK)
    assert os.access(work / "pkg", os.W_OK)
    assert not (work / ".hpclab-complete").exists()
    assert (snap / ".hpclab-complete").exists()
    assert not os.access(snap / "pkg" / "mod.py", os.W_OK)  # the snapshot's mode never changes


# ---------------------------------------------------------------------------
# setup, venv, preparation failures
# ---------------------------------------------------------------------------


def test_setup_lines_run_in_order_in_the_code_dir(harness: Harness) -> None:
    harness.write(
        runtime={
            "setup": ['export FROM_SETUP="$PWD"', "set -e", "cd /", "export SECOND=$FROM_SETUP-2"],
        }
    )
    assert harness.run() == 0, harness.stderr()
    env = harness.dump()["env"]
    assert env["FROM_SETUP"] == harness.layout.work
    assert env["SECOND"] == f"{harness.layout.work}-2"
    assert harness.dump()["cwd"] == harness.layout.work  # the program still starts in the code dir


def test_failing_setup_line_stops_before_the_program(harness: Harness) -> None:
    harness.write(runtime={"setup": ["true", "false", "touch should-not-run"]})
    assert harness.run() == EXIT_SETUP_FAILED
    status = harness.status()
    assert status["failed_phase"] == "setup" and status["exit_code"] == EXIT_SETUP_FAILED
    assert "setup line 2" in status["error"]
    assert not (Path(harness.layout.work) / "should-not-run").exists()
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()


def test_exit_inside_setup_is_still_recorded(harness: Harness) -> None:
    harness.write(runtime={"setup": ["exit 5"]})
    assert harness.run() == 5
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 5 and status["failed_phase"] == "setup"


def test_venv_is_activated_before_setup(harness: Harness, tmp_path: Path) -> None:
    venv = tmp_path / "venvs" / "proj"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'export VIRTUAL_ENV="{venv}"\nexport ACTIVATED=yes\n')
    harness.write(runtime={"venv": str(venv), "setup": ['test "$ACTIVATED" = yes']})
    assert harness.run() == 0, harness.stderr()
    assert harness.dump()["env"]["VIRTUAL_ENV"] == str(venv)


def test_missing_venv_fails_fast_with_guard_marker(harness: Harness, tmp_path: Path) -> None:
    harness.write(runtime={"venv": str(tmp_path / "no-such-venv")})
    assert harness.run() == EXIT_GUARD_FAILED
    assert "HPCLAB_GUARD_FAILED:venv\n" in harness.stderr()
    status = harness.status()
    assert status["guard_failed"] == "venv" and status["failed_phase"] == "guard"


def test_incomplete_snapshot_is_refused(harness: Harness) -> None:
    harness.publish(complete=False)
    harness.write()
    assert harness.run() == EXIT_PREPARE_FAILED
    status = harness.status()
    assert status["failed_phase"] == "prepare" and ".hpclab-complete" in status["error"]


def test_missing_entrypoint_workdir_is_a_prepare_failure(harness: Harness) -> None:
    harness.write(workdir="no/such/dir")
    assert harness.run() == EXIT_PREPARE_FAILED
    assert "no/such/dir" in harness.status()["error"]


# ---------------------------------------------------------------------------
# GPU guard (fake nvidia-smi / fake torch)
# ---------------------------------------------------------------------------


def gpu_env(
    harness: Harness, tmp_path: Path, *, smi: bool = True, torch: dict | None = None
) -> dict[str, str]:
    env = {"PATH": f"{fake_bin_dir(tmp_path / 'bin', nvidia_smi=smi)}:{harness.base_env['PATH']}"}
    if torch is not None:
        env["PYTHONPATH"] = str(fake_torch_dir(tmp_path / "site", **torch))
    return env


def test_guard_fails_fast_without_nvidia_smi(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    assert harness.run(env=gpu_env(harness, tmp_path, smi=False)) == EXIT_GUARD_FAILED
    assert "HPCLAB_GUARD_FAILED:nvidia_smi\n" in harness.stderr()
    status = harness.status()
    assert status["guard_failed"] == "nvidia_smi" and status["exit_code"] == EXIT_GUARD_FAILED
    assert harness.env_json()["guard"]["failed"] == "nvidia_smi"
    assert not (Path(harness.layout.output_dir) / "dump.json").exists()


def test_guard_explains_a_torch_cuda_build_newer_than_the_driver(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    env = gpu_env(harness, tmp_path, torch={"version": "2.8.0+cu130", "cuda": "13.0", "available": False})
    assert harness.run(env=env) == EXIT_GUARD_FAILED
    assert "HPCLAB_GUARD_FAILED:torch_cuda_available" in harness.stderr()
    status = harness.status()
    assert status["guard_failed"] == "torch_cuda_available"
    assert "whl/cu128" in status["error"] and "12.8" in status["error"]
    torch = harness.env_json()["torch"]
    assert torch["version"] == "2.8.0+cu130" and "driver on your system is too old" in torch["cuda_error"]


def test_guard_rejects_a_gpu_architecture_missing_from_the_torch_build(
    harness: Harness, tmp_path: Path
) -> None:
    harness.write(resources={"gpus": 1})
    torch = {
        "cuda": "12.4",
        "available": True,
        "arch_list": ["sm_80", "sm_90"],
        "name": "NVIDIA B200",
        "cap": (10, 0),
    }
    assert harness.run(env=gpu_env(harness, tmp_path, torch=torch)) == EXIT_GUARD_FAILED
    assert harness.status()["guard_failed"] == "torch_gpu_arch"
    assert "sm_100" in harness.status()["error"]


def test_guard_passes_with_binary_compatible_arch_and_records_nvidia_info(
    harness: Harness, tmp_path: Path
) -> None:
    harness.write(resources={"gpus": 1})
    torch = {
        "cuda": "12.8",
        "available": True,
        "arch_list": ["sm_80", "sm_90"],
        "name": "NVIDIA A10",
        "cap": (8, 6),
    }
    assert harness.run(env=gpu_env(harness, tmp_path, torch=torch)) == 0, harness.stderr()
    env = harness.env_json()
    assert env["guard"]["status"] == "pass"
    assert env["nvidia_smi"]["driver_version"] == "570.86.15" and env["nvidia_smi"]["cuda_version"] == "12.8"
    assert env["nvidia_smi"]["gpus"][0]["memory_total_mib"] == 81559.0
    assert env["torch"]["device_capabilities"] == [[8, 6]]


def test_guard_without_torch_is_unchecked_not_failed(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})
    assert harness.run(env=gpu_env(harness, tmp_path)) == 0, harness.stderr()
    checks = {c["check"]: c["status"] for c in harness.env_json()["guard"]["checks"]}
    assert checks == {"nvidia_smi": "pass", "torch_cuda_available": "unchecked"}


def test_disabled_guard_only_warns(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1}, runtime={"guard": False})
    assert harness.run(env=gpu_env(harness, tmp_path, smi=False)) == 0, harness.stderr()
    env = harness.env_json()
    assert env["guard"]["enabled"] is False and env["guard"]["status"] == "fail"
    assert any("nvidia_smi" in w for w in env["warnings"])
    assert "WARNING: nvidia_smi" in harness.stderr()


# ---------------------------------------------------------------------------
# telemetry sidecar, probe timeouts, degraded mode
# ---------------------------------------------------------------------------


def test_heartbeat_and_gpu_telemetry_sidecar(harness: Harness, tmp_path: Path) -> None:
    harness.write(resources={"gpus": 1})  # CPU-only attempts sample no GPUs (see review tests)
    env = {
        **gpu_env(harness, tmp_path),
        "HPCLAB_TELEMETRY_INTERVAL_S": "1",
        "PROG_MODE": "sleep",
        "PROG_SLEEP": "2.5",
    }
    assert harness.run(env=env) == 0, harness.stderr()

    beats = Path(harness.layout.heartbeat).read_text().split()
    assert len(beats) >= 2 and all(b.isdigit() for b in beats)
    rows = Path(harness.layout.gpu_telemetry).read_text().splitlines()
    assert len(rows) >= 2 and rows[0].split(", ")[3] == "97"


def test_hanging_probe_is_bounded_and_not_fatal(harness: Harness, tmp_path: Path) -> None:
    slow = tmp_path / "slowpython"
    slow.write_text("#!/bin/bash\nsleep 30\n")
    slow.chmod(0o755)
    harness.write(runtime={"python": str(slow)}, argv=[sys.executable, "prog.py"])
    t0 = time.monotonic()
    assert harness.run(env={"HPCLAB_PROBE_TIMEOUT_S": "1"}) == 0, harness.stderr()
    assert time.monotonic() - t0 < 20
    env = harness.env_json()
    assert env["python"] is None
    assert any("timed out" in e for e in env["errors"])


def test_degraded_mode_without_python_helper(harness: Harness) -> None:
    harness.write(
        argv=[sys.executable, "prog.py", "a b", "$X"],
        outputs=[
            OutputDecl(name="metrics", path="{output_dir}/metrics.jsonl"),
            OutputDecl(name="preds", path="{output_dir}/preds", kind="dir"),
            OutputDecl(name="gone", path="{output_dir}/gone.txt"),
        ],
    )
    rc = harness.run(env={"HPCLAB_HELPER_PYTHON": "none", "PROG_MODE": "outputs"})
    assert rc == 0, harness.stderr()

    assert harness.dump()["argv"] == ["a b", "$X"]
    status = harness.status()
    assert status["degraded"] is True and status["state"] == "exited" and status["exit_code"] == 0
    assert status["token"] == TOKEN and status["signals_received"] == []
    env = harness.env_json()
    assert env["degraded"] is True and env["slurm"]["job_id"] == "4242" and env["python"] is None
    outputs = harness.outputs()
    files = {f["name"]: f for f in outputs["files"]}
    assert files["metrics"]["sha256"] == sha(b'{"step": 1}\n')
    assert files["preds"]["exists"] is True and files["preds"]["skipped"] == "no_python"
    assert outputs["missing_required"] == ["gone"]


def test_degraded_mode_records_a_killed_program(harness: Harness) -> None:
    harness.write()
    proc = harness.start(env={"HPCLAB_HELPER_PYTHON": "none", "PROG_MODE": "sleep"})
    assert harness.signal_when_ready(proc, signal.SIGTERM) == 143
    status = harness.status()
    assert (
        status["state"] == "killed" and status["signal"] == "TERM" and status["signals_received"] == ["TERM"]
    )
