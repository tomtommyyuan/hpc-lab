"""Pure tests: attempt layout, job identity, #SBATCH directives, quoting, validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from jobscript_support import ATTEMPT_ID, BASH, RUN_ID, TOKEN, Harness
from pydantic import ValidationError

from hpclab.domain.errors import InvalidArgument, Unsupported
from hpclab.domain.models import InputRef, ResilienceSpec
from hpclab.services.jobscript import (
    AttemptLayout,
    attempt_layout,
    bash_word,
    job_name_for,
    render_job_script,
    token_from_comment,
    token_from_job_name,
)

# ---------------------------------------------------------------------------
# layout + identity
# ---------------------------------------------------------------------------


def test_attempt_layout_matches_the_shared_remote_layout() -> None:
    lay = attempt_layout("/scratch/u/hpclab/", "att_01ABC")
    d = "/scratch/u/hpclab/attempts/att_01ABC"
    assert lay == AttemptLayout(
        dir=d,
        script=f"{d}/job.sbatch",
        stdout=f"{d}/logs/stdout.log",
        stderr=f"{d}/logs/stderr.log",
        status_json=f"{d}/status.json",
        env_json=f"{d}/env.json",
        outputs_manifest=f"{d}/outputs.json",
        receipt=f"{d}/submit.receipt",
        work=f"{d}/work",
        output_dir=f"{d}/outputs",
        checkpoint_dir=f"{d}/checkpoints",
        token_file=f"{d}/hpclab-plan.json",
    )
    assert lay.heartbeat == f"{d}/telemetry/heartbeat"
    assert lay.gpu_telemetry == f"{d}/telemetry/gpu.csv"
    assert lay.stop_file == f"{d}/STOP"
    assert lay.dirs_to_create() == (d, f"{d}/logs")


@pytest.mark.parametrize(
    ("root", "attempt"),
    [
        ("relative/root", "att_1"),
        ("/has space", "att_1"),
        ("/ok", "../escape"),
        ("/ok", "a/b"),
        ("/a/../b", "att_1"),
    ],
)
def test_attempt_layout_rejects_unsafe_values(root: str, attempt: str) -> None:
    with pytest.raises(InvalidArgument):
        attempt_layout(root, attempt)


def test_job_name_roundtrip_and_limits() -> None:
    name = job_name_for("lr=0.1 method/mixup", TOKEN)
    assert name == "hl.lr-0.1-method-mixup.k7m2p9q4"
    assert token_from_job_name(name) == TOKEN[:8]
    long_name = job_name_for("x" * 500, TOKEN)
    assert len(long_name) <= 64 and token_from_job_name(long_name) == TOKEN[:8]


@pytest.mark.parametrize(
    "name", ["", "train", "hl.x", "hl.x.SHORT", "other.x.abcdefgh", "hl.x.abcdefgh.extra!"]
)
def test_token_from_foreign_job_names_is_none(name: str) -> None:
    assert token_from_job_name(name) is None


def test_token_from_comment() -> None:
    assert token_from_comment(f"hpclab:{TOKEN}") == TOKEN
    assert token_from_comment(None) is None
    assert token_from_comment("someone else's comment") is None
    assert token_from_comment("hpclab:NOT A TOKEN") is None


def test_job_name_requires_a_real_token() -> None:
    with pytest.raises(InvalidArgument):
        job_name_for("x", "BAD TOKEN")


# ---------------------------------------------------------------------------
# #SBATCH directives
# ---------------------------------------------------------------------------


def directives(script: str) -> list[str]:
    return [line.removeprefix("#SBATCH ") for line in script.splitlines() if line.startswith("#SBATCH ")]


def test_directives_carry_identity_paths_and_resources(harness: Harness) -> None:
    lay = harness.layout
    script = harness.render(
        resources={
            "gpus": 2,
            "gpu_type": "h100",
            "cpus": 8,
            "mem": "64GB",
            "time": "90",
            "nodes": 1,
            "exclude_nodes": ["node[1-2]", "node7"],
            "constraint": "ib&h100",
            "extra_sbatch": ["--mail-type=END", "--hint=nomultithread"],
        }
    )
    assert script.startswith("#!/bin/bash\n#SBATCH ")
    assert directives(script) == [
        "--job-name=hl.test.k7m2p9q4",
        f"--comment=hpclab:{TOKEN}",
        f"--output={lay.stdout}",
        f"--error={lay.stderr}",
        "--open-mode=append",
        f"--chdir={lay.dir}",
        "--account=lab",
        "--partition=gpu",
        "--gres=gpu:h100:2",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=01:30:00",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--exclude=node[1-2],node7",
        "--constraint=ib&h100",
        "--mail-type=END",
        "--hint=nomultithread",
    ]


def test_spec_values_override_cluster_defaults_and_gpu_type_is_optional(harness: Harness) -> None:
    run_spec, plan = harness.specs(
        resources={"account": "other", "qos": "high", "partition": "scavenge", "gpus": 1}
    )
    script = render_job_script(
        run_spec=run_spec,
        plan=plan,
        cluster=Harness.cluster(account="lab", qos="normal"),
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        token=TOKEN,
        job_name=job_name_for("t", TOKEN),
        layout=harness.layout,
    )
    flags = directives(script)
    assert "--account=other" in flags and "--qos=high" in flags and "--partition=scavenge" in flags
    assert "--gres=gpu:1" in flags


def test_overlay_qos_default_and_no_gres_without_gpus(harness: Harness) -> None:
    run_spec, plan = harness.specs()
    script = render_job_script(
        run_spec=run_spec,
        plan=plan,
        cluster=Harness.cluster(account="lab", qos="normal"),
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        token=TOKEN,
        job_name=job_name_for("t", TOKEN),
        layout=harness.layout,
    )
    flags = directives(script)
    assert "--qos=normal" in flags
    assert not any(f.startswith("--gres") for f in flags)
    assert not any(f.startswith("--mem") for f in flags)


@pytest.mark.parametrize(
    ("mem", "expected"), [("512M", "512M"), ("64G", "64G"), ("1.5G", "1536M"), ("100", "100")]
)
def test_memory_is_normalized_for_sbatch(harness: Harness, mem: str, expected: str) -> None:
    assert f"--mem={expected}" in directives(harness.render(resources={"mem": mem}))


def test_resilience_adds_batch_signal_and_requeue_flags(harness: Harness) -> None:
    res = ResilienceSpec(signal="USR1", lead_s=600, stop_behavior="checkpoint_and_exit", resumable=True)
    flags = directives(harness.render(resilience=res, requeue=True))
    assert "--signal=B:USR1@600" in flags
    assert "--requeue" in flags
    assert "--no-requeue" in directives(harness.render(requeue=False))
    assert not any("requeue" in f or "signal" in f for f in directives(harness.render()))


def test_resilience_without_checkpoint_behaviour_adds_no_signal(harness: Harness) -> None:
    res = ResilienceSpec(stop_behavior="none")
    assert not any(f.startswith("--signal") for f in directives(harness.render(resilience=res)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resources": {"partition": "gpu part"}},
        {"resources": {"account": "lab;rm"}},
        {"resources": {"constraint": "a b"}},
        {"resources": {"exclude_nodes": ["node 1"]}},
        {"resources": {"extra_sbatch": ["--mail-user=a b"]}},
        {"resources": {"extra_sbatch": ["--no-requeue"]}, "requeue": True},
        {
            "resources": {"extra_sbatch": ["--signal=USR2@60"]},
            "resilience": ResilienceSpec(stop_behavior="checkpoint_and_exit"),
        },
        {"runtime": {"env": {"HPCLAB_OUTPUT_DIR": "/elsewhere"}}},
        {"runtime": {"env": {"BAD-NAME": "x"}}},
        {"argv": ["python", "a\0b"]},
        {"workdir": "../outside"},
        {"runtime": {"lockfiles": ["/etc/passwd"]}},
        {"recovery_path": "relative/ckpt"},
        {"runtime": {"env": {"CACHE": "{home}/cache"}}},  # the test cluster has no home role
        {"runtime": {"venv": "{home}/venv"}},
        {"inputs": [InputRef(name="d", uri="{home}/data")]},
    ],
)
def test_unsafe_or_contradictory_values_are_rejected(harness: Harness, kwargs: dict) -> None:
    # Some of these are now rejected earlier, by the domain model (pydantic ValidationError);
    # the renderer keeps its own checks as defence in depth.
    with pytest.raises((InvalidArgument, ValidationError)):
        harness.render(**kwargs)


def test_container_runtime_is_unsupported_with_a_hint(harness: Harness) -> None:
    with pytest.raises(Unsupported) as err:
        harness.render(runtime={"container": "/images/torch.sif"})
    assert "apptainer" in err.value.details["hint"]


def test_plan_for_another_run_is_rejected(harness: Harness) -> None:
    run_spec, plan = harness.specs()
    with pytest.raises(InvalidArgument):
        render_job_script(
            run_spec=run_spec,
            plan=plan,
            cluster=Harness.cluster(),
            run_id="run_OTHER",
            attempt_id=ATTEMPT_ID,
            token=TOKEN,
            job_name=job_name_for("t", TOKEN),
            layout=harness.layout,
        )


def test_render_is_deterministic_and_embeds_provenance(harness: Harness) -> None:
    a = harness.render(params={"lr": 0.1})
    b = harness.render(params={"lr": 0.1})
    assert a == b
    assert f"# run {RUN_ID} | attempt {ATTEMPT_ID}" in a
    assert "hpclab-wrapper/1" in a


def test_attempt_paths_are_written_relative_to_the_attempt_dir(harness: Harness) -> None:
    """The long attempt-dir prefix appears once in the values block, and bash expands each
    derived path back to the layout's (the script must fit in one 64 KiB plan_script page)."""
    script = harness.render()
    start = script.index("# ---- values (generated) ----\n") + len("# ---- values (generated) ----\n")
    values = script[start : script.index("\n\n", start)]
    lay = harness.layout
    assert values.count(lay.dir) == 1
    assert 'HL_OUTPUT_DIR="$HL_ATTEMPT_DIR"/outputs' in values.splitlines()
    names = {
        "HL_ATTEMPT_DIR": lay.dir,
        "HL_LOGS_DIR": lay.logs_dir,
        "HL_WORK_DIR": lay.work,
        "HL_OUTPUT_DIR": lay.output_dir,
        "HL_CHECKPOINT_DIR": lay.checkpoint_dir,
        "HL_TELEMETRY_DIR": lay.telemetry_dir,
        "HL_INTERNAL_DIR": lay.internal_dir,
        "HL_STATUS_JSON": lay.status_json,
        "HL_ENV_JSON": lay.env_json,
        "HL_OUTPUTS_JSON": lay.outputs_manifest,
        "HL_HEARTBEAT": lay.heartbeat,
        "HL_GPU_CSV": lay.gpu_telemetry,
        "HL_STOP_FILE": lay.stop_file,
    }
    show = "".join(f'printf "%s\\n" "${name}"\n' for name in names)
    proc = subprocess.run([BASH, "-c", values + "\n" + show], capture_output=True, text=True, check=True)
    assert proc.stdout.splitlines() == list(names.values())


def test_rendered_script_passes_bash_syntax_check_with_hostile_values(
    harness: Harness, tmp_path: Path
) -> None:
    script = harness.render(
        argv=[
            "python",
            "x.py",
            'it\'s "quoted"',
            "$(rm -rf /)",
            "`id`",
            "a\nb",
            "EOF",
            "HPCLAB_HELPER_PY_EOF",
            "{",
        ],
        runtime={
            "setup": ["echo 'hi' && export A=1", "if true; then\n  echo multi-line\nfi"],
            "env": {"WEIRD": "$(nope) `nope` 'q' \"dq\" \\", "WANDB_DIR": "{output_dir}/wandb"},
        },
        params={"text": "HPCLAB_CONFIG_JSON_EOF\n'\"$"},
    )
    path = tmp_path / "job.sbatch"
    path.write_text(script)
    proc = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# bash_word: byte-exact values through real bash
# ---------------------------------------------------------------------------

STATIC = {
    "output_dir": "/r/att/outputs",
    "checkpoint_dir": "/r/att/checkpoints",
    "attempt_dir": "/r/att",
    "code_dir": "/r/att/work",
    "run_id": "run_1",
    "attempt_id": "att_1",
}


def eval_words(templates: list[str], env: dict[str, str]) -> list[str]:
    words = " ".join(bash_word(t, STATIC) for t in templates)
    script = f'set -u\nx=( {words} )\nprintf "%s\\0" "${{x[@]}}"\n'
    proc = subprocess.run(
        [BASH, "-c", script], capture_output=True, env={**os.environ, **env, "LC_ALL": "C"}, check=True
    )
    return proc.stdout.decode("utf-8").split("\0")[:-1]


def test_bash_word_is_byte_exact_for_hostile_strings() -> None:
    hostile = [
        "",
        " ",
        "a b",
        "it's",
        '"double"',
        "$HOME",
        "${PATH}",
        "$(id)",
        "`id`",
        "back\\slash",
        "tab\there",
        "new\nline",
        "*",
        "~",
        "-",
        "--",
        "[0]=x",
        "üñíçødé ✓",
        "{not_a_placeholder}",
        '{"json": 1}',
        "{{output_dir}}",
    ]
    assert eval_words(hostile, {}) == [
        *hostile[:-1],
        "{/r/att/outputs}",
    ]


def test_bash_word_renders_static_and_runtime_placeholders() -> None:
    templates = [
        "{output_dir}/x",
        "--job={job_id}",
        "{resume_from}",
        "{attempt_dir}:{run_id}:{code_dir}",
        "{job_id}",
    ]
    assert eval_words(templates, {"SLURM_JOB_ID": "77", "HPCLAB_RESUME_FROM": "/ck/step 5"}) == [
        "/r/att/outputs/x",
        "--job=77",
        "/ck/step 5",
        "/r/att:run_1:/r/att/work",
        "77",
    ]
    # unset runtime values become empty *arguments* (never disappear)
    assert eval_words(["{resume_from}", "a{job_id}b"], {}) == ["", "ab"]
