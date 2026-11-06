"""usage / digest / vault / backup / notify / events / sim reset / demo / onboard."""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from cli_support import Cli, env, err, ok

from hpclab.cli import commands
from hpclab.domain.models import ExperimentSpec

# ---------------------------------------------------------------------------
# usage / digest / vault / backup / notify / events
# ---------------------------------------------------------------------------


def test_usage_self_report(cli: Cli) -> None:
    cli.client.responses["clusters_list"] = env("clusters_list")
    cli.client.responses["usage_report"] = env("usage_report.self")
    r = cli.run("usage", "--since", "14d", "--by", "partition,state")
    assert r.exit_code == 0, r.text
    assert "visibility: full" in r.stdout and "your total: 6 job(s)" in r.stdout
    assert "allocated hours" in r.stdout
    p = cli.client.params("usage_report")
    assert p == {"cluster_id": "sim", "scope": "self", "since": "14d", "group_by": ["partition", "state"]}


def test_usage_account_self_only_never_shows_lab_totals(cli: Cli) -> None:
    cli.client.responses["usage_report"] = env("usage_report.account_self_only")
    r = cli.run("usage", "--cluster", "sim", "--account", "lab", "--by", "user,state")
    assert r.exit_code == 0, r.text
    assert "only your jobs are visible" in r.stdout
    assert "no lab totals" in r.stdout
    assert "account total" not in r.stdout and "lab total" not in r.stdout.replace("no lab totals", "")
    p = cli.client.params("usage_report")
    assert p["scope"] == "account" and p["account"] == "lab" and "clusters_list" not in cli.client.methods()


def test_usage_errors(cli: Cli) -> None:
    assert cli.run("usage", "--cluster", "sim", "--by", "color").exit_code == 2
    assert cli.run("usage", "--cluster", "sim", "--account", "lab", "--mine").exit_code == 2
    cli.client.responses["usage_report"] = err(
        "auth_required", "ssh master is gone", remediation="hpclab auth hai"
    )
    r = cli.run("usage", "--cluster", "hai")
    assert r.exit_code == 4 and "hpclab auth hai" in r.stderr
    cli.client.responses["clusters_list"] = ok({"clusters": []})
    r = cli.run("usage")
    assert r.exit_code == 1 and "no cluster registered" in r.stderr


def test_digest_stdout_file_and_json(cli: Cli, tmp_path: Path) -> None:
    d = env("lab_digest")
    cli.client.responses["lab_digest"] = d
    r = cli.run("digest", "--since", "7d", "--account", "lab", "--cluster", "sim")
    assert r.exit_code == 0
    assert r.stdout.startswith("# HPC Lab digest")  # raw Markdown when stdout is not a terminal
    assert cli.client.params("lab_digest") == {"since": "7d", "account": "lab", "cluster_id": "sim"}
    out = tmp_path / "week.md"
    r = cli.run("digest", "--out", str(out))
    assert out.read_text().startswith("# HPC Lab digest") and r.stdout == ""
    r = cli.run("--json", "digest")
    assert r.json == d


def test_vault_backup_notify(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["vault_status"] = env("vault_status")
    r = cli.run("vault", "--experiment", "exp_01M3K9Q1CC33H1R96NRPJEDAVW")
    assert r.exit_code == 0 and "fully captured" in r.stdout
    assert cli.client.params("vault_status") == {"experiment_id": "exp_01M3K9Q1CC33H1R96NRPJEDAVW"}

    cli.client.responses["backup"] = env("backup")
    r = cli.run("backup", str(tmp_path / "bk"), "--no-vault")
    assert r.exit_code == 0 and "integrity" in r.stdout
    p = cli.client.params("backup")
    assert p["dest"] == str((tmp_path / "bk").resolve()) and p["include_vault"] is False
    assert p["idempotency_key"].startswith("backup-")

    cli.client.responses["notify_test"] = env("notify_test")
    r = cli.run("notify", "test", "--cluster", "sim")
    assert r.exit_code == 0 and "notify:" in r.stdout
    failed = env("notify_test")
    failed["data"].update(
        channels=[
            {
                "channel_id": "c1",
                "kind": "ntfy",
                "target": "https://ntfy.sh/x",
                "clusters": [],
                "events": [],
                "ok": False,
                "error": "HTTP 500",
            }
        ],
        sent=0,
        failed=1,
    )
    cli.client.responses["notify_test"] = failed
    r = cli.run("notify", "test")
    assert r.exit_code == 1 and "FAILED: HTTP 500" in r.stdout


def test_events_filters_and_paging(cli: Cli) -> None:
    e = env("events_list")
    e["next_cursor"] = "abc"
    cli.client.responses["events_list"] = e
    r = cli.run("events", "--entity", "att_01M3K9Q7QY2C92358BRYJ7C2R7", "--type", "attempt.", "--limit", "4")
    assert r.exit_code == 0, r.text
    assert e["data"]["events"][0]["event_type"] in r.stdout
    assert "more: --cursor abc" in r.stdout
    assert cli.client.params("events_list") == {
        "entity_id": "att_01M3K9Q7QY2C92358BRYJ7C2R7",
        "event_type": "attempt.",
        "limit": 4,
    }
    cli.client.responses["events_list"] = ok({"events": [], "entity_id": None, "event_type": None})
    assert "no events match" in cli.run("events").stdout


def test_controller_unavailable_is_exit_3_everywhere(cli: Cli) -> None:
    cli.client.unavailable = True
    for args in (
        ("experiments",),
        ("events",),
        ("usage", "--cluster", "sim"),
        ("grid", "exp_01M3K9Q1CC33H1R96NRPJEDAVW"),
    ):
        r = cli.run(*args)
        assert r.exit_code == 3, args
        assert "controller unavailable" in r.stderr and "hpclab controller start" in r.stderr


# ---------------------------------------------------------------------------
# sim reset
# ---------------------------------------------------------------------------


def test_sim_reset_kills_jobs_deletes_read_only_snapshots_and_reinitializes(
    cli: Cli, hpclab_home: Path
) -> None:
    from hpclab.sim.cluster import SimCluster

    root = hpclab_home / "sim" / "sim"
    SimCluster.ensure(root)
    snap = root / "fs" / "scratch" / "alice" / "hpclab" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "train.py").write_text("print(1)\n")
    (snap / "train.py").chmod(0o444)
    snap.chmod(0o555)
    cli.client.responses["init_sim"] = env("init_sim")
    r = cli.run("sim", "reset")
    assert r.exit_code == 2 and snap.exists()  # refused without --yes: nothing deleted
    try:
        r = cli.run("sim", "reset", "--yes")
        assert r.exit_code == 0, r.text
        assert not (root / "fs" / "scratch" / "alice").exists()
        assert (root / "state.db").exists() and (root / "bin" / "sbatch").exists()
        assert "init_sim" in cli.client.methods()
        assert "reset" in r.stdout
    finally:
        if snap.exists():
            commands.make_writable(root)


def test_sim_commands_use_the_relocated_simulator(cli: Cli, hpclab_home: Path, tmp_path: Path) -> None:
    """init_sim places the simulator outside a home whose path job scripts cannot carry and
    records it in the overlay: `hpclab sim ...` must act on that one, not on <home>/sim/sim."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path
    from hpclab.sim.cluster import SimCluster

    relocated = tmp_path / "relocated-sim"
    SimCluster.ensure(relocated)
    (relocated / "fs" / "marker").write_text("x")
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {relocated}\n")
    ov.chmod(0o600)
    r = cli.run("sim", "jobs")
    assert r.exit_code == 0, r.text  # <home>/sim/sim does not exist: it would say "no simulator"
    r = cli.run("sim", "fault", "sbatch_lose_reply=1")
    assert r.exit_code == 0, r.text
    assert SimCluster(relocated).faults().sbatch_lose_reply == 1
    cli.client.responses["init_sim"] = env("init_sim")
    r = cli.run("sim", "reset", "--yes")
    assert r.exit_code == 0, r.text
    assert not (relocated / "fs" / "marker").exists() and (relocated / "state.db").exists()
    assert str(relocated) in r.stdout and not (hpclab_home / "sim" / "sim").exists()
    # a relocated simulator lives under a shared, predictable /var/tmp name: re-created private
    assert stat.S_IMODE(os.lstat(relocated).st_mode) == 0o700


def test_sim_fault_unknown_key_or_bad_value_is_a_usage_error(
    cli: Cli, hpclab_home: Path, tmp_path: Path
) -> None:
    """A typo in a fault name is a usage error listing the valid names, never a traceback,
    and nothing is written to the simulator."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path
    from hpclab.sim.cluster import SimCluster
    from hpclab.sim.config import SimFaults

    relocated = tmp_path / "relocated-sim"
    SimCluster.ensure(relocated)
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {relocated}\n")
    ov.chmod(0o600)

    r = cli.run("sim", "fault", "sbatch_loose_reply=1", "preempt_next=1")
    assert r.exit_code == 2, r.text
    assert "Traceback" not in r.text
    assert "unknown fault 'sbatch_loose_reply'" in r.stderr
    for key in SimFaults.field_names():
        assert key in r.stderr
    assert SimCluster(relocated).faults().preempt_next == 0  # all-or-nothing

    for bad in ("preempt_next=-1", "preempt_next=lots", "preempt_next=null", "auth_required=maybe"):
        r = cli.run("sim", "fault", bad)
        assert r.exit_code == 2, (bad, r.text)
        assert "Traceback" not in r.text and "usage" in r.stderr, bad
    assert SimCluster(relocated).faults() == SimFaults()

    r = cli.run("--json", "sim", "fault", "nope=1")
    assert r.exit_code == 2
    doc = json.loads(r.stdout)
    assert doc["ok"] is False and doc["error"]["code"] == "invalid_argument"


def test_sim_fault_rejects_values_that_would_be_silently_mangled(
    cli: Cli, hpclab_home: Path, tmp_path: Path
) -> None:
    """A count must be a whole number (1.7 is not 1, true is not 1) and a duration or
    utilization a finite number: NaN/Infinity make every comparison in the simulator false."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path
    from hpclab.sim.cluster import SimCluster
    from hpclab.sim.config import SimFaults

    relocated = tmp_path / "relocated-sim"
    SimCluster.ensure(relocated)
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {relocated}\n")
    ov.chmod(0o600)

    for bad in (
        "preempt_next=1.7",
        "sbatch_lose_reply=true",
        "accounting_lag_s=NaN",
        "min_job_age_s=Infinity",
        "gpu_util=-Infinity",
    ):
        r = cli.run("sim", "fault", bad)
        assert r.exit_code == 2, (bad, r.text)
        assert "Traceback" not in r.text and "usage" in r.stderr, bad
    assert SimCluster(relocated).faults() == SimFaults()

    # the forms people actually type keep working
    r = cli.run(
        "sim",
        "fault",
        "preempt_next=2",
        "oom_next=1.0",
        "accounting_lag_s=2.5",
        "min_job_age_s=null",
        "auth_required=false",
        "transport_timeout_cmds=squeue,sacct",
    )
    assert r.exit_code == 0, r.text
    f = SimCluster(relocated).faults()
    assert (f.preempt_next, f.oom_next, f.accounting_lag_s, f.min_job_age_s) == (2, 1, 2.5, None)
    assert f.transport_timeout_cmds == ["squeue", "sacct"] and f.auth_required is False


def test_sim_fault_clear_and_shutdown_print_one_json_document_with_json(
    cli: Cli, hpclab_home: Path, tmp_path: Path
) -> None:
    """--json prints one JSON document on stdout for the simulator commands too (a script that
    injects a fault with --json must be able to parse the reply), errors included."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path
    from hpclab.sim.cluster import SimCluster
    from hpclab.sim.config import SimFaults

    relocated = tmp_path / "relocated-sim"
    SimCluster.ensure(relocated)
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {relocated}\n")
    ov.chmod(0o600)

    r = cli.run("--json", "sim", "fault", "preempt_next=2", "transport_timeout_cmds=squeue")
    assert r.exit_code == 0, r.text
    doc = json.loads(r.stdout)
    assert doc["ok"] is True and doc["cluster_id"] == "sim"
    assert doc["faults"]["preempt_next"] == 2 and doc["faults"]["transport_timeout_cmds"] == ["squeue"]

    r = cli.run("--json", "sim", "clear")
    assert r.exit_code == 0, r.text
    doc = json.loads(r.stdout)
    assert doc["ok"] is True and doc["faults"] == SimFaults().to_dict()
    assert SimCluster(relocated).faults() == SimFaults()

    r = cli.run("--json", "sim", "shutdown")
    assert r.exit_code == 0, r.text
    doc = json.loads(r.stdout)
    assert doc == {"ok": True, "cluster_id": "sim", "jobs_stopped": 0}

    r = cli.run("--json", "sim", "fault", "preempt_next")  # no '=': a usage error, still JSON
    assert r.exit_code == 2
    doc = json.loads(r.stdout)
    assert doc["ok"] is False and doc["error"]["code"] == "invalid_argument"
    r = cli.run("sim", "fault", "preempt_next")
    assert r.exit_code == 2 and "expected key=value" in r.stderr and "Traceback" not in r.text

    ov.write_text(f"cluster_id: sim\nsim_root: {tmp_path / 'nowhere'}\n")
    for args in (("fault", "preempt_next=1"), ("clear",), ("shutdown",)):
        r = cli.run("--json", "sim", *args)
        assert r.exit_code == 1, (args, r.text)
        doc = json.loads(r.stdout)
        assert doc["ok"] is False and doc["error"]["code"] == "not_found", args
        assert "no simulator at" in doc["error"]["message"]


def test_sim_reset_refuses_a_simulator_path_planted_meanwhile(
    cli: Cli, hpclab_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between deleting and re-creating a relocated simulator, another user could plant a symlink
    under its predictable name: the reset must refuse it, never write the simulator through it."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path
    from hpclab.sim.cluster import SimCluster

    relocated = tmp_path / "relocated-sim"
    SimCluster.ensure(relocated)
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {relocated}\n")
    ov.chmod(0o600)
    elsewhere = tmp_path / "someone-elses"
    elsewhere.mkdir()
    real_rmtree = commands.shutil.rmtree

    def rmtree_then_plant(path: Any, *args: Any, **kwargs: Any) -> None:
        real_rmtree(path, *args, **kwargs)
        if Path(path) == relocated:
            relocated.symlink_to(elsewhere)

    monkeypatch.setattr(commands.shutil, "rmtree", rmtree_then_plant)
    cli.client.responses["init_sim"] = env("init_sim")
    r = cli.run("sim", "reset", "--yes")
    assert r.exit_code == 1, r.text
    assert "Traceback" not in r.text and "refusing to use" in r.stderr
    assert list(elsewhere.iterdir()) == []  # nothing was written through the planted link


def test_sim_reset_never_deletes_a_directory_that_is_not_a_simulator(
    cli: Cli, hpclab_home: Path, tmp_path: Path
) -> None:
    """The simulator root now comes from the cluster overlay (it may live outside the home): a
    mistaken or edited sim_root must never make `sim reset --yes` delete someone's directory."""
    from hpclab.config import get_paths
    from hpclab.profiles.loader import overlay_path

    project = tmp_path / "my-project"
    project.mkdir()
    (project / "paper.tex").write_text("precious\n")
    ov = overlay_path("sim", get_paths(str(hpclab_home)))
    ov.parent.mkdir(parents=True, exist_ok=True)
    ov.write_text(f"cluster_id: sim\nsim_root: {project}\n")
    ov.chmod(0o600)
    r = cli.run("sim", "reset", "--yes")
    assert r.exit_code == 1, r.text
    assert "not a simulator" in r.stderr
    assert (project / "paper.tex").read_text() == "precious\n"
    for args in (("jobs",), ("fault", "sbatch_lose_reply=1"), ("clear",), ("shutdown",)):
        r = cli.run("sim", *args)
        assert r.exit_code == 1 and "no simulator at" in r.stderr, (args, r.text)
    assert sorted(p.name for p in project.iterdir()) == ["paper.tex"]  # nothing created in it


def test_sim_reset_without_simulator_and_make_writable(cli: Cli, tmp_path: Path) -> None:
    r = cli.run("sim", "reset", "--yes")
    assert r.exit_code == 1 and "hpclab init --sim" in r.stderr
    d = tmp_path / "ro" / "deep"
    d.mkdir(parents=True)
    f = d / "x"
    f.write_text("x")
    link = tmp_path / "ro" / "link"
    target = tmp_path / "outside"
    target.write_text("y")
    target.chmod(0o444)
    link.symlink_to(target)
    f.chmod(0o444)
    d.chmod(0o555)
    commands.make_writable(tmp_path / "ro")
    assert os.stat(d).st_mode & stat.S_IWUSR and os.stat(f).st_mode & stat.S_IWUSR
    assert not os.stat(target).st_mode & stat.S_IWUSR  # symlink targets are not touched
    target.chmod(0o644)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def test_demo_delegates_to_run_demo(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def run_demo(paths: Any, *, console: Any, fast: bool, keep: bool) -> int:
        seen.update(paths=paths, fast=fast, keep=keep, console=console)
        return 3

    monkeypatch.setitem(sys.modules, "hpclab.demo", types.SimpleNamespace(run_demo=run_demo))
    r = cli.run("demo", "--fast", "--keep")
    assert r.exit_code == 3
    assert seen["fast"] is True and seen["keep"] is True
    assert seen["console"] is commands.console


def test_demo_missing_module_is_a_clear_error(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hpclab.demo", None)  # import raises ImportError
    r = cli.run("demo")
    assert r.exit_code == 1 and "not available" in r.stderr and "hpclab init --sim" in r.stderr
    assert "hpclab examples copy tiny-classifier" in r.stderr


def test_json_demo_is_a_usage_error(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setitem(
        sys.modules, "hpclab.demo", types.SimpleNamespace(run_demo=lambda *a, **k: called.append(True) or 0)
    )
    r = cli.run("--json", "demo")
    assert r.exit_code == 2 and not called
    assert "demo is interactive; --json is not supported" in r.stderr
    assert r.json["ok"] is False and r.json["error"]["code"] == "invalid_argument"


def test_demo_ctrl_c_does_not_send_the_user_to_reset_their_simulator(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_demo(paths: Any, *, console: Any, fast: bool, keep: bool) -> int:
        raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "hpclab.demo", types.SimpleNamespace(run_demo=run_demo))
    r = cli.run("demo")
    assert r.exit_code == 130
    assert "sim reset" not in r.text and "cleans up" in r.stderr


def test_demo_keep_hint_quotes_a_home_with_spaces(tmp_path: Path) -> None:
    """With --keep the tour prints how to reopen its home: one pasteable command even when the
    temp directory it fell back to has a space or a quote in its path."""
    import io
    import shlex

    from rich.console import Console

    from hpclab import demo as demo_mod
    from hpclab.config import get_paths

    buf = io.StringIO()
    home = tmp_path / "it's a dir" / "home"
    tour = demo_mod._Tour(
        console=Console(file=buf, width=400, color_system=None),
        fast=True,
        paths=get_paths(str(home)),
        client=None,  # type: ignore[arg-type]
    )
    demo_mod._next_steps(tour, True, "exp_01M3K9Q1CC33H1R96NRPJEDAVW")
    line = next(ln for ln in buf.getvalue().splitlines() if "demo home is kept" in ln)
    cmd = line.split("kept: ", 1)[1].split(" (or export", 1)[0]
    argv = shlex.split(cmd)
    assert argv == [
        "hpclab",
        "--home",
        str(get_paths(str(home)).home),
        "grid",
        "exp_01M3K9Q1CC33H1R96NRPJEDAVW",
    ]
    assert f"export HPCLAB_HOME={shlex.quote(str(get_paths(str(home)).home))}" in line


# ---------------------------------------------------------------------------
# onboard
# ---------------------------------------------------------------------------


def _onboard_client(cli: Cli, final: str = "ok") -> dict[str, Any]:
    seen: dict[str, Any] = {}
    cli.client.responses["cluster_inspect"] = env("cluster_inspect")
    ready = env("experiment_prepare.ready")
    run = dict(ready["data"]["runs"][1])
    ready["data"]["runs"] = [run]

    def prepare(p: dict[str, Any]) -> dict[str, Any]:
        spec = p["spec"]
        seen["spec"] = spec
        root = Path(spec["code"]["project_root"])
        seen["script_existed"] = (root / "hpclab_gpu_smoke.py").is_file()
        seen["script"] = (root / "hpclab_gpu_smoke.py").read_text()
        seen["root"] = root
        return ready

    cli.client.responses["experiment_prepare"] = prepare
    cli.client.responses["run_submit"] = env("run_submit")
    staging = env("run_inspect.pending")
    staging["data"]["attempts"][0].update(phase="staging", scheduler_state="UNKNOWN")
    running = env("run_inspect.pending")
    running["data"]["attempts"][0].update(phase="observing", scheduler_state="RUNNING")
    running["data"]["pending"] = None
    done = env("run_inspect.succeeded" if final == "ok" else "run_inspect.failed")
    cli.client.responses["run_inspect"] = [staging, running, done]
    cli.client.responses["run_logs"] = ok(
        {"text": "host=gpu-node\nCUDA_VISIBLE_DEVICES=0\nGPU 0: NVIDIA A100\n"}
    )
    return seen


def test_onboard_runs_a_smoke_job_and_reports_pass(cli: Cli) -> None:
    seen = _onboard_client(cli)
    r = cli.run("onboard", "sim", "--yes", "--partition", "gpu", "--gpu-type", "a100")
    assert r.exit_code == 0, r.text
    assert "PASS: sim ran a 1-GPU job end to end" in r.stdout
    assert "GPU 0: NVIDIA A100" in r.stdout
    spec = ExperimentSpec.model_validate(seen["spec"])  # the generated spec is a valid ExperimentSpec
    assert spec.cluster == "sim" and spec.resources.gpus == 1 and spec.resources.time == "00:10:00"
    assert spec.resources.partition == "gpu" and spec.resources.gpu_type == "a100"
    assert spec.outputs[0].path == "{output_dir}/metrics.json"
    assert seen["script_existed"] and "nvidia-smi" in seen["script"] and "hpclab.metrics/v1" in seen["script"]
    assert not seen["root"].exists()  # the temporary project is cleaned up
    assert [m for m in cli.client.methods() if m == "run_inspect"] == ["run_inspect"] * 3
    assert all(p.get("refresh") for m, p in cli.client.calls if m == "run_inspect")
    compile(seen["script"], "hpclab_gpu_smoke.py", "exec")


def test_onboard_reports_failure_with_diagnosis(cli: Cli) -> None:
    _onboard_client(cli, final="failed")
    r = cli.run("onboard", "sim", "--yes")
    assert r.exit_code == 1
    assert "FAIL: the smoke job ended NODE_FAIL" in r.stdout and "Node failure" in r.stdout


def test_onboard_needs_confirmation_and_stops_on_failed_inspection(cli: Cli) -> None:
    _onboard_client(cli)
    r = cli.run("onboard", "sim")
    assert r.exit_code == 2 and "experiment_prepare" not in cli.client.methods()
    bad = env("cluster_inspect")
    bad["data"]["status"] = "fail"
    cli.client.responses["cluster_inspect"] = bad
    r = cli.run("onboard", "sim", "--yes")
    assert r.exit_code == 1 and "cluster check failed" in r.stderr
    assert "experiment_prepare" not in cli.client.methods()


def test_onboard_times_out_honestly(cli: Cli) -> None:
    _onboard_client(cli)
    pending = env("run_inspect.pending")
    cli.client.responses["run_inspect"] = pending
    r = cli.run("onboard", "sim", "--yes", "--timeout", "20", "--interval", "5")
    assert r.exit_code == 1
    assert "did not finish within 20s (it keeps running)" in r.stderr


# ---------------------------------------------------------------------------
# mcp config
# ---------------------------------------------------------------------------


def test_mcp_config_quotes_paths_with_spaces_and_passes_the_home(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A home or an install path with spaces/quotes: the Claude Code line must be ONE pasteable
    shell command that also passes HPCLAB_HOME, and the JSON config must not split the path."""
    import json
    import shlex

    from hpclab.cli import main as cli_main
    from hpclab.config import get_paths

    home = tmp_path / "it's my home" / "hpclab"
    exe = str(tmp_path / "My Tools" / "bin" / "hpclab")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: exe)
    r = cli.run("--home", str(home), "mcp", "config")
    assert r.exit_code == 0, r.text
    want_home = str(get_paths(str(home)).home)
    line = next(ln for ln in r.stdout.splitlines() if "claude mcp add" in ln).strip()
    argv = shlex.split(line)
    assert argv[:4] == ["claude", "mcp", "add", "hpclab"]
    assert f"HPCLAB_HOME={want_home}" in argv
    assert argv[argv.index("--") + 1 :] == [exe, "mcp", "serve"]
    server = json.loads(r.stdout[r.stdout.index("{") :])["mcpServers"]["hpclab"]
    assert server["command"] == exe and server["args"] == ["mcp", "serve"]
    assert server["env"] == {"HPCLAB_HOME": want_home}


def test_mcp_config_without_hpclab_on_path_uses_this_interpreter(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shlex

    from hpclab.cli import main as cli_main

    monkeypatch.setattr(cli_main.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli_main.sys, "executable", "/opt/my python/bin/python3")
    r = cli.run("mcp", "config")
    assert r.exit_code == 0, r.text
    line = next(ln for ln in r.stdout.splitlines() if "claude mcp add" in ln).strip()
    argv = shlex.split(line)
    assert argv[argv.index("--") + 1 :] == ["/opt/my python/bin/python3", "-m", "hpclab.cli", "mcp", "serve"]
    server = json.loads(r.stdout[r.stdout.index("{") :])["mcpServers"]["hpclab"]
    assert server["command"] == "/opt/my python/bin/python3"
    assert server["args"] == ["-m", "hpclab.cli", "mcp", "serve"]
