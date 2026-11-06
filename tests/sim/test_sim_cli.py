"""Fake Slurm commands, exercised in-process: output formats, validation texts, admin plane.

None of these tests start a job, so they run without the ``sim`` marker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hpclab.sim.cluster import SimCluster

FS = "\x1f"


def test_shims_are_installed_and_executable(sim: SimCluster) -> None:
    for command in ("sbatch", "squeue", "sacct", "scancel", "sinfo", "sacctmgr", "sstat", "srun", "scontrol"):
        path = sim.bin_dir / command
        assert path.exists() and path.stat().st_mode & 0o111
    done = sim.run("squeue", "--version")
    assert done.returncode == 0 and done.stdout.strip() == "slurm 23.02.7"
    again = SimCluster.ensure(sim.root)  # idempotent
    assert again.root == sim.root and again.config == sim.config


def test_cli_requires_a_simulator(tmp_path: Path) -> None:
    from hpclab.sim import cli

    assert cli.main(["squeue"], env={}) == 2
    assert cli.main(["--root", str(tmp_path / "nope"), "squeue"], env={}) == 2


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["-p", "nope"],
            "sbatch: error: invalid partition specified: nope\n"
            "sbatch: error: Batch job submission failed: Invalid partition name specified\n",
        ),
        (
            ["-A", "ghost"],
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified\n",
        ),
        (
            ["-p", "interactive", "-t", "5:00:00"],
            "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)\n",
        ),
        (
            ["--gres=gpu:v100:1"],
            "sbatch: error: Batch job submission failed: Invalid generic resource (gres) specification\n",
        ),
        (
            ["--gres=gpu:a100:8"],
            "sbatch: error: Batch job submission failed: Requested node configuration is not available\n",
        ),
        (
            ["--mem=2T"],
            "sbatch: error: Memory specification can not be satisfied\n"
            "sbatch: error: Batch job submission failed: Requested node configuration is not available\n",
        ),
        (
            ["-p", "interactive", "--gres=gpu:2"],
            "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
            "(job submit limit, user's size and/or time limits)\n",
        ),
        (["--qos=vip"], "sbatch: error: Batch job submission failed: Invalid qos specification\n"),
        (
            ["--dependency=afterok:424242"],
            "sbatch: error: Batch job submission failed: Job dependency problem\n",
        ),
        (["-t", "soon"], "sbatch: error: Invalid --time specification\n"),
        (["--bogus"], "sbatch: unrecognized option '--bogus'\nTry \"sbatch --help\" for more information\n"),
        (["--constraint=tpu"], "sbatch: error: Batch job submission failed: Invalid feature specification\n"),
        (["-w", "ghost-01"], "sbatch: error: Batch job submission failed: Invalid node name specified\n"),
    ],
)
def test_sbatch_rejections_use_real_texts(
    sim: SimCluster, run_cli, write_script, args: list[str], expected: str
) -> None:
    script = write_script("echo hi")
    res = run_cli("sbatch", "--parsable", *args, str(script), cwd=script.parent)
    assert (res.rc, res.stdout, res.stderr) == (1, "", expected)
    assert sim.jobs() == []


def test_sbatch_script_problems(sim: SimCluster, run_cli, workdir: Path) -> None:
    res = run_cli("sbatch", "missing.sbatch", cwd=workdir)
    assert res.rc == 1 and res.stderr == "sbatch: error: Unable to open file missing.sbatch\n"
    (workdir / "noshebang.sh").write_text("echo hi\n")
    res = run_cli("sbatch", "noshebang.sh", cwd=workdir)
    assert res.rc == 1 and "This does not look like a batch script" in res.stderr
    res = run_cli("sbatch", "--wrap", "echo hi", "extra", cwd=workdir)
    assert res.rc == 1 and "Script arguments not permitted with --wrap option" in res.stderr
    assert sim.jobs() == []


def test_sbatch_reject_fault(sim: SimCluster, run_cli, write_script) -> None:
    sim.set_faults(
        sbatch_reject="sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"
    )
    res = run_cli("sbatch", str(write_script("echo hi")))
    assert res.rc == 1
    assert (
        res.stderr
        == "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified\n"
    )
    assert sim.jobs() == []


def test_sbatch_directives_are_validated_like_cli_options(sim: SimCluster, run_cli, write_script) -> None:
    script = write_script("echo hi", "--partition=nope")
    res = run_cli("sbatch", str(script))
    assert res.rc == 1 and "invalid partition specified: nope" in res.stderr
    res = run_cli("sbatch", "--test-only", "-p", "gpu", str(script))  # CLI wins over the directive
    assert res.rc == 0


def test_test_only_queues_nothing(sim: SimCluster, run_cli, write_script) -> None:
    res = run_cli(
        "sbatch", "--test-only", "--parsable", "--gres=gpu:a100:2", "-c", "4", str(write_script("echo hi"))
    )
    assert res.rc == 0 and res.stdout == ""
    m = re.fullmatch(
        r"sbatch: Job (\d+) to start at (\d+) using (\d+) processors on nodes (\S+) in partition (\S+)\n",
        res.stderr,
    )
    assert m is not None
    assert (m.group(1), m.group(3), m.group(4), m.group(5)) == ("1000", "4", "sim-a100-01", "gpu")
    assert sim.jobs() == []
    again = run_cli("sbatch", "--test-only", str(write_script("echo hi")))
    assert "Job 1000 to start" in again.stderr  # the id was not consumed


def test_squeue_on_empty_queue_and_unknown_ids(sim: SimCluster, run_cli) -> None:
    res = run_cli("squeue")
    assert res.rc == 0 and res.stdout.split() == [
        "JOBID",
        "PARTITION",
        "NAME",
        "USER",
        "ST",
        "TIME",
        "NODES",
        "NODELIST(REASON)",
    ]
    assert run_cli("squeue", "-h").stdout == ""
    single = run_cli("squeue", "-h", "--jobs=999999")
    assert (single.rc, single.stderr) == (1, "slurm_load_jobs error: Invalid job id specified\n")
    several = run_cli("squeue", "-h", "--jobs=999998,999999")
    assert (several.rc, several.stdout, several.stderr) == (0, "", "")
    bad = run_cli("squeue", "-h", "-O", "JobID,Bogus")
    assert bad.rc == 1 and "Invalid job format specification: Bogus" in bad.stderr


def test_squeue_unavailable_fault(sim: SimCluster, run_cli) -> None:
    sim.set_faults(squeue_unavailable=True)
    res = run_cli("squeue", "-h")
    assert (
        res.rc == 1
        and res.stderr == "slurm_load_jobs error: Unable to contact slurm controller (connect failure)\n"
    )


def test_sinfo_per_node_format(sim: SimCluster, run_cli) -> None:
    fmt = f"NodeList:0{FS},Partition:0{FS},StateLong:0{FS},Time:0{FS},Gres:0{FS},GresUsed:0{FS},Reason:0"
    res = run_cli("sinfo", "--noheader", "--Node", f"--Format={fmt}")
    rows = [line.split(FS) for line in res.stdout.splitlines()]
    assert len(rows) == 7
    assert rows[0] == [
        "sim-a100-01",
        "gpu*",
        "idle",
        "2-00:00:00",
        "gpu:a100:4(S:0-1)",
        "gpu:a100:0(IDX:N/A)",
        "none",
    ]
    assert [
        "sim-int-01",
        "interactive",
        "idle",
        "4:00:00",
        "gpu:a100:2(S:0-1)",
        "gpu:a100:0(IDX:N/A)",
        "none",
    ] in rows
    sim.set_node_state("sim-h100-02", "drain", "bad NVLink")
    rows = [
        line.split(FS)
        for line in run_cli("sinfo", "-h", "-N", "-p", "scavenge", f"-O{fmt}").stdout.splitlines()
    ]
    assert [r[0] for r in rows] == ["sim-h100-01", "sim-h100-02"]
    assert rows[1][2] == "drained" and rows[1][6] == "bad NVLink"


def test_sinfo_default_summary(sim: SimCluster, run_cli) -> None:
    lines = run_cli("sinfo").stdout.splitlines()
    assert lines[0].split() == ["PARTITION", "AVAIL", "TIMELIMIT", "NODES", "STATE", "NODELIST"]
    assert lines[1].split() == ["gpu*", "up", "2-00:00:00", "4", "idle", "sim-a100-[01-04]"]


def test_scontrol_show(sim: SimCluster, run_cli) -> None:
    parts = run_cli("scontrol", "--oneliner", "show", "partition").stdout.splitlines()
    assert len(parts) == 3
    gpu = dict(tok.split("=", 1) for tok in parts[0].split())
    assert (gpu["PartitionName"], gpu["Default"], gpu["MaxTime"], gpu["PreemptMode"]) == (
        "gpu",
        "YES",
        "2-00:00:00",
        "OFF",
    )
    scav = dict(tok.split("=", 1) for tok in parts[2].split())
    assert (scav["PreemptMode"], scav["PriorityTier"]) == ("CANCEL", "1")
    config = run_cli("scontrol", "show", "config").stdout
    assert "AccountingStorageType   = accounting_storage/slurmdbd" in config
    assert "AccountingStoreFlags    = job_comment" in config
    missing = run_cli("scontrol", "show", "job", "123")
    assert missing.rc == 1 and "Invalid job id specified" in missing.stderr
    assert run_cli("scontrol", "update", "JobId=1").rc == 1


def test_sacctmgr_associations(sim: SimCluster, run_cli) -> None:
    res = run_cli(
        "sacctmgr",
        "--noheader",
        "--parsable2",
        "show",
        "associations",
        "where",
        f"user={sim.user}",
        "format=Cluster,Account,User,Partition,QOS,DefaultQOS",
    )
    assert res.rc == 0
    assert res.stdout.splitlines() == [
        f"simcluster|lab|{sim.user}||normal|normal",
        f"simcluster|other|{sim.user}||normal|normal",
    ]
    assert run_cli("sacctmgr", "-n", "-P", "show", "assoc", "where", "user=nobody-else").stdout == ""
    bad = run_cli("sacctmgr", "-n", "-P", "show", "assoc", "format=Bogus")
    assert bad.rc == 1


def test_sacct_help_and_errors(sim: SimCluster, run_cli) -> None:
    fields = run_cli("sacct", "--helpformat").stdout.split()
    assert {"JobID", "JobIDRaw", "AllocTRES", "SubmitLine", "Comment", "WorkDir", "TimelimitRaw"} <= set(
        fields
    )
    bad = run_cli("sacct", "-X", "--format=JobID,Nope")
    assert bad.rc == 1 and bad.stderr == 'sacct: error: Invalid field requested: "Nope"\n'
    assert run_cli("sacct", "-X", "-n", "-P", "--format=JobID").stdout == ""
    sim.set_faults(sacct_unavailable=True)
    down = run_cli("sacct", "-X", "-n", "-P")
    assert down.rc == 1 and "Problem talking to the database" in down.stderr


def test_accounting_disabled_config(tmp_path: Path) -> None:
    from hpclab.sim import cli
    from hpclab.sim.testing import sim_cluster

    with sim_cluster(tmp_path, accounting_enabled=False) as sim:
        env = {"HPCLAB_SIM_ROOT": str(sim.root)}
        import io

        err = io.StringIO()
        assert cli.main(["sacct", "-X"], stdout=io.StringIO(), stderr=err, env=env) == 1
        assert err.getvalue() == "sacct: error: Slurm accounting storage is disabled\n"
        out = io.StringIO()
        cli.main(["scontrol", "show", "config"], stdout=out, stderr=io.StringIO(), env=env)
        assert "accounting_storage/none" in out.getvalue()


def test_scancel_srun_sstat_on_unknown_jobs(sim: SimCluster, run_cli) -> None:
    res = run_cli("scancel", "4242")
    assert (
        res.rc == 1
        and res.stderr == "scancel: error: Kill job error on job id 4242: Invalid job id specified\n"
    )
    assert run_cli("scancel").rc == 1
    srun = run_cli("srun", "--jobid=4242", "--overlap", "nvidia-smi")
    assert (
        srun.rc == 1 and "Unable to confirm allocation for job 4242: Invalid job id specified" in srun.stderr
    )
    assert "only supports steps inside an existing job" in run_cli("srun", "hostname").stderr
    sstat = run_cli("sstat", "--allsteps", "--jobs=4242")
    assert sstat.rc == 1 and "couldn't get steps for job 4242" in sstat.stderr


def test_admin_plane(sim: SimCluster, run_cli) -> None:
    res = run_cli("_admin", "set-fault", "sbatch_lose_reply=2", "transport_timeout_cmds=squeue,sacct")
    assert res.rc == 0
    faults = json.loads(res.stdout)
    assert faults["sbatch_lose_reply"] == 2 and faults["transport_timeout_cmds"] == ["squeue", "sacct"]
    assert sim.faults().sbatch_lose_reply == 2
    assert run_cli("_admin", "set-fault", "no_such_fault=1").rc == 2
    assert run_cli("_admin", "set-fault", "garbage").rc == 2
    assert json.loads(run_cli("_admin", "clear-faults").stdout)["sbatch_lose_reply"] == 0
    assert json.loads(run_cli("_admin", "jobs").stdout) == []
    assert run_cli("_admin", "set-node", "sim-a100-01", "down", "cable", "pulled").rc == 0
    assert sim.state.node_admin()["sim-a100-01"] == ("down", "cable pulled")
    assert run_cli("_admin", "set-node", "ghost", "down").rc == 2
    assert json.loads(run_cli("_admin", "wait-idle", "1").stdout) == {"idle": True}
    info = json.loads(run_cli("_admin", "info").stdout)
    assert info["config"]["cluster_name"] == "simcluster"
    assert run_cli("_admin", "frobnicate").rc == 2
