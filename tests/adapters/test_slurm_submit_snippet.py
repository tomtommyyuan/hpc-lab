"""The sbatch journaling snippet, executed by real bash against a scripted fake ``sbatch``."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hpclab.adapters.base import CommandResult, ConnectionLost, SubmissionRejected
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.slurm import SlurmScheduler
from hpclab.profiles.schema import ClusterProfile, ResolvedCluster, UserOverlay

FAKE_SBATCH = r"""#!/bin/bash
printf '%s\n' "$@" > "$FAKE_DIR/args"
printf '%s' "${SLURM_TIME_FORMAT-<unset>}" > "$FAKE_DIR/time_format"
env | grep '^SBATCH_' | sort > "$FAKE_DIR/sbatch_env"
pwd > "$FAKE_DIR/cwd"
if read -t 1 line; then printf 'stdin:%s' "$line" > "$FAKE_DIR/stdin"; fi
case "$FAKE_MODE" in
  ok) echo "sbatch: warning: noisy site banner" >&2; echo "777" ;;
  cluster) echo "778;cluster1" ;;
  multiline) printf '779\nextra\n' ;;
  reject)
    echo "sbatch: error: invalid partition specified: nope" >&2
    echo "sbatch: error: Batch job submission failed: Invalid partition name specified" >&2
    exit 1 ;;
  timeout) echo "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation" >&2; exit 1 ;;
  testonly) echo "sbatch: Job 780 to start at 1790000000 using 8 processors on nodes n1 in partition p" >&2 ;;
esac
"""


class LosingTransport(LocalTransport):
    """Runs the command for real, then drops the reply (like an SSH connection dying)."""

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        super().run_script(script, timeout=timeout, args=args)
        raise ConnectionLost("reply lost")


@pytest.fixture
def fake_dir(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(FAKE_SBATCH)
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "work").mkdir()
    return tmp_path


def make(fake_dir: Path, mode: str, cls: type[LocalTransport] = LocalTransport) -> SlurmScheduler:
    transport = cls(
        "local",
        env={"FAKE_DIR": str(fake_dir), "FAKE_MODE": mode},
        path_prefix=[str(fake_dir / "bin")],
    )
    cluster = ResolvedCluster(
        profile=ClusterProfile(cluster_id="local", display_name="local"),
        overlay=UserOverlay(cluster_id="local"),
    )
    return SlurmScheduler(transport, cluster)


def test_success_is_journaled_and_parsed(fake_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_TIME_FORMAT", raising=False)
    sched = make(fake_dir, "ok")
    receipt = fake_dir / "work" / "submit.receipt"
    resp = sched.submit("job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt))
    assert resp.job_id == "777"
    assert receipt.read_text() == "0\n777\nsbatch: warning: noisy site banner"
    assert (fake_dir / "args").read_text().split() == ["--parsable", "job.sbatch"]
    # A real submission runs sbatch in the user's own environment: with --export=ALL an
    # SLURM_TIME_FORMAT=%s would leak into the job and change what scontrol/squeue print there.
    assert (fake_dir / "time_format").read_text() == "<unset>"
    assert Path((fake_dir / "cwd").read_text().strip()).resolve() == (fake_dir / "work").resolve()
    assert not (fake_dir / "stdin").exists()  # sbatch must not eat the rest of the script on stdin
    assert not [p for p in receipt.parent.iterdir() if ".tmp." in p.name or ".err." in p.name]
    recovered = sched.read_receipt(str(receipt))
    assert recovered is not None and recovered.job_id == "777"


def test_test_only_gets_epoch_times(fake_dir: Path) -> None:
    est = make(fake_dir, "testonly").test_submit("job.sbatch", workdir=str(fake_dir / "work"))
    assert (fake_dir / "time_format").read_text() == "%s"  # nothing is queued, so nothing leaks
    assert est.start_time == "2026-09-21T14:13:20.000Z" and est.partition == "p"


def test_cluster_suffix_and_multiline_stdout(fake_dir: Path) -> None:
    receipt = fake_dir / "work" / "r1"
    resp = make(fake_dir, "cluster").submit(
        "job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt)
    )
    assert (resp.job_id, resp.cluster_name) == ("778", "cluster1")
    receipt2 = fake_dir / "work" / "r2"
    resp2 = make(fake_dir, "multiline").submit(
        "job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt2)
    )
    assert resp2.job_id == "779"
    assert receipt2.read_text().splitlines()[:2] == ["0", "779 extra"]  # stdout folded onto line 2


def test_refusal_is_journaled_and_raised(fake_dir: Path) -> None:
    sched = make(fake_dir, "reject")
    receipt = fake_dir / "work" / "submit.receipt"
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt))
    assert exc.value.details["reason_code"] == "invalid_partition"
    assert receipt.read_text().startswith("1\n\nsbatch: error: invalid partition specified: nope")
    with pytest.raises(SubmissionRejected):
        sched.read_receipt(str(receipt))


def test_ambiguous_failure_is_uncertain(fake_dir: Path) -> None:
    with pytest.raises(ConnectionLost):
        make(fake_dir, "timeout").submit("job.sbatch", workdir=str(fake_dir / "work"))


def test_missing_workdir_never_calls_sbatch(fake_dir: Path) -> None:
    with pytest.raises(SubmissionRejected) as exc:
        make(fake_dir, "ok").submit(
            "job.sbatch", workdir=str(fake_dir / "nope"), receipt_path=str(fake_dir / "r")
        )
    assert exc.value.details["reason_code"] == "workdir_unavailable"
    assert not (fake_dir / "args").exists()


def test_missing_workdir_leaves_no_scratch_files_next_to_the_receipt(fake_dir: Path) -> None:
    attempt = fake_dir / "attempt"
    attempt.mkdir()  # staged by submit.stage in production
    with pytest.raises(SubmissionRejected):
        make(fake_dir, "ok").submit(
            "job.sbatch", workdir=str(fake_dir / "nope"), receipt_path=str(attempt / "submit.receipt")
        )
    assert list(attempt.iterdir()) == []


def test_sbatch_missing_from_path_is_a_definite_refusal(fake_dir: Path) -> None:
    # Non-interactive SSH shells often lack the `module load slurm` PATH: env cannot exec sbatch,
    # so nothing was submitted and the attempt must not be parked as "uncertain".
    transport = LocalTransport("local", env={"PATH": "/usr/bin:/bin"})
    cluster = ResolvedCluster(
        profile=ClusterProfile(cluster_id="local", display_name="local"),
        overlay=UserOverlay(cluster_id="local"),
    )
    receipt = fake_dir / "work" / "submit.receipt"
    sched = SlurmScheduler(transport, cluster)
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt))
    assert exc.value.details["reason_code"] == "sbatch_unavailable"
    assert receipt.read_text().startswith("127\n")
    with pytest.raises(SubmissionRejected):  # the journal tells the same story
        sched.read_receipt(str(receipt))


def test_lost_reply_is_recoverable_from_the_journal(fake_dir: Path) -> None:
    receipt = fake_dir / "work" / "submit.receipt"
    sched = make(fake_dir, "ok", LosingTransport)
    with pytest.raises(ConnectionLost):
        sched.submit("job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt))
    recovered = sched.read_receipt(str(receipt))
    assert recovered is not None and recovered.job_id == "777"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_journal_does_not_hide_a_success(fake_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    locked = fake_dir / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        resp = make(fake_dir, "ok").submit(
            "job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(locked / "submit.receipt")
        )
    finally:
        locked.chmod(0o700)
    assert resp.job_id == "777"
    assert "could not be journaled" in caplog.text


@pytest.mark.parametrize("test_only", [False, True])
def test_sbatch_input_environment_never_overrides_the_script(
    fake_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, test_only: bool
) -> None:
    # sbatch applies SBATCH_* input variables OVER #SBATCH lines. Exported from ~/.bashrc (read
    # by the non-interactive SSH shell) they would rename the job (losing the hl.<label>.<token>
    # identity observe/cancel check), move it to another partition or redirect its logs, while
    # the ledger records the script. The snippet must run sbatch without any of them.
    monkeypatch.setenv("SBATCH_JOB_NAME", "myexp")
    monkeypatch.setenv("SBATCH_PARTITION", "scavenge")
    monkeypatch.setenv("SBATCH_OUTPUT", "logs/%j.out")
    monkeypatch.setenv("SBATCH_EXPORT", "NONE")
    monkeypatch.setenv("SLURM_ACCOUNT", "kept")  # not an sbatch input variable: untouched
    sched = make(fake_dir, "testonly" if test_only else "ok")
    caplog.set_level("WARNING", logger="hpclab.adapters.slurm")
    if test_only:
        sched.test_submit("job.sbatch", workdir=str(fake_dir / "work"))
    else:
        receipt = fake_dir / "work" / "submit.receipt"
        resp = sched.submit("job.sbatch", workdir=str(fake_dir / "work"), receipt_path=str(receipt))
        assert resp.job_id == "777"
        assert receipt.read_text() == "0\n777\nsbatch: warning: noisy site banner"
    assert (fake_dir / "sbatch_env").read_text() == ""
    assert "SBATCH_EXPORT,SBATCH_JOB_NAME,SBATCH_OUTPUT,SBATCH_PARTITION" in caplog.text
