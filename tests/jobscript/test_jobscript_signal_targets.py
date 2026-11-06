"""A checkpoint signal must reach the processes that handle it, not just the wrapper's child.

``--signal=B:USR1@<lead>`` and ``scancel --batch --signal=USR1`` (request_checkpoint) signal
only the batch shell, i.e. the wrapper. When the entrypoint is a launcher that does not handle
the signal itself (torchrun, accelerate, deepspeed, a ``bash train.sh`` without ``exec``),
forwarding it to the launcher alone kills the launcher (USR1's default action) and the
training processes - torchrun even starts them in their own sessions - never checkpoint.
"""

from __future__ import annotations

import os
import shlex
import signal
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from jobscript_support import Harness

from hpclab.domain.models import ResilienceSpec
from hpclab.services.jobscript import HELPER_SOURCE

RES = ResilienceSpec(signal="USR1", lead_s=300, stop_behavior="checkpoint_and_exit", resumable=True)

# Like torchrun's PContext: handles (and forwards) only TERM/INT, leaves USR1/USR2 at their
# default action, and starts every worker in its own session (so a process-group signal
# would not reach it either).
LAUNCHER = r"""
import os, signal, subprocess, sys
nproc = int(os.environ.get("LAUNCH_NPROC", "1"))
procs = [subprocess.Popen([sys.executable] + sys.argv[1:], env=dict(os.environ, RANK=str(r)),
                          start_new_session=True) for r in range(nproc)]
def forward(signum, frame):
    for p in procs:
        try:
            p.send_signal(signum)
        except OSError:
            pass
signal.signal(signal.SIGTERM, forward)
signal.signal(signal.SIGINT, forward)
rc = 0
for p in procs:
    rc = p.wait() or rc
sys.exit(rc)
"""

# A rank that checkpoints on USR1 and owns a data-loader-like helper forked AFTER installing
# the handler (so the helper inherits it): the rank is responsible for its own subtree and
# the helper must not be signalled as well.
WORKER = r"""
import os, signal, sys, time
out = os.environ["HPCLAB_OUTPUT_DIR"]
rank = os.environ.get("RANK", "0")
child = None
def mark(name, text=""):
    with open(os.path.join(out, name), "w") as fh:
        fh.write(text)
def on_usr1(signum, frame):
    mark("got_usr1.%s.%d" % (rank, os.getpid()), "stop_file=%s" % os.path.exists(os.environ["HPCLAB_STOP_FILE"]))
    if child:
        os.kill(child, signal.SIGKILL)
        os.waitpid(child, 0)
    sys.exit(0)
signal.signal(signal.SIGUSR1, on_usr1)
pid = os.fork()
if pid == 0:
    time.sleep(30)
    os._exit(0)
child = pid
mark("ready.%s" % rank)
time.sleep(30)
sys.exit(9)
"""


def _got_usr1(harness: Harness) -> list[str]:
    return sorted(p.name for p in Path(harness.layout.output_dir).glob("got_usr1*"))


# ---------------------------------------------------------------------------
# wrapper behaviour (real bash, real processes)
# ---------------------------------------------------------------------------


def test_checkpoint_signal_reaches_the_program_behind_a_launcher(harness: Harness) -> None:
    harness.publish({"launch.py": LAUNCHER})
    harness.write(argv=[sys.executable, "launch.py", "prog.py"], resilience=RES)
    proc = harness.start(env={"PROG_MODE": "usr1"})
    rc = harness.signal_when_ready(proc, signal.SIGUSR1)  # sent to the wrapper only, like B:USR1

    assert rc == 0, harness.stderr()
    assert (Path(harness.layout.output_dir) / "got_usr1").read_text() == "stop_file=True"
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 0 and status["signal"] is None
    assert status["signals_received"] == ["USR1"] and status["stop_signal"] == "USR1"
    assert "handle it" in harness.stderr()


def test_checkpoint_signal_reaches_every_rank_but_not_their_forked_helpers(harness: Harness) -> None:
    harness.publish({"launch.py": LAUNCHER, "worker.py": WORKER})
    harness.write(argv=[sys.executable, "launch.py", "worker.py"], resilience=RES)
    proc = harness.start(env={"LAUNCH_NPROC": "2"})
    out = Path(harness.layout.output_dir)
    harness.wait_for(out / "ready.0")
    harness.wait_for(out / "ready.1")
    proc.send_signal(signal.SIGUSR1)
    rc = proc.wait(timeout=30)

    assert rc == 0, harness.stderr()
    got = _got_usr1(harness)
    assert sorted(name.split(".")[1] for name in got) == ["0", "1"], got  # one per rank, none per helper
    assert all((out / name).read_text() == "stop_file=True" for name in got)
    status = harness.status()
    assert status["state"] == "exited" and status["exit_code"] == 0
    assert status["signals_received"] == ["USR1"]


def test_checkpoint_signal_reaches_the_program_behind_a_shell_script(harness: Harness) -> None:
    script = f'{shlex.quote(sys.executable)} prog.py\necho "rc=$?" > "$HPCLAB_OUTPUT_DIR/after"\n'
    harness.publish({"train.sh": script})
    harness.write(argv=["/bin/bash", "train.sh"], resilience=RES)
    proc = harness.start(env={"PROG_MODE": "usr1"})
    rc = harness.signal_when_ready(proc, signal.SIGUSR1)

    assert rc == 0, harness.stderr()
    out = Path(harness.layout.output_dir)
    assert (out / "got_usr1").read_text() == "stop_file=True"
    assert (out / "after").read_text() == "rc=0\n"  # the script itself was not killed
    assert harness.status()["state"] == "exited"


def test_request_checkpoint_signal_reaches_the_program_behind_a_launcher(harness: Harness) -> None:
    """USR2 as the checkpoint signal (run_cancel request_checkpoint sends it with --batch)."""
    usr2_prog = (
        "import os, signal, sys, time\n"
        "out = os.environ['HPCLAB_OUTPUT_DIR']\n"
        "def on(s, f):\n"
        "    open(os.path.join(out, 'got_usr1'), 'w').write('usr2')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGUSR2, on)\n"
        "open(os.path.join(out, 'ready'), 'w').close()\n"
        "time.sleep(30)\n"
        "sys.exit(9)\n"
    )
    harness.publish({"launch.py": LAUNCHER, "usr2.py": usr2_prog})
    res = ResilienceSpec(signal="USR2", lead_s=300, stop_behavior="checkpoint_and_exit")
    harness.write(argv=[sys.executable, "launch.py", "usr2.py"], resilience=res)
    proc = harness.start()
    rc = harness.signal_when_ready(proc, signal.SIGUSR2)

    assert rc == 0, harness.stderr()
    assert (Path(harness.layout.output_dir) / "got_usr1").read_text() == "usr2"
    assert harness.status()["stop_signal"] == "USR2"


def test_direct_program_still_gets_the_signal_itself(harness: Harness) -> None:
    harness.write(resilience=RES)
    proc = harness.start(env={"PROG_MODE": "usr1"})
    assert harness.signal_when_ready(proc, signal.SIGUSR1) == 0
    assert _got_usr1(harness) == ["got_usr1"]
    assert "forwarded to the program (pid" in harness.stderr()


def test_program_without_any_handler_still_gets_the_default_action(harness: Harness) -> None:
    """Nothing in the tree handles USR1: the main process receives it, as before."""
    harness.write(resilience=RES)
    proc = harness.start(env={"PROG_MODE": "sleep"})
    rc = harness.signal_when_ready(proc, signal.SIGUSR1)

    assert rc == 128 + signal.SIGUSR1
    status = harness.status()
    assert status["state"] == "killed" and status["signal"] == "USR1"
    assert "forwarded to the program (pid" in harness.stderr()


def test_degraded_wrapper_forwards_to_the_program(harness: Harness) -> None:
    """Without the python helper the wrapper cannot inspect the tree: it forwards to the child."""
    harness.write(resilience=RES)
    proc = harness.start(env={"PROG_MODE": "usr1", "HPCLAB_HELPER_PYTHON": "none"})
    assert harness.signal_when_ready(proc, signal.SIGUSR1) == 0
    assert _got_usr1(harness) == ["got_usr1"]


# ---------------------------------------------------------------------------
# helper: choosing the recipients
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def helper() -> types.ModuleType:
    module = types.ModuleType("hpclab_wrapper_helper")
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), module.__dict__)
    return module


USR1_BIT = 1 << (signal.SIGUSR1 - 1)


def _targets(helper: types.ModuleType, children: dict[int, list[int]], catching: set[int]) -> list[int]:
    result: list[int] = helper.signal_targets(
        1, signal.SIGUSR1, children, lambda pid: USR1_BIT if pid in catching else 0
    )
    return result


def test_signal_targets_root_that_handles_it_owns_its_tree(helper: types.ModuleType) -> None:
    assert _targets(helper, {1: [2, 3], 2: [4]}, {1, 2, 4}) == [1]


def test_signal_targets_skips_launchers_and_stops_at_the_topmost_handlers(helper: types.ModuleType) -> None:
    # 1 = torchrun, 2/3 = ranks (handle it), 4/5 = their forked helpers (inherited the handler),
    # 7 = a bash script whose python child 8 handles it
    children = {1: [2, 3, 7], 2: [4], 3: [5], 7: [8]}
    assert _targets(helper, children, {2, 3, 4, 5, 8}) == [2, 3, 8]


def test_signal_targets_none_when_nothing_handles_it(helper: types.ModuleType) -> None:
    assert _targets(helper, {1: [2], 2: [3]}, set()) == []


def test_signal_targets_unknown_masks_and_cycles_are_tolerated(helper: types.ModuleType) -> None:
    result = helper.signal_targets(
        1, signal.SIGUSR1, {1: [2], 2: [1, 3]}, lambda pid: None if pid != 3 else USR1_BIT
    )
    assert result == [3]


def _fake_proc(root: Path, procs: dict[int, tuple[str, int, str]]) -> Path:
    """A /proc look-alike: pid -> (comm, ppid, SigCgt hex)."""
    for pid, (comm, ppid, sigcgt) in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_bytes(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194560 100".encode())
        (d / "status").write_text(
            f"Name:\t{comm}\nPPid:\t{ppid}\nSigIgn:\t0000000001001000\nSigCgt:\t{sigcgt}\n"
        )
    (root / "self").mkdir()
    (root / "meminfo").write_text("MemTotal: 1 kB\n")
    return root


def test_linux_proc_tree_and_masks(
    helper: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helper, "PLATFORM", "linux")
    usr1 = 1 << (10 - 1)  # Linux SIGUSR1 = 10
    proc = _fake_proc(
        tmp_path / "proc",
        {
            100: ("torchrun", 1, "0000000180004007"),  # TERM/INT/HUP/QUIT handled, not USR1
            101: ("pt_main_thread", 100, f"{0x180004002 | usr1:016x}"),
            102: ("weird) (name", 100, f"{usr1:016x}"),  # comm with spaces and parens
            103: ("pt_data_worker", 101, f"{usr1:016x}"),
        },
    )
    children = helper.process_children(str(proc))
    assert children == {1: [100], 100: [101, 102], 101: [103]}
    assert helper.caught_mask(100, str(proc)) & usr1 == 0
    assert helper.caught_mask(101, str(proc)) & usr1
    assert helper.caught_mask(999, str(proc)) is None
    targets = helper.signal_targets(100, 10, children, lambda pid: helper.caught_mask(pid, str(proc)))
    assert targets == [101, 102]


def test_process_children_sees_this_process(helper: types.ModuleType) -> None:
    children = helper.process_children()
    assert os.getpid() in children.get(os.getppid(), [])


@pytest.fixture
def usr1_handler() -> Iterator[None]:
    previous = signal.signal(signal.SIGUSR1, lambda s, f: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGUSR1, previous)


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"), reason="linux/macOS only"
)
def test_caught_mask_sees_an_installed_handler(helper: types.ModuleType, usr1_handler: None) -> None:
    mask = helper.caught_mask(os.getpid())
    assert mask is not None and mask & USR1_BIT


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"), reason="linux/macOS only"
)
def test_caught_mask_of_a_default_process(helper: types.ModuleType) -> None:
    import subprocess

    child = subprocess.Popen(["sleep", "5"])
    try:
        mask = helper.caught_mask(child.pid)
        assert mask is not None and not mask & USR1_BIT
    finally:
        child.kill()
        child.wait()
    assert helper.caught_mask(2**22 + 12345) is None  # no such process


def test_sigtargets_command_prints_the_recipients(
    helper: types.ModuleType, usr1_handler: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert helper.main(["helper.py", "sigtargets", str(os.getpid()), "USR1"]) == 0
    printed = capsys.readouterr().out.split()
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert printed == [str(os.getpid())]
    assert helper.main(["helper.py", "sigtargets", "nope", "USR1"]) == 1
