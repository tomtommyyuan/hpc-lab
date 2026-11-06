"""A home shared by two machines (an NFS home) must never make one machine signal a process of
the other, and a controller must not start on a network home by default.

Before the fix the pidfile held only the pid. On host B, which sees host A's lock over NFS
but cannot reach host A's socket, ``hpclab controller stop [--force]`` sent SIGTERM (and
SIGKILL) to whatever process of the user on host B had host A's controller pid. Now the pidfile
records the host, boot session and process start time, and a pid recorded by another machine,
or one that has been reused since, is never signalled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from hpclab_ctl_toy import ToyApp, spawned_pids, toy_spawn_cmd, wait_for

from hpclab import config
from hpclab.config import ALLOW_NETWORK_HOME_ENV, FileSystem, Paths
from hpclab.controller.client import HpcLabClient, controller_status, stop_controller
from hpclab.controller.daemon import Controller
from hpclab.controller.files import (
    InstanceLock,
    boot_id,
    lock_is_held,
    pid_record_problem,
    process_start_token,
    read_pid,
    read_pid_record,
    recorded_elsewhere,
    this_host,
    write_pidfile,
)
from hpclab.domain.errors import ControllerUnavailable, ErrorCode, HpcLabError

pytestmark = pytest.mark.timeout(30)

OTHER_HOST = "login-2.other-cluster.example"


@pytest.fixture
def victim() -> Iterator[subprocess.Popen[bytes]]:
    """An unrelated process of the same user (an editor, a Jupyter kernel)."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    proc.kill()
    proc.wait(timeout=5)


@pytest.fixture
def held_lock(home: Paths) -> Iterator[InstanceLock]:
    """The instance lock as host A's controller holds it, seen from host B over NFS."""
    home.ensure()
    lock = InstanceLock(home.lock)
    assert lock.acquire()
    yield lock
    lock.release()


def _foreign_pidfile(home: Paths, pid: int) -> None:
    home.pidfile.write_text(f"{pid}\nhost={OTHER_HOST}\nboot={uuid.uuid4()}\nstart=12345\n")


# -- the pidfile ---------------------------------------------------------------------------------


def test_pidfile_records_this_machine_and_stays_readable_by_older_clients(tmp_path: Path) -> None:
    pidfile = tmp_path / "controller.pid"
    write_pidfile(pidfile, os.getpid())
    assert pidfile.read_text().splitlines()[0] == str(os.getpid())  # older clients read line 1
    assert read_pid(pidfile) == os.getpid()
    rec = read_pid_record(pidfile)
    assert rec is not None and rec.pid == os.getpid() and rec.host == this_host()
    assert rec.boot == boot_id() and rec.start == process_start_token(os.getpid())
    assert not recorded_elsewhere(rec) and pid_record_problem(rec) is None


def test_a_legacy_pid_only_pidfile_is_still_trusted(tmp_path: Path) -> None:
    pidfile = tmp_path / "controller.pid"
    pidfile.write_text(f"{os.getpid()}\n")
    rec = read_pid_record(pidfile)
    assert rec is not None and rec.pid == os.getpid() and rec.host is None
    assert pid_record_problem(rec) is None


def test_a_pid_recorded_by_another_machine_is_foreign(tmp_path: Path) -> None:
    pidfile = tmp_path / "controller.pid"
    pidfile.write_text(f"{os.getpid()}\nhost={OTHER_HOST}\nboot={uuid.uuid4()}\n")
    rec = read_pid_record(pidfile)
    assert rec is not None and recorded_elsewhere(rec)
    problem = pid_record_problem(rec)
    assert problem is not None and OTHER_HOST in problem


def test_a_new_hostname_on_the_same_boot_is_still_this_machine(tmp_path: Path) -> None:
    """Laptops change hostname with the network (DHCP); the boot session decides."""
    if boot_id() is None:
        pytest.skip("no boot session id on this platform")
    pidfile = tmp_path / "controller.pid"
    pidfile.write_text(f"{os.getpid()}\nhost=renamed-by-dhcp\nboot={boot_id()}\n")
    rec = read_pid_record(pidfile)
    assert rec is not None and not recorded_elsewhere(rec) and pid_record_problem(rec) is None


def test_a_reused_pid_is_detected_by_its_start_time(tmp_path: Path, victim: subprocess.Popen[bytes]) -> None:
    if process_start_token(victim.pid) is None:
        pytest.skip("process start times are not available on this platform")
    pidfile = tmp_path / "controller.pid"
    pidfile.write_text(f"{victim.pid}\nhost={this_host()}\nboot={boot_id() or ''}\nstart=1.000001\n")
    rec = read_pid_record(pidfile)
    assert rec is not None and not recorded_elsewhere(rec)
    problem = pid_record_problem(rec)
    assert problem is not None and str(victim.pid) in problem


# -- stop / status from the other machine --------------------------------------------------------


def test_stop_never_signals_a_pid_recorded_by_another_machine(
    home: Paths, held_lock: InstanceLock, victim: subprocess.Popen[bytes]
) -> None:
    _foreign_pidfile(home, victim.pid)
    with pytest.raises(ControllerUnavailable) as ei:
        stop_controller(home, timeout=0.5, force=True)
    time.sleep(0.2)
    assert victim.poll() is None, "an unrelated process on this machine was signalled"
    err = ei.value
    assert OTHER_HOST in err.message and not err.retryable
    assert OTHER_HOST in err.details["hint"] and "local disk" in err.details["hint"]
    assert lock_is_held(home.lock)


def test_stop_never_signals_a_reused_pid(
    home: Paths, held_lock: InstanceLock, victim: subprocess.Popen[bytes]
) -> None:
    if process_start_token(victim.pid) is None:
        pytest.skip("process start times are not available on this platform")
    home.pidfile.write_text(f"{victim.pid}\nhost={this_host()}\nboot={boot_id() or ''}\nstart=1.000001\n")
    with pytest.raises(ControllerUnavailable) as ei:
        stop_controller(home, timeout=0.5, force=True)
    time.sleep(0.2)
    assert victim.poll() is None, "a process that reused the controller's pid was signalled"
    assert str(victim.pid) in ei.value.message and "lsof" in ei.value.details["hint"]


_HOLDER = """
import sys, time
from pathlib import Path
from hpclab.controller.files import InstanceLock, write_pidfile
lock_path, pidfile, legacy = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
lock = InstanceLock(lock_path)
assert lock.acquire()
if legacy:
    import os
    pidfile.write_text(f"{os.getpid()}\\n")
else:
    import os
    write_pidfile(pidfile, os.getpid())
print("ready", flush=True)
time.sleep(60)
"""


@pytest.mark.parametrize("legacy", [False, True], ids=["identity-pidfile", "legacy-pidfile"])
def test_stop_still_signals_an_unresponsive_controller_on_this_machine(home: Paths, legacy: bool) -> None:
    """The guard must not break the SIGTERM fallback for a local controller that does not answer."""
    home.ensure()
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(home.lock), str(home.pidfile), "1" if legacy else "0"],
        stdout=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == b"ready"
        result = stop_controller(home, timeout=5.0)
        assert result["stopped"] is True and result["method"] == "sigterm" and result["pid"] == proc.pid
        assert proc.wait(timeout=5) != 0
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_status_names_the_other_machine(home: Paths, held_lock: InstanceLock) -> None:
    _foreign_pidfile(home, 4242)
    env = controller_status(home, timeout=1.0)
    data = env["data"]
    assert data["running"] is False and data["state"] == "unresponsive"
    assert data["pid"] == 4242 and data["host"] == OTHER_HOST
    assert OTHER_HOST in env["summary"]


def test_client_hint_for_a_lock_held_by_another_machine_is_not_stop_force(
    home: Paths, held_lock: InstanceLock
) -> None:
    _foreign_pidfile(home, 4242)
    c = HpcLabClient(home, spawn_cmd=toy_spawn_cmd(home), startup_timeout=0.5, timeout=1.0)
    try:
        with pytest.raises(ControllerUnavailable) as ei:
            c.call("toy.pid")
    finally:
        c.close()
    hint = ei.value.details["hint"]
    assert OTHER_HOST in hint and "--force" not in hint


# -- no controller on a network home -------------------------------------------------------------


@pytest.fixture
def nfs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_NETWORK_HOME_ENV, raising=False)
    monkeypatch.setattr(config, "filesystem_of", lambda p: FileSystem("nfs4", "/home", network=True))


def test_controller_refuses_a_network_home(home: Paths, nfs: None) -> None:
    ctl = Controller(home, ToyApp, lock_wait_s=0.0, shutdown_timeout_s=2.0)
    with pytest.raises(HpcLabError) as ei:
        ctl.start()
    assert ei.value.code == ErrorCode.PRECONDITION_FAILED and "nfs4" in ei.value.message
    assert not lock_is_held(home.lock) and not home.pidfile.exists()


def test_controller_on_an_allowed_network_home_starts_and_warns(
    home: Paths, nfs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ALLOW_NETWORK_HOME_ENV, "1")
    ctl = Controller(home, ToyApp, lock_wait_s=0.0, shutdown_timeout_s=2.0, tick_interval_s=0.05)
    ctl.start()
    try:
        assert lock_is_held(home.lock)
        assert wait_for(lambda: "network file system" in home.log.read_text(), 5)
    finally:
        ctl.stop(timeout=5.0)


def test_client_does_not_spawn_a_controller_on_a_network_home(home: Paths, nfs: None) -> None:
    c = HpcLabClient(home, spawn_cmd=toy_spawn_cmd(home), startup_timeout=5.0, timeout=2.0)
    try:
        with pytest.raises(ControllerUnavailable) as ei:
            c.call("toy.pid")
    finally:
        c.close()
    assert "network file system" in ei.value.message and not ei.value.retryable
    assert ALLOW_NETWORK_HOME_ENV in ei.value.details["hint"]
    assert spawned_pids(home) == []
