"""An unusable HPCLAB_HOME (unreadable, a file in place of the directory, an unreadable lock file)
must never make the client or the lifecycle helpers raise a raw ``OSError``.

``HpcLabClient.call`` promises an Envelope or ``ControllerUnavailable``/``HpcLabError``; before
the fix ``lock_is_held`` (used to build every ControllerUnavailable's diagnostics) raised
``PermissionError``/``NotADirectoryError`` itself, so the MCP server and the CLI saw a traceback
instead of an actionable "HPCLAB_HOME <path> is not usable: <reason>" hint.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from hpclab_ctl_toy import toy_spawn_cmd

from hpclab.config import Paths, get_paths
from hpclab.controller.client import HpcLabClient, controller_status, stop_controller
from hpclab.controller.files import check_private_dir, home_problem, lock_is_held, wait_lock_free
from hpclab.domain.errors import ControllerUnavailable, HpcLabError

pytestmark = [
    pytest.mark.timeout(30),
    pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits"),
]


@pytest.fixture
def locked_out(make_home: Callable[..., Paths]) -> Iterator[Callable[[str], Paths]]:
    """Homes made unreadable (mode 000) for the test; permissions restored before teardown."""
    restore: list[Path] = []

    def make(name: str = "home") -> Paths:
        p = make_home(name).ensure()
        p.home.chmod(0)
        restore.append(p.home)
        return p

    yield make
    for home in restore:
        home.chmod(stat.S_IRWXU)


@pytest.fixture
def file_home(make_home: Callable[..., Paths]) -> Paths:
    """A regular file where the home directory should be."""
    p = make_home("home-file")
    p.home.write_text("not a directory\n")
    return p


def _assert_actionable(exc: HpcLabError, paths: Paths) -> None:
    # the wording the CLI uses for an unusable home, with the reason in the hint
    assert f"cannot use the hpclab home {paths.home}" in exc.message, exc.message
    details: dict[str, Any] = exc.details
    assert details["home"] == str(paths.home)
    assert f"HPCLAB_HOME {paths.home} is not usable" in details["hint"], details
    assert details["home_problem"], details
    assert details["lock_held"] is None  # unknown, never a guess


# -- files.py helpers ------------------------------------------------------------------------


def test_lock_is_held_never_raises_on_an_unusable_home(
    locked_out: Callable[[str], Paths], file_home: Paths, home: Paths
) -> None:
    unreadable = locked_out("unreadable")
    assert lock_is_held(unreadable.lock) is False
    assert lock_is_held(file_home.lock) is False
    assert wait_lock_free(file_home.lock, 0.1) is True
    # an unreadable lock file inside a usable home
    home.ensure()
    home.lock.write_text("")
    home.lock.chmod(0)
    try:
        assert lock_is_held(home.lock) is False
    finally:
        home.lock.chmod(0o600)


def test_home_problem_names_the_reason(
    locked_out: Callable[[str], Paths], file_home: Paths, home: Paths, tmp_path: Path
) -> None:
    assert home_problem(home.home) is None  # does not exist yet but can be created
    home.ensure()
    assert home_problem(home.home) is None
    assert "not a directory" in (home_problem(file_home.home) or "")
    assert "not a directory" in (home_problem(file_home.home / "nested") or "")
    unreadable = locked_out("unreadable")
    assert "permission" in (home_problem(unreadable.home) or "").lower()
    assert "permission" in (home_problem(unreadable.home / "child") or "").lower()


def test_check_private_dir_turns_os_errors_into_hpclab_errors(
    locked_out: Callable[[str], Paths], file_home: Paths
) -> None:
    unreadable = locked_out("unreadable")
    with pytest.raises(HpcLabError, match="cannot inspect"):
        check_private_dir(unreadable.home / "run")
    with pytest.raises(HpcLabError):
        check_private_dir(file_home.home / "run")


# -- the client --------------------------------------------------------------------------------


@pytest.mark.parametrize("autostart", [True, False])
def test_call_on_an_unreadable_home_is_controller_unavailable(
    locked_out: Callable[[str], Paths], autostart: bool
) -> None:
    paths = locked_out("home")
    c = HpcLabClient(
        paths, autostart=autostart, spawn_cmd=toy_spawn_cmd(paths), timeout=5.0, startup_timeout=3.0
    )
    with pytest.raises(ControllerUnavailable) as ei:
        c.call("toy.pid")
    _assert_actionable(ei.value, paths)


@pytest.mark.parametrize("autostart", [True, False])
def test_call_with_a_file_in_place_of_the_home_is_controller_unavailable(
    file_home: Paths, autostart: bool
) -> None:
    c = HpcLabClient(
        file_home, autostart=autostart, spawn_cmd=toy_spawn_cmd(file_home), timeout=5.0, startup_timeout=3.0
    )
    with pytest.raises(ControllerUnavailable) as ei:
        c.call("toy.pid")
    _assert_actionable(ei.value, file_home)
    assert "not a directory" in ei.value.details["hint"]
    assert file_home.home.is_file()  # never replaced or "repaired"


def test_home_below_a_file_is_controller_unavailable(file_home: Paths) -> None:
    nested = get_paths(file_home.home / "nested")
    with pytest.raises(ControllerUnavailable) as ei:
        HpcLabClient(nested, spawn_cmd=toy_spawn_cmd(nested), timeout=5.0, startup_timeout=3.0).call(
            "toy.pid"
        )
    _assert_actionable(ei.value, nested)


def test_spawn_lock_failure_is_controller_unavailable(home: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """``paths.ensure()`` succeeded but the spawn lock cannot be created (e.g. run/ lost its
    permissions between the two): still ControllerUnavailable, never a raw OSError."""
    from hpclab.controller import client as client_mod

    def denied(self: Any, wait_s: float = 0.0) -> bool:
        raise PermissionError(13, "Permission denied", str(home.home / "run" / "spawn.lock"))

    monkeypatch.setattr(client_mod.InstanceLock, "acquire", denied)
    with pytest.raises(ControllerUnavailable) as ei:
        HpcLabClient(home, spawn_cmd=toy_spawn_cmd(home), timeout=5.0, startup_timeout=3.0).call("toy.pid")
    assert "spawn.lock" in ei.value.message and ei.value.details["hint"]


def test_unserializable_params_are_an_error_envelope(home: Paths) -> None:
    env = HpcLabClient(home, autostart=False).call("toy.echo", {"value": object()})
    assert env["ok"] is False and env["error"]["code"] == "invalid_argument", env
    assert not home.home.exists()  # rejected before touching the home


# -- lifecycle helpers ---------------------------------------------------------------------------


def test_status_and_stop_on_an_unusable_home_never_raise_os_errors(
    locked_out: Callable[[str], Paths], file_home: Paths
) -> None:
    """Whether a controller runs cannot be told there: both raise ControllerUnavailable (what the
    CLI maps to exit 3), never OSError, and never claim "stopped"."""
    unreadable = locked_out("unreadable")
    for paths in (unreadable, file_home):
        with pytest.raises(ControllerUnavailable) as ei:
            controller_status(paths, timeout=2.0)
        _assert_actionable(ei.value, paths)
        with pytest.raises(ControllerUnavailable) as ei:
            stop_controller(paths, timeout=1.0)
        _assert_actionable(ei.value, paths)


def test_status_of_a_missing_home_is_still_stopped(home: Paths) -> None:
    env = controller_status(home, timeout=2.0)
    assert env["ok"] and env["data"]["running"] is False and env["data"]["state"] == "stopped", env
    assert not home.home.exists()


def test_status_with_an_unreadable_lock_file_does_not_claim_stopped(home: Paths) -> None:
    """A usable home whose lock file cannot be read (left behind by `sudo hpclab controller
    start`, say): a controller may hold it, so status must not report "stopped" — it says it
    cannot tell, like ``stop_controller``, and names the lock file."""
    home.ensure()
    home.lock.write_text("")
    home.lock.chmod(0)
    try:
        with pytest.raises(ControllerUnavailable) as ei:
            controller_status(home, timeout=2.0)
        assert str(home.lock) in ei.value.message and "cannot tell" in ei.value.message
        assert ei.value.details["lock_held"] is None
        assert str(home.lock) in ei.value.details["hint"]
        with pytest.raises(ControllerUnavailable) as ei:
            stop_controller(home, timeout=1.0)
        assert str(home.lock) in ei.value.message
    finally:
        home.lock.chmod(0o600)


def test_lock_probe_never_blocks_on_a_fifo(home: Paths) -> None:
    """Opening a FIFO for reading blocks until a writer appears: a FIFO where the lock file
    should be must not hang the client's error paths."""
    import threading

    home.ensure()
    os.mkfifo(home.lock)
    answers: list[bool | None] = []
    probe = threading.Thread(target=lambda: answers.append(lock_is_held(home.lock)), daemon=True)
    try:
        probe.start()
        probe.join(timeout=3.0)
        blocked = probe.is_alive()
        if blocked:  # unblock the stuck open() so the thread can finish
            os.close(os.open(home.lock, os.O_WRONLY | os.O_NONBLOCK))
            probe.join(timeout=3.0)
    finally:
        home.lock.unlink()  # the fixture's teardown probes the lock file too
    assert not blocked, "lock_is_held blocked on a FIFO"
    assert answers == [False]


def test_stop_when_the_pid_cannot_be_signalled_is_controller_unavailable(
    home: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock is held but the pidfile names a process of another user (a stale pidfile, or a
    controller started with sudo): SIGTERM is refused with EPERM. That is ControllerUnavailable
    with a hint, never a raw PermissionError."""
    from hpclab.controller.files import InstanceLock, write_pidfile

    home.ensure()
    lock = InstanceLock(home.lock)
    assert lock.acquire()
    foreign_pid = 999_999_1
    write_pidfile(home.pidfile, foreign_pid)
    real_kill = os.kill

    def kill(pid: int, sig: int) -> None:
        if pid == foreign_pid:
            raise PermissionError(1, "Operation not permitted")
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)
    try:
        with pytest.raises(ControllerUnavailable) as ei:
            stop_controller(home, timeout=0.5)
        assert str(foreign_pid) in ei.value.message
        assert ei.value.details["hint"]
    finally:
        lock.release()
