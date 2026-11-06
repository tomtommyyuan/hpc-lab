"""``SimTransport``: a ``LocalTransport`` whose PATH starts with the simulator's fake Slurm
commands, plus transport-level fault injection.

Faults (see ``SimFaults``) applied here, before/after the command runs:

* ``auth_required``: every call raises ``AuthRequiredError`` (as an expired SSH ControlMaster).
* ``transport_timeout_cmds``: calls whose command (``squeue``, ``sbatch``...; for ``run_script``
  the first Slurm command in its arguments) or method name (``read_range``, ``stat``...) is listed,
  or ``*``, raise ``TransportTimeout`` *without running anything*.
* ``sbatch_lose_reply``: the fake sbatch really queues the job and tags this call (via the
  per-call ``HPCLAB_SIM_CALL_ID``); after the command returns, the reply is dropped and
  ``ConnectionLost(maybe_executed=True)`` is raised. Any on-cluster receipt was already written.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandResult,
    ConnectionLost,
    RangeRead,
    StatInfo,
    TransportTimeout,
)
from hpclab.adapters.local import LocalTransport
from hpclab.sim.cluster import SHIM_COMMANDS, SimCluster

_SHELLS = frozenset({"bash", "sh"})


def command_name(argv: list[str]) -> str:
    """The Slurm command an argv runs (skipping ``env`` and ``VAR=value``; for ``bash -s`` scripts
    the first Slurm command among its arguments), else the program's basename."""
    tokens = list(argv)
    while tokens and (
        os.path.basename(tokens[0]) == "env" or ("=" in tokens[0] and not tokens[0].startswith("-"))
    ):
        tokens.pop(0)
    if not tokens:
        return ""
    head = os.path.basename(tokens[0])
    if head in _SHELLS:
        for tok in tokens[1:]:
            base = os.path.basename(tok)
            if base in SHIM_COMMANDS:
                return base
    return head


class SimTransport(LocalTransport):
    """Transport to a simulated cluster rooted at ``sim_root`` (created on first use)."""

    def __init__(self, cluster_id: str, sim_root: Path) -> None:
        self.sim = SimCluster.ensure(Path(sim_root))
        super().__init__(
            cluster_id,
            env={"HPCLAB_SIM_ROOT": str(self.sim.root)},
            path_prefix=[str(self.sim.bin_dir)],
            user=self.sim.user,
        )
        self._local = threading.local()

    # -- fault gate ------------------------------------------------------------------
    def _gate(self, name: str) -> None:
        faults = self.sim.faults()
        if faults.auth_required:
            raise AuthRequiredError(
                "simulated: SSH needs interactive authentication (ControlMaster expired)",
                details={
                    "cluster_id": self.cluster_id,
                    "login_command": ["hpclab", "sim", "fault", "clear", self.cluster_id],
                    "simulated": True,
                },
            )
        listed = set(faults.transport_timeout_cmds)
        if "*" in listed or name in listed:
            raise TransportTimeout(
                f"simulated: {name} timed out on the login node",
                details={"command": name, "simulated": True},
            )

    def _env(self) -> dict[str, str]:
        env = super()._env()
        call_id = getattr(self._local, "call_id", None)
        if call_id:
            env["HPCLAB_SIM_CALL_ID"] = call_id
        return env

    # -- commands -----------------------------------------------------------------------
    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        name = command_name(argv)
        self._gate(name)
        call_id = uuid.uuid4().hex
        self._local.call_id = call_id
        try:
            # Empty stdin instead of None: never let a child read the caller's stdin (MCP channel).
            result = super().run(argv, timeout=timeout, stdin=stdin if stdin is not None else b"")
        finally:
            self._local.call_id = None
        if name == "sbatch" and self.sim.state.pop_lost_reply(call_id):
            raise ConnectionLost(
                "simulated: connection lost after sbatch was accepted; reply never arrived",
                details={"command": name, "simulated": True},
            )
        return result

    def remote_user(self) -> str:
        self._gate("remote_user")
        return super().remote_user()

    # -- files (same fault gate; the "remote" filesystem is local) ------------------------
    def upload_dir(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        self._gate("upload_dir")
        super().upload_dir(local_dir, remote_dir, timeout=timeout)

    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        self._gate("write_file")
        super().write_file(remote_path, content, mode=mode, timeout=timeout)

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._gate("read_range")
        return super().read_range(remote_path, offset, length, timeout=timeout)

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._gate("read_tail")
        return super().read_tail(remote_path, length, timeout=timeout)

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        self._gate("stat")
        return super().stat(remote_paths, timeout=timeout)

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        self._gate("sha256")
        return super().sha256(remote_paths, timeout=timeout)

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        self._gate("download")
        super().download(remote_path, local_path, timeout=timeout)

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        self._gate("list_dir")
        return super().list_dir(remote_dir, max_entries=max_entries, timeout=timeout)
