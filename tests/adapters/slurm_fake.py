"""Scripted ``Transport`` for SlurmScheduler unit tests (imported by name, pytest prepend mode).

Rules are keyed by Slurm command; the last matching rule wins. Fixture files under
tests/fixtures/slurm show the ASCII unit separator as U+241F; ``load`` restores ``\\x1f``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hpclab.adapters.base import CommandResult, RangeRead
from hpclab.adapters.slurm import FS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "slurm"
ENV = ["env", "SLURM_TIME_FORMAT=%s"]

Response = CommandResult | Exception
Matcher = Callable[[list[str]], bool]


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8").replace("␟", FS)


@dataclass
class FakeTransport:
    cluster_id: str = "cluster1"
    user: str = "jdoe"
    calls: list[list[str]] = field(default_factory=list)
    rules: list[tuple[str, Matcher, Callable[[list[str]], Response]]] = field(default_factory=list)

    def on(
        self,
        command: str,
        stdout: str = "",
        stderr: str = "",
        rc: int = 0,
        *,
        when: Matcher | None = None,
        raises: Exception | None = None,
    ) -> None:
        def respond(argv: list[str]) -> Response:
            return raises if raises is not None else CommandResult(argv, rc, stdout, stderr, 0.01)

        self.rules.append((command, when or (lambda _argv: True), respond))

    def _dispatch(self, command: str, argv: list[str]) -> CommandResult:
        for name, match, respond in reversed(self.rules):
            if name == command and match(argv):
                out = respond(argv)
                if isinstance(out, Exception):
                    raise out
                return out
        return CommandResult(argv, 127, "", f"env: '{command}': No such file or directory", 0.01)

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        self.calls.append(list(argv))
        command = argv[2] if argv[:2] == ENV else argv[0]
        return self._dispatch(command, argv)

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        return self._dispatch("run_script", list(args or []))

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        raise FileNotFoundError(remote_path)

    def remote_user(self) -> str:
        return self.user

    def last(self, command: str) -> list[str]:
        for argv in reversed(self.calls):
            if (argv[2] if argv[:2] == ENV else argv[0]) == command:
                return argv
        raise AssertionError(f"{command} was never called")


def probe_ok(fake: FakeTransport, *, config: str | None = None) -> None:
    fake.on("sbatch", load("sbatch_version.txt"), when=lambda a: "--version" in a)
    fake.on("squeue", load("squeue_help.txt"), when=lambda a: "--help" in a)
    fake.on(
        "scontrol",
        config if config is not None else load("scontrol_show_config.txt"),
        when=lambda a: "config" in a,
    )
    fake.on("sacct", load("sacct_helpformat.txt"), when=lambda a: "--helpformat" in a)
    fake.on("sstat", "JobID MaxRSS", when=lambda a: "--helpformat" in a)
    fake.on("srun", "slurm 23.02.7\n", when=lambda a: "--version" in a)
