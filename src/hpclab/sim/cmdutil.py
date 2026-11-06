"""Shared plumbing for the fake Slurm commands: invocation context and getopt-style parsing.

Parse errors reproduce GNU getopt's messages (``sbatch: unrecognized option '--foo'``) and
Slurm's exit status 1, so callers exercising error paths see what a real login node prints.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TextIO

from hpclab.sim.state import SimState

ArgMode = Literal["none", "required", "optional"]


@dataclass
class Ctx:
    """One command invocation."""

    state: SimState
    env: Mapping[str, str]
    cwd: str
    out: TextIO
    err: TextIO
    stdin: TextIO | None = None
    now: float = field(default_factory=time.time)

    @property
    def call_id(self) -> str | None:
        return self.env.get("HPCLAB_SIM_CALL_ID") or None

    def fail(self, prog: str, message: str, rc: int = 1) -> int:
        self.err.write(f"{prog}: error: {message}\n")
        return rc


class UsageError(Exception):
    def __init__(self, message: str, rc: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.rc = rc


@dataclass(frozen=True)
class Opt:
    long: str
    short: str | None = None
    arg: ArgMode = "required"


Parsed = dict[str, "str | bool"]


def parse_args(
    prog: str, argv: list[str], opts: list[Opt], *, stop_at_positional: bool = False
) -> tuple[Parsed, list[str]]:
    """Parse ``argv``; later occurrences override earlier ones. Raises UsageError.

    ``stop_at_positional``: everything from the first positional on is returned untouched
    (sbatch script arguments, srun commands).
    """
    by_long = {o.long: o for o in opts}
    by_short = {o.short: o for o in opts if o.short}
    values: Parsed = {}
    positionals: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            positionals.extend(argv[i + 1 :])
            break
        if arg.startswith("--") and len(arg) > 2:
            name, eq, inline = arg[2:].partition("=")
            opt = by_long.get(name) or _unique_prefix(by_long, name)
            if opt is None:
                raise UsageError(
                    f"{prog}: unrecognized option '{arg}'\nTry \"{prog} --help\" for more information"
                )
            if opt.arg == "none":
                if eq:
                    raise UsageError(f"{prog}: option '--{opt.long}' doesn't allow an argument")
                values[opt.long] = True
            elif eq:
                values[opt.long] = inline
            elif opt.arg == "optional":
                values[opt.long] = True
            else:
                if i + 1 >= len(argv):
                    raise UsageError(f"{prog}: option '--{opt.long}' requires an argument")
                i += 1
                values[opt.long] = argv[i]
            i += 1
            continue
        if arg.startswith("-") and len(arg) > 1:
            j = 1
            while j < len(arg):
                ch = arg[j]
                opt = by_short.get(ch)
                if opt is None:
                    raise UsageError(
                        f"{prog}: invalid option -- '{ch}'\nTry \"{prog} --help\" for more information"
                    )
                rest = arg[j + 1 :]
                if opt.arg == "none":
                    values[opt.long] = True
                    j += 1
                    continue
                if rest:
                    values[opt.long] = rest
                elif opt.arg == "optional":
                    values[opt.long] = True
                else:
                    if i + 1 >= len(argv):
                        raise UsageError(f"{prog}: option requires an argument -- '{ch}'")
                    i += 1
                    values[opt.long] = argv[i]
                break
            i += 1
            continue
        if stop_at_positional:
            positionals.extend(argv[i:])
            break
        positionals.append(arg)
        i += 1
    return values, positionals


def _unique_prefix(by_long: Mapping[str, Opt], name: str) -> Opt | None:
    """getopt_long accepts unambiguous abbreviations (``--part`` for ``--partition``)."""
    matches = [o for n, o in by_long.items() if n.startswith(name)] if name else []
    return matches[0] if len(matches) == 1 else None


def sval(values: Parsed, key: str) -> str | None:
    """String value of an option (flags given without argument -> None)."""
    v = values.get(key)
    return v if isinstance(v, str) else None


def flag(values: Parsed, key: str) -> bool:
    return key in values and values[key] is not False


def split_list(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]
