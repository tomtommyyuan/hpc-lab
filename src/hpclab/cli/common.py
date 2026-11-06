"""Shared CLI plumbing: talking to the controller and rendering Envelopes.

Every command that reads or changes the ledger goes through the controller (the single
writer) via ``call()``. Output modes:

* default: the envelope ``summary`` line, then a readable rendering of ``data`` (commands
  with a dedicated view pass ``render=``), then warnings and incomplete-coverage notes;
* ``--json`` (global option): the raw Envelope as JSON on stdout, for scripts.

Exit codes: 0 ok; 1 domain error; 2 usage error; 3 controller unavailable; 4 auth required.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from hpclab.config import Paths, get_paths
from hpclab.domain.errors import ControllerUnavailable, HpcLabError
from hpclab.domain.ids import new_id

console = Console(highlight=False, soft_wrap=False)
err_console = Console(stderr=True, highlight=False)

EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CONTROLLER = 3
EXIT_AUTH = 4


@dataclass
class CliState:
    json_output: bool = False
    home: str | None = None
    timeout: float = 120.0
    _client: Any = field(default=None, repr=False)

    @property
    def paths(self) -> Paths:
        return get_paths(self.home)

    def client(self) -> Any:
        if self._client is None:
            from hpclab.controller.client import HpcLabClient

            self._client = HpcLabClient(self.paths, timeout=self.timeout)
        return self._client


STATE = CliState()


def new_key(prefix: str = "cli") -> str:
    """A fresh idempotency key for a one-shot CLI mutation."""
    return f"{prefix}-{new_id('request')}"


def _client_call(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """``STATE.client().call`` with every failure to get an answer turned into an exit code.

    An unusable home (permissions, an expired AFS token, a file where the directory should be)
    surfaces from the client as a bare ``OSError``: it is reported like any other unavailable
    controller, never as a traceback."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        env: dict[str, Any] = STATE.client().call(method, clean)
    except ControllerUnavailable as exc:
        _print_controller_down(exc)
        raise typer.Exit(EXIT_CONTROLLER) from exc
    except HpcLabError as exc:
        if STATE.json_output:
            print_json(_exc_envelope(exc))
        err_console.print(f"[red]error[/red] {exc.code}: {escape(exc.message)}")
        raise typer.Exit(EXIT_AUTH if exc.code == "auth_required" else EXIT_ERROR) from exc
    except OSError as exc:
        raise unusable_home(exc) from exc
    return env


def unusable_home(exc: OSError) -> typer.Exit:
    """Report a home the controller cannot use (exit 3), instead of an OSError traceback."""
    home = STATE.paths.home
    down = ControllerUnavailable(
        f"cannot use the hpclab home {home}: {exc}",
        details={
            "home": str(home),
            "hint": "HPCLAB_HOME (or --home) must be a directory you own and can write "
            "(on AFS/Kerberos homes, renew your token or point HPCLAB_HOME at local or scratch storage)",
        },
    )
    _print_controller_down(down)
    return typer.Exit(EXIT_CONTROLLER)


def _exc_envelope(exc: HpcLabError) -> dict[str, Any]:
    from hpclab.controller.protocol import envelope_from_exception

    return envelope_from_exception(exc, new_id("request"))


def print_json_error(code: str, message: str, details: dict[str, Any] | None = None) -> None:
    """An error Envelope on stdout for errors the CLI itself detects (``--json`` mode), so a
    script always gets one JSON document with ``ok: false``, never empty output."""
    from hpclab.controller.protocol import error_envelope

    print_json(error_envelope(new_id("request"), code, message, details=details or {}))


def call(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    render: Callable[[dict[str, Any]], None] | None = None,
    exit_on_error: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Call a controller method, print the result, exit non-zero on error."""
    env = _client_call(method, params)
    show(env, render=render, quiet=quiet)
    if not env.get("ok") and exit_on_error:
        raise typer.Exit(exit_code_for(env))
    return env


def exit_code_for(env: dict[str, Any]) -> int:
    """Process exit code for an Envelope (0 ok, 4 auth required, 1 other domain errors)."""
    if env.get("ok"):
        return 0
    code = (env.get("error") or {}).get("code")
    return EXIT_AUTH if code == "auth_required" else EXIT_ERROR


def fetch(method: str, params: dict[str, Any] | None = None, *, allow_error: bool = False) -> dict[str, Any]:
    """Call a controller method *without* printing a successful result.

    For lookups inside composite commands (e.g. read a version before a cancel). A failed
    call is printed (as JSON in ``--json`` mode) and ends the command with the matching exit
    code, unless ``allow_error`` (then the error Envelope is returned for the caller to handle).
    """
    env = _client_call(method, params)
    if not env.get("ok") and not allow_error:
        show(env)
        raise typer.Exit(exit_code_for(env))
    return env


def _unknown_method(env: dict[str, Any]) -> bool:
    err = env.get("error") or {}
    return err.get("code") == "invalid_argument" and str(err.get("message", "")).startswith("unknown method")


def auth_resume(cluster_id: str) -> tuple[bool, str, str | None]:
    """Ask the controller to retry, now, the steps that wait for SSH authentication on a cluster
    whose connection is open again (method ``auth_resume``). Never fails the caller: a controller
    that predates the method, or any error, gives a hint instead.

    Returns (resumed, message, the controller's note if any)."""
    env = fetch("auth_resume", {"cluster_id": cluster_id}, allow_error=True)
    if env.get("ok"):
        data = env.get("data") or {}
        woken = data.get("woken")
        n = len(woken) if isinstance(woken, (list, tuple)) else woken
        note = str(data["note"]) if data.get("note") else None
        return True, f"resumed {n if isinstance(n, int) else '?'} waiting step(s)", note
    later = "blocked steps are retried automatically within about 2 minutes"
    if _unknown_method(env):
        return (
            False,
            f"{later} (this controller cannot resume them on demand; `hpclab controller restart` updates it)",
            None,
        )
    err = env.get("error") or {}
    return (
        False,
        f"could not resume waiting steps now ({err.get('code')}: {err.get('message')}); {later}",
        None,
    )


def print_json(obj: Any, *, compact: bool = False) -> None:
    """JSON on stdout (``--json`` output); ``compact`` = one line (for streams)."""
    if compact:
        text = json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def show(
    env: dict[str, Any], *, render: Callable[[dict[str, Any]], None] | None = None, quiet: bool = False
) -> None:
    if STATE.json_output:
        sys.stdout.write(json.dumps(env, indent=2, ensure_ascii=False, default=str) + "\n")
        return
    if not env.get("ok"):
        print_error(env)
        return
    if not quiet and env.get("summary"):
        console.print(f"[bold]{escape(str(env['summary']))}[/bold]")
    data = env.get("data")
    if render is not None:
        render(env)
    elif data not in (None, {}, []) and not quiet:
        print_data(data)
    print_notes(env)


def print_error(env: dict[str, Any]) -> None:
    err = env.get("error") or {}
    code = err.get("code", "error")
    err_console.print(f"[red bold]{code}[/red bold]: {escape(str(err.get('message', '')))}")
    details = err.get("details") or {}
    for key in ("remediation", "hint"):
        if details.get(key):
            err_console.print(f"  [yellow]{key}[/yellow]: {escape(str(details[key]))}")
    if details.get("login_command_str"):
        err_console.print(f"  [yellow]run[/yellow]: {escape(str(details['login_command_str']))}")
    rest = {
        k: v
        for k, v in details.items()
        if k not in ("remediation", "hint", "login_command_str", "login_command")
    }
    if rest:
        err_console.print("  details: " + escape(json.dumps(rest, default=str)[:2000]))
    if err.get("operation_ref"):
        err_console.print(f"  operation: {err['operation_ref']}")


def print_notes(env: dict[str, Any]) -> None:
    for w in env.get("warnings") or []:
        console.print(f"[yellow]warning[/yellow] {escape(str(w))}")
    cov = env.get("coverage")
    if cov and not cov.get("complete", True):
        bad = [s for s in cov.get("sources", []) if s.get("status") not in ("ok", "cached", "skipped")]
        parts = [f"{s['name']}={s['status']}" + (f" ({s['detail']})" if s.get("detail") else "") for s in bad]
        console.print("[yellow]incomplete[/yellow] " + escape("; ".join(parts) or incomplete_reason(cov)))
        for m in cov.get("missing") or []:
            console.print(f"  [yellow]missing[/yellow] {escape(str(m))}")
    if env.get("next_cursor"):
        console.print(f"[dim]more: --cursor {env['next_cursor']}[/dim]")


def incomplete_reason(cov: dict[str, Any]) -> str:
    """Why coverage is incomplete when every source answered (e.g. runs missing or excluded)."""
    notes = [str(n) for n in cov.get("notes") or [] if n]
    if notes:
        return "; ".join(notes)
    missing = cov.get("missing") or []
    if missing:
        return f"{len(missing)} item(s) not included (listed below)"
    return "some sources were not consulted"


def print_data(data: Any, *, indent: int = 0, max_items: int = 50) -> None:
    """Readable nested key/value rendering (generic fallback)."""
    pad = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if v in (None, [], {}):
                continue
            if isinstance(v, (dict, list)):
                console.print(f"{pad}[cyan]{escape(str(k))}[/cyan]:")
                print_data(v, indent=indent + 1, max_items=max_items)
            else:
                console.print(f"{pad}[cyan]{escape(str(k))}[/cyan]: {escape(_short(v))}")
    elif isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data) and _tabular(data):
            print_table(data[:max_items])
        else:
            for x in data[:max_items]:
                if isinstance(x, (dict, list)):
                    console.print(f"{pad}-")
                    print_data(x, indent=indent + 1, max_items=max_items)
                else:
                    console.print(f"{pad}- {escape(_short(x))}")
        if len(data) > max_items:
            console.print(f"{pad}[dim]… {len(data) - max_items} more[/dim]")
    else:
        console.print(f"{pad}{escape(_short(data))}")


def _short(v: Any, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def _tabular(rows: list[dict[str, Any]]) -> bool:
    keys = set(rows[0])
    return (
        all(set(r) == keys for r in rows)
        and all(not isinstance(v, (dict, list)) for r in rows for v in r.values())
        and len(keys) <= 10
    )


def print_table(
    rows: list[dict[str, Any]], columns: list[str] | None = None, title: str | None = None
) -> None:
    if not rows:
        console.print("[dim](none)[/dim]")
        return
    cols = columns or list(rows[0].keys())
    table = Table(title=title, show_lines=False, header_style="bold")
    for c in cols:
        table.add_column(c)
    for r in rows:
        table.add_row(*[escape(_short(r.get(c), 60)) if r.get(c) is not None else "" for c in cols])
    console.print(table)


def controller_down(exc: HpcLabError) -> typer.Exit:
    """Report an unavailable controller (or unusable home) and return the exit (3) to raise."""
    _print_controller_down(exc)
    return typer.Exit(EXIT_CONTROLLER)


def _print_controller_down(exc: HpcLabError) -> None:
    if STATE.json_output:
        print_json(_exc_envelope(exc))
    err_console.print(f"[red bold]controller unavailable[/red bold]: {escape(exc.message)}")
    for key in ("hint", "log"):
        if exc.details.get(key):
            err_console.print(f"  {key}: {escape(str(exc.details[key]))}")
    tail = exc.details.get("spawn_log_tail") or exc.details.get("log_tail")
    if tail:
        err_console.print("  last log lines:")
        for line in tail[-10:]:
            err_console.print(f"    {escape(str(line))}")


def parse_kv(pairs: list[str]) -> dict[str, Any]:
    """``k=v`` pairs -> dict with JSON-ish value parsing (numbers, bools, null, lists)."""
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise typer.BadParameter(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def parse_csv_ints(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"expected comma-separated integers, got {text!r}") from exc
