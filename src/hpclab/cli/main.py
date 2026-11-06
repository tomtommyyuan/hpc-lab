"""``hpclab`` command line.

Setup and plumbing commands live here; experiment/run/result commands are registered from
``hpclab.cli.commands`` (same controller methods the MCP tools use).
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

from hpclab import __version__
from hpclab.bundled import bundled_path
from hpclab.cli import views as V
from hpclab.cli.common import (
    EXIT_CONTROLLER,
    EXIT_ERROR,
    EXIT_USAGE,
    STATE,
    auth_resume,
    call,
    console,
    controller_down,
    err_console,
    parse_kv,
    print_json,
    print_json_error,
    print_table,
    show,
    unusable_home,
)
from hpclab.domain.errors import ControllerUnavailable, HpcLabError

app = typer.Typer(
    name="hpclab",
    help="Reliable Slurm execution and traceable experiment evidence for ML research.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
controller_app = typer.Typer(
    help="Manage the local controller (single writer of the ledger).", no_args_is_help=True
)
cluster_app = typer.Typer(help="Register and inspect clusters.", no_args_is_help=True)
mcp_app = typer.Typer(help="Run or configure the MCP server.", no_args_is_help=True)
skills_app = typer.Typer(help="Claude Code skills shipped with hpclab.", no_args_is_help=True)
sim_app = typer.Typer(help="Local Slurm simulator (faults, jobs).", no_args_is_help=True)
examples_app = typer.Typer(
    help="Example research projects shipped with hpclab (copy one to start).", no_args_is_help=True
)
app.add_typer(controller_app, name="controller")
app.add_typer(cluster_app, name="cluster")
app.add_typer(mcp_app, name="mcp")
app.add_typer(skills_app, name="skills")
app.add_typer(sim_app, name="sim")
app.add_typer(examples_app, name="examples")


def _version(value: bool) -> None:
    if value:
        console.print(f"hpclab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    json_output: Annotated[bool, typer.Option("--json", help="Print raw result envelopes as JSON.")] = False,
    home: Annotated[
        str | None, typer.Option("--home", envvar="HPCLAB_HOME", help="hpclab home directory.")
    ] = None,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for the controller per call.")] = 120.0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    STATE.json_output = json_output
    STATE.home = home
    STATE.timeout = timeout
    if home:
        os.environ["HPCLAB_HOME"] = str(Path(home).expanduser().resolve())


# ---------------------------------------------------------------------------
# init / status
# ---------------------------------------------------------------------------


@app.command()
def init(
    sim: Annotated[
        bool, typer.Option("--sim/--no-sim", help="Register the local Slurm simulator as cluster 'sim'.")
    ] = False,
) -> None:
    """Create the hpclab home, start the controller, optionally set up the simulator."""
    try:
        paths = STATE.paths.ensure()
    except OSError as exc:
        raise unusable_home(exc) from exc
    console.print(f"hpclab home: {paths.home}")
    if sim:

        def render(env: dict[str, Any]) -> None:
            # an older service names repository paths (examples/...), absent from an installed wheel
            steps = [x for x in (env.get("data") or {}).get("next_steps") or [] if "examples/" not in str(x)]
            if not any("hpclab examples copy" in str(x) for x in steps):
                steps.append(
                    "or start from the example project: hpclab examples copy tiny-classifier "
                    "&& hpclab prepare tiny-classifier/hpclab.yaml"
                )
            _render_cluster_data({**env, "data": {**(env.get("data") or {}), "next_steps": steps}})

        call("init_sim", {}, render=render)
    else:
        call("controller.ping", {}, quiet=True)
        console.print(
            "controller is running. Next: [bold]hpclab cluster profiles[/bold] / [bold]hpclab init --sim[/bold]"
        )


@app.command()
def status() -> None:
    """Controller health and anything that needs attention."""
    from hpclab.controller.client import controller_status

    try:
        env = controller_status(STATE.paths)
    except ControllerUnavailable as exc:  # the home is unusable: whether a controller runs is unknown
        raise controller_down(exc) from exc
    except OSError as exc:
        raise unusable_home(exc) from exc
    if STATE.json_output:
        show(env)
        return
    data = env.get("data") or {}
    if not data.get("running", True) and "pid" in data and data.get("state"):
        console.print(f"[yellow]{escape(env.get('summary', ''))}[/yellow]")
        console.print(f"log: {data.get('log')}")
        raise typer.Exit(EXIT_CONTROLLER if data.get("state") == "unresponsive" else 0)
    console.print(
        f"[bold]controller[/bold] pid {data.get('pid')} · v{data.get('version')} · up {int(data.get('uptime_s') or 0)}s"
    )
    ob = data.get("outbox") or {}
    if ob:
        console.print(
            f"background work: {json.dumps(ob.get('counts') or ob.get('by_state') or {}, default=str)}"
        )
    for c in data.get("clusters") or []:
        console.print(f"cluster {c.get('cluster_id') or c}")
    attention = data.get("attention") or []
    for a in attention:
        console.print(f"[yellow]attention[/yellow] {escape(str(a))}")
    if not attention:
        console.print("[green]nothing needs attention[/green]")


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------


@controller_app.command("start")
def controller_start() -> None:
    """Start the controller in the background (clients also autostart it)."""
    from hpclab.controller.client import start_controller

    info = start_controller(STATE.paths.ensure())
    if STATE.json_output:
        console.print_json(data=info)
    else:
        console.print(f"controller running (pid {info.get('pid')}, socket {info.get('socket')})")


@controller_app.command("stop")
def controller_stop(
    force: Annotated[bool, typer.Option(help="SIGKILL if it does not stop in time.")] = False,
) -> None:
    """Stop the controller (jobs on the cluster keep running)."""
    from hpclab.controller.client import stop_controller

    try:
        res = stop_controller(STATE.paths, force=force)
    except HpcLabError as exc:  # e.g. an unusable home or an unreadable lock file
        err_console.print(f"[red bold]{exc.code}[/red bold]: {escape(exc.message)}")
        hint = exc.details.get("hint") if isinstance(exc.details, dict) else None
        if hint:
            err_console.print(f"  hint: {escape(str(hint))}")
        raise typer.Exit(EXIT_CONTROLLER) from exc
    if STATE.json_output:
        console.print_json(data=res)
    elif not res["was_running"]:
        console.print("controller was not running")
    elif res["stopped"]:
        console.print(f"controller stopped (pid {res.get('pid')}, via {res.get('method')})")
    else:
        err_console.print(f"[yellow]controller still stopping[/yellow]: {res.get('hint')}")
        raise typer.Exit(EXIT_ERROR)


@controller_app.command("restart")
def controller_restart() -> None:
    """Stop then start the controller."""
    controller_stop(force=False)
    controller_start()


@controller_app.command("status")
def controller_status_cmd() -> None:
    """Same as `hpclab status`."""
    status()


@controller_app.command("logs")
def controller_logs(lines: Annotated[int, typer.Option("-n", help="Number of lines.")] = 50) -> None:
    """Show the end of the controller log."""
    from hpclab.controller.files import tail_lines

    for line in tail_lines(STATE.paths.log, max_bytes=max(4096, lines * 400), max_lines=lines):
        console.print(escape(line))


@controller_app.command("run", hidden=True)
def controller_run() -> None:
    """Run the controller in the foreground (for debugging / service managers)."""
    os.execv(  # noqa: S606 - replace this process with the controller
        sys.executable,
        [sys.executable, "-m", "hpclab.controller", "--home", str(STATE.paths.home), "--foreground"],
    )


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


def _render_cluster_data(env: dict[str, Any]) -> None:
    data = env.get("data") or {}
    c = data.get("cluster") or data
    console.print(
        f"cluster [bold]{c.get('cluster_id')}[/bold] ({escape(str(c.get('display_name')))}) · transport {c.get('transport')}"
        f" · user {c.get('user') or '?'} · account {c.get('account') or '-'}"
    )
    console.print(f"remote root: {c.get('remote_root') or '[red]unresolved[/red]'}")
    if c.get("missing"):
        console.print(f"[yellow]missing[/yellow]: {', '.join(c['missing'])}")
    for step in data.get("next_steps") or []:
        console.print(f"next: {escape(step)}")


@cluster_app.command("profiles")
def cluster_profiles() -> None:
    """Bundled and local site profiles."""

    def render(env: dict[str, Any]) -> None:
        rows = [
            {
                "name": p["name"],
                "cluster_id": p["cluster_id"],
                "display_name": p["display_name"],
                "source": p["source"],
                "error": p.get("error") or "",
            }
            for p in env["data"]["profiles"]
        ]
        print_table(rows)

    call("profiles_list", {}, render=render)


@cluster_app.command("list")
def cluster_list() -> None:
    """Registered clusters."""

    def render(env: dict[str, Any]) -> None:
        rows = [
            {
                "cluster": c["cluster_id"],
                "transport": c["transport"],
                "host": c.get("ssh_host") or "",
                "user": c.get("user") or "",
                "account": c.get("account") or "",
                "remote_root": c.get("remote_root") or "",
                "missing": ",".join(c.get("missing") or []),
            }
            for c in env["data"]["clusters"]
        ]
        print_table(rows)

    call("clusters_list", {}, render=render)


@cluster_app.command("add")
def cluster_add(
    cluster_id: Annotated[str, typer.Argument(help="Cluster id (must match the profile's cluster_id).")],
    profile: Annotated[
        str | None, typer.Option(help="Profile name or YAML path (default: the cluster id).")
    ] = None,
    ssh_host: Annotated[str | None, typer.Option(help="SSH host or ~/.ssh/config alias.")] = None,
    user: Annotated[str | None, typer.Option(help="Your username on the cluster.")] = None,
    account: Annotated[str | None, typer.Option(help="Slurm account.")] = None,
    qos: Annotated[str | None, typer.Option(help="Slurm QoS.")] = None,
    remote_root: Annotated[
        str | None, typer.Option(help="Absolute cluster path for hpclab data (on scratch).")
    ] = None,
    partition: Annotated[str | None, typer.Option(help="Default partition.")] = None,
    ssh_option: Annotated[
        list[str] | None, typer.Option(help="Extra ssh option Key=Value (repeatable).")
    ] = None,
    lab_user: Annotated[
        list[str] | None, typer.Option(help="Lab member usernames for usage reports (repeatable).")
    ] = None,
) -> None:
    """Register or update a cluster: public site profile + your private overlay."""
    call(
        "cluster_register",
        {
            "profile": profile or cluster_id,
            "cluster_id": cluster_id,
            "ssh_host": ssh_host,
            "user": user,
            "account": account,
            "qos": qos,
            "remote_root": remote_root,
            "default_partition": partition,
            "ssh_options": ssh_option or None,
            "lab_users": lab_user or None,
        },
        render=_render_cluster_data,
    )


@cluster_app.command("show")
def cluster_show(cluster_id: str) -> None:
    """Effective settings of one cluster."""
    call("cluster_get", {"cluster_id": cluster_id}, render=_render_cluster_data)


@cluster_app.command("inspect")
def cluster_inspect(
    cluster_id: str,
    check: Annotated[list[str] | None, typer.Option(help="Only these checks (repeatable).")] = None,
) -> None:
    """Auth, identity, scheduler, storage, limits, capacity, nodes, queue."""

    def render(env: dict[str, Any]) -> None:
        data = env.get("data") or {}
        checks = data.get("checks") or {}
        items = checks.items() if isinstance(checks, dict) else [(c.get("name"), c) for c in checks]
        for name, c in items:
            st = c.get("status", "?")
            color = {"ok": "green", "warn": "yellow", "fail": "red"}.get(st, "yellow")
            console.print(
                f"[{color}]{st:>11}[/{color}]  [bold]{name}[/bold]  {escape(str(c.get('summary') or ''))}"
            )
            for f in (c.get("findings") or [])[:8]:
                msg = f.get("message") if isinstance(f, dict) else f
                console.print(f"             · {escape(str(msg))}")
        if not checks:
            show({**env, "summary": ""})

    call("cluster_inspect", {"cluster_id": cluster_id, "checks": check or None}, render=render)


@app.command()
def auth(
    cluster_id: str,
    print_only: Annotated[bool, typer.Option("--print", help="Only print the command.")] = False,
) -> None:
    """Open the SSH master connection hpclab reuses (you complete MFA/Duo in this terminal)."""
    env = call("auth_command", {"cluster_id": cluster_id}, quiet=True, render=lambda e: None)
    data = env.get("data") or {}
    # --json: stdout holds the auth_command envelope alone; the human lines go to stderr
    say = err_console if STATE.json_output else console
    if not data.get("argv"):
        say.print(escape(str(data.get("explanation", "no login needed"))))
        return
    if data.get("master_alive"):
        say.print(f"[green]already connected[/green] to {escape(cluster_id)}; hpclab will reuse it")
        _resume_waiting(cluster_id, say)
        return
    if print_only:
        if not STATE.json_output:  # the envelope already carries `command`
            console.print(escape(str(data["command"])), soft_wrap=True)  # copied: never wrapped
        return
    say.print(f"running: {escape(data['command'])}")
    rc = subprocess.call(data["argv"])  # interactive: inherits this terminal for the MFA prompt
    if rc == 0:
        say.print(f"[green]connected[/green]: hpclab will reuse this connection for {escape(cluster_id)}")
        _resume_waiting(cluster_id, say)
    else:
        err_console.print(f"[red]ssh exited with {rc}[/red]")
        raise typer.Exit(EXIT_ERROR)


def _resume_waiting(cluster_id: str, say: Console) -> None:
    """Wake the submissions/cancels/polls that were blocked on authentication (never fails)."""
    resumed, message, note = auth_resume(cluster_id)
    say.print(escape(message), style="green" if resumed else "yellow")
    if note:
        say.print(f"  {escape(note)}", style="dim")


# ---------------------------------------------------------------------------
# mcp / skills
# ---------------------------------------------------------------------------


@mcp_app.command("serve")
def mcp_serve() -> None:
    """Run the MCP server on stdio (what `claude mcp add hpclab -- hpclab mcp serve` runs)."""
    from hpclab.mcp.server import main as mcp_main

    mcp_main()


@mcp_app.command("config")
def mcp_config() -> None:
    """Print how to register the MCP server with Claude Code and other clients."""
    # an argv, never a string split on spaces: install paths and homes may contain spaces/quotes
    found = shutil.which("hpclab")
    command = [found] if found else [sys.executable, "-m", "hpclab.cli"]
    home = STATE.paths.home
    env = {"HPCLAB_HOME": str(home)} if STATE.home else {}
    cfg = {
        "mcpServers": {
            "hpclab": {
                "command": command[0],
                "args": [*command[1:], "mcp", "serve"],
                **({"env": env} if env else {}),
            }
        }
    }
    add = ["claude", "mcp", "add", "hpclab"]
    for key, value in env.items():  # the server must use the same home as this CLI
        add += ["-e", f"{key}={value}"]
    add += ["--", *command, "mcp", "serve"]
    console.print("[bold]Claude Code[/bold]:")
    console.print("  " + escape(shlex.join(add)), soft_wrap=True)  # pasteable: never wrapped
    console.print("[bold]JSON config[/bold] (Claude Desktop, Cursor, ...):")
    console.print_json(data=cfg)


@skills_app.command("path")
def skills_path() -> None:
    """Directory containing the bundled skills."""
    p = bundled_path("skills")
    if p is None:
        err_console.print("skills are not bundled with this installation")
        raise typer.Exit(EXIT_ERROR)
    console.print(str(p))


@skills_app.command("install")
def skills_install(
    dest: Annotated[Path, typer.Option(help="Skills directory.")] = Path("~/.claude/skills"),
    force: Annotated[bool, typer.Option(help="Overwrite existing skills of the same name.")] = False,
) -> None:
    """Copy the hpclab skills into a Claude Code skills directory."""
    src = bundled_path("skills")
    if src is None:
        err_console.print("skills are not bundled with this installation")
        raise typer.Exit(EXIT_ERROR)
    target = dest.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    for skill in sorted(p for p in src.iterdir() if (p / "SKILL.md").is_file()):
        out = target / skill.name
        if out.exists() and not force:
            console.print(f"skip {skill.name} (exists; --force to overwrite)")
            continue
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(skill, out)
        console.print(f"installed {skill.name} -> {out}")


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------

_SKIP_DIRS = frozenset({"__pycache__"})
_SKIP_SUFFIXES = (".pyc",)


def _stop(message: str, *, code: int = EXIT_ERROR, error: str = "precondition_failed") -> typer.Exit:
    """Report a CLI-detected error (one JSON document in --json mode) and exit."""
    if STATE.json_output:
        print_json_error(error, message)
    label = "usage" if code == EXIT_USAGE else "error"
    err_console.print(f"[red bold]{label}[/red bold]: {escape(message)}")
    return typer.Exit(code)


def _examples_root() -> Path:
    root = bundled_path("examples")
    if root is None:
        raise _stop("examples are not bundled with this installation", error="unsupported")
    return root


def _example_dirs(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.is_symlink() and not p.name.startswith(".") and p.name not in _SKIP_DIRS
    )


def _example_info(path: Path) -> dict[str, Any]:
    """Name, one-line description (README title) and the ExperimentSpec files with their cluster."""
    description = ""
    readme = path / "README.md"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                description = title.partition(": ")[2] or title
                break
    import yaml

    specs = []
    for f in sorted(path.glob("hpclab*.yaml"), key=lambda p: (p.name != "hpclab.yaml", p.name)):
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            doc = None
        doc = doc if isinstance(doc, dict) else {}
        specs.append({"file": f.name, "name": doc.get("name"), "cluster": doc.get("cluster")})
    return {"name": path.name, "path": str(path), "description": description, "specs": specs}


def _copy_tree(src: Path, dest: Path) -> int:
    """Copy ``src`` into ``dest`` (created as needed), skipping bytecode and symlinks; files of
    the same name are overwritten, nothing else in ``dest`` is touched. Returns files copied.

    ``dest`` itself may be a symlink the user chose (e.g. ``~/work -> /scratch/work``); a symlink
    *inside* it is never written through."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not (here / d).is_symlink())
        rel = here.relative_to(src)
        out_dir = dest / rel
        if rel != Path(".") and out_dir.is_symlink():
            raise _stop(f"refusing to write through the symlink {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(filenames):
            f = here / name
            if name.endswith(_SKIP_SUFFIXES) or f.is_symlink():
                continue
            target = out_dir / name
            if target.is_symlink():
                raise _stop(f"refusing to write through the symlink {target}")
            shutil.copyfile(f, target)  # contents only: the copy is the user's, writable
            n += 1
    return n


@examples_app.command("list")
def examples_list() -> None:
    """Bundled example projects and their ExperimentSpec files."""
    root = _examples_root()
    items = [_example_info(p) for p in _example_dirs(root)]
    if STATE.json_output:
        print_json({"examples": items})
        return
    if not items:
        console.print("no examples are bundled")
        return
    for it in items:
        console.print(f"[bold]{escape(it['name'])}[/bold]  {escape(it['description'])}")
        for spec in it["specs"]:
            cluster = f" (cluster {spec['cluster']})" if spec.get("cluster") else ""
            console.print(f"    {escape(spec['file'])}{escape(cluster)}")
    console.print(V.next_steps([f"hpclab examples copy {items[0]['name']} [DEST]"], title="copy one"))


@examples_app.command("copy")
def examples_copy(
    name: Annotated[str, typer.Argument(help="Example name (see `hpclab examples list`).")],
    dest: Annotated[
        Path | None, typer.Argument(help="Directory to copy into (default: ./NAME).", show_default=False)
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Copy into a non-empty DEST, overwriting files of the same name.")
    ] = False,
) -> None:
    """Copy a bundled example project to start from (bytecode is left out)."""
    root = _examples_root()
    known = {p.name: p for p in _example_dirs(root)}
    if name not in known:
        raise _stop(
            f"unknown example {name!r} (available: {', '.join(sorted(known)) or 'none'})",
            code=EXIT_USAGE,
            error="invalid_argument",
        )
    target = dest if dest is not None else Path(name)
    if target.exists() and not target.is_dir():
        raise _stop(f"{target} exists and is not a directory")
    if target.is_dir() and any(target.iterdir()) and not force:
        raise _stop(f"{target} exists and is not empty: choose another DEST, or pass --force to overwrite")
    info = _example_info(known[name])
    try:
        count = _copy_tree(known[name], target)
    except OSError as exc:
        raise _stop(f"copying {name} to {target} failed: {exc}") from exc
    if STATE.json_output:
        print_json({"ok": True, "name": name, "dest": str(target.absolute()), "files": count})
        return
    console.print(f"[green]copied[/green] {escape(name)} → {escape(str(target))} ({count} file(s))")
    specs = info["specs"]
    steps: list[str] = []
    if specs:
        main_spec = specs[0]
        if main_spec.get("cluster") == "sim":
            steps.append("hpclab init --sim   # once: the local Slurm simulator, cluster 'sim'")
        steps += [
            f"hpclab prepare {shlex.quote(str(target / main_spec['file']))}   # validates and freezes the runs",
            "hpclab submit <plan id>   # printed by prepare: smoke gate first, then the matrix",
            "hpclab grid <workflow id> --watch",
        ]
        for spec in specs[1:]:
            where = f"   # for cluster {spec['cluster']}" if spec.get("cluster") else ""
            steps.append(f"hpclab prepare {shlex.quote(str(target / spec['file']))}{where}")
    console.print(V.next_steps(steps))


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------


def sim_root(cluster_id: str) -> Path:
    """The simulator a registered sim cluster really uses: ``init_sim`` records it in the
    overlay (it lives outside a home whose path job scripts cannot carry)."""
    from hpclab.services.clusters import effective_sim_root

    try:
        return effective_sim_root(STATE.paths, cluster_id)
    except HpcLabError as exc:  # an unreadable overlay: the default place
        err_console.print(f"[yellow]warning[/yellow]: {escape(exc.message)}")
        return STATE.paths.sim_root(cluster_id)


def _sim(cluster_id: str) -> Any:
    from hpclab.sim.cluster import SimCluster

    root = sim_root(cluster_id)
    # the root may come from the cluster overlay (outside the home): never turn some other
    # directory into a simulator
    if not (root / "state.db").is_file():
        raise _stop(f"no simulator at {root}; run: hpclab init --sim", error="not_found")
    return SimCluster.ensure(root)


def _check_fault_value(key: str, value: Any) -> None:
    """Raise ValueError for a value ``SimFaults.coerce`` refuses or would silently mangle:
    a count given as 1.7 or true (it would become 1), or a NaN/Infinity duration or
    utilization (every comparison with NaN is false)."""
    from hpclab.sim.config import SimFaults

    if key in SimFaults.COUNTERS and (
        isinstance(value, bool) or (isinstance(value, float) and not value.is_integer())
    ):
        raise ValueError(f"{key} is a count: a whole number >= 0")
    coerced = SimFaults.coerce(key, value)
    if isinstance(coerced, float) and not math.isfinite(coerced):
        raise ValueError(f"{key} must be a finite number")


@sim_app.command("fault")
def sim_fault(
    settings: Annotated[list[str], typer.Argument(help="key=value fault settings, e.g. sbatch_lose_reply=1")],
    cluster_id: Annotated[str, typer.Option("--cluster")] = "sim",
) -> None:
    """Set simulator faults (lost sbatch replies, accounting lag, preemption, OOM, auth...)."""
    from hpclab.sim.config import SimFaults

    try:
        requested = parse_kv(settings)
    except typer.BadParameter as exc:
        raise _stop(exc.message, code=EXIT_USAGE, error="invalid_argument") from None
    known = sorted(SimFaults.field_names())
    # validate every setting before touching the simulator: a typo sets nothing
    for key, value in requested.items():
        if key not in known:
            raise _stop(
                f"unknown fault {key!r}; valid: {', '.join(known)}", code=EXIT_USAGE, error="invalid_argument"
            )
        try:
            _check_fault_value(key, value)
        except (TypeError, ValueError) as exc:
            raise _stop(
                f"bad value for fault {key}: {value!r} ({exc})", code=EXIT_USAGE, error="invalid_argument"
            ) from None
    faults = _sim(cluster_id).set_faults(**requested)
    if STATE.json_output:
        print_json({"ok": True, "cluster_id": cluster_id, "faults": faults.to_dict()})
        return
    console.print(f"faults: {faults}")


@sim_app.command("clear")
def sim_clear(cluster_id: Annotated[str, typer.Option("--cluster")] = "sim") -> None:
    """Clear all simulator faults."""
    sim = _sim(cluster_id)
    sim.clear_faults()
    if STATE.json_output:
        print_json({"ok": True, "cluster_id": cluster_id, "faults": sim.faults().to_dict()})
        return
    console.print("faults cleared")


@sim_app.command("jobs")
def sim_jobs(cluster_id: Annotated[str, typer.Option("--cluster")] = "sim") -> None:
    """Ground-truth list of simulated jobs (what Slurm really accepted)."""
    jobs = _sim(cluster_id).jobs()
    if STATE.json_output:
        console.print_json(data=jobs)
        return
    rows = [
        {k: j.get(k) for k in ("job_id", "name", "state", "partition", "exit_code") if k in j} for j in jobs
    ]
    print_table(rows)


@sim_app.command("shutdown")
def sim_shutdown(cluster_id: Annotated[str, typer.Option("--cluster")] = "sim") -> None:
    """Kill all running simulated jobs."""
    killed = _sim(cluster_id).shutdown()
    if STATE.json_output:
        print_json({"ok": True, "cluster_id": cluster_id, "jobs_stopped": len(killed)})
        return
    console.print(f"stopped {len(killed)} job process group(s)")


def _register_commands() -> None:
    import importlib
    import importlib.util

    if importlib.util.find_spec("hpclab.cli.commands") is not None:
        importlib.import_module("hpclab.cli.commands")  # registers experiment/run/result commands


_register_commands()

if __name__ == "__main__":
    app()
