"""Shared helpers for the OpenSSH transport tests: fake ``ssh``/``rsync`` executables.

The fake ``ssh`` parses ssh's command line like the real client (options, ``--``, host) and
then runs the remote command string locally with ``<shell> -c`` — exactly what sshd does with
the user's login shell. Failure modes are driven by environment variables so the transport
under test is exercised unmodified:

``FAKE_SSH_FAIL_STDERR``  print this to stderr and exit ``FAKE_SSH_FAIL_CODE`` (default 255)
                          without running anything (connect/auth failures)
``FAKE_SSH_DROP_AFTER``   run the command, then print this and exit 255 (connection dropped)
``FAKE_SSH_SLEEP``        sleep this many seconds first (timeouts)
``FAKE_SSH_NOISE``        print this to stdout first (chatty ``.bashrc``)
``FAKE_SSH_WARN``         print this to stderr first and carry on (ssh client warnings)
``FAKE_SSH_REMOTE_PATH``  PATH for the remote command (e.g. without python3)
``FAKE_SSH_SHELL``        the "login shell" that runs the remote command (default /bin/bash)
``FAKE_SSH_MASTER``       "1": ``-O check``/``-O exit`` report a running master
``FAKE_SSH_LOG``          append one JSON line per invocation (argv, host, command)
``FAKE_SSH_PIDFILE``      write the fake ssh's pid here (to check it gets killed)

The fake ``rsync`` logs its argv to ``FAKE_RSYNC_LOG`` and, after reaching the remote side
through the ``-e`` command, can fail like a transfer error would:
``FAKE_RSYNC_FAIL_STDERR`` / ``FAKE_RSYNC_FAIL_CODE`` (default 23).
"""

from __future__ import annotations

import json
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hpclab.adapters.openssh import OpenSSHTransport

FAKE_SSH = r"""
import json, os, subprocess, sys, time

WITH_ARG = set("BbcDEeFIiJLlmOoPpQRSWw")


def parse(argv):
    opts, i = [], 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            i += 1
            break
        if not arg.startswith("-") or arg == "-":
            break
        if arg[1] in WITH_ARG:
            if len(arg) > 2:
                opts.append((arg[1], arg[2:]))
            else:
                i += 1
                opts.append((arg[1], argv[i]))
        else:
            opts.extend((ch, None) for ch in arg[1:])
        i += 1
    return opts, argv[i], argv[i + 1:]


def main():
    opts, host, command = parse(sys.argv[1:])
    env = os.environ
    if env.get("FAKE_SSH_LOG"):
        with open(env["FAKE_SSH_LOG"], "a") as fh:
            fh.write(json.dumps({"argv": sys.argv[1:], "host": host, "command": " ".join(command)}) + "\n")
    control = [v for k, v in opts if k == "O"]
    if control:
        if env.get("FAKE_SSH_MASTER") == "1":
            sys.stderr.write("Master running (pid=4242)\n" if control[0] == "check" else "Exit request sent.\n")
            return 0
        sys.stderr.write("Control socket connect(/x/ssh-abc): No such file or directory\n")
        return 255
    if env.get("FAKE_SSH_PIDFILE"):
        with open(env["FAKE_SSH_PIDFILE"], "w") as fh:
            fh.write(str(os.getpid()))
    if env.get("FAKE_SSH_SLEEP"):
        time.sleep(float(env["FAKE_SSH_SLEEP"]))
    if "FAKE_SSH_FAIL_STDERR" in env:
        sys.stderr.write(env["FAKE_SSH_FAIL_STDERR"] + "\n")
        return int(env.get("FAKE_SSH_FAIL_CODE", "255"))
    if env.get("FAKE_SSH_WARN"):
        sys.stderr.write(env["FAKE_SSH_WARN"] + "\n")
        sys.stderr.flush()
    if env.get("FAKE_SSH_NOISE"):
        sys.stdout.write(env["FAKE_SSH_NOISE"])
        sys.stdout.flush()
    remote_env = dict(env)
    if env.get("FAKE_SSH_REMOTE_PATH"):
        remote_env["PATH"] = env["FAKE_SSH_REMOTE_PATH"]
    shell = env.get("FAKE_SSH_SHELL") or "/bin/bash"
    rc = subprocess.call([shell, "-c", " ".join(command)], env=remote_env)
    if env.get("FAKE_SSH_DROP_AFTER"):
        sys.stderr.write(env["FAKE_SSH_DROP_AFTER"] + "\n")
        return 255
    return rc


sys.exit(main())
"""

FAKE_RSYNC = r"""
import json, os, shlex, shutil, subprocess, sys


def sync_dir(src, dst, delete):
    if os.path.lexists(dst) and not os.path.isdir(dst):
        os.unlink(dst)
    os.makedirs(dst, exist_ok=True)
    names = set(os.listdir(src))
    for name in names:
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.islink(s) or (os.path.lexists(d) and os.path.isdir(s) != os.path.isdir(d)):
            if os.path.isdir(d) and not os.path.islink(d):
                shutil.rmtree(d)
            elif os.path.lexists(d):
                os.unlink(d)
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)
        elif os.path.isdir(s):
            sync_dir(s, d, delete)
        else:
            shutil.copy2(s, d)
    if delete:
        for name in set(os.listdir(dst)) - names:
            d = os.path.join(dst, name)
            if os.path.isdir(d) and not os.path.islink(d):
                shutil.rmtree(d)
            else:
                os.unlink(d)


def split_remote(spec):
    if ":" in spec and not spec.startswith("/"):
        return spec.split(":", 1)
    return None, spec


def main():
    args = sys.argv[1:]
    flags, positional, shell, rsync_path = set(), [], None, None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-e":
            i += 1
            shell = args[i]
        elif arg.startswith("--rsync-path="):
            rsync_path = arg[len("--rsync-path="):]
        elif arg.startswith("-"):
            flags.add(arg)
        else:
            positional.append(arg)
        i += 1
    if os.environ.get("FAKE_RSYNC_LOG"):
        with open(os.environ["FAKE_RSYNC_LOG"], "a") as fh:
            fh.write(json.dumps({"argv": args}) + "\n")
    src, dst = positional
    (src_host, src_path), (dst_host, dst_path) = split_remote(src), split_remote(dst)
    host = src_host or dst_host
    if host:
        remote = (rsync_path[: -len("rsync")] + "true") if rsync_path else "true"
        rc = subprocess.call([*shlex.split(shell), host, remote])
        if rc != 0:
            sys.stderr.write("rsync: connection unexpectedly closed (0 bytes received so far) [sender]\n")
            return 255 if rc == 255 else 12
    if "FAKE_RSYNC_FAIL_STDERR" in os.environ:
        sys.stderr.write(os.environ["FAKE_RSYNC_FAIL_STDERR"] + "\n")
        return int(os.environ.get("FAKE_RSYNC_FAIL_CODE", "23"))
    if src_path.endswith("/"):
        sync_dir(src_path, dst_path, "--delete-after" in flags)
    else:
        if os.path.isdir(dst_path):
            dst_path = os.path.join(dst_path, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path, follow_symlinks="-L" in flags)
    return 0


sys.exit(main())
"""


def install_shim(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable} -S\n{source}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@dataclass
class FakeSsh:
    """Controls the fake ssh through environment variables (reset for every test)."""

    monkeypatch: pytest.MonkeyPatch
    log_path: Path

    def set(self, **env: str) -> None:
        for key, value in env.items():
            self.monkeypatch.setenv(f"FAKE_SSH_{key.upper()}", value)

    def clear(self, *keys: str) -> None:
        for key in keys:
            self.monkeypatch.delenv(f"FAKE_SSH_{key.upper()}", raising=False)

    def calls(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]


MakeSsh = Callable[..., OpenSSHTransport]
