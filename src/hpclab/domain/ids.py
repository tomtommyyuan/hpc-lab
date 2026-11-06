"""Identifiers.

Every entity id is ``<prefix>_<26 chars>``: a Crockford-base32 ULID (48-bit millisecond
timestamp + 80 random bits). Ids sort by creation time, which keeps listings stable, and
the prefix makes a pasted id self-describing ("att_..." is an Attempt).

Slurm job ids are *not* global identities; see ``SchedulerJobRef`` in ``models``.
"""

from __future__ import annotations

import os
import re
import secrets
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

PREFIXES: dict[str, str] = {
    "experiment": "exp",
    "plan": "pln",
    "run": "run",
    "attempt": "att",
    "operation": "op",
    "workflow": "wf",
    "artifact": "art",
    "evaluation": "ev",
    "comparison": "cmp",
    "evidence": "evb",
    "reservation": "bud",
    "request": "req",
    "event": "evt",
    "notification": "ntf",
    "project": "prj",
}

_ID_RE = re.compile(r"^(?P<prefix>[a-z]{2,4})_(?P<body>[0-9A-HJKMNP-TV-Z]{26})$")


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def ulid(now_ms: int | None = None) -> str:
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    rand = int.from_bytes(os.urandom(10), "big")
    return _encode(ms, 10) + _encode(rand, 16)


def new_id(kind: str) -> str:
    """Return a new id for an entity kind listed in ``PREFIXES``."""
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:  # programming error, not user input
        raise ValueError(f"unknown id kind {kind!r}") from exc
    return f"{prefix}_{ulid()}"


def kind_of(entity_id: str) -> str | None:
    """Return the entity kind for an id, or None if it is not an hpclab id."""
    m = _ID_RE.match(entity_id or "")
    if not m:
        return None
    prefix = m.group("prefix")
    for kind, p in PREFIXES.items():
        if p == prefix:
            return kind
    return None


def is_id(entity_id: str, kind: str) -> bool:
    return kind_of(entity_id) == kind


def submit_token() -> str:
    """A short random token embedded in the Slurm job name/comment and the execution manifest.

    It lets reconciliation prove that a queued job is *this* submission and not a
    look-alike. 12 lowercase base32 chars = 60 bits.
    """
    return "".join(secrets.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(12))


_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_label(label: str, max_len: int = 40) -> str:
    """Make a human label safe for Slurm job names and directory names."""
    cleaned = _LABEL_RE.sub("-", label).strip("-.") or "run"
    return cleaned[:max_len]
