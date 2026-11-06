"""Slurm output conventions the simulator reproduces.

Everything here mirrors real Slurm behavior for the flags ``adapters.slurm`` relies on:

* ``SLURM_TIME_FORMAT`` (``%s`` -> epoch seconds; unset/``standard`` -> ``%Y-%m-%dT%H:%M:%S``
  local time; any other value is a strftime format).
* ``--Format``/``-O`` tokens ``type[:[.][size][suffix]]``: no ``:`` -> width 20; width ``0`` ->
  the full value without padding; ``.`` right-justifies. Values longer than the width are cut.
* ``-o``/``--format`` ``%[.][size]type`` tokens with literal text in between.
* Duration styles: squeue/sinfo ``secs2time_str`` (``M:SS``, ``H:MM:SS``, ``D-HH:MM:SS``),
  sacct elapsed (``HH:MM:SS``, ``D-HH:MM:SS``).
* Hostlists (``sim-a100-[01-03]``), GPU index ranges (``0-2,5``), memory (``64G``).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

INFINITE = None


def slurm_time(epoch: float | None, env: Mapping[str, str], *, missing: str = "Unknown") -> str:
    """Format a timestamp as Slurm would under ``SLURM_TIME_FORMAT``."""
    if epoch is None or epoch <= 0:
        return missing
    fmt = env.get("SLURM_TIME_FORMAT", "") or "standard"
    if fmt in ("standard", "relative"):
        fmt = "%Y-%m-%dT%H:%M:%S"
    if fmt == "%s":
        return str(int(epoch))
    return time.strftime(fmt, time.localtime(epoch))


def secs2time_str(secs: float | None) -> str:
    """squeue/sinfo duration style; None -> UNLIMITED."""
    if secs is None:
        return "UNLIMITED"
    s = max(0, int(secs))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def elapsed_str(secs: float | None) -> str:
    """sacct ``Elapsed``/``Timelimit`` style (``00:05:00``, ``2-00:00:00``); None -> UNLIMITED."""
    if secs is None:
        return "UNLIMITED"
    s = max(0, int(secs))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


_TIME_RE = re.compile(r"^(?:(?P<d>\d+)-)?(?P<rest>[\d:]+)$")


def parse_time_limit(value: str) -> int | None:
    """sbatch ``--time`` -> seconds. Accepts minutes, M:S, H:M:S, D-H, D-H:M, D-H:M:S.

    ``0``/``UNLIMITED``/``infinite`` -> None (no limit). Raises ValueError when invalid.
    """
    v = value.strip()
    if v.lower() in ("unlimited", "infinite", "-1"):
        return None
    m = _TIME_RE.match(v)
    if not m:
        raise ValueError(value)
    parts = [int(p) for p in m.group("rest").split(":") if p != ""]
    if not parts or len(parts) > 3:
        raise ValueError(value)
    if m.group("d") is not None:
        days = int(m.group("d"))
        while len(parts) < 3:
            parts.append(0)
        h, mi, s = parts
        total = days * 86400 + h * 3600 + mi * 60 + s
    elif len(parts) == 1:
        total = parts[0] * 60
    elif len(parts) == 2:
        total = parts[0] * 60 + parts[1]
    else:
        total = parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None if total == 0 else total


_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)B?$", re.IGNORECASE)


def parse_mem_mb(value: str) -> int:
    """sbatch memory (``64G``, ``512M``, ``4096`` = MB) -> MB. Raises ValueError."""
    m = _MEM_RE.match(value.strip())
    if not m:
        raise ValueError(value)
    num = float(m.group(1))
    unit = m.group(2).upper() or "M"
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    return max(1, int(num * factor)) if num > 0 else 0


def format_mem(mb: int) -> str:
    """TRES memory: ``64G`` when whole gigabytes, else ``500M``."""
    if mb and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def format_rss_k(n_bytes: int) -> str:
    """sstat/sacct MaxRSS style (kilobytes with K suffix; 0 prints as 0)."""
    if n_bytes <= 0:
        return "0"
    return f"{n_bytes // 1024}K"


def compress_ranges(values: Iterable[int], width: int = 0) -> str:
    """``[0,1,2,5]`` -> ``0-2,5`` (zero-padded to ``width``)."""
    nums = sorted(set(values))
    out: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        a, b = f"{nums[i]:0{width}d}", f"{nums[j]:0{width}d}"
        out.append(a if i == j else f"{a}-{b}")
        i = j + 1
    return ",".join(out)


_HOST_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


def compress_hostlist(names: Iterable[str]) -> str:
    """``[sim-a100-01, sim-a100-02]`` -> ``sim-a100-[01-02]``; single host stays bare."""
    groups: dict[tuple[str, int], list[int]] = {}
    plain: list[str] = []
    order: list[tuple[str, int] | str] = []
    for name in names:
        m = _HOST_RE.match(name)
        if not m:
            plain.append(name)
            order.append(name)
            continue
        key = (m.group("prefix"), len(m.group("num")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(m.group("num")))
    parts: list[str] = []
    for item in order:
        if isinstance(item, str):
            parts.append(item)
            continue
        prefix, width = item
        nums = groups[item]
        if len(nums) == 1:
            parts.append(f"{prefix}{nums[0]:0{width}d}")
        else:
            parts.append(f"{prefix}[{compress_ranges(nums, width)}]")
    return ",".join(parts)


def expand_hostlist(expr: str) -> list[str]:
    """Inverse of ``compress_hostlist`` for ``a[01-03,07],b1``."""
    out: list[str] = []
    for token in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", expr):
        m = re.match(r"^(?P<prefix>[^\[]*)\[(?P<body>[^\]]*)\](?P<suffix>.*)$", token)
        if not m:
            if token:
                out.append(token)
            continue
        for part in m.group("body").split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                width = len(a)
                out.extend(
                    f"{m.group('prefix')}{i:0{width}d}{m.group('suffix')}" for i in range(int(a), int(b) + 1)
                )
            elif part:
                out.append(f"{m.group('prefix')}{part}{m.group('suffix')}")
    return out


# ---------------------------------------------------------------------------
# --Format / --format rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    width: int
    right: bool
    suffix: str


def parse_long_format(spec: str) -> list[FieldSpec]:
    """``-O``/``--Format`` value -> field specs (Slurm's ``_parse_long_token``)."""
    out: list[FieldSpec] = []
    for token in spec.split(","):
        if not token:
            continue
        name, sep, rest = token.partition(":")
        if not sep:
            out.append(FieldSpec(name=name, width=20, right=False, suffix=""))
            continue
        right = rest.startswith(".")
        if right:
            rest = rest[1:]
        digits = re.match(r"\d*", rest)
        assert digits is not None
        width = int(digits.group(0)) if digits.group(0) else 0
        out.append(FieldSpec(name=name, width=width, right=right, suffix=rest[len(digits.group(0)) :]))
    return out


_SHORT_RE = re.compile(r"%(?P<right>\.)?(?P<width>\d*)(?P<type>[A-Za-z%])")


def parse_short_format(spec: str) -> list[FieldSpec | str]:
    """``-o "%.18i %T|%j"`` -> specs and literal strings. ``%%`` is a literal percent."""
    out: list[FieldSpec | str] = []
    pos = 0
    for m in _SHORT_RE.finditer(spec):
        if m.start() > pos:
            out.append(spec[pos : m.start()])
        if m.group("type") == "%":
            out.append("%")
        else:
            width = int(m.group("width")) if m.group("width") else 0
            out.append(FieldSpec(name=m.group("type"), width=width, right=bool(m.group("right")), suffix=""))
        pos = m.end()
    if pos < len(spec):
        out.append(spec[pos:])
    return out


def render_value(value: str, spec: FieldSpec) -> str:
    """Slurm's ``_print_str``: width 0 -> as is; else cut to width (left) and pad."""
    if spec.width == 0:
        text = value
    elif spec.right:
        text = value[: spec.width].rjust(spec.width)
    else:
        text = value[: spec.width].ljust(spec.width)
    return text + spec.suffix


def format_datetime_local(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%S")


# sacct/sinfo -S/-E parsing -------------------------------------------------------

_REL_RE = re.compile(r"^now(?:(?P<sign>[+-])(?P<num>\d+)(?P<unit>[a-z]*))?$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
    "week": 7 * 86400,
    "weeks": 7 * 86400,
}


def parse_timespec(value: str, now: float) -> float:
    """Slurm time spec: ``now[+-N[unit]]``, ``YYYY-MM-DD[THH:MM[:SS]]`` (local), ``MM/DD[/YY]``,
    ``HH:MM[:SS]`` (today). Raises ValueError."""
    v = value.strip()
    m = _REL_RE.match(v)
    if m:
        if not m.group("sign"):
            return now
        unit = m.group("unit").lower()
        if unit not in _UNITS:
            raise ValueError(value)
        delta = int(m.group("num")) * _UNITS[unit]
        return now + delta if m.group("sign") == "+" else now - delta
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).timestamp()
        except ValueError:
            continue
    today = datetime.fromtimestamp(now)
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(v, fmt)
            return today.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0).timestamp()
        except ValueError:
            continue
    for fmt in ("%m/%d/%y", "%m/%d"):
        try:
            d = datetime.strptime(v, fmt)
            year = d.year if fmt.endswith("%y") else today.year
            return d.replace(year=year).timestamp()
        except ValueError:
            continue
    raise ValueError(value)
