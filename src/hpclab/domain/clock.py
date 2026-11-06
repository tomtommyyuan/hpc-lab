"""Time helpers. All persisted timestamps are UTC ISO-8601 with a trailing ``Z``.

Services take a ``Clock`` so tests can control time (reconcile windows, poll intervals,
budget settlement) without sleeping.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """Manually advanced clock for tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
        self._mono = 1000.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


_SLURM_TIME_RE = re.compile(r"^(?:(?P<d>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+)(?::(?P<s>\d+))?$")


def parse_slurm_duration(value: str | None) -> int | None:
    """Parse Slurm time strings to seconds.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``.
    Returns None for ``UNLIMITED``, ``INVALID``, ``Partition_Limit``, empty or unparsable.
    """
    if value is None:
        return None
    v = value.strip()
    if not v or v.upper() in {"UNLIMITED", "INVALID", "NONE", "N/A", "PARTITION_LIMIT"}:
        return None
    if "-" in v:
        days_s, rest = v.split("-", 1)
        if not days_s.isdigit():
            return None
        days = int(days_s)
        parts = rest.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        while len(nums) < 3:
            nums.append(0)
        h, m, s = nums[:3]
        return days * 86400 + h * 3600 + m * 60 + s
    parts = v.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0] * 60
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def format_slurm_duration(seconds: int) -> str:
    """Format seconds as Slurm ``D-HH:MM:SS`` (or ``HH:MM:SS`` under a day)."""
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if days:
        return f"{days}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"
