"""The test counts quoted in the docs match the suite.

README, CHANGELOG, CONTRIBUTING and docs/guarantees.md quote how many tests there are (in
total, marked ``sim``, marked ``slow``, the fast subset). Those numbers drifted by hundreds once;
this collects the suite (collection only, no test runs) and requires every quoted count to be
within 5% of the real one, so a doc edit is due long before a claim becomes misleading.
Recount with ``uv run pytest --co | tail -1`` (add ``-m "not slow"`` etc. for a subset).
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOLERANCE = 0.05


@cache
def collected(marker: str | None) -> int:
    argv = [sys.executable, "-m", "pytest", "--co", "-q", "-o", "addopts=", "-p", "no:cacheprovider"]
    if marker:
        argv += ["-m", marker]
    out = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=240, check=False)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    m = re.match(r"^(\d+)(?:/\d+)? tests? collected", last)
    assert m, f"unexpected collection output: {last!r}\n{out.stderr[-2000:]}"
    return int(m.group(1))


def _n(text: str) -> int:
    return int(text.replace(",", ""))


# (file, regex capturing a count, marker expression the count is about)
CLAIMS: list[tuple[str, str, str | None]] = [
    ("README.md", r"and (?:about )?([\d,]+) tests\.", None),
    ("CHANGELOG.md", r"Slurm output and (?:about )?([\d,]+) tests", None),
    ("CHANGELOG.md", r"Slurm output and [\d,]+ tests \(([\d,]+) without the slow ones\)", "not slow"),
    ("CONTRIBUTING.md", r"(?m)^(?:About )?([\d,]+) tests\b", None),
    ("CONTRIBUTING.md", r"The fast suite \((?:about )?([\d,]+) tests\)", "not sim and not slow"),
    ("CONTRIBUTING.md", r"through the simulator \(([\d,]+)\)", "sim"),
    ("CONTRIBUTING.md", r"Everything except the slow simulator tests \(([\d,]+) tests\)", "not slow"),
    ("docs/guarantees.md", r"([\d,]+) tests at 0\.1\.0", None),
    ("docs/guarantees.md", r"([\d,]+) are marked `sim`", "sim"),
    ("docs/guarantees.md", r"([\d,]+) of which are also `slow`", "sim and slow"),
    ("docs/guarantees.md", r"\(`-m \"not sim and not slow\"`,\s+([\d,]+) tests\)", "not sim and not slow"),
]


@pytest.mark.parametrize(("doc", "pattern", "marker"), CLAIMS, ids=[f"{d}:{m}" for d, _, m in CLAIMS])
def test_quoted_test_counts_are_current(doc: str, pattern: str, marker: str | None) -> None:
    text = (REPO / doc).read_text(encoding="utf-8")
    m = re.search(pattern, text)
    assert m, f"{doc} no longer quotes this count ({pattern}); update CLAIMS"
    quoted, real = _n(m.group(1)), collected(marker)
    assert abs(quoted - real) <= TOLERANCE * real, (
        f"{doc} says {quoted} tests{f' for -m {marker!r}' if marker else ''}; the suite has {real} "
        "(recount with `uv run pytest --co | tail -1`)"
    )
