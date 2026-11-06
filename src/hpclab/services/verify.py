"""Evidence verification (``evidence_verify``) and cell tracing (``trace``).

``evidence_verify`` re-derives what a bundle claims from the ledger and checks a manuscript's
``\\hpcval{<bundle>:<cell>}{<number>}`` tags against it. It **verifies record consistency, not
scientific correctness**: a clean report means every number traces to archived records and was
copied faithfully, not that the experiment answers its question.

Findings (``code`` → default severity):

=============================  ========  ==================================================
AGGREGATE_MISMATCH             error     bundle/archive disagrees with a recomputation from
                                         the evaluations (or was modified after archiving)
N_BELOW_EXPECTED_UNFOOTNOTED   error     a tex cell with n < expected_n has no footnote mark
SEED_SELECTION                 warn      post-hoc seeds, exclusions decided after observing the
                                         result, evaluations pinned by hand (evaluation_refs)
                                         instead of attempt_selection (info: several valid
                                         attempts per run)
MIXED_CODE_DIGEST              warn/info code differs within a group (info: only across groups)
MIXED_GPU_TYPE                 warn      included values ran on different GPU models
INVARIANT_MISMATCH             error     the comparison flagged incomparable results
ARTIFACT_DIGEST_CHANGED        error     the artifact behind a value changed (ledger record, or
                                         the file on the cluster with ``recheck_remote``)
SELECTION_LEAK                 error     best-of selection on the reported split
STALE_TEX_NUMBER               error     a newer comparison that answers the SAME question (same
                                         run set, contract, exclusions and evaluation filter; the
                                         newest such one that has the metric), or a read-only
                                         recomputation of the bundle's own request from the
                                         current ledger, gives a different number than the tex
                                         shows. Exploratory comparisons (other exclusions or
                                         contract) never make a published table stale.
VALUE_MISMATCH                 error     tex number ≠ bundle value at the tex's own precision
UNRESOLVED_REF                 error     tag/bundle/cell/evaluation/run reference not found
NO_TAGS_FOUND                  error     a manuscript was given but none of its scanned files has
                                         an ``\\hpcval`` tag: it was NOT checked
UNRESOLVED_INPUT               warn      an ``\\input``/``\\include``/``\\subfile``/``\\import``
                                         could not be read (or a scan bound was hit): the numbers
                                         in that file were not checked (coverage incomplete)
UNTAGGED_TEX_NUMBER            info      a decimal number in a tabular without a tag
=============================  ========  ==================================================

Manuscripts: every ``tex_paths`` file (and ``tex_content``) is scanned together with the files
it pulls in with ``\\input{f}``, ``\\include{f}``, ``\\subfile{f}`` and ``\\import{dir}{f}``
(also ``\\subimport``), resolved relative to the including file, then to the main file's
directory, with ``.tex`` appended when missing (quotes and grouping braces in a name are
dropped, as LaTeX does); each file is read once (cycles are harmless), at most :data:`MAX_INCLUDE_DEPTH` levels
deep, :data:`MAX_INCLUDE_FILES` files and :data:`MAX_TOTAL_TEX_BYTES` bytes per call. Text
LaTeX never typesets — ``%`` comments, ``comment`` environments and plain ``\\iffalse ... \\fi``
blocks — is skipped: its tags are not checked and its includes are not followed.
``files_scanned`` lists the files in scan order; ``checked`` says whether any ``\\hpcval`` tag
was found at all.

``trace`` resolves ``<bundle_id>:<cell_id>`` to the whole chain of every seed behind the
cell: the frozen provenance row plus the live ledger records, with any drift listed.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hpclab.adapters.base import AuthRequiredError
from hpclab.api.envelope import Coverage, SourceStatus
from hpclab.domain.errors import HpcLabError, InvalidArgument, NotFound
from hpclab.services.compare import (
    ARCHIVE_SCHEMA,
    ExclusionSpec,
    Reader,
    archive_content_hash,
    comparison_request_key,
    compute_comparison,
    env_facts,
    load_archive,
    strip_ref,
)
from hpclab.services.context import ServiceContext
from hpclab.services.localfiles import FileTooLarge, NotRegularFile, read_regular_file
from hpclab.services.render import CELL_STATS, cell_ids, parse_bundle, parse_cell_id
from hpclab.services.stats import summarize
from hpclab.storage.db import jload

log = logging.getLogger("hpclab.verify")

__all__ = [
    "STATEMENT",
    "Finding",
    "TexTag",
    "TraceData",
    "VerifyData",
    "evidence_verify",
    "outbox_handlers",
    "parse_hpcval_tags",
    "parse_tex_number",
    "periodic_tasks",
    "trace",
]

STATEMENT = (
    "evidence_verify checks record consistency (ledger, archived comparison, bundle and manuscript "
    "numbers); it does not verify scientific correctness."
)
MAX_FINDINGS = 200
MAX_TEX_FILES = 20  # tex_paths per call (each may pull in more files through includes)
MAX_TEX_BYTES = 4 * 1024 * 1024  # per file
MAX_INCLUDE_DEPTH = 8  # nesting of \input/\include/\subfile/\import below a given file
MAX_INCLUDE_FILES = 200  # files scanned per call, given ones included
MAX_TOTAL_TEX_BYTES = 16 * 1024 * 1024  # bytes scanned per call
MAX_UNTAGGED_FINDINGS = 20
MAX_REMOTE_ARTIFACTS = 64
MAX_REMOTE_FILE_BYTES = 16 * 1024 * 1024
MAX_NEWER_SCAN = 200  # newer comparisons over the run set searched (newest first) for the same request
MAX_LIVE_RECOMPUTES = 10  # bundles per call re-derived from the current ledger
_REL_TOL = 1e-9


def _load_bundle(reader: Reader, bundle_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``evidence`` re-exports :func:`trace` (the facade resolves it there), so this import is deferred.
    from hpclab.services.evidence import load_bundle

    return load_bundle(reader, bundle_ref)


Severity = Literal["error", "warn", "info"]
FindingCode = Literal[
    "AGGREGATE_MISMATCH",
    "N_BELOW_EXPECTED_UNFOOTNOTED",
    "SEED_SELECTION",
    "MIXED_CODE_DIGEST",
    "MIXED_GPU_TYPE",
    "INVARIANT_MISMATCH",
    "ARTIFACT_DIGEST_CHANGED",
    "SELECTION_LEAK",
    "STALE_TEX_NUMBER",
    "VALUE_MISMATCH",
    "UNRESOLVED_REF",
    "UNTAGGED_TEX_NUMBER",
    "UNRESOLVED_INPUT",
    "NO_TAGS_FOUND",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(_Model):
    code: FindingCode
    severity: Severity
    message: str
    refs: list[str] = Field(default_factory=list)


class TexSourceReport(_Model):
    source: str
    tags: int
    untagged_numbers: int
    bundles: list[str] = Field(default_factory=list)
    included_from: str | None = None  # "<file>:<line>" of the include that pulled this file in


class VerifyData(_Model):
    bundle_id: str
    bundle_ref: str
    comparison_id: str | None
    comparison_ref: str | None
    ok: bool
    statement: str = STATEMENT
    findings: list[Finding]
    findings_truncated: int = 0
    counts_by_code: dict[str, int] = Field(default_factory=dict)
    counts_by_severity: dict[str, int] = Field(default_factory=dict)
    tex: list[TexSourceReport] = Field(default_factory=list)
    # Absolute paths of every manuscript file read (given ones and their includes), in scan order.
    files_scanned: list[str] = Field(default_factory=list)
    # True iff at least one \hpcval tag was found in the scanned manuscript (False without one).
    checked: bool = False
    newer_comparison_ref: str | None = None
    # Would re-running the bundle's comparison on today's ledger change any cell? None = unknown.
    ledger_changed: bool | None = None
    values_checked: int = 0
    tags_checked: int = 0
    artifacts_rechecked: int = 0
    summary: str
    coverage: Coverage
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TeX parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TexTag:
    ref: str
    value: str
    line: int
    start: int
    end: int


_HPCVAL_RE = re.compile(r"\\hpcval(?![A-Za-z@])")
_FOOTNOTE_MARKS = (
    r"\dag",
    r"\ddag",
    r"\textdagger",
    r"\textdaggerdbl",
    r"\footnotemark",
    r"\tnote",
    r"\textsuperscript",
    r"\S",
    "^",
    "†",
    "‡",
)
_TABULAR_RE = re.compile(
    r"\\begin\{(tabular\*?|tabularx|tabulary|longtable\*?|array)\}(.*?)\\end\{\1\}", re.S
)
_DECIMAL_RE = re.compile(r"(?<![\w.\\])[-+−]?\d+\.\d+(?![\w.])")


def _strip_comments(text: str) -> str:
    """Blank out what LaTeX never typesets, keeping every offset and newline in place: ``%``
    comments (not ``\\%``), ``comment`` environments and plain ``\\iffalse ... \\fi`` blocks
    (see :func:`_blank_dead_blocks`). Tags and includes in them are neither checked nor followed."""
    out = []
    for line in text.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            ch = line[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut] + " " * (len(line) - cut))
    return _blank_dead_blocks("\n".join(out))


_DEAD_START_RE = re.compile(r"\\begin\s*\{comment\}|^[ \t]*\\iffalse(?![A-Za-z@])", re.M)
_COMMENT_END_RE = re.compile(r"\\end\s*\{comment\}")
_CONDITIONAL_RE = re.compile(r"\\(?:if[A-Za-z@]*|else|fi)(?![A-Za-z@])")


def _blank_dead_blocks(text: str) -> str:
    """Blank ``\\begin{comment}...\\end{comment}`` and ``\\iffalse ... \\fi`` (up to an ``\\else``,
    whose branch is typeset).

    Conservative on purpose — hiding a typeset table would turn a check into a false "ok",
    while scanning dead text only risks a visible false finding: ``\\iffalse`` counts only at
    the start of a line (so ``\\let\\ifdraft\\iffalse`` and macro bodies are not blocks), only
    when the next conditional token is its own ``\\fi``/``\\else`` (no nested conditionals), and
    a block without its end is left alone.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while (m := _DEAD_START_RE.search(text, pos)) is not None:
        if m.group(0).lstrip().startswith("\\begin"):
            end = _COMMENT_END_RE.search(text, m.end())
        else:
            nxt = _CONDITIONAL_RE.search(text, m.end())
            end = nxt if nxt is not None and nxt.group(0) in ("\\fi", "\\else") else None
        if end is None:
            pos = m.end()
            continue
        spans.append((m.start(), end.end()))
        pos = end.end()
    if not spans:
        return text
    chars = list(text)
    for start, stop in spans:
        for i in range(start, stop):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _group(text: str, pos: int) -> tuple[str, int] | None:
    """Balanced ``{...}`` starting at ``pos`` (after optional spaces) → (content, end)."""
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None
    depth = 0
    i = pos
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    return None


def parse_hpcval_tags(text: str) -> list[TexTag]:
    """Every ``\\hpcval{ref}{value}`` outside comments (definitions like
    ``\\providecommand{\\hpcval}...`` are not tags)."""
    clean = _strip_comments(text)
    tags = []
    for m in _HPCVAL_RE.finditer(clean):
        first = _group(clean, m.end())
        if first is None:
            continue
        second = _group(clean, first[1])
        if second is None:
            continue
        tags.append(
            TexTag(
                ref=" ".join(first[0].split()),
                value=second[0],
                line=clean.count("\n", 0, m.start()) + 1,
                start=m.start(),
                end=second[1],
            )
        )
    return tags


_NUM_RE = re.compile(
    r"^(?P<sign>[+-]?)(?P<int>\d+)(?:\.(?P<frac>\d+))?"
    r"(?:(?:\\times|\\cdot|x|×)10\^(?P<exp>[+-]?\d+)|[eE](?P<exp2>[+-]?\d+))?$"
)
_STRIP_MACROS = re.compile(
    r"\\(?:mathbf|textbf|bm|boldsymbol|mathrm|text|textrm|underline|emph|num|textit|mathit)(?![A-Za-z])"
)


def parse_tex_number(text: str) -> tuple[float, float] | None:
    """(value, half a unit of the last shown digit) for a typeset number, else None.

    Accepts ``0.919``, ``$\\mathbf{0.919}$``, ``-0.013``, ``91.9\\%``, ``1{,}234.5``,
    ``1.2\\times10^{-5}`` and ``1.2e-05``.
    """
    t = _STRIP_MACROS.sub("", text)
    t = t.replace("\\%", "").replace("%", "").replace("−", "-").replace("\\!", "")
    for token in ("\\,", "\\;", "\\:", "\\ ", "~", "$", "{", "}", " ", "\t", "\n"):
        t = t.replace(token, "")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    m = _NUM_RE.match(t)
    if m is None:
        return None
    frac = m.group("frac") or ""
    exp = int(m.group("exp") or m.group("exp2") or 0)
    try:
        value = float(f"{m.group('sign')}{m.group('int')}.{frac or '0'}") * 10.0**exp
    except OverflowError:
        return None
    if not math.isfinite(value):
        return None
    return value, 0.5 * 10.0 ** (exp - len(frac))


def _matches(shown: tuple[float, float], expected: float) -> bool:
    value, half = shown
    slack = half * (1 + 1e-9) + 1e-12 * max(1.0, abs(expected))
    return abs(value - expected) <= slack


def _cell_context(text: str, end: int) -> str:
    """Text after a tag up to the end of its table cell (``&`` or ``\\\\``)."""
    window = text[end : end + 400]
    stops = [i for i in (window.find("&"), window.find("\\\\")) if i >= 0]
    return window[: min(stops)] if stops else window.split("\n", 1)[0]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class _Findings:
    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, code: FindingCode, severity: Severity, message: str, refs: Iterable[str] = ()) -> None:
        self.items.append(
            Finding(code=code, severity=severity, message=message, refs=[r for r in refs if r][:20])
        )


@dataclass
class _Newer:
    """Numbers newer than a bundle: an archived comparison, or the current ledger (``comparison_id`` None)."""

    comparison_id: str | None
    groups: dict[tuple[str, str], dict[str, Any]]  # (label, metric) -> cell
    metrics: set[str]

    def gives(self) -> str:
        if self.comparison_id is None:
            return "the current ledger (the bundle's request recomputed read-only) gives"
        return f"the newer comparison {self.comparison_id} of the same request gives"


@dataclass
class _Freshness:
    """What is newer than one bundle's comparison."""

    # metric -> newest newer archived comparison of the same request (request_key) with the metric
    archived: dict[str, _Newer]
    newest_ref: str | None  # newest newer same-request comparison sharing a metric with the bundle
    live: _Newer | None  # set only when today's ledger changes some cell of the bundle's comparison
    ledger_changed: bool | None  # None: the recomputation was not possible (see ``note``)
    changed_cells: list[str]
    note: str | None = None
    changed_fields: dict[str, list[str]] = field(default_factory=dict)  # cell -> differing signature fields
    # the MAX_NEWER_SCAN window of newer comparisons was exhausted before every metric of the
    # bundle had a same-request match: an older same-request comparison may have been missed
    scan_truncated: bool = False


class _BundleView:
    """A stored bundle with lookups by cell id (shared by tex checks and trace)."""

    def __init__(self, manifest: dict[str, Any], row: Mapping[str, Any]) -> None:
        self.manifest = manifest
        self.row = row
        self.bundle_id: str = row["bundle_id"]
        self.groups = {g["label"]: g for g in manifest.get("groups") or [] if isinstance(g, dict)}
        cids = manifest.get("cell_ids")
        if not isinstance(cids, dict):
            cids = {base: {"group_label": g, "metric": m} for (g, m), base in cell_ids(manifest).items()}
        self.cell_ids: dict[str, dict[str, Any]] = cids
        table = manifest.get("table") if isinstance(manifest.get("table"), dict) else {}
        self.scale = float((table or {}).get("scale") or 1.0)
        self.provenance = [p for p in manifest.get("provenance") or [] if isinstance(p, dict)]

    def cell(self, base: str) -> tuple[str, str, dict[str, Any]] | None:
        entry = self.cell_ids.get(base)
        if not entry:
            return None
        group = self.groups.get(entry.get("group_label", ""))
        metric = entry.get("metric", "")
        if group is None or metric not in (group.get("metrics") or {}):
            return None
        return group["label"], metric, group["metrics"][metric]

    def base_of(self, label: str, metric: str) -> str | None:
        for base, entry in self.cell_ids.items():
            if entry.get("group_label") == label and entry.get("metric") == metric:
                return base
        return None


def _stat_value(cell: Mapping[str, Any], stat: str) -> tuple[float | None, str]:
    """(value, description) of a statistic or ``seed-<s>`` of a bundle cell."""
    if stat.startswith("seed-"):
        seed = stat[len("seed-") :]
        v = (cell.get("values_by_seed") or {}).get(seed)
        return (float(v) if v is not None else None), f"seed {seed}"
    v = cell.get(stat)
    return (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None), stat


def evidence_verify(
    ctx: ServiceContext,
    *,
    bundle_ref: str,
    tex_paths: list[str] | None = None,
    tex_content: str | None = None,
    recheck_remote: bool = False,
) -> VerifyData:
    """Recompute a bundle from the ledger and check manuscript ``\\hpcval`` numbers against it.

    Verifies record consistency, not scientific correctness. ``tex_paths`` are absolute paths
    readable by the controller; the files they ``\\input``/``\\include``/``\\subfile``/``\\import``
    are scanned too (``files_scanned``). A manuscript without any ``\\hpcval`` tag was not
    checked: ``NO_TAGS_FOUND`` (error), ``checked=False``. ``recheck_remote`` re-hashes small
    metrics artifacts on the cluster (bounded; an unreachable cluster makes coverage
    incomplete, never a finding).
    """
    manifest, row = _load_bundle(ctx.db, bundle_ref)
    view = _BundleView(manifest, row)
    findings = _Findings()
    coverage = Coverage(sources=[SourceStatus(name="ledger", status="ok")])
    warnings: list[str] = []
    values_checked = _check_bundle(ctx.db, view, findings)
    fresh = _Fresher(ctx)
    newer = fresh.of(view)
    has_tex = bool(tex_paths) or tex_content is not None
    if newer.newest_ref is not None and not has_tex:
        warnings.append(
            f"a newer comparison of the same request (run set, contract, exclusions, evaluation filter) "
            f"exists ({newer.newest_ref}); pass the manuscript (tex_paths/tex_content) to find stale "
            "numbers, or re-export the bundle"
        )
    if newer.ledger_changed:
        changed = newer.changed_fields
        cells = ", ".join(f"{c} ({', '.join(changed.get(c) or [])})" for c in newer.changed_cells[:10])
        only_buckets = all(set(f) <= _BUCKET_FIELDS for f in changed.values())
        warnings.append(
            f"the ledger changed since this bundle's comparison was archived: {len(newer.changed_cells)} "
            f"cell(s) would differ today ({cells}{' ...' if len(newer.changed_cells) > 10 else ''})"
            + (
                "; only which seeds count as missing or failed changed (e.g. a pending evaluation came "
                "out invalid), not the numbers, so the footnotes are out of date"
                if only_buckets
                else ""
            )
            + "; re-run results_compare and re-export the bundle"
        )
    if newer.ledger_changed is None:
        coverage.add("ledger_recompute", "unavailable", detail=newer.note)
        warnings.append(f"could not re-derive the comparison from the current ledger: {newer.note}")
    else:
        coverage.add("ledger_recompute", "ok")
    scan = _scan_manuscript(tex_paths, tex_content, coverage, warnings, findings)
    reports: list[TexSourceReport] = []
    tags_checked = 0
    views = {view.bundle_id: view}
    for name, text, included_from in scan.sources:
        report = _check_tex(ctx.db, name, text, views, fresh, findings)
        tags_checked += report.tags
        reports.append(report.model_copy(update={"included_from": included_from}))
    checked = tags_checked > 0
    if has_tex and not checked:
        given = scan.files + (["tex_content"] if tex_content is not None else [])
        given = given or [str(p) for p in tex_paths or []]
        listing = ", ".join(given[:5]) + (f" and {len(given) - 5} more" if len(given) > 5 else "")
        findings.add(
            "NO_TAGS_FOUND",
            "error",
            f"no \\hpcval tags found in {listing}: the manuscript was NOT checked; pass the file that "
            "contains the table or \\input it",
            given,
        )
    if fresh.skipped:
        coverage.add(
            "ledger_recompute:other_bundles",
            "partial",
            detail=f"{fresh.skipped} further bundle(s) referenced by the tex were not re-derived "
            f"(at most {MAX_LIVE_RECOMPUTES} per call); only archived comparisons were checked for them",
        )
    _disclose_truncated_scans(fresh, coverage)
    rechecked = _check_remote(ctx, view, findings, coverage) if recheck_remote else 0
    if not recheck_remote:
        coverage.add("remote", "skipped", detail="recheck_remote=false: cluster files were not re-hashed")

    items = findings.items
    counts_code: dict[str, int] = {}
    counts_sev: dict[str, int] = {}
    for f in items:
        counts_code[f.code] = counts_code.get(f.code, 0) + 1
        counts_sev[f.severity] = counts_sev.get(f.severity, 0) + 1
    order = {"error": 0, "warn": 1, "info": 2}
    items = sorted(items, key=lambda f: order[f.severity])
    ok = counts_sev.get("error", 0) == 0
    cid = manifest.get("comparison_id") if isinstance(manifest.get("comparison_id"), str) else None
    summary = (
        f"bundle {view.bundle_id}: {'no errors' if ok else str(counts_sev.get('error', 0)) + ' error(s)'}, "
        f"{counts_sev.get('warn', 0)} warning(s), {counts_sev.get('info', 0)} info; {values_checked} value(s) "
        f"recomputed, {tags_checked} tex tag(s) checked (record consistency only)"
    )
    if has_tex and not checked:
        summary += "; the manuscript was NOT checked (no \\hpcval tags found)"
    if not coverage.complete:
        gaps = [s.name for s in coverage.sources if s.status not in ("ok", "cached", "skipped")]
        summary += f"; INCOMPLETE: not checked: {', '.join(gaps[:5])}"
    return VerifyData(
        bundle_id=view.bundle_id,
        bundle_ref=f"hpclab://evidence/{view.bundle_id}",
        comparison_id=cid,
        comparison_ref=f"hpclab://comparisons/{cid}" if cid else None,
        ok=ok,
        findings=items[:MAX_FINDINGS],
        findings_truncated=max(0, len(items) - MAX_FINDINGS),
        counts_by_code=dict(sorted(counts_code.items())),
        counts_by_severity=counts_sev,
        tex=reports,
        files_scanned=scan.files,
        checked=checked,
        newer_comparison_ref=f"hpclab://comparisons/{newer.newest_ref}" if newer.newest_ref else None,
        ledger_changed=newer.ledger_changed,
        values_checked=values_checked,
        tags_checked=tags_checked,
        artifacts_rechecked=rechecked,
        summary=summary,
        coverage=coverage,
        warnings=warnings,
    )


def _disclose_truncated_scans(fresh: _Fresher, coverage: Coverage) -> None:
    """Coverage-honest staleness: say when the bounded search for a newer same-request
    comparison may have missed one. Harmless (a note) for a bundle whose own request was
    recomputed from the current ledger — that catches every later change; a gap otherwise."""
    truncated = [bid for bid, f in fresh.cache.items() if f.scan_truncated]
    if not truncated:
        return
    text = (
        f"only the newest {MAX_NEWER_SCAN} newer comparisons of the run set were searched for one of the "
        f"same request (bundle(s) {', '.join(truncated[:5])}{' ...' if len(truncated) > 5 else ''})"
    )
    blind = [bid for bid in truncated if fresh.cache[bid].ledger_changed is None]
    if blind:
        coverage.add(
            "newer_comparisons",
            "partial",
            detail=f"{text}; bundle(s) {', '.join(blind[:5])} were not recomputed from the ledger either",
        )
    else:
        coverage.notes.append(f"{text}; the recomputation from the current ledger still checked each one")


# Cell-signature fields that only say which seeds are missing vs failed (footnotes, not numbers).
_BUCKET_FIELDS = frozenset({"missing_seeds", "failed_seeds"})


def _close(a: float | None, b: float | None, scale: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_REL_TOL * max(scale, 1e-300))


def _first_metric_value(metrics_json: str | None, name: str) -> tuple[bool, float | None]:
    for m in jload(metrics_json, []) or []:
        if isinstance(m, dict) and m.get("name") == name:
            v = m.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return True, float(v)
            return True, None
    return False, None


def _rows_by_id(reader: Reader, table: str, key: str, ids: Iterable[str | None]) -> dict[str, dict[str, Any]]:
    uniq = sorted({i for i in ids if i})
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(uniq), 500):
        chunk = uniq[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        for r in reader.all(f"SELECT * FROM {table} WHERE {key} IN ({marks})", chunk):
            out[r[key]] = r
    return out


def _check_bundle(reader: Reader, view: _BundleView, findings: _Findings) -> int:
    """Everything that needs only the ledger. Returns the number of per-seed values recomputed."""
    manifest = view.manifest
    bref = f"hpclab://evidence/{view.bundle_id}"
    try:
        parse_bundle(manifest)
    except HpcLabError as exc:
        findings.add(
            "AGGREGATE_MISMATCH",
            "error",
            f"the stored bundle is internally inconsistent: {exc.message}",
            [bref],
        )

    # -- the archived comparison ------------------------------------------------
    cid = manifest.get("comparison_id")
    contract_selection: dict[str, Any] = {}
    archived_pins: Any = None  # the comparison's evaluation_refs (hand-picked evaluations) or None
    if isinstance(cid, str):
        crow = reader.one("SELECT * FROM comparisons WHERE comparison_id=?", (cid,))
        if crow is None:
            findings.add(
                "UNRESOLVED_REF", "error", f"the bundle's comparison {cid} is not in the ledger", [cid]
            )
        else:
            archive = jload(crow["result_json"], {}) or {}
            if isinstance(archive.get("data"), dict):
                archived_pins = archive["data"].get("evaluation_refs")
            intact = (
                archive.get("schema") == ARCHIVE_SCHEMA
                and archive_content_hash(archive) == crow["content_hash"]
            )
            if not intact:
                findings.add(
                    "AGGREGATE_MISMATCH",
                    "error",
                    f"archived comparison {cid} no longer matches its content hash (modified after archiving)",
                    [cid],
                )
            if manifest.get("comparison_content_hash") not in (None, crow["content_hash"]):
                findings.add(
                    "AGGREGATE_MISMATCH",
                    "error",
                    f"the bundle was built from a different version of comparison {cid}",
                    [cid, bref],
                )
            elif intact:
                _check_against_archive(view, archive.get("data") or {}, cid, bref, findings)
            contract = jload(crow["contract_json"], {}) or {}
            contract_selection = contract.get("selection") or {} if isinstance(contract, dict) else {}
    else:
        findings.add(
            "UNRESOLVED_REF",
            "error",
            "the bundle names no comparison; its numbers cannot be re-derived",
            [bref],
        )

    # -- per-seed values against the evaluations ----------------------------------
    prov_index = {(p.get("group_label"), p.get("metric"), str(p.get("seed"))): p for p in view.provenance}
    evals = _rows_by_id(
        reader, "evaluations", "evaluation_id", (p.get("evaluation_id") for p in view.provenance)
    )
    runs = _rows_by_id(reader, "runs", "run_id", (p.get("run_id") for p in view.provenance))
    arts = _rows_by_id(reader, "artifacts", "artifact_id", (p.get("artifact_id") for p in view.provenance))
    checked = 0
    for label, group in view.groups.items():
        for metric, cell in (group.get("metrics") or {}).items():
            base = view.base_of(label, metric) or f"{label}/{metric}"
            cref = f"{view.bundle_id}:{base}"
            ledger_values: dict[str, float] = {}
            complete = True
            for seed, value in (cell.get("values_by_seed") or {}).items():
                checked += 1
                p = prov_index.get((label, metric, str(seed)))
                if p is None or not p.get("evaluation_id"):
                    findings.add(
                        "UNRESOLVED_REF",
                        "error",
                        f"{label}/{metric} seed {seed} has no evaluation to trace",
                        [cref],
                    )
                    complete = False
                    continue
                ev = evals.get(p["evaluation_id"])
                if ev is None:
                    findings.add(
                        "UNRESOLVED_REF",
                        "error",
                        f"evaluation {p['evaluation_id']} behind {label}/{metric} seed {seed} is not in the ledger",
                        [cref, p["evaluation_id"]],
                    )
                    complete = False
                    continue
                present, ledger_value = _first_metric_value(ev["metrics_json"], metric)
                if (
                    not present
                    or ledger_value is None
                    or not _close(ledger_value, float(value), abs(float(value)))
                ):
                    shown = "absent" if not present else repr(ledger_value)
                    findings.add(
                        "AGGREGATE_MISMATCH",
                        "error",
                        f"{label}/{metric} seed {seed}: bundle value {value!r} but evaluation "
                        f"{ev['evaluation_id']} records {shown}",
                        [cref, ev["evaluation_id"]],
                    )
                if ledger_value is None or not math.isfinite(ledger_value):
                    complete = False  # the aggregate cannot be re-derived without this value
                else:
                    ledger_values[str(seed)] = ledger_value
                if ev["status"] != "valid":
                    findings.add(
                        "AGGREGATE_MISMATCH",
                        "error",
                        f"evaluation {ev['evaluation_id']} is {ev['status']}, not valid",
                        [cref, ev["evaluation_id"]],
                    )
                if p.get("attempt_id") and ev["attempt_id"] != p["attempt_id"]:
                    findings.add(
                        "UNRESOLVED_REF",
                        "error",
                        f"evaluation {ev['evaluation_id']} belongs to attempt {ev['attempt_id']}, "
                        f"not {p['attempt_id']} as the bundle says",
                        [cref, ev["evaluation_id"]],
                    )
                run = runs.get(p.get("run_id") or "")
                if run is None:
                    findings.add(
                        "UNRESOLVED_REF", "error", f"run {p.get('run_id')} is not in the ledger", [cref]
                    )
                elif p.get("run_spec_hash") and run["spec_hash"] != p["run_spec_hash"]:
                    findings.add(
                        "UNRESOLVED_REF",
                        "error",
                        f"run {run['run_id']} spec hash differs from the bundle's provenance",
                        [cref, run["run_id"]],
                    )
                art = arts.get(p.get("artifact_id") or "")
                if p.get("artifact_id") and art is None:
                    findings.add(
                        "UNRESOLVED_REF", "error", f"artifact {p['artifact_id']} is not in the ledger", [cref]
                    )
                elif (
                    art is not None and p.get("artifact_digest") and art.get("digest") != p["artifact_digest"]
                ):
                    findings.add(
                        "ARTIFACT_DIGEST_CHANGED",
                        "error",
                        f"artifact {art['artifact_id']} is now recorded with digest {art.get('digest')}, "
                        f"the bundle traced {p['artifact_digest']}",
                        [cref, art["artifact_id"]],
                    )
            if complete and ledger_values:
                _check_aggregate(
                    label,
                    metric,
                    cell,
                    ledger_values,
                    float(manifest.get("ci_level") or 0.95),
                    cref,
                    findings,
                )

    # -- disclosures -----------------------------------------------------------------
    for item in manifest.get("incomparable") or []:
        reason = str(item.get("reason", "")) if isinstance(item, dict) else str(item)
        details = item.get("details") if isinstance(item, dict) else None
        if reason.startswith("selection_leak"):
            findings.add("SELECTION_LEAK", "error", reason, [bref])
        else:
            text = reason + (f" ({_short(details)})" if details else "")
            findings.add("INVARIANT_MISMATCH", "error", text, [bref])
    if contract_selection.get("rule") == "best_by" and (
        contract_selection.get("select_split") is None
        or contract_selection.get("select_split") == contract_selection.get("report_split")
    ):
        findings.add(
            "SELECTION_LEAK",
            "error",
            "the comparison contract selects the best record on the split it reports",
            [str(cid)],
        )
    raw_integrity = manifest.get("integrity")
    integrity: dict[str, Any] = raw_integrity if isinstance(raw_integrity, dict) else {}
    post = integrity.get("post_hoc") or []
    if post:
        seeds = "; ".join(f"{p.get('group_label')} seed {p.get('seed')}" for p in post[:10])
        findings.add(
            "SEED_SELECTION",
            "warn",
            f"{len(post)} run(s) were added after results existed (post hoc): {seeds}",
            [str(p.get("run_id")) for p in post],
        )
    for e in integrity.get("late_exclusions") or []:
        findings.add(
            "SEED_SELECTION",
            "warn",
            f"{e.get('group_label')} seed {e.get('seed')} was excluded after its result was observed "
            f"(reason: {e.get('reason')})",
            [str(e.get("run_id"))],
        )
    multi = [m for m in integrity.get("multiple_valid_attempts") or [] if isinstance(m, dict)]
    pinned = (
        archived_pins is not None
        or manifest.get("evaluation_refs") is not None
        or bool(integrity.get("evaluations_pinned"))
        or any(m.get("rule") == "evaluation_refs" for m in multi)
    )
    if pinned:
        # evaluation_refs bypass attempt_selection (and the check for newer attempts awaiting
        # evaluation): the reported attempts were chosen by hand
        rule = integrity.get("attempt_selection")
        over = [m for m in multi if m.get("overrides_rule")]
        text = f"the evaluations were pinned by the caller (evaluation_refs), not chosen by the contract's attempt_selection {rule}"
        if over:
            who = "; ".join(f"{m.get('group_label')} seed {m.get('seed')}" for m in over[:10])
            text += f"; {len(over)} run(s) report an attempt other than the one {rule} picks: {who}"
        elif multi:
            text += f"; {len(multi)} run(s) had several valid attempts (the pinned one is reported)"
        findings.add("SEED_SELECTION", "warn", text, [str(m.get("run_id")) for m in over or multi] or [bref])
    elif multi:
        findings.add(
            "SEED_SELECTION",
            "info",
            f"{len(multi)} run(s) had several valid attempts; the {integrity.get('attempt_selection')} one is "
            "reported",
            [str(m.get("run_id")) for m in multi],
        )
    _check_mixed(view, findings)
    return checked


def _short(details: Any) -> str:
    text = str(details)
    return text if len(text) <= 300 else text[:297] + "..."


def _cell_signature(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The denominator and per-seed values of a cell (every other statistic derives from them)."""
    values = cell.get("values_by_seed") or {}
    return {
        "n": cell.get("n"),
        "expected_n": cell.get("expected_n"),
        "values_by_seed": {
            str(k): float(v)
            for k, v in (values.items() if isinstance(values, Mapping) else [])
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        "missing_seeds": sorted(str(s) for s in cell.get("missing_seeds") or []),
        "failed_seeds": sorted(str(s) for s in cell.get("failed_seeds") or []),
        "excluded": sorted(
            (str(e.get("seed")), str(e.get("reason")))
            for e in cell.get("excluded") or []
            if isinstance(e, Mapping)
        ),
    }


def _cells_of(groups: Any) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for g in groups or []:
        if not isinstance(g, Mapping):
            continue
        for m, c in (g.get("metrics") or {}).items():
            if isinstance(c, Mapping):
                out[(str(g.get("label")), str(m))] = c
    return out


def _diff_cells(
    old: Mapping[tuple[str, str], Mapping[str, Any]], new: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, list[str]]:
    """``label/metric`` → the signature fields that differ (``absent``/``added`` for missing cells)."""
    out: dict[str, list[str]] = {}
    for key in sorted(set(old) | set(new)):
        name = f"{key[0]}/{key[1]}"
        if key not in new:
            out[name] = ["absent"]
        elif key not in old:
            out[name] = ["added"]
        else:
            a, b = _cell_signature(old[key]), _cell_signature(new[key])
            fields = [f for f in a if a[f] != b[f]]
            if fields:
                out[name] = fields
    return out


def _check_against_archive(
    view: _BundleView, data: Mapping[str, Any], cid: str, bref: str, findings: _Findings
) -> None:
    """A stored bundle must show exactly its archived comparison's cells and counts: an edited
    manifest can stay internally consistent (e.g. a failed seed dropped together with its
    expected_n) and still hide a failure."""
    diffs = _diff_cells(_cells_of(data.get("groups")), _cells_of(view.manifest.get("groups")))
    for name, fields in list(diffs.items())[:20]:
        what = (
            "is missing from the bundle"
            if fields == ["absent"]
            else "is not in the archived comparison"
            if fields == ["added"]
            else f"differs in {', '.join(fields)}"
        )
        findings.add(
            "AGGREGATE_MISMATCH",
            "error",
            f"{name} {what} compared with its archived comparison {cid} (the bundle was edited)",
            [bref, cid],
        )
    counts = data.get("counts") if isinstance(data.get("counts"), Mapping) else {}
    cov = view.manifest.get("coverage") if isinstance(view.manifest.get("coverage"), Mapping) else {}
    keys = ("planned_runs", "included", "missing", "failed", "excluded")
    off = [k for k in keys if (counts or {}).get(k) != (cov or {}).get(k)]
    if off:
        findings.add(
            "AGGREGATE_MISMATCH",
            "error",
            f"the bundle's coverage counts ({', '.join(off)}) differ from its archived comparison {cid}",
            [bref, cid],
        )


def _check_aggregate(
    label: str,
    metric: str,
    cell: Mapping[str, Any],
    ledger_values: Mapping[str, float],
    ci_level: float,
    cref: str,
    findings: _Findings,
) -> None:
    ordered = [ledger_values[s] for s in sorted(ledger_values, key=_seed_sort)]
    try:
        s = summarize(ordered, ci_level)
    except HpcLabError as exc:  # pragma: no cover - finite values were checked
        findings.add(
            "AGGREGATE_MISMATCH", "error", f"{label}/{metric}: cannot recompute ({exc.message})", [cref]
        )
        return
    scale = max((abs(v) for v in ordered), default=0.0)
    bad = []
    if cell.get("n") != s.n:
        bad.append(f"n={cell.get('n')} vs {s.n}")
    for stat in ("mean", "std", "sem", "ci_low", "ci_high", "min", "max"):
        got = cell.get(stat)
        want = getattr(s, stat)
        if not _close(float(got) if got is not None else None, want, scale):
            bad.append(f"{stat}={got!r} vs recomputed {want!r}")
    if bad:
        findings.add(
            "AGGREGATE_MISMATCH",
            "error",
            f"{label}/{metric}: bundle aggregates differ from the evaluations: " + "; ".join(bad[:4]),
            [cref],
        )


def _seed_sort(s: str) -> tuple[int, int, str]:
    try:
        return (0, int(s), "")
    except ValueError:
        return (1, 0, s)


def _check_mixed(view: _BundleView, findings: _Findings) -> None:
    included = []
    for p in view.provenance:
        group = view.groups.get(p.get("group_label", ""))
        cell = (group or {}).get("metrics", {}).get(p.get("metric", ""))
        if cell is not None and str(p.get("seed")) in (cell.get("values_by_seed") or {}):
            included.append(p)
    per_run: dict[str, dict[str, Any]] = {}
    for p in included:  # one entry per run (a run appears once per metric)
        per_run.setdefault(str(p.get("run_id")), p)
    checks: list[tuple[str, FindingCode]] = [
        ("code_digest", "MIXED_CODE_DIGEST"),
        ("env_gpu", "MIXED_GPU_TYPE"),
    ]
    for attr, code in checks:
        overall: dict[str, int] = {}
        within: list[str] = []
        per_group: dict[str, set[str]] = {}
        unknown = 0
        for p in per_run.values():
            v = p.get(attr)
            if v is None:
                unknown += 1
                continue
            overall[str(v)] = overall.get(str(v), 0) + 1
            per_group.setdefault(str(p.get("group_label")), set()).add(str(v))
        within = sorted(g for g, vs in per_group.items() if len(vs) > 1)
        what = "code digests" if attr == "code_digest" else "GPU models"
        listing = ", ".join(f"{k} ×{n}" for k, n in sorted(overall.items()))
        if within:
            findings.add(
                code,
                "warn",
                f"included values within group(s) {within} ran with different {what}: {listing}",
                [f"hpclab://evidence/{view.bundle_id}"],
            )
        elif len(overall) > 1:
            findings.add(
                code,
                "info" if attr == "code_digest" else "warn",
                f"groups ran with different {what} ({listing}); a confound unless it is the treatment",
                [f"hpclab://evidence/{view.bundle_id}"],
            )
        if unknown and overall:
            findings.add(
                code,
                "info",
                f"{what} not recorded for {unknown} included run(s); homogeneity is only partially checked",
                [f"hpclab://evidence/{view.bundle_id}"],
            )


def _newer_view(comparison_id: str | None, data: Mapping[str, Any]) -> _Newer:
    return _Newer(
        comparison_id=comparison_id,
        groups={k: dict(v) for k, v in _cells_of(data.get("groups")).items()},
        metrics={str(m) for m in data.get("metric_names") or []},
    )


class _Fresher:
    """Per-call cache of what is newer than each bundle referenced (bounded live recomputes)."""

    def __init__(self, ctx: ServiceContext) -> None:
        self.ctx = ctx
        self.cache: dict[str, _Freshness] = {}
        self.recomputed = 0
        self.skipped = 0

    def of(self, view: _BundleView) -> _Freshness:
        if view.bundle_id not in self.cache:
            live = self.recomputed < MAX_LIVE_RECOMPUTES
            self.recomputed += int(live)
            self.skipped += int(not live)
            self.cache[view.bundle_id] = _freshness(self.ctx, view, live=live)
        return self.cache[view.bundle_id]


def _freshness(ctx: ServiceContext, view: _BundleView, *, live: bool) -> _Freshness:
    reader: Reader = ctx.db
    cid = view.manifest.get("comparison_id")
    run_set_ref = view.manifest.get("run_set_ref") or view.row.get("run_set_ref")
    none = _Freshness(archived={}, newest_ref=None, live=None, ledger_changed=None, changed_cells=[])
    if not isinstance(cid, str) or not run_set_ref:
        none.note = "the bundle names no comparison"
        return none
    base = reader.one(
        "SELECT created_at, rowid AS rid, content_hash, result_json, run_set_ref, contract_json, "
        "request_json FROM comparisons WHERE comparison_id=?",
        (cid,),
    )
    if base is None:
        none.note = f"comparison {cid} is not in the ledger"
        return none
    metrics = {str(e.get("metric")) for e in view.cell_ids.values() if isinstance(e, Mapping)}

    # -- archived: per metric, the newest newer comparison of the SAME request that has it ------
    # Only a comparison that answers the same question (run set, contract, exclusions,
    # evaluation filter: request_key) supersedes the bundle's; an exploratory one never does.
    key = comparison_request_key(base)
    rows = reader.all(
        "SELECT comparison_id, run_set_ref, contract_json, request_json FROM comparisons WHERE run_set_ref=? "
        "AND (created_at > ? OR (created_at = ? AND rowid > ?)) ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (base["run_set_ref"], base["created_at"], base["created_at"], base["rid"], MAX_NEWER_SCAN),
    )
    chosen: dict[str, str] = {}
    newest_ref: str | None = None
    for r in rows:
        if key is None or comparison_request_key(r) != key:
            continue
        request = jload(r["request_json"], {}) or {}
        names = {str(m) for m in (request.get("metric_names") or [])} if isinstance(request, dict) else set()
        shared = names & metrics
        if shared and newest_ref is None:
            newest_ref = r["comparison_id"]
        for m in shared:
            chosen.setdefault(m, r["comparison_id"])
        if len(chosen) == len(metrics):
            break
    archived: dict[str, _Newer] = {}
    loaded: dict[str, _Newer] = {}
    for m, other in chosen.items():
        if other not in loaded:
            row = reader.one("SELECT result_json FROM comparisons WHERE comparison_id=?", (other,))
            data = (jload(row["result_json"], {}) or {}).get("data") if row else None
            loaded[other] = _newer_view(other, data if isinstance(data, dict) else {})
        archived[m] = loaded[other]
    out = _Freshness(
        archived=archived,
        newest_ref=newest_ref,
        live=None,
        ledger_changed=None,
        changed_cells=[],
        scan_truncated=len(rows) >= MAX_NEWER_SCAN and len(chosen) < len(metrics),
    )

    # -- live: the bundle's comparison re-derived from today's ledger (read-only) ---------------
    if not live:
        out.note = "not re-derived (per-call limit)"
        return out
    archive = jload(base["result_json"], {}) or {}
    if archive.get("schema") != ARCHIVE_SCHEMA or archive_content_hash(archive) != base["content_hash"]:
        out.note = f"archived comparison {cid} fails its content hash"
        return out
    try:
        data_model, _chains = load_archive(reader, cid)
        request_row = reader.one("SELECT request_json FROM comparisons WHERE comparison_id=?", (cid,))
        request = jload(request_row["request_json"], {}) if request_row else {}
        exclusions = [
            ExclusionSpec(run_id=str(e["run_id"]), reason=str(e["reason"]))
            for e in (request.get("exclusions") if isinstance(request, dict) else None) or []
            if isinstance(e, dict)
        ]
        current = compute_comparison(
            reader,
            planned_run_set_ref=data_model.run_set.ref,
            metric_names=list(data_model.metric_names),
            expected_seeds=None,  # a check, not a filter: seeds added since are exactly what staleness is about
            contract=data_model.contract,
            evaluation_ids=set(data_model.evaluation_refs)
            if data_model.evaluation_refs is not None
            else None,
            exclusions=exclusions,
            now=ctx.clock.now(),
        )
    except (HpcLabError, ValidationError, KeyError, TypeError) as exc:
        out.note = (
            exc.message if isinstance(exc, HpcLabError) else f"unreadable archive ({type(exc).__name__})"
        )
        return out
    now_data = current.data.model_dump(mode="json")
    diffs = _diff_cells(_cells_of(archive["data"].get("groups")), _cells_of(now_data.get("groups")))
    out.ledger_changed = bool(diffs)
    out.changed_cells = list(diffs)
    out.changed_fields = diffs
    if diffs:
        out.live = _newer_view(None, now_data)
    return out


# ---------------------------------------------------------------------------
# Manuscript scanning (given files + everything they \input)
# ---------------------------------------------------------------------------

_INCLUDE_RE = re.compile(r"\\(input|include|subfile|import|subimport)(?![A-Za-z@])")
_BARE_INPUT_RE = re.compile(r"[ \t]+([^\s{}\\%$&#]+)")


@dataclass(frozen=True)
class _Include:
    target: str  # the path as written (``dir/file`` for \import)
    line: int
    text: str  # the command as written, for messages


def _file_name(arg: str) -> str:
    """An include argument as a path: LaTeX drops the quotes of ``{"my table"}`` and the grouping
    braces of ``{{table.v2}}``."""
    return arg.replace('"', "").replace("{", "").replace("}", "").strip()


def _find_includes(clean: str) -> list[_Include]:
    """File includes of comment-stripped TeX, in document order (``\\includeonly``,
    ``\\includegraphics`` and ``#1`` parameters of macro definitions are not includes)."""
    out: list[_Include] = []
    for m in _INCLUDE_RE.finditer(clean):
        cmd = m.group(1)
        pos = m.end()
        if cmd in ("import", "subimport"):
            if clean.startswith("*", pos):
                pos += 1
            first = _group(clean, pos)
            second = _group(clean, first[1]) if first is not None else None
            if first is None or second is None:
                continue
            folder, name = _file_name(first[0]), _file_name(second[0])
            target = name if not folder else folder.rstrip("/") + "/" + name
            text = f"\\{cmd}{{{first[0]}}}{{{second[0]}}}"
        else:
            arg = _group(clean, pos)
            bare = _BARE_INPUT_RE.match(clean, pos) if arg is None and cmd == "input" else None
            if arg is not None:
                target, text = _file_name(arg[0]), f"\\{cmd}{{{arg[0]}}}"
            elif bare is not None:  # TeX's own form: \input file
                target, text = bare.group(1), f"\\input {bare.group(1)}"
            else:
                continue
        if not target or "#" in target:
            continue
        out.append(_Include(target=target, line=clean.count("\n", 0, m.start()) + 1, text=text))
    return out


def _normal(path: Path) -> Path:
    return Path(os.path.normpath(str(path.expanduser())))


def _identity(path: Path) -> str:
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(_normal(path))


class _Manuscript:
    """Reads the given manuscript files and, depth first, every file they include.

    Each file is read once (by resolved path), so include cycles and repeated includes are
    harmless; the scan is bounded by :data:`MAX_INCLUDE_DEPTH`, :data:`MAX_INCLUDE_FILES` and
    :data:`MAX_TOTAL_TEX_BYTES`. An include that cannot be read (or is cut by a bound) becomes
    ``UNRESOLVED_INPUT`` and an ``unavailable`` coverage source.
    """

    def __init__(self, coverage: Coverage, warnings: list[str], findings: _Findings) -> None:
        self.coverage = coverage
        self.warnings = warnings
        self.findings = findings
        self.sources: list[tuple[str, str, str | None]] = []  # (name, text, included_from)
        self.files: list[str] = []
        self.seen: set[str] = set()
        self.total = 0
        self.first_dir: Path | None = None

    # -- roots ---------------------------------------------------------------------------
    def add_file(self, path: Path) -> None:
        path = _normal(path)
        if self.first_dir is None:
            self.first_dir = path.parent
        if _identity(path) in self.seen:
            return  # already pulled in by an earlier file
        text, error = self._read(path)
        if text is None:
            self.coverage.add(f"tex:{path}", "unavailable", detail=error)
            self.warnings.append(f"{path} was not checked: {error}")
            return
        self._scan(str(path), text, path.parent, path.parent, 0, None)

    def add_content(self, text: str) -> None:
        self.total += len(text.encode("utf-8"))
        self.coverage.add("tex:content", "ok")
        self._scan("tex_content", text, None, None, 0, None)

    # -- scanning -------------------------------------------------------------------------
    def _scan(
        self,
        name: str,
        text: str,
        here: Path | None,
        root: Path | None,
        depth: int,
        included_from: str | None,
    ) -> None:
        self.sources.append((name, text, included_from))
        for inc in _find_includes(_strip_comments(text)):
            self._follow(inc, f"{name}:{inc.line}", here, root, depth + 1)

    def _candidates(self, target: str, here: Path | None, root: Path | None) -> list[Path]:
        names = [target] if target.lower().endswith(".tex") else [target + ".tex", target]
        if Path(target).expanduser().is_absolute():
            dirs: list[Path | None] = [None]
        else:
            dirs = []
            for d in (here, root, self.first_dir):
                if d is not None and d not in dirs:
                    dirs.append(d)
        out: list[Path] = []
        for d in dirs:
            for n in names:
                candidate = _normal(Path(n) if d is None else d / n)
                if candidate not in out:
                    out.append(candidate)
        return out

    def _follow(self, inc: _Include, where: str, here: Path | None, root: Path | None, depth: int) -> None:
        candidates = self._candidates(inc.target, here, root)
        if not candidates:
            self._unresolved(
                inc,
                where,
                None,
                "a relative path in tex_content cannot be resolved without a file; pass the "
                "manuscript as tex_paths (or \\input it by absolute path)",
            )
            return
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            others = [str(c) for c in candidates[1:] if c.suffix == ".tex"]
            self._unresolved(
                inc,
                where,
                candidates[0],
                "not found" + (f" (also looked for {', '.join(others)})" if others else ""),
            )
            return
        if _identity(path) in self.seen:
            return  # a cycle or a repeated include: every file is scanned once
        if depth > MAX_INCLUDE_DEPTH:
            self._unresolved(
                inc, where, path, f"not read: include depth limit ({MAX_INCLUDE_DEPTH} levels) reached"
            )
            return
        text, error = self._read(path)
        if text is None:
            self._unresolved(inc, where, path, error or "unreadable")
            return
        self._scan(str(path), text, path.parent, root, depth, where)

    def _read(self, path: Path) -> tuple[str | None, str | None]:
        try:
            size = path.stat().st_size
        except OSError as exc:
            return None, f"could not be read: {exc}"
        if size > MAX_TEX_BYTES:
            return None, f"not read: larger than {MAX_TEX_BYTES} bytes"
        if len(self.files) >= MAX_INCLUDE_FILES:
            return None, f"not read: at most {MAX_INCLUDE_FILES} manuscript files are scanned per call"
        if self.total + size > MAX_TOTAL_TEX_BYTES:
            return None, f"not read: the manuscript files would exceed {MAX_TOTAL_TEX_BYTES} bytes in total"
        # stat() says 0 bytes for /dev/zero and a FIFO: read the opened descriptor, non-blocking,
        # regular files only, never more than the bound (a root path is not filtered by is_file())
        try:
            raw = read_regular_file(path, max_bytes=MAX_TEX_BYTES)
        except FileTooLarge:
            return None, f"not read: larger than {MAX_TEX_BYTES} bytes"
        except NotRegularFile as exc:
            return None, f"not read: not a regular file ({exc.kind})"
        except OSError as exc:
            return None, f"could not be read: {exc}"
        size = len(raw)
        if self.total + size > MAX_TOTAL_TEX_BYTES:
            return None, f"not read: the manuscript files would exceed {MAX_TOTAL_TEX_BYTES} bytes in total"
        # the universal-newline translation read_text() did (line numbers in refs count "\n")
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        self.total += size
        self.seen.add(_identity(path))
        self.files.append(str(path))
        self.coverage.add(f"tex:{path}", "ok")
        return text, None

    def _unresolved(self, inc: _Include, where: str, path: Path | None, reason: str) -> None:
        shown = str(path) if path is not None else inc.target
        source = f"tex:{shown}"
        if not any(
            s.name == source for s in self.coverage.sources
        ):  # one gap per file, a finding per include
            self.coverage.add(source, "unavailable", detail=f"{inc.text} at {where}: {reason}")
        self.findings.add(
            "UNRESOLVED_INPUT",
            "warn",
            f"{inc.text} at {where}: {shown} {reason}; the numbers in it were NOT checked",
            [where, shown],
        )


def _scan_manuscript(
    tex_paths: Sequence[str] | None,
    tex_content: str | None,
    coverage: Coverage,
    warnings: list[str],
    findings: _Findings,
) -> _Manuscript:
    paths = list(tex_paths or [])
    if len(paths) > MAX_TEX_FILES:
        raise InvalidArgument(f"at most {MAX_TEX_FILES} tex files per call")
    roots: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.is_absolute():
            raise InvalidArgument(
                f"tex path {p!r} must be absolute (the controller's working directory differs)"
            )
        roots.append(path)
    if tex_content is not None and len(tex_content.encode("utf-8")) > MAX_TEX_BYTES:
        raise InvalidArgument(f"tex_content exceeds {MAX_TEX_BYTES} bytes")
    scan = _Manuscript(coverage, warnings, findings)
    for path in roots:
        scan.add_file(path)
    if tex_content is not None:
        scan.add_content(tex_content)
    return scan


def _check_tex(
    reader: Reader,
    name: str,
    text: str,
    views: dict[str, _BundleView],
    fresh: _Fresher,
    findings: _Findings,
) -> TexSourceReport:
    clean = _strip_comments(text)
    tags = parse_hpcval_tags(text)
    masked = _mask_tags(clean, tags)  # a tagged number (e.g. 1.2\times10^{-5}) is never a footnote mark
    bundles: list[str] = []
    for tag in tags:
        where = f"{name}:{tag.line}"
        bid, sep, cell_id = tag.ref.partition(":")
        if not sep or not bid or not cell_id:
            findings.add("UNRESOLVED_REF", "error", f"malformed \\hpcval reference {tag.ref!r}", [where])
            continue
        view = views.get(bid)
        if view is None:
            try:
                manifest, row = _load_bundle(reader, bid)
            except HpcLabError:
                findings.add(
                    "UNRESOLVED_REF", "error", f"\\hpcval references unknown bundle {bid!r}", [where]
                )
                continue
            view = views[bid] = _BundleView(manifest, row)
        if bid not in bundles:
            bundles.append(bid)
        try:
            base, stat = parse_cell_id(cell_id)
        except HpcLabError as exc:
            findings.add("UNRESOLVED_REF", "error", exc.message, [where, tag.ref])
            continue
        resolved = view.cell(base)
        if resolved is None:
            findings.add("UNRESOLVED_REF", "error", f"bundle {bid} has no cell {base!r}", [where, tag.ref])
            continue
        label, metric, cell = resolved
        expected, what = _stat_value(cell, stat)
        if expected is None:
            findings.add(
                "VALUE_MISMATCH",
                "error",
                f"{label}/{metric} {what} is undefined in the bundle (e.g. std with one seed, or no included "
                f"value) but the tex prints {tag.value.strip()!r}",
                [where, tag.ref],
            )
            continue
        shown = parse_tex_number(tag.value)
        if shown is None:
            findings.add(
                "VALUE_MISMATCH",
                "error",
                f"cannot read a number from \\hpcval{{{tag.ref}}}{{{tag.value.strip()}}}",
                [where, tag.ref],
            )
            continue
        scale = 1.0 if stat == "n" else view.scale
        if not _matches(shown, expected * scale):
            findings.add(
                "VALUE_MISMATCH",
                "error",
                f"{label}/{metric} {what}: tex shows {tag.value.strip()} but the bundle has {expected * scale!r}"
                + (f" (value × {scale:g})" if scale != 1.0 else ""),
                [where, tag.ref],
            )
        state = fresh.of(view)
        for newer in (state.archived.get(metric), state.live):  # at most one stale finding per tag
            if newer is None or metric not in newer.metrics:
                continue
            ncell = newer.groups.get((label, metric))
            nvalue = _stat_value(ncell, stat)[0] if ncell is not None else None
            refs = [where, tag.ref, *([newer.comparison_id] if newer.comparison_id else [])]
            if nvalue is None:
                findings.add(
                    "STALE_TEX_NUMBER",
                    "error",
                    f"{label}/{metric} {what}: tex shows {tag.value.strip()} but {newer.gives()} no such "
                    "value; re-export the bundle",
                    refs,
                )
                break
            if not _matches(shown, nvalue * scale):
                findings.add(
                    "STALE_TEX_NUMBER",
                    "error",
                    f"{label}/{metric} {what}: tex shows {tag.value.strip()} but {newer.gives()} "
                    f"{nvalue * scale:.6g}; "
                    + (
                        "re-export the bundle"
                        if newer.comparison_id
                        else "re-run results_compare and re-export"
                    ),
                    refs,
                )
                break
        n, expected_n = cell.get("n"), cell.get("expected_n")
        if (
            stat == "mean"
            and isinstance(n, int)
            and isinstance(expected_n, int)
            and n < expected_n
            and not any(mark in _cell_context(masked, tag.end) for mark in _FOOTNOTE_MARKS)
        ):
            findings.add(
                "N_BELOW_EXPECTED_UNFOOTNOTED",
                "error",
                f"{label}/{metric} uses {n} of {expected_n} planned seeds but the tex cell has no footnote mark",
                [where, tag.ref],
            )
    untagged = _untagged_numbers(clean, tags)
    for line, number in untagged[:MAX_UNTAGGED_FINDINGS]:
        findings.add(
            "UNTAGGED_TEX_NUMBER",
            "info",
            f"number {number} in a table is not tagged with \\hpcval",
            [f"{name}:{line}"],
        )
    if len(untagged) > MAX_UNTAGGED_FINDINGS:
        findings.add(
            "UNTAGGED_TEX_NUMBER",
            "info",
            f"{len(untagged) - MAX_UNTAGGED_FINDINGS} more untagged table numbers in {name}",
            [name],
        )
    return TexSourceReport(source=name, tags=len(tags), untagged_numbers=len(untagged), bundles=bundles)


def _mask_tags(clean: str, tags: Sequence[TexTag]) -> str:
    """``clean`` with every tag blanked out (offsets and newlines kept)."""
    masked = list(clean)
    for t in tags:
        for i in range(t.start, t.end):
            if masked[i] != "\n":
                masked[i] = " "
    return "".join(masked)


def _untagged_numbers(clean: str, tags: Sequence[TexTag]) -> list[tuple[int, str]]:
    text = _mask_tags(clean, tags)
    out = []
    for env in _TABULAR_RE.finditer(text):
        body_start = env.start(2)
        body = env.group(2)
        if body.startswith("{"):  # skip the column specification
            spec = _group(body, 0)
            skip = spec[1] if spec else 0
        else:
            skip = 0
        for m in _DECIMAL_RE.finditer(body, skip):
            out.append((text.count("\n", 0, body_start + m.start()) + 1, m.group(0)))
    return out


def _check_remote(ctx: ServiceContext, view: _BundleView, findings: _Findings, coverage: Coverage) -> int:
    """Re-hash small artifacts behind included values on the cluster (never inside a transaction)."""
    wanted: dict[str, str] = {}
    for p in view.provenance:
        if p.get("artifact_id") and p.get("artifact_digest") and p.get("value") is not None:
            wanted.setdefault(p["artifact_id"], p["artifact_digest"])
    ids = sorted(wanted)
    if len(ids) > MAX_REMOTE_ARTIFACTS:
        coverage.add(
            "remote",
            "partial",
            detail=f"only the first {MAX_REMOTE_ARTIFACTS} of {len(ids)} artifacts were re-hashed",
        )
        ids = ids[:MAX_REMOTE_ARTIFACTS]
    arts = _rows_by_id(ctx.db, "artifacts", "artifact_id", ids)
    attempts = _rows_by_id(ctx.db, "attempts", "attempt_id", (a["attempt_id"] for a in arts.values()))
    by_cluster: dict[str, dict[str, str]] = {}
    skipped = 0
    for aid in ids:
        art = arts.get(aid)
        if art is None:
            continue  # already reported as UNRESOLVED_REF
        size = art.get("size_bytes")
        if art.get("kind") != "file" or size is None or size > MAX_REMOTE_FILE_BYTES:
            skipped += 1
            continue
        att = attempts.get(art["attempt_id"])
        if att is None:
            continue
        by_cluster.setdefault(att["cluster_id"], {})[art["remote_path"]] = aid
    if skipped:
        coverage.notes.append(f"{skipped} artifact(s) were not re-hashed (directories or larger than 16 MiB)")
    checked = 0
    timeout = max(5.0, min(float(ctx.settings.request_deadline_s), 60.0))
    for cluster_id, paths in sorted(by_cluster.items()):
        source = f"remote:{cluster_id}"
        try:
            digests = ctx.clusters.transport(cluster_id).sha256(sorted(paths), timeout=timeout)
        except AuthRequiredError as exc:
            coverage.add(source, "unauthorized", detail=exc.message)
            continue
        except HpcLabError as exc:  # TransportError included
            # unreachable, unregistered or misconfigured (e.g. no ssh host): a coverage gap for
            # this cluster, never a verification error and never evidence of change
            coverage.add(source, "unavailable", detail=exc.message)
            continue
        except OSError as exc:
            coverage.add(source, "unavailable", detail=str(exc))
            continue
        coverage.add(source, "ok")
        for path, aid in sorted(paths.items()):
            checked += 1
            now = digests.get(path)
            if now is None:
                findings.add(
                    "ARTIFACT_DIGEST_CHANGED",
                    "error",
                    f"artifact {aid} ({path}) is missing or unreadable on {cluster_id} now",
                    [aid],
                )
            elif now != wanted[aid]:
                findings.add(
                    "ARTIFACT_DIGEST_CHANGED",
                    "error",
                    f"artifact {aid} ({path}) now hashes to {now}; the bundle traced {wanted[aid]}",
                    [aid],
                )
    return checked


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceLinks(_Model):
    evaluation: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    attempt: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    run: dict[str, Any] | None = None
    code: dict[str, Any] | None = None
    env: dict[str, Any] | None = None


class TraceSeed(_Model):
    seed: int | str
    status: Literal["included", "missing", "failed", "excluded"]
    # Cell code of the run when the comparison was made (V, E, C!, F, X, P, ...; see
    # services/coverage) and why the seed is not included: ``E`` is missing (completed, no valid
    # evaluation yet), ``C!`` failed (bad outputs or a partial/invalid evaluation).
    code: str | None = None
    reason: str | None = None
    value: float | None = None
    exclusion_reason: str | None = None
    provenance: dict[str, Any] | None = None
    chain: TraceLinks | None = None
    complete: bool = False
    issues: list[str] = Field(default_factory=list)


class TraceData(_Model):
    cell_ref: str
    bundle_id: str
    bundle_ref: str
    cell_id: str
    group_label: str
    metric: str
    stat: str
    value: float | None
    n: int
    expected_n: int
    comparison_id: str | None
    comparison_ref: str | None
    run_set_ref: str | None
    seeds: list[TraceSeed]
    complete: bool
    issues: list[str] = Field(default_factory=list)
    summary: str
    coverage: Coverage


_ATTEMPT_FIELDS = (
    "attempt_id",
    "seq",
    "phase",
    "scheduler_state",
    "app_state",
    "result_state",
    "exit_code",
    "exit_signal",
    "elapsed_s",
    "time_limit_s",
    "gpus",
    "gpu_type",
    "partition",
    "node_list",
    "start_time",
    "end_time",
    "exec_plan_hash",
    "restart_count",
)
_ARTIFACT_FIELDS = (
    "artifact_id",
    "name",
    "role",
    "kind",
    "remote_path",
    "size_bytes",
    "mtime",
    "digest",
    "digest_state",
    "digest_source",
    "integrity",
)


def trace(ctx: ServiceContext, *, cell_ref: str) -> TraceData:
    """The full provenance chain behind one table cell (``<bundle_id>:<cell_id>``, as in
    ``\\hpcval``); resource ``hpclab://trace/{cell_ref}``. Frozen bundle provenance is shown
    next to the live ledger records, and any drift is listed in ``issues``."""
    ref = (cell_ref or "").strip()
    if ref.startswith("hpclab://trace/"):
        ref = ref[len("hpclab://trace/") :]
    bid, sep, cell_id = ref.partition(":")
    if not sep or not bid or not cell_id:
        raise InvalidArgument(f"cell_ref must be <bundle_id>:<cell_id>, got {cell_ref!r}")
    manifest, row = _load_bundle(ctx.db, strip_ref(bid))
    view = _BundleView(manifest, row)
    base, stat = parse_cell_id(cell_id)
    resolved = view.cell(base)
    if resolved is None:
        raise NotFound(
            f"bundle {view.bundle_id} has no cell {base!r}",
            details={"known_cells": sorted(view.cell_ids)[:100], "stats": list(CELL_STATS)},
        )
    label, metric, cell = resolved
    value, _what = _stat_value(cell, stat)
    status = _seed_status(cell)
    if stat.startswith("seed-"):
        wanted = stat[len("seed-") :]
        if wanted not in status:
            raise NotFound(f"{label}/{metric} has no seed {wanted}", details={"seeds": list(status)})
        status = {wanted: status[wanted]}
    prov = {
        str(p.get("seed")): p
        for p in view.provenance
        if p.get("group_label") == label and p.get("metric") == metric
    }
    evals = _rows_by_id(
        ctx.db, "evaluations", "evaluation_id", (p.get("evaluation_id") for p in prov.values())
    )
    arts = _rows_by_id(ctx.db, "artifacts", "artifact_id", (p.get("artifact_id") for p in prov.values()))
    attempts = _rows_by_id(ctx.db, "attempts", "attempt_id", (p.get("attempt_id") for p in prov.values()))
    runs = _rows_by_id(ctx.db, "runs", "run_id", (p.get("run_id") for p in prov.values()))
    reasons = {str(e.get("seed")): e.get("reason") for e in cell.get("excluded") or [] if isinstance(e, dict)}
    run_rows = [r for r in manifest.get("runs") or [] if isinstance(r, dict)]
    seeds: list[TraceSeed] = []
    all_issues: list[str] = []
    for seed, st in status.items():
        p = prov.get(seed)
        v = (cell.get("values_by_seed") or {}).get(seed)
        run_row = _run_behind(run_rows, label, seed, st, p)
        entry = TraceSeed(
            seed=_seed_out(seed),
            status=st,
            code=str(run_row["code"]) if run_row and run_row.get("code") else None,
            reason=_seed_reason(st, metric, run_row, reasons.get(seed)),
            value=float(v) if v is not None else (p.get("value") if p else None),
            exclusion_reason=reasons.get(seed),
            provenance=p,
        )
        if p is not None:
            links, issues, complete = _links(p, metric, evals, arts, attempts, runs)
            entry = entry.model_copy(update={"chain": links, "issues": issues, "complete": complete})
            all_issues += [f"seed {seed}: {i}" for i in issues]
        elif st == "included":
            entry = entry.model_copy(update={"issues": ["no provenance row: this value cannot be traced"]})
            all_issues.append(f"seed {seed}: no provenance row")
        seeds.append(entry)
    included = [s for s in seeds if s.status == "included"]
    complete = (bool(included) and all(s.complete for s in included)) or (not included and not all_issues)
    cid = manifest.get("comparison_id") if isinstance(manifest.get("comparison_id"), str) else None
    return TraceData(
        cell_ref=f"{view.bundle_id}:{cell_id}",
        bundle_id=view.bundle_id,
        bundle_ref=f"hpclab://evidence/{view.bundle_id}",
        cell_id=cell_id,
        group_label=label,
        metric=metric,
        stat=stat,
        value=value,
        n=int(cell.get("n") or 0),
        expected_n=int(cell.get("expected_n") or 0),
        comparison_id=cid,
        comparison_ref=f"hpclab://comparisons/{cid}" if cid else None,
        run_set_ref=manifest.get("run_set_ref"),
        seeds=seeds,
        complete=complete,
        issues=all_issues,
        summary=(
            f"{label}/{metric} {stat} = {value!r}: {cell.get('n')}/{cell.get('expected_n')} seeds included; "
            f"chain {'complete' if complete else 'incomplete'}"
        ),
        coverage=Coverage(
            complete=not all_issues, sources=[SourceStatus(name="ledger", status="ok")], notes=all_issues[:20]
        ),
    )


def _run_behind(
    runs: Sequence[Mapping[str, Any]], label: str, seed: str, status: str, prov: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """The bundle's run row behind one seed of a cell. A seed can have several planned runs (e.g.
    in two plans of an experiment) when all but one are excluded: the traced run (provenance)
    wins, else one whose exclusion matches the seed's status."""
    if prov is not None and prov.get("run_id"):
        row = next((r for r in runs if r.get("run_id") == prov["run_id"]), None)
        if row is not None:
            return row
    candidates = [
        r
        for r in runs
        if str(r.get("group_label")) == label
        and ("none" if r.get("seed") is None else str(r.get("seed"))) == seed
    ]
    wanted = [r for r in candidates if (r.get("status") == "excluded") == (status == "excluded")]
    return next(iter(wanted or candidates), None)


def _seed_reason(
    status: str, metric: str, run: Mapping[str, Any] | None, exclusion_reason: str | None
) -> str | None:
    """Why a seed is not included in a cell (None when it is)."""
    if status == "included":
        return None
    if status == "excluded":
        return exclusion_reason
    if run is None:
        return None
    if run.get("status") == "included":  # the run has a valid evaluation; this metric is the gap
        ev = run.get("evaluation_id") or "?"
        if status == "missing":
            return f"evaluation {ev} does not report {metric}"
        return f"{metric} is not a finite number in evaluation {ev}"
    return str(run.get("reason") or run.get("code") or "") or None


def _seed_out(seed: str) -> int | str:
    try:
        return int(seed)
    except ValueError:
        return seed


def _seed_status(cell: Mapping[str, Any]) -> dict[str, str]:
    status = {str(s): "included" for s in cell.get("values_by_seed") or {}}
    status.update({str(s): "missing" for s in cell.get("missing_seeds") or []})
    status.update({str(s): "failed" for s in cell.get("failed_seeds") or []})
    status.update({str(e.get("seed")): "excluded" for e in cell.get("excluded") or [] if isinstance(e, dict)})
    return {s: status[s] for s in sorted(status, key=_seed_sort)}


def _pick(row: Mapping[str, Any] | None, fields: Sequence[str]) -> dict[str, Any] | None:
    return None if row is None else {f: row.get(f) for f in fields}


def _links(
    p: Mapping[str, Any],
    metric: str,
    evals: Mapping[str, dict[str, Any]],
    arts: Mapping[str, dict[str, Any]],
    attempts: Mapping[str, dict[str, Any]],
    runs: Mapping[str, dict[str, Any]],
) -> tuple[TraceLinks, list[str], bool]:
    issues: list[str] = []
    ev = evals.get(p.get("evaluation_id") or "")
    evaluation = None
    if p.get("evaluation_id") and ev is None:
        issues.append(f"evaluation {p['evaluation_id']} is not in the ledger")
    if ev is not None:
        row = next(
            (
                m
                for m in jload(ev["metrics_json"], []) or []
                if isinstance(m, dict) and m.get("name") == metric
            ),
            None,
        )
        evaluation = {
            "evaluation_id": ev["evaluation_id"],
            "evaluator": f"{ev['evaluator_id']}@{ev['evaluator_version']}",
            "config_hash": ev["config_hash"],
            "status": ev["status"],
            "created_at": ev["created_at"],
            "metric": row,
        }
        if p.get("value") is not None and (row is None or row.get("value") != p.get("value")):
            live = row.get("value") if row else None
            if not (
                isinstance(live, (int, float))
                and _close(float(live), float(p["value"]), abs(float(p["value"])))
            ):
                issues.append(f"evaluation records {live!r}, the bundle traced {p.get('value')!r}")
    art = arts.get(p.get("artifact_id") or "")
    if p.get("artifact_id") and art is None:
        issues.append(f"artifact {p['artifact_id']} is not in the ledger")
    if art is not None and p.get("artifact_digest") and art.get("digest") != p.get("artifact_digest"):
        issues.append(
            f"artifact digest is now {art.get('digest')}, the bundle traced {p.get('artifact_digest')}"
        )
    att = attempts.get(p.get("attempt_id") or "")
    if p.get("attempt_id") and att is None:
        issues.append(f"attempt {p['attempt_id']} is not in the ledger")
    run = runs.get(p.get("run_id") or "")
    if p.get("run_id") and run is None:
        issues.append(f"run {p['run_id']} is not in the ledger")
    code = None
    if run is not None:
        spec = jload(run["spec_json"], {}) or {}
        c = spec.get("code") if isinstance(spec, dict) else None
        if isinstance(c, dict):
            code = {
                k: c.get(k) for k in ("digest", "git_commit", "git_branch", "git_dirty", "dirty_patch_digest")
            }
        if p.get("run_spec_hash") and run["spec_hash"] != p.get("run_spec_hash"):
            issues.append("run spec hash differs from the bundle's provenance")
    env = None
    if att is not None:
        raw = jload(att.get("env_json"), None)
        gpu, torch = env_facts(att.get("env_json"))
        if isinstance(raw, dict):
            torch_info = raw.get("torch") if isinstance(raw.get("torch"), dict) else {}
            smi = raw.get("nvidia_smi") if isinstance(raw.get("nvidia_smi"), dict) else {}
            python = raw.get("python") if isinstance(raw.get("python"), dict) else {}
            env = {
                "gpu": gpu,
                "torch": torch,
                "cuda": (torch_info or {}).get("cuda"),
                "driver_version": (smi or {}).get("driver_version"),
                "python": (python or {}).get("version"),
                "packages_digest": raw.get("packages_digest"),
                "host": raw.get("host"),
                "captured_at": raw.get("captured_at"),
            }
    links = TraceLinks(
        evaluation=evaluation,
        artifact=_pick(art, _ARTIFACT_FIELDS),
        attempt=_pick(att, _ATTEMPT_FIELDS),
        job=(
            {
                "cluster_id": att.get("cluster_id"),
                "job_id": att.get("job_id"),
                "owner": att.get("job_owner"),
                "submit_time": att.get("job_submit_time"),
                "job_name": att.get("job_name"),
            }
            if att is not None
            else None
        ),
        run=(
            {
                "run_id": run["run_id"],
                "label": run["label"],
                "spec_hash": run["spec_hash"],
                "cell": jload(run["cell_key"], {}),
                "seed": run["seed"],
                "plan_id": run["plan_id"],
                "experiment_id": run["experiment_id"],
                "derived_from": run.get("derived_from"),
                "added_post_hoc": bool(run.get("added_post_hoc") or 0),
            }
            if run is not None
            else None
        ),
        code=code,
        env=env,
    )
    complete = (
        not issues
        and evaluation is not None
        and links.artifact is not None
        and bool((links.artifact or {}).get("digest"))
        and links.attempt is not None
        and bool((links.job or {}).get("job_id"))
        and links.run is not None
        and code is not None
        and bool(code.get("digest"))
    )
    if evaluation is not None and links.artifact is None:
        issues.append("no artifact is linked to the evaluation")
    return links, issues, complete


def outbox_handlers() -> dict[str, Any]:
    """Verification is read-only and bounded; no background work."""
    return {}


def periodic_tasks() -> list[Any]:
    return []
