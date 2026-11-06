"""Registered, bounded result extractors (tools §12 ``evaluation_collect``).

An evaluator turns the bytes of a declared output artifact into typed ``Metric`` rows plus
validation ``EvalIssue`` rows. Evaluators are pure — no I/O, clock or randomness — so the
evaluation service can fetch the artifact, call :func:`run_evaluator`, and archive the result
under ``(input digests, evaluator id/version, config hash)``; re-running on the same bytes
always yields the same output.

Failure semantics
-----------------
* **Request errors raise** and nothing should be archived: unknown evaluator
  (``NotFound``), invalid config or wrong number of inputs (``InvalidArgument``).
* **Data problems never raise.** They become issues and determine ``status``:

  - ``invalid`` — a *fatal* issue (input empty / oversized / undecodable / unparseable,
    digest mismatch, no selectable record, split mismatch), a ``require``-d metric without a
    finite value, or no metric with a finite value at all. Fatal issues yield no metrics.
  - ``partial`` — at least one finite metric, but some error issue (missing field,
    non-numeric or non-finite requested value, malformed JSONL lines, truncated input, or a
    ``final`` selection that had to fall back to a record with no evidence of being final).
    Partial evaluations are reported for inspection but never count as results
    (``services.coverage`` only counts ``valid``).
  - ``valid`` — no error issues; warnings are allowed and always reported.

  The rule behind the split: an *unknown* (was this the final number? was the value
  numeric?) must never be reported as ``valid``.

* Non-finite numbers (NaN/±inf, e.g. a diverged loss) are *kept* as metrics with
  ``value=None`` plus a ``non_finite`` error, so a diverged seed is visible downstream instead
  of silently turning into "missing".
* Every input is bounded: at most :data:`MAX_INPUT_BYTES`, at most :data:`MAX_JSONL_LINES`
  JSONL lines (then an ``input_truncated`` error), at most :data:`MAX_ISSUES` detailed issues
  (then an ``issues_truncated`` summary; the fatal issue and the selection issue —
  ``selection_on_split`` / ``selection_protocol`` — are listed even past it), at most :data:`MAX_METRICS` metrics from an
  unfiltered read (then a ``too_many_metrics`` error), and records are *streamed* so memory
  does not grow with the number of JSONL lines.

Best-checkpoint selection
-------------------------
A ``best:<metric>`` selection records how the reported record was chosen in
``EvalOutput.selection`` and, with the same dict as ``details``, in exactly one issue (so it
also survives in the stored validation record)::

    {"select_metric": "val_loss", "direction": "min",
     "select_split": <split the selection metric was measured on, or null>,
     "report_split": <split of the reported metrics, or null>,
     "selected_step": <step of the selected record, or null>,
     "select_value": <the selected metric's value>, "select_locator": "L42#",
     "on_reported_split": true | false | null}

The config's ``select_split`` declares the split the selection is made on; ``split`` stays the
split that is reported. Selecting on one split and reporting another is the early-stopping
protocol (choose the step by the best ``val`` loss, report the ``test`` metrics *at that
step*), not selection leakage:

* ``on_reported_split: true`` — the selection was made on the reported split (no
  ``select_split``, or one equal to ``split``, for ``hpclab_metrics@1``, whose records carry one
  split each; ``select_split`` equal to ``split`` for ``jsonl_record@1``). Issue
  ``selection_on_split``: a best-of-split number is optimistic (selection leakage), and
  ``results_compare`` refuses every evaluation carrying it.
* ``on_reported_split: null`` — ``jsonl_record@1`` without ``select_split``: one record may hold
  fields of several splits (early stopping logs ``val_loss`` next to ``test_acc``), so the
  evaluator cannot tell which split the selection field was measured on. Issue
  ``selection_on_split`` as well: an unknown must not be reported as leak-free.
* ``on_reported_split: false`` — ``select_split`` differs from ``split``. Issue
  ``selection_protocol`` (a disclosure, not a leak). ``hpclab_metrics@1`` picks the best record
  of ``select_split``, then reports the ``split`` record with the same ``step`` (``no_step`` /
  ``record_not_found`` when the step cannot be joined). ``jsonl_record@1`` does the same join
  when records carry their split (``split_path``; ``step_path`` required; a selection field the
  reported record does not log, e.g. ``val_loss`` on val records only or ``null`` on the test
  rows of a wide log, is read from the selected record and labelled ``select_split``); without
  ``split_path`` one record holds both
  splits, the selection field is labelled ``select_split`` and the other fields ``split``.

The join reads the input a second time (memory stays O(1)); issues are reported once.

Locators
--------
``Metric.source.locator`` / ``EvalIssue.locator`` point into the input: ``L<line>`` is a
1-based JSONL line, ``#<pointer>`` an RFC 6901 JSON pointer (``#/metrics/acc``); combined as
``L42#/metrics/acc``. A record inside a top-level JSON array is ``#/<index>/...``.

Path syntax (``json_path@1`` / ``jsonl_record@1`` config)
--------------------------------------------------------
``a.b[0].c``, negative indexes ``history[-1].loss`` and quoted keys for names containing
dots or slashes: ``metrics["val/acc"]``. Integer brackets index arrays only.

Determinism: output depends only on (config, input bytes). Metrics are sorted by name,
issues are in input order.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NoReturn, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.domain.hashing import sha256_hex

__all__ = [
    "EVALUATORS",
    "MAX_INPUT_BYTES",
    "MAX_ISSUES",
    "MAX_JSONL_LINES",
    "MAX_METRICS",
    "METRICS_SCHEMA",
    "EvalInput",
    "EvalIssue",
    "EvalOutput",
    "Evaluator",
    "EvaluatorInfo",
    "HpclabMetricsConfig",
    "JsonPathConfig",
    "JsonlRecordConfig",
    "Metric",
    "MetricSource",
    "get_evaluator",
    "list_evaluators",
    "parse_path",
    "register_evaluator",
    "run_evaluator",
    "validate_config",
]

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINES = 1_000_000
MAX_ISSUES = 50
MAX_METRICS = 10_000  # metrics reported by an "all metrics" read (no explicit filter)
MAX_SELECTED_RECORD_BYTES = 16 * 1024
# JSON parsed by the C scanner can be nested deeper than Python's recursion limit; anything
# deeper than this is summarized instead of walked recursively.
MAX_SELECTED_RECORD_DEPTH = 32
METRICS_SCHEMA = "hpclab.metrics/v1"

_BOM = b"\xef\xbb\xbf"
_DIGEST_RE = re.compile(r"(?:sha256:)?([0-9a-f]{64})")

# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricSource(_Model):
    """Where a metric value came from: input name, sha256 of the bytes read, locator."""

    input_name: str
    digest: str
    locator: str


class Metric(_Model):
    """One extracted value. ``value is None`` only for non-finite numbers (see issues)."""

    name: str
    value: float | None
    unit: str | None = None
    split: str | None = None
    step: int | None = None
    seed: int | None = None
    dataset: str | None = None
    higher_is_better: bool | None = None
    source: MetricSource


Severity = Literal["error", "warning"]
EvalStatus = Literal["valid", "partial", "invalid"]
InputFormat = Literal["json", "jsonl"]


class EvalIssue(_Model):
    """One validation finding. ``details`` is machine-readable context for the codes that
    downstream checks act on (``selection_on_split`` / ``selection_protocol``: see the module
    docstring); ``None`` for every other issue."""

    severity: Severity
    code: str
    message: str
    locator: str | None = None
    details: dict[str, Any] | None = None


class EvalInput(_Model):
    """Bytes of one artifact. ``digest`` (``sha256:<hex>``) is verified when given.

    The digest is normalized (case, optional ``sha256:`` prefix, surrounding spaces) so a
    spelling difference is never reported as corrupted data; a value that is not a sha256
    digest at all is a caller bug and fails validation here.

    ``format`` overrides inference (``.jsonl``/``.ndjson`` → jsonl, ``.json`` → json,
    otherwise a document that parses as one JSON value is json, else jsonl).
    """

    name: str
    data: bytes = Field(repr=False)
    digest: str | None = None
    format: InputFormat | None = None

    @field_validator("digest")
    @classmethod
    def _normalize_digest(cls, v: str | None) -> str | None:
        if v is None:
            return None
        match = _DIGEST_RE.fullmatch(v.strip().lower())
        if match is None:
            raise ValueError(f"digest must be 'sha256:<64 hex digits>', got {v!r}")
        return "sha256:" + match.group(1)


class EvalOutput(_Model):
    """Result of one evaluator run.

    ``config`` is the normalized config (defaults applied) — hash this, not the raw
    request, so ``{}`` and ``{"split": "val"}`` identify the same protocol.
    ``selected_record`` is the raw record the metrics came from (bounded to
    :data:`MAX_SELECTED_RECORD_BYTES`; non-finite floats become the strings
    ``"NaN"``/``"Infinity"``/``"-Infinity"`` so it is strict JSON).
    ``selection`` says how a ``best:<metric>`` selection chose that record (select/report
    split, selected step, whether it is on the reported split: see the module docstring);
    ``None`` for every other selection and for invalid evaluations.
    """

    evaluator_id: str
    evaluator_version: str
    config: dict[str, Any]
    status: EvalStatus
    metrics: list[Metric]
    issues: list[EvalIssue]
    selected_record: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None


class EvaluatorInfo(_Model):
    """Registry entry for docs / MCP discovery."""

    id: str
    version: str
    description: str
    config_schema: dict[str, Any]


@runtime_checkable
class Evaluator(Protocol):
    """A registered extractor. ``run`` receives an instance of ``config_model``."""

    @property
    def id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def config_model(self) -> type[BaseModel]: ...

    def run(self, config: Any, inputs: Sequence[EvalInput]) -> EvalOutput: ...


# ---------------------------------------------------------------------------
# Path expressions
# ---------------------------------------------------------------------------

PathToken = str | int

_KEY_RE = re.compile(r"[^.\[\]]+")
_INDEX_RE = re.compile(r"\[(-?\d+)\]")


def parse_path(expr: str) -> tuple[PathToken, ...]:
    """Parse ``a.b[0].c`` / ``metrics["val/acc"]`` / ``h[-1]`` into keys and indexes.

    Raises ``ValueError`` with the offending position for malformed expressions.
    """
    tokens, pos = _parse_path_prefix(expr, stop_at_equals=False)
    if pos != len(expr):
        raise ValueError(f"unexpected character {expr[pos]!r} at position {pos} in path {expr!r}")
    return tokens


def _parse_path_prefix(expr: str, *, stop_at_equals: bool) -> tuple[tuple[PathToken, ...], int]:
    tokens: list[PathToken] = []
    pos, n = 0, len(expr)
    need_key = False  # True right after '.', where only a bare key may follow
    while pos < n:
        ch = expr[pos]
        if stop_at_equals and ch == "=" and not need_key and tokens:
            break
        if ch == "[" and not need_key:
            token, pos = _parse_bracket(expr, pos)
            tokens.append(token)
        elif ch not in ".[]" and (need_key or not tokens):
            match = _KEY_RE.match(expr, pos)
            key = match.group(0) if match else ""
            if stop_at_equals:
                key = key.split("=", 1)[0]
            if not key:
                raise ValueError(f"empty key at position {pos} in path {expr!r}")
            tokens.append(key)
            pos += len(key)
            need_key = False
        elif ch == "." and tokens and not need_key:
            need_key = True
            pos += 1
            continue
        else:
            raise ValueError(f"unexpected character {ch!r} at position {pos} in path {expr!r}")
        if pos < n and expr[pos] not in ".[" and not (stop_at_equals and expr[pos] == "="):
            raise ValueError(f"unexpected character {expr[pos]!r} at position {pos} in path {expr!r}")
    if need_key:
        raise ValueError(f"path {expr!r} ends with '.'")
    if not tokens:
        raise ValueError("empty path")
    return tuple(tokens), pos


def _parse_bracket(expr: str, pos: int) -> tuple[PathToken, int]:
    index = _INDEX_RE.match(expr, pos)
    if index:
        return int(index.group(1)), index.end()
    if pos + 1 < len(expr) and expr[pos + 1] in "\"'":
        quote = expr[pos + 1]
        buf: list[str] = []
        i = pos + 2
        while i < len(expr) and expr[i] != quote:
            if expr[i] == "\\" and i + 1 < len(expr):
                i += 1
            buf.append(expr[i])
            i += 1
        if i + 1 >= len(expr) or expr[i + 1] != "]":
            raise ValueError(f"unterminated quoted key at position {pos} in path {expr!r}")
        return "".join(buf), i + 2
    raise ValueError(f'invalid bracket at position {pos} in path {expr!r} (use [0] or ["key"])')


class _PathMissing(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _resolve(doc: Any, tokens: Sequence[PathToken]) -> tuple[Any, str]:
    """Follow ``tokens``; return (value, JSON pointer) or raise ``_PathMissing``."""
    cur, pointer = doc, ""
    for tok in tokens:
        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise _PathMissing(f"expected an array at '{pointer or '/'}' for index [{tok}]")
            idx = tok if tok >= 0 else len(cur) + tok
            if not 0 <= idx < len(cur):
                raise _PathMissing(f"index [{tok}] out of range at '{pointer or '/'}' (length {len(cur)})")
            cur, pointer = cur[idx], f"{pointer}/{idx}"
        else:
            if not isinstance(cur, dict):
                raise _PathMissing(f"expected an object at '{pointer or '/'}' for key {tok!r}")
            if tok not in cur:
                raise _PathMissing(f"key {tok!r} not found at '{pointer or '/'}'")
            cur, pointer = cur[tok], f"{pointer}/{_pointer_escape(tok)}"
    return cur, pointer


def _pointer_escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


# ---------------------------------------------------------------------------
# Issue collection and value coercion
# ---------------------------------------------------------------------------


class _Fatal(Exception):
    """Aborts an extraction: the evaluation is ``invalid`` and yields no metrics."""

    def __init__(self, code: str, message: str, locator: str | None = None) -> None:
        super().__init__(message)
        self.issue = EvalIssue(severity="error", code=code, message=message, locator=locator)


class _IssueLog:
    """Bounded issue list; suppressed issues still count towards the status.

    Issues that carry ``details`` are the ones downstream checks act on (``results_compare``
    flags ``selection_on_split``), so they are never suppressed: dropping one past the bound
    would make a best-of-split number look leak-free (and would lose the disclosure of a
    ``selection_protocol``). At most one is logged per extraction.
    """

    def __init__(self) -> None:
        self.issues: list[EvalIssue] = []
        self.error_count = 0
        self.suppressed = 0

    def error(self, code: str, message: str, locator: str | None = None) -> None:
        self.error_count += 1
        self._add(EvalIssue(severity="error", code=code, message=message, locator=locator))

    def warning(
        self, code: str, message: str, locator: str | None = None, *, details: dict[str, Any] | None = None
    ) -> None:
        issue = EvalIssue(severity="warning", code=code, message=message, locator=locator, details=details)
        if details is not None:
            self.issues.append(issue)  # never suppressed: downstream checks act on it
        else:
            self._add(issue)

    def fatal(self, issue: EvalIssue) -> None:
        self.error_count += 1
        self.issues.append(issue)  # never suppressed: it explains the invalid status

    def _add(self, issue: EvalIssue) -> None:
        if len(self.issues) < MAX_ISSUES:
            self.issues.append(issue)
        else:
            self.suppressed += 1

    def finalized(self) -> list[EvalIssue]:
        if not self.suppressed:
            return list(self.issues)
        note = EvalIssue(
            severity="warning",
            code="issues_truncated",
            message=f"{self.suppressed} further issue(s) not listed (limit {MAX_ISSUES})",
        )
        return [*self.issues, note]


def _as_number(value: Any) -> tuple[float | None, str | None]:
    """Return (finite float, None) or (None, problem code: non_numeric | non_finite)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "non_numeric"
    try:
        f = float(value)
    except OverflowError:
        return None, "non_finite"
    if not math.isfinite(f):
        return None, "non_finite"
    return f, None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _json_equal(a: Any, b: Any) -> bool:
    """JSON-typed equality: ``true`` never equals ``1`` and ``"1"`` never equals ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _nesting_exceeds(value: Any, limit: int) -> bool:
    """True if ``value`` has a container nested more than ``limit`` levels deep.

    Iterative on purpose: the record may be deeper than Python's recursion limit.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict):
            children: Any = node.values()
        elif isinstance(node, list):
            children = node
        else:
            continue
        if depth > limit:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False


def _bounded_record(record: dict[str, Any], log: _IssueLog) -> dict[str, Any]:
    keys = sorted(record)[:100]
    if _nesting_exceeds(record, MAX_SELECTED_RECORD_DEPTH):
        log.warning(
            "selected_record_truncated",
            f"selected record is nested deeper than {MAX_SELECTED_RECORD_DEPTH} levels; only its "
            "top-level keys are kept",
        )
        return {"_truncated": True, "_max_depth": MAX_SELECTED_RECORD_DEPTH, "_keys": keys}
    safe = _json_safe(record)
    size = len(json.dumps(safe, ensure_ascii=False, default=str).encode("utf-8"))
    if size <= MAX_SELECTED_RECORD_BYTES:
        return safe
    log.warning(
        "selected_record_truncated",
        f"selected record is {size} bytes; only its top-level keys are kept (limit {MAX_SELECTED_RECORD_BYTES})",
    )
    return {"_truncated": True, "_bytes": size, "_keys": keys}


# ---------------------------------------------------------------------------
# Input decoding and record streaming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Record:
    data: dict[str, Any]
    line: int | None = None  # 1-based JSONL line
    index: int | None = None  # position in a top-level JSON array

    def locator(self, pointer: str = "") -> str:
        if self.line is not None:
            return f"L{self.line}#{pointer}"
        if self.index is not None:
            return f"#/{self.index}{pointer}"
        return f"#{pointer}"


def _resolve_format(inp: EvalInput) -> InputFormat:
    if inp.format is not None:
        return inp.format
    lower = inp.name.lower()
    if lower.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"
    try:
        json.loads(inp.data.removeprefix(_BOM).decode("utf-8"))
    except (ValueError, RecursionError):
        return "jsonl"
    return "json"


def _load_json_document(data: bytes) -> Any:
    try:
        text = data.removeprefix(_BOM).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Fatal("decode_error", f"input is not valid UTF-8 (byte offset {exc.start})") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Fatal(
            "parse_error",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            f"L{exc.lineno}",
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise _Fatal("parse_error", f"invalid JSON: {exc}") from exc


def _iter_records(data: bytes, fmt: InputFormat, log: _IssueLog) -> Iterator[_Record]:
    """Stream JSON-object records; JSON input yields one object or each array element."""
    if fmt == "jsonl":
        yield from _iter_jsonl(data, log)
        return
    doc = _load_json_document(data)
    if isinstance(doc, dict):
        yield _Record(doc)
    elif isinstance(doc, list):
        for i, element in enumerate(doc):
            if isinstance(element, dict):
                yield _Record(element, index=i)
            else:
                log.error(
                    "not_an_object", f"array element {i} is {_json_type(element)}, not an object", f"#/{i}"
                )
    else:
        raise _Fatal("not_an_object", f"JSON document is {_json_type(doc)}, expected an object or array")


def _iter_jsonl(data: bytes, log: _IssueLog) -> Iterator[_Record]:
    pos = len(_BOM) if data.startswith(_BOM) else 0
    lineno, size = 0, len(data)
    while pos < size:
        if lineno >= MAX_JSONL_LINES:
            if data[pos:].strip():
                log.error(
                    "input_truncated",
                    f"stopped after {MAX_JSONL_LINES} lines; the remaining {size - pos} bytes were not read, "
                    "so 'last'/'final' selection may not reflect the end of the file",
                    f"L{lineno + 1}",
                )
            return
        end = data.find(b"\n", pos)
        end = size if end < 0 else end
        raw = data[pos:end]
        pos, lineno = end + 1, lineno + 1
        if not raw.strip():
            continue
        record = _parse_jsonl_line(raw, lineno, log)
        if record is not None:
            yield record


def _parse_jsonl_line(raw: bytes, lineno: int, log: _IssueLog) -> _Record | None:
    loc = f"L{lineno}"
    try:
        obj = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        log.error("parse_error", f"line {lineno} is not valid UTF-8 (byte {exc.start})", loc)
        return None
    except json.JSONDecodeError as exc:
        log.error("parse_error", f"line {lineno}: {exc.msg} at column {exc.colno}", loc)
        return None
    except (ValueError, RecursionError) as exc:
        log.error("parse_error", f"line {lineno}: {exc}", loc)
        return None
    if not isinstance(obj, dict):
        log.error("not_an_object", f"line {lineno} is {_json_type(obj)}, not an object", loc)
        return None
    return _Record(obj, line=lineno)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    return "an object"


def _split_repr(value: Any) -> str:
    """Display form of a split label for messages (never walks containers: they may be deep)."""
    if value is None:
        return "<none>"
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return f"<{_json_type(value)}>"
    return json.dumps(value)


# ---------------------------------------------------------------------------
# Selection expressions
# ---------------------------------------------------------------------------

Direction = Literal["max", "min"]


@dataclass(frozen=True)
class _Selection:
    kind: Literal["final", "last", "best", "step", "where"]
    metric: str | None = None
    direction: Direction | None = None
    step: int | None = None
    where_path: tuple[PathToken, ...] = ()
    where_value: Any = None


def _parse_best(rest: str, *, require_direction: bool) -> _Selection:
    metric: str = rest
    direction: Direction | None = None
    head, sep, tail = rest.rpartition(":")
    if sep and tail == "max":
        metric, direction = head, "max"
    elif sep and tail == "min":
        metric, direction = head, "min"
    if not metric:
        raise ValueError("best:<metric> needs a metric name")
    if require_direction and direction is None:
        raise ValueError("best selection must be 'best:<metric>:max' or 'best:<metric>:min'")
    return _Selection("best", metric=metric, direction=direction)


def _selection_details(
    metric: str,
    direction: Direction,
    best: tuple[float, _Record],
    *,
    select_split: str | None,
    report_split: str | None,
    selected_step: int | None,
    on_reported_split: bool | None,
) -> dict[str, Any]:
    """``EvalOutput.selection`` and the ``details`` of its issue (see the module docstring).

    ``on_reported_split`` is ``True`` when the evaluator knows the selection metric shares
    the reported metrics' split, ``False`` when ``select_split`` declares another split, and
    ``None`` when it cannot tell.
    """
    value, record = best
    return {
        "select_metric": metric,
        "direction": direction,
        "select_split": select_split,
        "report_split": report_split,
        "selected_step": selected_step,
        "select_value": value,
        "select_locator": record.locator(),
        "on_reported_split": on_reported_split,
    }


def _flag_selection_on_split(log: _IssueLog, details: dict[str, Any], message: str) -> None:
    """The ``selection_on_split`` issue: the selection is (or may be) on the reported split."""
    log.warning("selection_on_split", message, details["select_locator"], details=details)


def _disclose_selection(log: _IssueLog, details: dict[str, Any]) -> None:
    """The ``selection_protocol`` issue: selected on ``select_split``, reported another split."""
    metric, direction, step = details["select_metric"], details["direction"], details["selected_step"]
    chosen = f"step {step}" if step is not None else "the record"
    reported = "at that step" if step is not None else "from that record"
    log.warning(
        "selection_protocol",
        f"selected {chosen} by the {direction} {metric!r} on split {details['select_split']!r}; "
        f"the {details['report_split']!r} metrics are reported {reported} (select on one split, "
        "report another: not selection leakage)",
        details["select_locator"],
        details=details,
    )


def _record_at_step(
    records: Iterator[_Record],
    is_reported: Callable[[_Record], bool],
    step_of: Callable[[_Record], int | None],
    step: int,
    split: str | None,
    log: _IssueLog,
) -> _Record:
    """The last record of the reported ``split`` (``is_reported``) at the selected ``step``;
    a resumed run may log a step twice (``duplicate_step``)."""
    found: _Record | None = None
    count = 0
    for rec in records:
        if is_reported(rec) and step_of(rec) == step:
            found, count = rec, count + 1
    if found is None:
        raise _Fatal(
            "record_not_found",
            f"no {split!r} record at step {step}, the step selected on select_split; the reported "
            "split must be logged at the steps the selection is made on",
        )
    if count > 1:
        log.warning(
            "duplicate_step",
            f"{count} {split!r} records have step {step} (restarted or resumed training?); using the last one",
            found.locator(),
        )
    return found


def _parse_metrics_select(expr: str) -> _Selection:
    if expr == "final":
        return _Selection("final")
    if expr == "last":
        return _Selection("last")
    if expr.startswith("best:"):
        return _parse_best(expr[len("best:") :], require_direction=False)
    raise ValueError(f"select must be 'final', 'last' or 'best:<metric>[:max|min]', got {expr!r}")


def _parse_record_select(expr: str) -> _Selection:
    if expr == "last":
        return _Selection("last")
    if expr.startswith("step:"):
        step = expr[len("step:") :]
        if not re.fullmatch(r"-?\d+", step):
            raise ValueError(f"step selection needs an integer, got {step!r}")
        return _Selection("step", step=int(step))
    if expr.startswith("best:"):
        return _parse_best(expr[len("best:") :], require_direction=True)
    if expr.startswith("where:"):
        body = expr[len("where:") :]
        tokens, pos = _parse_path_prefix(body, stop_at_equals=True)
        if pos >= len(body) or body[pos] != "=":
            raise ValueError("where selection must be 'where:<path>=<value>'")
        return _Selection("where", where_path=tokens, where_value=_parse_where_value(body[pos + 1 :]))
    raise ValueError(
        f"select must be 'last', 'step:<n>', 'best:<metric>:max|min' or 'where:<path>=<value>', got {expr!r}"
    )


def _parse_where_value(text: str) -> Any:
    """``true``/``1``/``null``/``"x"`` are JSON-typed; anything else is a plain string."""
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if value is None or isinstance(value, (str, int, float, bool)) else text


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Fields added after configs were archived: left out of the normalized config while unset,
    # so a config that does not use them keeps hashing to the same evaluation identity.
    _omit_when_unset: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _normalized(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for key in self._omit_when_unset:
            if data.get(key) is None:
                data.pop(key, None)
        return data


def _check_select_split(select_split: str | None, selection: _Selection, split: str | None) -> None:
    if select_split is None:
        return
    if selection.kind != "best":
        raise ValueError("select_split applies to a 'best:<metric>' selection only")
    if split is None:
        raise ValueError(
            "select_split needs split: the split whose metrics are reported at the selected step"
        )


class HpclabMetricsConfig(_Config):
    """Config of ``hpclab_metrics@1`` (the ``hpclab.metrics/v1`` format).

    * ``split``: records must carry this split (``None`` accepts any split).
    * ``select``: ``final`` (default: the last ``final: true`` record of the split. Without
      one, the last record of the split is reported; it is only ``valid`` (with a
      ``no_final_record`` warning) when its ``step`` reached ``total_steps``, otherwise
      ``no_final_record`` is an *error* — a run killed mid-training looks exactly like that),
      ``last`` (explicit opt-in for loggers that never mark final records), or
      ``best:<metric>[:max|min]`` (direction defaults to the record's ``higher_is_better``;
      flagged ``selection_on_split`` unless ``select_split`` names another split: picking
      the best checkpoint on the split you report is optimistic).
    * ``select_split``: with ``best:<metric>``, the split whose records choose the step (e.g.
      ``val``); the reported metrics are those of the ``split`` record (e.g. ``test``) with that
      ``step`` — early stopping, not selection leakage. Omitted (or equal to ``split``): the
      selection is made on the reported split and flagged ``selection_on_split``.
    * ``metrics``: only report these (``None`` = all numeric entries; non-numeric entries
      are then skipped with a warning). Absent requested metrics are warnings; requested
      non-numeric values are errors.
    * ``require``: these must have a finite value, else the evaluation is ``invalid``.
    """

    _omit_when_unset: ClassVar[tuple[str, ...]] = ("select_split",)

    split: str | None = "val"
    select: str = "final"
    select_split: str | None = Field(default=None, min_length=1)
    metrics: list[str] | None = None
    require: list[str] = Field(default_factory=list)

    @field_validator("select")
    @classmethod
    def _select(cls, v: str) -> str:
        _parse_metrics_select(v)
        return v

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not v:
            raise ValueError("metrics filter must be non-empty (use null for all metrics)")
        return v

    @model_validator(mode="after")
    def _select_split_refs(self) -> HpclabMetricsConfig:
        _check_select_split(self.select_split, self.selection, self.split)
        return self

    @property
    def selection(self) -> _Selection:
        return _parse_metrics_select(self.select)


class _FieldConfig(_Config):
    """Shared config of the path-based evaluators.

    ``fields`` maps metric name → path. ``split``/``dataset`` declare constants; the
    ``*_path`` variants read them from the document (a declared ``split`` must then match
    the value read, otherwise the evaluation is invalid).
    """

    fields: dict[str, str]
    split: str | None = None
    split_path: str | None = None
    step_path: str | None = None
    seed_path: str | None = None
    dataset: str | None = None
    dataset_path: str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    higher_is_better: dict[str, bool] = Field(default_factory=dict)

    @field_validator("fields")
    @classmethod
    def _fields(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("fields must map at least one metric name to a path")
        for name, path in v.items():
            if not name:
                raise ValueError("metric names must be non-empty")
            parse_path(path)
        return v

    @field_validator("split_path", "step_path", "seed_path", "dataset_path")
    @classmethod
    def _paths(cls, v: str | None) -> str | None:
        if v is not None:
            parse_path(v)
        return v

    @model_validator(mode="after")
    def _known_metrics(self) -> _FieldConfig:
        for attr in ("units", "higher_is_better"):
            unknown = sorted(set(getattr(self, attr)) - set(self.fields))
            if unknown:
                raise ValueError(f"{attr} names metrics not in fields: {unknown}")
        return self


class JsonPathConfig(_FieldConfig):
    """Config of ``json_path@1``: extract fields from one JSON document."""


class JsonlRecordConfig(_FieldConfig):
    """Config of ``jsonl_record@1``: select one JSONL record, then extract fields from it.

    ``select``: ``last`` | ``step:<n>`` (needs ``step_path``) | ``best:<metric>:max|min``
    (metric must be in ``fields``) | ``where:<path>=<value>`` (last record whose value at
    ``path`` equals ``value``, JSON-typed). When both ``split`` and ``split_path`` are set,
    only records of that split are considered.

    ``select_split`` (``best`` only; needs ``split``, the reported split) declares the split
    the selection metric was measured on. With ``split_path``, the best record of
    ``select_split`` picks the step and the ``split`` record at that step is reported
    (``step_path`` required; if that record does not log the selection field, e.g. val records
    carry ``val_loss`` and test records ``test_acc``, or a wide log writes ``null`` there, the
    field is read from the selected record with split ``select_split``); without it, one
    record holds both splits (early
    stopping logs ``val_loss`` next to ``test_acc``), and the selection field is reported with
    split ``select_split``, the other fields with ``split``. Without ``select_split`` a ``best``
    selection is flagged ``selection_on_split``: the evaluator cannot tell which split the
    metric came from.
    """

    _omit_when_unset: ClassVar[tuple[str, ...]] = ("select_split",)

    select: str = "last"
    select_split: str | None = Field(default=None, min_length=1)

    @field_validator("select")
    @classmethod
    def _select(cls, v: str) -> str:
        _parse_record_select(v)
        return v

    @model_validator(mode="after")
    def _select_refs(self) -> JsonlRecordConfig:
        sel = self.selection
        if sel.kind == "step" and self.step_path is None:
            raise ValueError("select 'step:<n>' requires step_path")
        if sel.kind == "best" and sel.metric not in self.fields:
            raise ValueError(f"best metric {sel.metric!r} must be one of fields {sorted(self.fields)}")
        _check_select_split(self.select_split, sel, self.split)
        if self.joins_by_step and self.step_path is None:
            raise ValueError(
                "select_split with split_path reports the split record at the selected step: "
                "step_path is required"
            )
        return self

    @property
    def joins_by_step(self) -> bool:
        """Records carry their split, and the selection is made on another split than the
        reported one: the reported record is joined to the selected one by step."""
        return (
            self.select_split is not None and self.split_path is not None and self.select_split != self.split
        )

    @property
    def selection(self) -> _Selection:
        return _parse_record_select(self.select)


# ---------------------------------------------------------------------------
# Evaluator base: bounds, digest verification, status
# ---------------------------------------------------------------------------


@dataclass
class _Extraction:
    metrics: list[Metric]
    selected: dict[str, Any] | None
    required: tuple[str, ...] = ()
    selection: dict[str, Any] | None = None


class _BaseEvaluator[C: BaseModel]:
    """Single-input evaluator skeleton; subclasses implement ``_extract``."""

    id: str
    version: str
    description: str
    config_model: type[C]

    def run(self, config: C, inputs: Sequence[EvalInput]) -> EvalOutput:
        if len(inputs) != 1:
            raise InvalidArgument(
                f"{self.id}@{self.version} takes exactly one input, got {len(inputs)}",
                details={"inputs": [i.name for i in inputs]},
            )
        inp = inputs[0]
        log = _IssueLog()
        try:
            digest = self._check_input(inp)
            extraction = self._extract(config, inp, digest, log)
        except _Fatal as fatal:
            log.fatal(fatal.issue)
            return self._output(config, "invalid", [], log, None)
        status = _status(extraction, log)
        return self._output(
            config, status, extraction.metrics, log, extraction.selected, extraction.selection
        )

    def _extract(self, config: C, inp: EvalInput, digest: str, log: _IssueLog) -> _Extraction:
        raise NotImplementedError

    @staticmethod
    def _check_input(inp: EvalInput) -> str:
        if len(inp.data) > MAX_INPUT_BYTES:
            raise _Fatal("input_too_large", f"input is {len(inp.data)} bytes; the limit is {MAX_INPUT_BYTES}")
        actual = "sha256:" + sha256_hex(inp.data)
        if inp.digest is not None and inp.digest != actual:
            raise _Fatal(
                "input_digest_mismatch",
                f"input bytes hash to {actual}, expected {inp.digest}; the artifact changed or the "
                "transfer was incomplete",
            )
        if not inp.data.removeprefix(_BOM).strip():
            raise _Fatal("empty_input", "input is empty")
        return actual

    def _output(
        self,
        config: C,
        status: EvalStatus,
        metrics: list[Metric],
        log: _IssueLog,
        selected: dict[str, Any] | None,
        selection: dict[str, Any] | None = None,
    ) -> EvalOutput:
        return EvalOutput(
            evaluator_id=self.id,
            evaluator_version=self.version,
            config=config.model_dump(mode="json"),
            status=status,
            metrics=sorted(metrics, key=lambda m: m.name),
            issues=log.finalized(),
            selected_record=selected,
            selection=selection,
        )


def _status(extraction: _Extraction, log: _IssueLog) -> EvalStatus:
    finite = {m.name for m in extraction.metrics if m.value is not None}
    missing_required = [name for name in extraction.required if name not in finite]
    for name in missing_required:
        log.error("missing_required_metric", f"required metric {name!r} has no finite value")
    if not finite:
        if not log.error_count:
            log.error("no_finite_metrics", "no metric with a finite value was extracted")
        return "invalid"
    if missing_required:
        return "invalid"
    return "partial" if log.error_count else "valid"


# ---------------------------------------------------------------------------
# hpclab_metrics@1
# ---------------------------------------------------------------------------


class _MetricsScan:
    """Streaming selection over ``hpclab.metrics/v1`` records (O(1) memory)."""

    def __init__(self, split: str | None, selection: _Selection) -> None:
        self.split = split
        self.sel = selection
        self.candidates = 0
        self.splits_seen: set[str] = set()
        self.last_match: _Record | None = None
        self.last_final: _Record | None = None
        self.final_count = 0
        self.final_other_splits: set[str] = set()
        self.best_max: tuple[float, _Record] | None = None
        self.best_min: tuple[float, _Record] | None = None
        self.declared_directions: set[bool] = set()

    def offer(self, rec: _Record, log: _IssueLog) -> None:
        schema = rec.data.get("schema")
        if schema is not None and schema != METRICS_SCHEMA:
            log.warning("foreign_schema", f"record with schema {schema!r} ignored", rec.locator("/schema"))
            return
        metrics = rec.data.get("metrics")
        if metrics is None:
            return  # event lines without metrics are allowed and ignored
        if not isinstance(metrics, dict):
            log.error("invalid_record", "'metrics' must be an object", rec.locator("/metrics"))
            return
        self.candidates += 1
        split = rec.data.get("split")
        self.splits_seen.add(_split_repr(split))
        matches = self.split is None or split == self.split
        if rec.data.get("final") is True:
            if matches:
                self.last_final, self.final_count = rec, self.final_count + 1
            else:
                self.final_other_splits.add(_split_repr(split))
        if matches:
            self.last_match = rec
            if self.sel.kind == "best":
                self._offer_best(rec, metrics)

    def _offer_best(self, rec: _Record, metrics: dict[str, Any]) -> None:
        metric = self.sel.metric or ""
        hib = rec.data.get("higher_is_better")
        if isinstance(hib, dict) and isinstance(hib.get(metric), bool):
            self.declared_directions.add(hib[metric])
        value, _ = _as_number(metrics.get(metric))
        if value is None:
            return
        if self.best_max is None or value > self.best_max[0]:
            self.best_max = (value, rec)
        if self.best_min is None or value < self.best_min[0]:
            self.best_min = (value, rec)

    def choose(self, log: _IssueLog) -> _Record:
        """The record of a ``final``/``last`` selection (``best``: :meth:`choose_best`)."""
        if self.sel.kind == "final":
            return self._choose_final(log)
        if self.last_match is None:
            self._fail_no_match()
        return self.last_match

    def _choose_final(self, log: _IssueLog) -> _Record:
        if self.last_final is not None:
            if self.final_count > 1:
                log.warning(
                    "multiple_final_records",
                    f"{self.final_count} records are marked final for split {_split_repr(self.split)}; "
                    "using the last one",
                    self.last_final.locator(),
                )
            return self.last_final
        if self.final_other_splits:
            raise _Fatal(
                "split_mismatch",
                f"final record(s) exist only for split(s) {sorted(self.final_other_splits)}, not "
                f"{self.split!r}; set split to the reported split, or select='last' to use the last "
                f"{self.split!r} record",
            )
        if self.last_match is None:
            self._fail_no_match()
        chosen = self.last_match
        reached, progress = _training_progress(chosen.data)
        split = _split_repr(self.split)
        if reached:
            log.warning(
                "no_final_record",
                f"no record is marked final; using the last record of split {split}, which {progress}",
                chosen.locator(),
            )
        else:
            log.error(
                "no_final_record",
                f"no record of split {split} is marked final and the last one {progress}: it may be a "
                "mid-training value (e.g. a run killed at its time limit), so it is shown for inspection "
                'but is not a valid result. Mark the final record with "final": true, or set '
                "select='last' to accept the last record explicitly",
                chosen.locator(),
            )
        return chosen

    def choose_best(self) -> tuple[Direction, tuple[float, _Record]]:
        """(direction, (value, record)) of a ``best:<metric>`` selection; logs nothing (the
        evaluator knows which split is reported)."""
        metric = self.sel.metric or ""
        if self.last_match is None:
            self._fail_no_match()
        direction = self.sel.direction or self._declared_direction(metric)
        chosen = self.best_max if direction == "max" else self.best_min
        if chosen is None:
            raise _Fatal("no_record", f"no record of split {_split_repr(self.split)} has a finite {metric!r}")
        return direction, chosen

    def _declared_direction(self, metric: str) -> Direction:
        if len(self.declared_directions) > 1:
            raise _Fatal("conflicting_direction", f"records disagree on higher_is_better for {metric!r}")
        if not self.declared_directions:
            raise _Fatal(
                "direction_unknown",
                f"no record declares higher_is_better for {metric!r}; use 'best:{metric}:max' or ':min'",
            )
        return "max" if next(iter(self.declared_directions)) else "min"

    def _fail_no_match(self) -> NoReturn:
        if not self.candidates:
            raise _Fatal("no_record", "no record with a 'metrics' object was found")
        raise _Fatal(
            "split_mismatch",
            f"no record for split {self.split!r}; splits present: {sorted(self.splits_seen)}",
        )


def _training_progress(data: Mapping[str, Any]) -> tuple[bool, str]:
    """(reached total_steps, human-readable progress) from a record's step / total_steps."""
    step, total = _as_int(data.get("step")), _as_int(data.get("total_steps"))
    if step is None or total is None or total <= 0:
        return False, "carries no step/total_steps showing that training finished"
    if step >= total:
        return True, f"is at step {step} of {total} (training reached total_steps)"
    return False, f"is at step {step} of {total}"


class _HpclabMetricsEvaluator(_BaseEvaluator[HpclabMetricsConfig]):
    id = "hpclab_metrics"
    version = "1"
    description = (
        "Reads the hpclab.metrics/v1 JSONL/JSON format: selects the last final:true record of the "
        "configured split (or last / best:<metric>, optionally chosen on select_split and reported "
        "at that step) and reports its metrics with units, direction, split, step, seed and dataset."
    )
    config_model = HpclabMetricsConfig

    def _extract(
        self, config: HpclabMetricsConfig, inp: EvalInput, digest: str, log: _IssueLog
    ) -> _Extraction:
        fmt = _resolve_format(inp)
        selection: dict[str, Any] | None = None
        if config.selection.kind == "best":
            chosen, selection = self._choose_best(config, inp.data, fmt, log)
        else:
            scan = _MetricsScan(config.split, config.selection)
            for rec in _iter_records(inp.data, fmt, log):
                scan.offer(rec, log)
            chosen = scan.choose(log)
        wanted = None if config.metrics is None else [*config.metrics, *config.require]
        metrics = _metrics_from_record(chosen, wanted, config.require, (inp.name, digest), log)
        return _Extraction(metrics, _bounded_record(chosen.data, log), tuple(config.require), selection)

    @staticmethod
    def _choose_best(
        config: HpclabMetricsConfig, data: bytes, fmt: InputFormat, log: _IssueLog
    ) -> tuple[_Record, dict[str, Any]]:
        """The reported record of a ``best:<metric>`` selection and its selection details.

        A record carries one split, so selecting on the reported split means the selection
        metric and every reported metric share it (``on_reported_split`` true). With another
        ``select_split`` the best record of that split picks the step, and the ``split`` record
        at that step is reported (early stopping).
        """
        sel = config.selection
        metric = sel.metric or ""
        other_split = config.select_split is not None and config.select_split != config.split
        scan = _MetricsScan(config.select_split if other_split else config.split, sel)
        for rec in _iter_records(data, fmt, log):
            scan.offer(rec, log)
        direction, best = scan.choose_best()
        selected = best[1]
        step = _as_int(selected.data.get("step"))
        if not other_split:
            split = selected.data.get("split")
            label = split if isinstance(split, str) else None
            details = _selection_details(
                metric,
                direction,
                best,
                select_split=label,
                report_split=label,
                selected_step=step,
                on_reported_split=True,
            )
            remedy = (
                "set select_split"
                if config.select_split is None
                # split defaults to 'val': declaring only select_split 'val' still reports val
                else f"select_split {config.select_split!r} is also the reported split: set split to "
                "the split you report (e.g. 'test')"
            )
            _flag_selection_on_split(
                log,
                details,
                f"selected the record with the {direction} {metric!r} on split {_split_repr(config.split)}; "
                "choosing the best checkpoint on the split you report is optimistic (selection leakage). "
                f"To choose the step on a validation split and report another split at that step, {remedy}",
            )
            return selected, details
        if step is None:
            raise _Fatal(
                "no_step",
                f"the record with the {direction} {metric!r} on split {config.select_split!r} has no "
                f"integer 'step', so the {config.split!r} record at that step cannot be found",
                selected.locator("/step"),
            )

        def is_reported(rec: _Record) -> bool:
            schema = rec.data.get("schema")
            return (
                (schema is None or schema == METRICS_SCHEMA)
                and isinstance(rec.data.get("metrics"), dict)
                and rec.data.get("split") == config.split
            )

        # Second pass (O(1) memory); problems of the input were reported by the first one.
        reported = _record_at_step(
            _iter_records(data, fmt, _IssueLog()),
            is_reported,
            lambda rec: _as_int(rec.data.get("step")),
            step,
            config.split,
            log,
        )
        details = _selection_details(
            metric,
            direction,
            best,
            select_split=config.select_split,
            report_split=config.split,
            selected_step=step,
            on_reported_split=False,
        )
        _disclose_selection(log, details)
        return reported, details


@dataclass(frozen=True)
class _Context:
    """Per-record attributes shared by every metric extracted from it."""

    split: str | None = None
    dataset: str | None = None
    step: int | None = None
    seed: int | None = None


def _make_metric(
    name: str,
    value: float | None,
    *,
    unit: str | None,
    higher_is_better: bool | None,
    ctx: _Context,
    source: MetricSource,
) -> Metric:
    return Metric(
        name=name,
        value=value,
        unit=unit,
        split=ctx.split,
        step=ctx.step,
        seed=ctx.seed,
        dataset=ctx.dataset,
        higher_is_better=higher_is_better,
        source=source,
    )


def _metrics_from_record(
    rec: _Record,
    wanted: list[str] | None,
    required: Sequence[str],
    source: tuple[str, str],
    log: _IssueLog,
) -> list[Metric]:
    """Metrics of an ``hpclab.metrics/v1`` record; ``source`` is (input name, digest)."""
    values: dict[str, Any] = rec.data["metrics"]
    units = _name_map(rec, "units", str, log)
    directions = _name_map(rec, "higher_is_better", bool, log)
    ctx = _Context(
        split=_optional_str(rec, "split", log),
        dataset=_optional_str(rec, "dataset", log),
        step=_optional_int(rec, "step", log),
        seed=_optional_int(rec, "seed", log),
    )
    names = sorted(values) if wanted is None else sorted(set(wanted))
    if wanted is None and len(names) > MAX_METRICS:
        log.error(
            "too_many_metrics",
            f"the selected record has {len(names)} metrics; only the first {MAX_METRICS} by name are "
            "reported. Name the ones you need with the metrics filter",
            rec.locator("/metrics"),
        )
        names = names[:MAX_METRICS]
    out: list[Metric] = []
    for name in names:
        locator = rec.locator(f"/metrics/{_pointer_escape(name)}")
        if name not in values:
            if name not in required:  # required ones get an error from _status
                log.warning("metric_absent", f"metric {name!r} is not in the selected record", locator)
            continue
        requested = wanted is not None or name in required
        keep, value = _metric_value(name, values[name], locator, log, requested=requested)
        if keep:
            out.append(
                _make_metric(
                    name,
                    value,
                    unit=units.get(name),
                    higher_is_better=directions.get(name),
                    ctx=ctx,
                    source=MetricSource(input_name=source[0], digest=source[1], locator=locator),
                )
            )
    return out


def _metric_value(
    name: str, raw: Any, locator: str, log: _IssueLog, *, requested: bool = True
) -> tuple[bool, float | None]:
    """(keep, value): finite → (True, v); NaN/inf → (True, None) + error; other → (False, None).

    A non-number is an error when the metric was asked for by name, and only a warning when
    it merely sat next to the metrics in an "all metrics" read (per-class arrays, notes).
    Non-finite numbers are always errors: a diverged value is a result, not noise.
    """
    value, problem = _as_number(raw)
    if problem == "non_finite":
        log.error("non_finite", f"metric {name!r} is not finite ({_json_safe(raw)!r})", locator)
        return True, None
    if problem == "non_numeric":
        message = f"metric {name!r} is {_json_type(raw)}, not a number"
        if requested:
            log.error("non_numeric", message, locator)
        else:
            log.warning("non_numeric", message + "; skipped (not requested by name)", locator)
        return False, None
    return True, value


def _name_map[T](rec: _Record, key: str, kind: type[T], log: _IssueLog) -> dict[str, T]:
    raw = rec.data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        log.warning("invalid_field", f"'{key}' must be an object; ignored", rec.locator(f"/{key}"))
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, kind)}


def _optional_str(rec: _Record, key: str, log: _IssueLog) -> str | None:
    raw = rec.data.get(key)
    if raw is None or isinstance(raw, str):
        return raw
    log.warning("invalid_field", f"'{key}' must be a string; ignored", rec.locator(f"/{key}"))
    return None


def _optional_int(rec: _Record, key: str, log: _IssueLog) -> int | None:
    raw = rec.data.get(key)
    if raw is None:
        return None
    value = _as_int(raw)
    if value is None:
        log.warning("invalid_field", f"'{key}' must be an integer; ignored", rec.locator(f"/{key}"))
    return value


# ---------------------------------------------------------------------------
# json_path@1 and jsonl_record@1
# ---------------------------------------------------------------------------

_Locate = Callable[[str], str]  # JSON pointer inside the document -> full locator


def _field_context(cfg: _FieldConfig, doc: Any, locate: _Locate, log: _IssueLog) -> _Context:
    return _Context(
        split=_read_label(cfg.split, cfg.split_path, "split", doc, locate, log),
        dataset=_read_label(cfg.dataset, cfg.dataset_path, "dataset", doc, locate, log),
        step=_read_int(cfg.step_path, "step", doc, locate, log),
        seed=_read_int(cfg.seed_path, "seed", doc, locate, log),
    )


def _extract_fields(
    cfg: _FieldConfig,
    doc: Any,
    locate: _Locate,
    source: tuple[str, str],
    log: _IssueLog,
    *,
    ctx: _Context | None = None,
    names: Sequence[str] | None = None,
) -> list[Metric]:
    """Metrics named in ``cfg.fields`` (or only ``names``); ``source`` is (input name, digest).

    ``ctx`` (split/dataset/step/seed) is read from ``doc`` unless given.
    """
    if ctx is None:
        ctx = _field_context(cfg, doc, locate, log)
    out: list[Metric] = []
    for name in sorted(cfg.fields if names is None else names):
        path = cfg.fields[name]
        try:
            raw, pointer = _resolve(doc, parse_path(path))
        except _PathMissing as missing:
            log.error("missing_field", f"metric {name!r}: {missing.reason} (path {path!r})", locate(""))
            continue
        locator = locate(pointer)
        keep, value = _metric_value(name, raw, locator, log)
        if keep:
            out.append(
                _make_metric(
                    name,
                    value,
                    unit=cfg.units.get(name),
                    higher_is_better=cfg.higher_is_better.get(name),
                    ctx=ctx,
                    source=MetricSource(input_name=source[0], digest=source[1], locator=locator),
                )
            )
    return out


def _read_at(path: str, what: str, doc: Any, locate: _Locate, log: _IssueLog) -> tuple[Any, str] | None:
    try:
        return _resolve(doc, parse_path(path))
    except _PathMissing as missing:
        log.error("missing_field", f"{what}: {missing.reason} (path {path!r})", locate(""))
        return None


def _read_int(path: str | None, what: str, doc: Any, locate: _Locate, log: _IssueLog) -> int | None:
    found = _read_at(path, what, doc, locate, log) if path is not None else None
    if found is None:
        return None
    value = _as_int(found[0])
    if value is None:
        log.error(
            "invalid_field", f"{what} at {path!r} is {_json_type(found[0])}, not an integer", locate(found[1])
        )
    return value


def _read_label(
    declared: str | None, path: str | None, what: str, doc: Any, locate: _Locate, log: _IssueLog
) -> str | None:
    """A split/dataset label: the declared constant, or the document's value at ``path``.

    When both are given they must agree — reporting a 'val' number that the document says
    is 'test' is exactly the mix-up the comparison invariants exist to catch.
    """
    found = _read_at(path, what, doc, locate, log) if path is not None else None
    if found is None:
        return declared
    value, pointer = found
    if not isinstance(value, str):
        log.error(
            "invalid_field", f"{what} at {path!r} is {_json_type(value)}, not a string", locate(pointer)
        )
        return declared
    if declared is not None and value != declared:
        raise _Fatal(
            f"{what}_mismatch", f"document {what} is {value!r}, config expects {declared!r}", locate(pointer)
        )
    return value


class _JsonPathEvaluator(_BaseEvaluator[JsonPathConfig]):
    id = "json_path"
    version = "1"
    description = (
        'Extracts named numeric fields from one JSON document by path (a.b[0].c, x["val/acc"]), '
        "with declared split, units and direction; optional step/seed/split/dataset paths."
    )
    config_model = JsonPathConfig

    def _extract(self, config: JsonPathConfig, inp: EvalInput, digest: str, log: _IssueLog) -> _Extraction:
        if _resolve_format(inp) != "json":
            raise _Fatal(
                "unsupported_format",
                "json_path@1 reads one JSON document; use jsonl_record@1 or hpclab_metrics@1 for JSONL",
            )
        doc = _load_json_document(inp.data)

        def locate(pointer: str) -> str:
            return f"#{pointer}"

        metrics = _extract_fields(config, doc, locate, (inp.name, digest), log)
        selected = _bounded_record(doc, log) if isinstance(doc, dict) else None
        return _Extraction(metrics, selected)


class _RecordScan:
    """Streaming selection for ``jsonl_record@1`` (O(1) memory)."""

    def __init__(self, cfg: JsonlRecordConfig, split: str | None) -> None:
        self.cfg = cfg
        self.sel = cfg.selection
        self.split = split  # only records of this split are considered (when split_path is set)
        self.split_filter = parse_path(cfg.split_path) if cfg.split_path and split is not None else None
        self.step_path = parse_path(cfg.step_path) if cfg.step_path else None
        self.best_path = parse_path(cfg.fields[self.sel.metric]) if self.sel.metric else None
        self.records = 0
        self.splits_seen: set[str] = set()
        self.last: _Record | None = None
        self.match: _Record | None = None
        self.match_count = 0
        self.best: tuple[float, _Record] | None = None

    def offer(self, rec: _Record) -> None:
        self.records += 1
        if self.split_filter is not None:
            split = _lookup(rec.data, self.split_filter)
            self.splits_seen.add(_split_repr(split))
            if not isinstance(split, str) or split != self.split:
                return
        self.last = rec
        if self.sel.kind == "step":
            if self.step_path is not None and _as_int(_lookup(rec.data, self.step_path)) == self.sel.step:
                self.match, self.match_count = rec, self.match_count + 1
        elif self.sel.kind == "where":
            if _json_equal(_lookup(rec.data, self.sel.where_path), self.sel.where_value):
                self.match, self.match_count = rec, self.match_count + 1
        elif self.sel.kind == "best" and self.best_path is not None:
            value, _ = _as_number(_lookup(rec.data, self.best_path))
            if value is not None and (self.best is None or self._better(value, self.best[0])):
                self.best = (value, rec)

    def _better(self, value: float, incumbent: float) -> bool:
        return value > incumbent if self.sel.direction == "max" else value < incumbent

    def _require_candidates(self) -> _Record:
        if self.records == 0:
            raise _Fatal("no_record", "input contains no JSON object records")
        if self.last is None:
            raise _Fatal(
                "split_mismatch",
                f"no record has split {self.split!r} at {self.cfg.split_path!r}; "
                f"splits present: {sorted(self.splits_seen)}",
            )
        return self.last

    def choose(self, log: _IssueLog) -> _Record:
        """The record of a ``last``/``step``/``where`` selection (``best``: :meth:`choose_best`)."""
        last = self._require_candidates()
        if self.sel.kind == "last":
            return last
        if self.match is None:
            raise _Fatal("record_not_found", f"no record matches select {self.cfg.select!r}")
        if self.sel.kind == "step" and self.match_count > 1:
            log.warning(
                "duplicate_step",
                f"{self.match_count} records have step {self.sel.step} (restarted or resumed training?); "
                "using the last one",
                self.match.locator(),
            )
        return self.match

    def choose_best(self) -> tuple[float, _Record]:
        """(value, record) of a ``best`` selection; logs nothing (the evaluator knows which
        split is reported)."""
        self._require_candidates()
        if self.best is None:
            raise _Fatal("no_record", f"no record has a finite value for {self.sel.metric!r}")
        return self.best


def _lookup(doc: Any, tokens: Sequence[PathToken]) -> Any:
    try:
        return _resolve(doc, tokens)[0]
    except _PathMissing:
        return None


def _logs_value(doc: Any, path: str) -> bool:
    """Whether ``doc`` logs a value at ``path``. ``null`` is not a value: wide logs (a CSV
    logger exported with pandas) write it for the columns a row did not log. A non-finite
    number is a value (a diverged loss) and stays visible as ``non_finite``."""
    try:
        value, _ = _resolve(doc, parse_path(path))
    except _PathMissing:
        return False
    return value is not None


class _JsonlRecordEvaluator(_BaseEvaluator[JsonlRecordConfig]):
    id = "jsonl_record"
    version = "1"
    description = (
        "Selects one JSONL record (last | step:<n> | best:<metric>:max|min | where:<path>=<value>) "
        "and extracts named numeric fields from it by path, like json_path@1."
    )
    config_model = JsonlRecordConfig

    def _extract(self, config: JsonlRecordConfig, inp: EvalInput, digest: str, log: _IssueLog) -> _Extraction:
        fmt = _resolve_format(inp)
        scan = _RecordScan(config, config.select_split if config.joins_by_step else config.split)
        for rec in _iter_records(inp.data, fmt, log):
            scan.offer(rec)
        selection: dict[str, Any] | None = None
        source = (inp.name, digest)
        if config.selection.kind == "best":
            best = scan.choose_best()
            chosen, selection = self._choose_best(config, best, inp.data, fmt, log)
            if config.joins_by_step and not _logs_value(
                chosen.data, config.fields[config.selection.metric or ""]
            ):
                metrics = self._join_fields(config, best[1], chosen, source, log)
                return _Extraction(metrics, _bounded_record(chosen.data, log), selection=selection)
        else:
            chosen = scan.choose(log)
        metrics = _extract_fields(config, chosen.data, chosen.locator, source, log)
        if selection is not None and selection["on_reported_split"] is False and not config.joins_by_step:
            # One record holds both splits: the selection field was measured on select_split.
            metrics = [
                m.model_copy(update={"split": config.select_split})
                if m.name == selection["select_metric"]
                else m
                for m in metrics
            ]
        return _Extraction(metrics, _bounded_record(chosen.data, log), selection=selection)

    @staticmethod
    def _join_fields(
        config: JsonlRecordConfig,
        selected: _Record,
        reported: _Record,
        source: tuple[str, str],
        log: _IssueLog,
    ) -> list[Metric]:
        """Fields of a step join whose reported record does not log the selection field (val
        records carry ``val_loss``, test records ``test_acc``; absent or ``null``): that field
        is read from the selected record and labelled ``select_split``; every other field, and
        the context (step, seed, dataset) of all of them, from the reported record."""
        metric = config.selection.metric or ""
        ctx = _field_context(config, reported.data, reported.locator, log)
        others = [name for name in config.fields if name != metric]
        return [
            *_extract_fields(config, reported.data, reported.locator, source, log, ctx=ctx, names=others),
            *_extract_fields(
                config,
                selected.data,
                selected.locator,
                source,
                log,
                ctx=_Context(split=config.select_split, dataset=ctx.dataset, step=ctx.step, seed=ctx.seed),
                names=[metric],
            ),
        ]

    @staticmethod
    def _choose_best(
        config: JsonlRecordConfig, best: tuple[float, _Record], data: bytes, fmt: InputFormat, log: _IssueLog
    ) -> tuple[_Record, dict[str, Any]]:
        """The reported record of a ``best`` selection and its selection details.

        Without ``select_split`` the selection field's split is unknown (``on_reported_split``
        null). With ``select_split`` != ``split`` and ``split_path``, the ``split`` record at
        the selected step is reported (``joins_by_step``); without ``split_path`` the selected
        record itself holds the reported fields.
        """
        sel = config.selection
        metric = sel.metric or ""
        direction: Direction = sel.direction or "max"  # always set: the config requires it
        selected = best[1]
        step_tokens = parse_path(config.step_path) if config.step_path else None
        step = _as_int(_lookup(selected.data, step_tokens)) if step_tokens is not None else None
        split_tokens = parse_path(config.split_path) if config.split_path else None
        reported = selected
        if config.joins_by_step and step_tokens is not None and split_tokens is not None:
            if step is None:
                raise _Fatal(
                    "no_step",
                    f"the record with the {direction} {metric!r} on split {config.select_split!r} has no "
                    f"integer step at {config.step_path!r}, so the {config.split!r} record at that step "
                    "cannot be found",
                    selected.locator(),
                )
            tokens = step_tokens
            # Second pass (O(1) memory); problems of the input were reported by the first one.
            reported = _record_at_step(
                _iter_records(data, fmt, _IssueLog()),
                lambda rec: _lookup(rec.data, split_tokens) == config.split,
                lambda rec: _as_int(_lookup(rec.data, tokens)),
                step,
                config.split,
                log,
            )
        report_split = config.split
        if report_split is None and split_tokens is not None:
            label = _lookup(reported.data, split_tokens)
            report_split = label if isinstance(label, str) else None
        on_reported = None if config.select_split is None else config.select_split == config.split
        details = _selection_details(
            metric,
            direction,
            best,
            select_split=config.select_split,
            report_split=report_split,
            selected_step=step,
            on_reported_split=on_reported,
        )
        if on_reported is False:
            _disclose_selection(log, details)
        elif on_reported is None:
            # A record can hold fields of several splits (val_loss next to test_acc): which split
            # the selection field was measured on is unknown, so it may be a leak.
            _flag_selection_on_split(
                log,
                details,
                f"selected the record with the {direction} {metric!r}; jsonl_record cannot tell which "
                f"split {metric!r} was measured on. If it is the split you report, choosing the best "
                "checkpoint by it is optimistic (selection leakage); for early stopping on a validation "
                "metric, declare the split it was measured on with select_split (e.g. select_split "
                "'val', split 'test')",
            )
        else:
            _flag_selection_on_split(
                log,
                details,
                f"selected the record with the {direction} {metric!r} on split {config.select_split!r}, "
                "the split that is reported; choosing the best checkpoint on the split you report is "
                "optimistic (selection leakage)",
            )
        return reported, details


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVALUATORS: dict[tuple[str, str], Evaluator] = {}


def register_evaluator(evaluator: Evaluator, *, replace: bool = False) -> None:
    """Add an evaluator (e.g. from a project adapter). ``(id, version)`` is immutable
    unless ``replace`` — a changed protocol must get a new version so archived
    evaluations keep meaning what they meant."""
    key = (evaluator.id, evaluator.version)
    if key in EVALUATORS and not replace:
        raise InvalidArgument(f"evaluator {evaluator.id}@{evaluator.version} is already registered")
    EVALUATORS[key] = evaluator


for _builtin in (_HpclabMetricsEvaluator(), _JsonPathEvaluator(), _JsonlRecordEvaluator()):
    register_evaluator(_builtin)


def get_evaluator(evaluator_id: str, version: str) -> Evaluator:
    try:
        return EVALUATORS[(evaluator_id, version)]
    except KeyError:
        available = [f"{i}@{v}" for i, v in sorted(EVALUATORS)]
        raise NotFound(f"no evaluator {evaluator_id}@{version}", details={"available": available}) from None


def list_evaluators() -> list[EvaluatorInfo]:
    """All registered evaluators with their config JSON schema, sorted by (id, version)."""
    return [
        EvaluatorInfo(
            id=ev.id,
            version=ev.version,
            description=ev.description,
            config_schema=ev.config_model.model_json_schema(),
        )
        for _, ev in sorted(EVALUATORS.items())
    ]


def validate_config(evaluator_id: str, version: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalized config (defaults applied) — the value to hash for evaluation identity."""
    evaluator = get_evaluator(evaluator_id, version)
    return _parse_config(evaluator, config).model_dump(mode="json")


def run_evaluator(
    evaluator_id: str, version: str, config: Mapping[str, Any] | None, inputs: Sequence[EvalInput]
) -> EvalOutput:
    """Validate ``config`` and run the evaluator; see the module docstring for semantics."""
    evaluator = get_evaluator(evaluator_id, version)
    return evaluator.run(_parse_config(evaluator, config), inputs)


def _parse_config(evaluator: Evaluator, config: Mapping[str, Any] | None) -> BaseModel:
    if config is not None and not isinstance(config, Mapping):
        raise InvalidArgument(
            f"config for {evaluator.id}@{evaluator.version} must be an object, got {type(config).__name__}"
        )
    try:
        return evaluator.config_model.model_validate(dict(config or {}))
    except ValidationError as exc:
        errors = [
            {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]} for err in exc.errors()[:20]
        ]
        raise InvalidArgument(
            f"invalid config for {evaluator.id}@{evaluator.version}: {errors[0]['msg'] if errors else exc}",
            details={"errors": errors},
        ) from None
