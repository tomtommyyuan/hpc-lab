"""Bounded, cursor-based reads of an attempt's stdout/stderr (tool ``run_logs``).

* One bounded remote read per call (``max_bytes``: default 32 KiB, max 64 KiB); never follows a
  log while the job runs - the caller comes back with ``next_cursor``.
* The cursor encodes ``(offset, generation)``; the generation is the file's inode. A replaced
  file (new inode) or a file shorter than the cursor (truncated in place) makes the old cursor
  invalid: the read restarts at byte 0 with ``rotated=True`` and a note.
* UTF-8 safe: a chunk never ends inside a multi-byte character (at most 3 trailing bytes are held
  back for the next read) and a window that starts mid-character skips its continuation bytes.
* A log that does not exist yet is "not started yet" (a note, not an error). Transport problems
  raise typed errors (``unavailable`` / ``auth_required``) - never an empty success.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hpclab.adapters.base import RangeRead
from hpclab.api.envelope import clamp_bytes, decode_cursor, encode_cursor
from hpclab.domain.clock import iso
from hpclab.domain.errors import HpcLabError, InvalidArgument, NotFound, Unavailable
from hpclab.domain.states import AttemptPhase, SchedulerState
from hpclab.services.context import ServiceContext
from hpclab.services.observe import OBSERVED_PHASES, layout_for

__all__ = ["LogChunkData", "run_logs"]

MAX_TAIL_LINES = 10_000
_LOOKAHEAD = 3  # bytes read past max_bytes so a character cut at the limit can be completed
Stream = Literal["stdout", "stderr"]
# Phases in which no job can have run, so a missing log is expected and nothing is read.
_NEVER_RAN = frozenset(
    {
        AttemptPhase.STAGING.value,
        AttemptPhase.SUBMISSION_REJECTED.value,
        AttemptPhase.CANCELLED_BEFORE_SUBMIT.value,
    }
)


class LogChunkData(BaseModel):
    """One chunk of a log. ``cursor`` points at ``text``'s first byte, ``next_cursor`` after its
    last byte (``None`` once a finished attempt's log was read to the end)."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    stream: Stream
    text: str
    cursor: str | None
    next_cursor: str | None
    offset: int
    next_offset: int
    generation: int | None
    size: int | None
    truncated: bool  # more bytes exist than were returned (after this chunk, or before a tail window)
    rotated: bool
    eof: bool
    observed_at: str | None
    note: str | None = None


def _cursor(attempt_id: str, stream: str, offset: int, generation: int | None) -> str:
    return encode_cursor({"a": attempt_id, "s": stream, "o": offset, "g": generation})


def _parse_cursor(cursor: str, attempt_id: str, stream: str) -> tuple[int, int | None]:
    data = decode_cursor(cursor)
    if data.get("a") != attempt_id or data.get("s") != stream:
        raise InvalidArgument(
            "cursor belongs to another attempt or stream",
            details={"attempt_id": attempt_id, "stream": stream},
        )
    offset, gen = data.get("o"), data.get("g")
    if not isinstance(offset, int) or offset < 0 or (gen is not None and not isinstance(gen, int)):
        raise InvalidArgument("malformed log cursor")
    return offset, gen


def leading_continuations(data: bytes) -> int:
    """UTF-8 continuation bytes at the start of a window that began mid-character (<= 3)."""
    n = 0
    while n < min(3, len(data)) and 0x80 <= data[n] <= 0xBF:
        n += 1
    return n


def incomplete_tail(data: bytes) -> int:
    """Bytes of an incomplete multi-byte character at the end of ``data`` (0..3)."""
    for back in range(1, min(4, len(data)) + 1):
        b = data[-back]
        if 0x80 <= b <= 0xBF:
            continue
        if b >= 0xC0:
            return back if back < _char_len(b) else 0
        return 0
    return 0


def _char_len(lead_byte: int) -> int:
    return 2 if lead_byte < 0xE0 else 3 if lead_byte < 0xF0 else 4


def _finished(attempt: dict[str, Any]) -> bool:
    return attempt["phase"] not in OBSERVED_PHASES and attempt["phase"] not in (
        AttemptPhase.SUBMISSION_IN_FLIGHT.value,
        AttemptPhase.SUBMISSION_UNCERTAIN.value,
        AttemptPhase.DUPLICATE_DETECTED.value,
    )


def _not_started(attempt: dict[str, Any]) -> bool:
    if attempt["phase"] in _NEVER_RAN or not attempt.get("job_id"):
        return True
    return attempt["phase"] in OBSERVED_PHASES and attempt.get("scheduler_state") in (
        SchedulerState.PENDING.value,
        SchedulerState.UNKNOWN.value,
        SchedulerState.CONFIGURING.value,
    )


def _missing(
    attempt: dict[str, Any], stream: Stream, observed_at: str | None, rotated_from: bool
) -> LogChunkData:
    if _not_started(attempt):
        note = "not started yet: the job has not written this log"
    else:
        note = (
            f"{stream} log not found (it may have been removed from scratch, or the job failed before "
            "Slurm opened it)"
        )
    if rotated_from:
        note += "; the file the cursor pointed to is gone"
    finished = _finished(attempt)
    return LogChunkData(
        attempt_id=attempt["attempt_id"],
        stream=stream,
        text="",
        cursor=None,
        next_cursor=None if finished else _cursor(attempt["attempt_id"], stream, 0, None),
        offset=0,
        next_offset=0,
        generation=None,
        size=None,
        truncated=False,
        rotated=rotated_from,
        eof=True,
        observed_at=observed_at,
        note=note,
    )


def run_logs(
    ctx: ServiceContext,
    *,
    attempt_id: str,
    stream: Stream = "stdout",
    cursor: str | None = None,
    max_bytes: int | None = None,
    tail_lines: int | None = None,
) -> LogChunkData:
    """Read a bounded chunk of an attempt's log: from ``cursor`` (default: the start), or the last
    ``tail_lines`` lines of the final ``max_bytes`` window. Continue with ``next_cursor``."""
    attempt = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    if attempt is None:
        raise NotFound(f"attempt {attempt_id} not found", details={"attempt_id": attempt_id})
    if stream not in ("stdout", "stderr"):
        raise InvalidArgument("stream must be 'stdout' or 'stderr'")
    limit = clamp_bytes(max_bytes)
    if tail_lines is not None:
        if cursor is not None:
            raise InvalidArgument("give either cursor or tail_lines, not both")
        if tail_lines < 1:
            raise InvalidArgument("tail_lines must be >= 1")
        tail_lines = min(tail_lines, MAX_TAIL_LINES)
    offset, generation = _parse_cursor(cursor, attempt_id, stream) if cursor else (0, None)
    observed_at = iso(ctx.clock.now())
    if attempt["phase"] in _NEVER_RAN:
        return _missing(attempt, stream, observed_at, rotated_from=False)
    layout = layout_for(attempt)
    path = layout.stdout if stream == "stdout" else layout.stderr
    transport = ctx.clusters.transport(attempt["cluster_id"])
    finished = _finished(attempt)

    if tail_lines is not None:
        try:
            rr = _remote(lambda: transport.read_tail(path, limit), path)
        except FileNotFoundError:
            return _missing(attempt, stream, observed_at, rotated_from=False)
        return _tail_chunk(attempt, stream, rr, tail_lines, finished, observed_at)

    rotated = False
    note: str | None = None
    try:
        rr = _remote(lambda: transport.read_range(path, offset, limit + _LOOKAHEAD), path)
    except FileNotFoundError:
        return _missing(attempt, stream, observed_at, rotated_from=cursor is not None)
    if (generation is not None and rr.inode is not None and rr.inode != generation) or offset > rr.size:
        rotated = True
        note = (
            "the log file was replaced or truncated since the cursor was issued (rotation or requeue); "
            "restarting from the beginning of the current file"
        )
        offset = 0
        try:
            rr = _remote(lambda: transport.read_range(path, 0, limit + _LOOKAHEAD), path)
        except FileNotFoundError:
            return _missing(attempt, stream, observed_at, rotated_from=True)
    return _range_chunk(attempt, stream, rr, limit, finished, observed_at, rotated, note)


def _remote(read: Callable[[], RangeRead], path: str) -> RangeRead:
    """One remote read; OS errors other than 'missing' (permission, I/O) become typed errors."""
    try:
        return read()
    except (FileNotFoundError, HpcLabError):
        raise
    except OSError as exc:
        raise Unavailable(
            f"cannot read the log: {exc.strerror or exc}", details={"path": path, "errno": exc.errno}
        ) from exc


def _range_chunk(
    attempt: dict[str, Any],
    stream: Stream,
    rr: RangeRead,
    limit: int,
    finished: bool,
    observed_at: str | None,
    rotated: bool,
    note: str | None,
) -> LogChunkData:
    data = rr.data  # up to limit + _LOOKAHEAD bytes
    start = rr.offset
    lead = leading_continuations(data) if start > 0 else 0
    end = min(len(data), lead + limit)
    chunk = data[lead:end]
    at_end = start + end >= rr.size
    # A finished log never grows again: hand out trailing garbage rather than holding it back forever.
    trail = 0 if (finished and at_end) else incomplete_tail(chunk)
    if trail and trail == len(chunk):
        # max_bytes is smaller than the one character in the window: complete it from the look-ahead
        need = _char_len(chunk[0])
        if len(data) - lead >= need:
            chunk, trail = data[lead : lead + need], 0
    kept = chunk[: len(chunk) - trail]
    text_start = start + lead
    next_offset = text_start + len(kept)
    eof = next_offset >= rr.size
    aid = attempt["attempt_id"]
    return LogChunkData(
        attempt_id=aid,
        stream=stream,
        text=kept.decode("utf-8", "replace"),
        cursor=_cursor(aid, stream, text_start, rr.inode),
        next_cursor=None if (eof and finished) else _cursor(aid, stream, next_offset, rr.inode),
        offset=text_start,
        next_offset=next_offset,
        generation=rr.inode,
        size=rr.size,
        truncated=not eof,
        rotated=rotated,
        eof=eof,
        observed_at=observed_at,
        note=note,
    )


def _tail_chunk(
    attempt: dict[str, Any],
    stream: Stream,
    rr: RangeRead,
    tail_lines: int,
    finished: bool,
    observed_at: str | None,
) -> LogChunkData:
    data = rr.data
    start = rr.offset
    drop = 0
    if start > 0:
        # The window began mid-line (and maybe mid-character): skip to the next full line.
        nl = data.find(b"\n")
        drop = nl + 1 if 0 <= nl < len(data) - 1 else leading_continuations(data)
    trail = 0 if finished else incomplete_tail(data[drop:])
    body = data[drop : len(data) - trail]
    lines = body.splitlines(keepends=True)
    if len(lines) > tail_lines:
        skipped = sum(len(x) for x in lines[: len(lines) - tail_lines])
        drop += skipped
        body = body[skipped:]
    text_start = start + drop
    end = text_start + len(body)
    aid = attempt["attempt_id"]
    return LogChunkData(
        attempt_id=aid,
        stream=stream,
        text=body.decode("utf-8", "replace"),
        cursor=_cursor(aid, stream, text_start, rr.inode),
        next_cursor=None if (finished and trail == 0) else _cursor(aid, stream, end, rr.inode),
        offset=text_start,
        next_offset=end,
        generation=rr.inode,
        size=rr.size,
        truncated=text_start > 0,
        rotated=False,
        eof=end >= rr.size,
        observed_at=observed_at,
        note=None
        if text_start == 0
        else f"last {min(tail_lines, len(lines))} line(s); earlier output omitted",
    )
