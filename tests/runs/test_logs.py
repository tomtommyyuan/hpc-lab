"""run_logs: cursors, generations (rotation/truncation), UTF-8 boundaries, tail_lines, not-started."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from observe_support import CLUSTER, Env, seed

from hpclab.adapters.base import CommandFailed, CommandResult
from hpclab.adapters.local import LocalTransport
from hpclab.api.envelope import encode_cursor
from hpclab.domain.errors import HpcLabError, InvalidArgument
from hpclab.services.logs import incomplete_tail, leading_continuations, run_logs


@pytest.fixture
def local(env: Env) -> LocalTransport:
    lt = LocalTransport(CLUSTER)
    env.ctx.clusters.transport_factory = lambda c, p: lt
    env.ctx.clusters.invalidate(CLUSTER)
    return lt


def attempt_at(env: Env, tmp_path: Path, **kw: object) -> tuple[str, Path]:
    d = tmp_path / "remote" / "attempts" / "a1"
    (d / "logs").mkdir(parents=True, exist_ok=True)
    _, aid = seed(env, remote_dir=str(d), **kw)  # type: ignore[arg-type]
    return aid, d / "logs" / "stdout.log"


TEXT = "héllo wörld ✓ 日本語 🚀 end\n" * 7


@pytest.mark.parametrize("max_bytes", [1, 2, 3, 5, 7, 64])
def test_cursor_walk_reassembles_utf8_exactly(
    env: Env, local: LocalTransport, tmp_path: Path, max_bytes: int
) -> None:
    aid, log = attempt_at(env, tmp_path, phase="terminal", scheduler_state="COMPLETED")
    log.write_bytes(TEXT.encode())
    parts: list[str] = []
    cursor = None
    for _ in range(2000):
        chunk = run_logs(env.ctx, attempt_id=aid, cursor=cursor, max_bytes=max_bytes)
        assert "�" not in chunk.text  # never split a code point
        assert len(chunk.text.encode()) <= max(max_bytes, 4)
        parts.append(chunk.text)
        if chunk.next_cursor is None:
            break
        cursor = chunk.next_cursor
    assert "".join(parts) == TEXT
    assert chunk.eof and not chunk.truncated and chunk.size == len(TEXT.encode())


def test_running_log_holds_back_a_half_written_character(
    env: Env, local: LocalTransport, tmp_path: Path
) -> None:
    aid, log = attempt_at(env, tmp_path)
    emoji = "🚀".encode()
    log.write_bytes(b"step 1 " + emoji[:2])  # the writer flushed half of the character
    first = run_logs(env.ctx, attempt_id=aid)
    assert first.text == "step 1 " and not first.eof and first.next_offset == 7
    with log.open("ab") as fh:
        fh.write(emoji[2:] + b" done\n")
    second = run_logs(env.ctx, attempt_id=aid, cursor=first.next_cursor)
    assert second.text == "🚀 done\n" and second.eof and not second.rotated
    assert second.next_cursor is not None  # running: come back later for more


def test_finished_log_with_trailing_garbage_does_not_loop(
    env: Env, local: LocalTransport, tmp_path: Path
) -> None:
    aid, log = attempt_at(env, tmp_path, phase="terminal", scheduler_state="FAILED")
    log.write_bytes(b"crash \xe6\x97")
    chunk = run_logs(env.ctx, attempt_id=aid)
    assert chunk.text.startswith("crash ") and chunk.eof and chunk.next_cursor is None


def test_replaced_file_invalidates_the_cursor(env: Env, local: LocalTransport, tmp_path: Path) -> None:
    aid, log = attempt_at(env, tmp_path)
    log.write_text("old segment line 1\nold segment line 2\n")
    first = run_logs(env.ctx, attempt_id=aid, max_bytes=10)
    gen = first.generation
    new = log.with_name("new.log")
    new.write_text("requeued segment\n")
    os.replace(new, log)  # same path, new inode
    again = run_logs(env.ctx, attempt_id=aid, cursor=first.next_cursor)
    assert again.rotated and again.offset == 0 and again.generation != gen
    assert again.text == "requeued segment\n"
    assert again.note is not None and "replaced or truncated" in again.note


def test_truncation_in_place_is_detected(env: Env, local: LocalTransport, tmp_path: Path) -> None:
    aid, log = attempt_at(env, tmp_path)
    log.write_text("x" * 100)
    first = run_logs(env.ctx, attempt_id=aid, max_bytes=80)
    with log.open("w") as fh:  # same inode, shorter
        fh.write("short\n")
    again = run_logs(env.ctx, attempt_id=aid, cursor=first.next_cursor)
    assert again.rotated and again.text == "short\n"


def test_tail_lines_then_follow(env: Env, local: LocalTransport, tmp_path: Path) -> None:
    aid, log = attempt_at(env, tmp_path)
    log.write_text("".join(f"line {i}\n" for i in range(100)))
    tail = run_logs(env.ctx, attempt_id=aid, tail_lines=3)
    assert tail.text == "line 97\nline 98\nline 99\n"
    assert tail.truncated and tail.eof and tail.offset == log.stat().st_size - len(tail.text)
    with log.open("a") as fh:
        fh.write("line 100\n")
    follow = run_logs(env.ctx, attempt_id=aid, cursor=tail.next_cursor)
    assert follow.text == "line 100\n"


def test_tail_window_drops_the_partial_first_line(env: Env, local: LocalTransport, tmp_path: Path) -> None:
    aid, log = attempt_at(env, tmp_path, phase="terminal", scheduler_state="COMPLETED")
    log.write_text("".join(f"record number {i:03d}\n" for i in range(50)))
    tail = run_logs(env.ctx, attempt_id=aid, tail_lines=1000, max_bytes=50)
    lines = tail.text.splitlines()
    assert lines and all(line.startswith("record number ") and len(line) == 17 for line in lines)
    assert tail.text.endswith("record number 049\n") and tail.truncated
    assert tail.next_cursor is None  # finished and read to the end


def test_max_bytes_is_clamped_to_64k(env: Env, local: LocalTransport, tmp_path: Path) -> None:
    aid, log = attempt_at(env, tmp_path)
    log.write_text("a" * 200_000)
    chunk = run_logs(env.ctx, attempt_id=aid, max_bytes=10_000_000)
    assert len(chunk.text) == 64 * 1024 and chunk.truncated and not chunk.eof
    default = run_logs(env.ctx, attempt_id=aid)
    assert len(default.text) == 32 * 1024


def test_missing_log_before_start_is_a_note_not_an_error(
    env: Env, local: LocalTransport, tmp_path: Path
) -> None:
    aid, log = attempt_at(env, tmp_path, phase="observing", scheduler_state="PENDING")
    chunk = run_logs(env.ctx, attempt_id=aid, stream="stderr")
    assert chunk.text == "" and chunk.eof and chunk.note is not None and "not started yet" in chunk.note
    assert chunk.next_cursor is not None
    log.with_name("stderr.log").write_text("started\n")
    later = run_logs(env.ctx, attempt_id=aid, stream="stderr", cursor=chunk.next_cursor)
    assert later.text == "started\n" and not later.rotated


def test_staging_attempt_reads_nothing(env: Env) -> None:
    _, aid = seed(env, phase="staging", scheduler_state="UNKNOWN", job_id=None)
    chunk = run_logs(env.ctx, attempt_id=aid)
    assert "not started yet" in (chunk.note or "") and env.transport.calls == []


def test_missing_log_of_a_finished_attempt_says_so(env: Env) -> None:
    _, aid = seed(env, phase="terminal", scheduler_state="FAILED", node_list="n1")
    chunk = run_logs(env.ctx, attempt_id=aid)
    assert chunk.note is not None and "not found" in chunk.note and chunk.next_cursor is None


def test_transport_failure_is_an_error_not_an_empty_log(env: Env) -> None:
    _, aid = seed(env)
    env.transport.fail["read_range"] = CommandFailed("helper failed", CommandResult(["x"], 1, "", "", 0.1))
    with pytest.raises(HpcLabError) as exc:
        run_logs(env.ctx, attempt_id=aid)
    assert exc.value.code == "unavailable"


def test_cursor_validation(env: Env) -> None:
    _, aid = seed(env)
    _, other = seed(env)
    env.transport.put(f"{env.attempt(other)['remote_dir']}/logs/stdout.log", "x")
    foreign = run_logs(env.ctx, attempt_id=other).next_cursor
    with pytest.raises(InvalidArgument):
        run_logs(env.ctx, attempt_id=aid, cursor=foreign)
    with pytest.raises(InvalidArgument):
        run_logs(env.ctx, attempt_id=aid, cursor="!!!")
    with pytest.raises(InvalidArgument):
        run_logs(env.ctx, attempt_id=aid, cursor=encode_cursor({"a": aid, "s": "stdout", "o": -1}))
    with pytest.raises(InvalidArgument):
        run_logs(env.ctx, attempt_id=aid, cursor=foreign, tail_lines=5)
    with pytest.raises(InvalidArgument):
        run_logs(env.ctx, attempt_id=aid, tail_lines=0)


def test_utf8_helpers() -> None:
    assert incomplete_tail("✓".encode()[:2]) == 2
    assert incomplete_tail("✓".encode()) == 0
    assert incomplete_tail("🚀".encode()[:3]) == 3
    assert incomplete_tail(b"abc") == 0
    assert leading_continuations("🚀".encode()[1:]) == 3
    assert leading_continuations(b"abc") == 0
