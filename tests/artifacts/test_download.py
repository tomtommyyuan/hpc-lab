"""Binary artifacts: the ``artifact.download`` operation into the content-addressed cache."""

from __future__ import annotations

import hashlib

import pytest
from artifacts_support import Att, Lab, auth_error, keep_mtime

from hpclab.domain.errors import LimitExceeded, Stale
from hpclab.services import artifacts
from hpclab.services.artifacts import artifact_read, cache_path_for, download_to_cache
from hpclab.services.outbox import OutboxRunner


def model_ref(lab: Lab, att: Att) -> str:
    ck = lab.artifact(att, "checkpoints")["artifact_id"]
    return f"hpclab://artifacts/{ck}#ckpt-000000060/model.bin"


def runner(lab: Lab) -> OutboxRunner:
    r = OutboxRunner(lab.ctx)
    r.register_all(artifacts.outbox_handlers())
    return r


def binary(att: Att) -> None:
    att.write("checkpoints/ckpt-000000060/model.bin", b"\x00\x01binary-weights\xff" * 10)


def test_binary_read_queues_one_verified_download(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    binary(att)
    lab.manifest(att)
    artifacts.index_attempt(lab.ctx, attempt_id=att.attempt_id)

    first = artifact_read(lab.ctx, artifact_ref=model_ref(lab, att))
    assert first.binary and first.text is None and first.view == "metadata"
    assert first.download is not None and first.download.state == "queued"
    assert first.download.operation_ref == f"hpclab://operations/{first.download.operation_id}"
    again = artifact_read(lab.ctx, artifact_ref=model_ref(lab, att))
    assert again.download is not None and again.download.operation_id == first.download.operation_id

    assert runner(lab).drain() == 1
    op = lab.row("operations", "operation_id", first.download.operation_id or "")
    assert op["state"] == "succeeded"
    row = lab.artifact(att, "checkpoints/ckpt-000000060/model.bin")
    cached = cache_path_for(lab.paths, row["digest"])
    assert (
        row["cache_path"] == str(cached)
        and cached.read_bytes() == att.path("checkpoints/ckpt-000000060/model.bin").read_bytes()
    )
    assert "sha256:" + hashlib.sha256(cached.read_bytes()).hexdigest() == row["digest"]
    assert len(lab.events("artifact.cached")) == 1

    third = artifact_read(lab.ctx, artifact_ref=model_ref(lab, att))
    assert third.download is not None and third.download.state == "cached" and third.artifact.cached


def test_tampered_file_fails_the_download_and_caches_nothing(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    binary(att)
    lab.manifest(att)
    artifacts.index_attempt(lab.ctx, attempt_id=att.attempt_id)
    first = artifact_read(lab.ctx, artifact_ref=model_ref(lab, att))
    path = att.path("checkpoints/ckpt-000000060/model.bin")
    keep_mtime(path, path.read_bytes()[::-1])
    runner(lab).drain()
    op = lab.row("operations", "operation_id", first.download.operation_id or "")  # type: ignore[union-attr]
    assert op["state"] == "failed" and '"stale"' in op["error_json"]
    row = lab.artifact(att, "checkpoints/ckpt-000000060/model.bin")
    assert row["digest_state"] == "mismatch" and row["cache_path"] is None
    assert not any(
        p.is_file() and not p.name.startswith(".") for p in (lab.paths.cache / "sha256").rglob("*")
    )


def test_auth_required_blocks_the_operation_until_login(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    binary(att)
    lab.manifest(att)
    artifacts.index_attempt(lab.ctx, attempt_id=att.attempt_id)
    first = artifact_read(lab.ctx, artifact_ref=model_ref(lab, att))
    lab.transport.down = auth_error()
    runner(lab).run_once()
    op = lab.row("operations", "operation_id", first.download.operation_id or "")  # type: ignore[union-attr]
    assert op["state"] == "blocked" and "hpclab auth lab" in op["error_json"]
    assert lab.ctx.db.one("SELECT state FROM outbox WHERE kind='artifact.download'")["state"] == "pending"  # type: ignore[index]
    lab.transport.down = None
    lab.clock.advance(121)
    runner(lab).drain()
    assert lab.row("operations", "operation_id", first.download.operation_id or "")["state"] == "succeeded"  # type: ignore[union-attr]


def test_download_limits_and_preconditions(lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch) -> None:
    row = lab.artifact(indexed, "metrics")
    with pytest.raises(LimitExceeded):
        download_to_cache(lab.ctx, row, max_bytes=10)
    monkeypatch.setattr(artifacts, "DOWNLOAD_MAX_BYTES", 10)
    dl, why = artifacts.ensure_download(lab.ctx, row)
    assert dl is None and why and "limit" in why
    with indexed.path("outputs/metrics.jsonl").open("a") as fh:
        fh.write("{}\n")
    with pytest.raises(Stale):
        download_to_cache(lab.ctx, row)


def test_corrupt_cache_entry_is_replaced(lab: Lab, indexed: Att) -> None:
    row = lab.artifact(indexed, "summary")
    path = download_to_cache(lab.ctx, row)
    path.chmod(0o644)
    path.write_bytes(b"x" * path.stat().st_size)
    again = download_to_cache(lab.ctx, lab.artifact(indexed, "summary"))
    assert again.read_bytes() == indexed.path("outputs/metrics.json").read_bytes()
