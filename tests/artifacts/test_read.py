"""artifacts_list / artifact_get / artifact_read: bounded, confined, never silently stale."""

from __future__ import annotations

import json

import pytest
from artifacts_support import Att, Lab, conn_lost, keep_mtime, metrics_jsonl

from hpclab.adapters.base import TransportError
from hpclab.domain.errors import InvalidArgument, NotFound, Stale, Unauthorized
from hpclab.services.artifacts import (
    ByteRange,
    artifact_get,
    artifact_read,
    artifacts_list,
    download_to_cache,
    index_attempt,
)


def ref(lab: Lab, att: Att, name: str) -> str:
    return lab.artifact(att, name)["artifact_id"]


def test_list_is_paginated_and_filtered(lab: Lab, indexed: Att) -> None:
    first = artifacts_list(lab.ctx, attempt_id=indexed.attempt_id, limit=2)
    assert [a.name for a in first.artifacts] == ["checkpoints", "metrics"] and first.total == 3
    assert first.next_cursor and first.coverage.complete and first.result_state == "valid"
    second = artifacts_list(lab.ctx, attempt_id=indexed.attempt_id, limit=2, cursor=first.next_cursor)
    assert [a.name for a in second.artifacts] == ["summary"] and second.next_cursor is None
    assert [a.name for a in artifacts_list(lab.ctx, attempt_id=indexed.attempt_id, kind="dir").artifacts] == [
        "checkpoints"
    ]
    assert artifacts_list(lab.ctx, attempt_id=indexed.attempt_id, kind="metrics").total == 2
    everything = artifacts_list(lab.ctx, attempt_id=indexed.attempt_id, include_members=True, limit=100)
    assert everything.total == 7
    member = next(a for a in everything.artifacts if a.member_path == "ckpt-000000060/model.bin")
    assert member.output == "checkpoints" and member.integrity == "finalized" and not member.required
    metrics = next(a for a in first.artifacts if a.name == "metrics")
    assert metrics.artifact_ref == f"hpclab://artifacts/{metrics.artifact_id}"
    assert (
        metrics.digest
        and metrics.digest_state == "computed"
        and metrics.mtime
        and metrics.mtime.endswith("Z")
    )
    assert artifact_get(lab.ctx, artifact_id=metrics.artifact_ref).artifact_id == metrics.artifact_id


def test_list_before_indexing_is_not_an_empty_success(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    data = artifacts_list(lab.ctx, attempt_id=att.attempt_id)
    assert data.artifacts == [] and data.coverage.complete is False
    assert data.coverage.missing == ["artifacts_index"]
    refreshed = artifacts_list(lab.ctx, attempt_id=att.attempt_id, refresh=True)
    assert refreshed.total == 3 and refreshed.result_state == "valid"


def test_list_of_a_never_run_attempt_is_a_true_empty(lab: Lab) -> None:
    att = lab.attempt(phase="cancelled_before_submit", scheduler_state="UNKNOWN", job_id=None)
    data = artifacts_list(lab.ctx, attempt_id=att.attempt_id, refresh=True)
    assert data.artifacts == [] and data.coverage.complete is True and data.coverage.notes


def test_bad_ids_are_rejected(lab: Lab) -> None:
    with pytest.raises(InvalidArgument):
        artifacts_list(lab.ctx, attempt_id="run_01J00000000000000000000000")
    with pytest.raises(NotFound):
        artifacts_list(lab.ctx, attempt_id="att_01J00000000000000000000000")
    with pytest.raises(InvalidArgument):
        artifact_read(lab.ctx, artifact_ref="hpclab://artifacts/not-an-id")
    with pytest.raises(InvalidArgument):
        artifacts_list(lab.ctx, attempt_id="att_01J00000000000000000000000", limit=0)


def test_default_views_by_format(lab: Lab, indexed: Att) -> None:
    tail = artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "metrics"))
    assert tail.view == "jsonl_tail" and tail.records and tail.records[-1]["final"] is True
    assert tail.digest_verified and tail.media_type == "application/x-ndjson" and tail.parse_errors == 0
    doc = artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    assert doc.view == "json" and doc.value["split"] == "test" and doc.digest_verified


def test_head_then_range_continues_on_utf8_boundaries(lab: Lab) -> None:
    att = lab.attempt(outputs=[{"name": "notes", "path": "{output_dir}/notes.txt", "role": "other"}])
    text = "héllo wörld — ünïcode ✓\n" * 20
    att.write("outputs/notes.txt", text)
    lab.manifest(att, outputs=[{"name": "notes", "path": "{output_dir}/notes.txt", "role": "other"}])
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    art = ref(lab, att, "notes")
    got, offset = "", 0
    for _ in range(100):
        page = artifact_read(lab.ctx, artifact_ref=art, range=ByteRange(offset=offset, length=7), max_bytes=7)
        assert "�" not in (page.text or "") and page.bytes_read <= 7
        got += page.text or ""
        if page.next_offset is None:
            assert page.eof
            break
        offset = page.next_offset
    assert got == text
    tail = artifact_read(lab.ctx, artifact_ref=art, view="tail", max_bytes=10)
    assert "�" not in (tail.text or "") and text.endswith(tail.text or "x") and tail.truncated


def test_json_view_of_a_large_document_returns_its_shape(lab: Lab) -> None:
    outputs = [{"name": "big", "path": "{output_dir}/big.json", "role": "other"}]
    att = lab.attempt(outputs=outputs)
    att.write("outputs/big.json", json.dumps({f"k{i}": "x" * 100 for i in range(200)}))
    lab.manifest(att, outputs=outputs)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    data = artifact_read(lab.ctx, artifact_ref=ref(lab, att, "big"), view="json", max_bytes=1024)
    assert data.value is None and data.truncated and data.value_summary is not None
    summary = data.value_summary
    assert summary["type"] == "object" and summary["key_count"] == 200 and summary["keys_truncated"]
    keys = summary["keys"]
    assert keys and keys == [f"k{i}" for i in range(len(keys))]  # a prefix, bounded by max_bytes
    wide = artifact_read(lab.ctx, artifact_ref=ref(lab, att, "big"), view="json", max_bytes=64 * 1024)
    assert wide.value is not None and len(wide.value) == 200 and wide.value_summary is None  # fits


def test_member_reference_and_directory_listing(lab: Lab, indexed: Att) -> None:
    ck = ref(lab, indexed, "checkpoints")
    listing = artifact_read(lab.ctx, artifact_ref=ck)
    assert listing.view == "members" and listing.members_total == 4 and listing.source == "ledger"
    manifest = artifact_read(lab.ctx, artifact_ref=f"hpclab://artifacts/{ck}#ckpt-000000060/hpclab-ckpt.json")
    assert manifest.view == "json" and manifest.value == {"schema": "hpclab.checkpoint/v1", "step": 60}
    assert manifest.artifact.member_path == "ckpt-000000060/hpclab-ckpt.json" and manifest.digest_verified
    for bad in ("../../outside", "/etc/passwd", "a//b", "./x"):
        with pytest.raises(InvalidArgument):
            artifact_read(lab.ctx, artifact_ref=f"{ck}#{bad}")
    with pytest.raises(NotFound):
        artifact_read(lab.ctx, artifact_ref=f"{ck}#ckpt-000000999/model.bin")
    with pytest.raises(InvalidArgument):
        artifact_read(lab.ctx, artifact_ref=f"{ref(lab, indexed, 'metrics')}#x")


def test_changed_size_is_stale(lab: Lab, indexed: Att) -> None:
    with indexed.path("outputs/metrics.jsonl").open("a") as fh:
        fh.write('{"late": 1}\n')
    with pytest.raises(Stale) as err:
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "metrics"))
    details = err.value.details
    assert details["recorded"]["size_bytes"] != details["current"]["size_bytes"]
    assert "refresh=true" in details["hint"]


def test_silent_tampering_is_stale_and_recorded(lab: Lab, indexed: Att) -> None:
    summary = indexed.path("outputs/metrics.json")
    original = summary.read_bytes()
    keep_mtime(summary, original.replace(b"0.9", b"0.1"))  # same size, same mtime, other bytes
    with pytest.raises(Stale):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    row = lab.artifact(indexed, "summary")
    assert row["digest_state"] == "mismatch" and row["integrity"] == "unknown"
    assert lab.row("attempts", "attempt_id", indexed.attempt_id)["result_state"] == "invalid"
    assert len(lab.events("artifact.digest_mismatch")) == 1
    res = index_attempt(lab.ctx, attempt_id=indexed.attempt_id)  # the manifest still attests the old bytes
    assert res.result_state == "invalid" and lab.artifact(indexed, "summary")["digest_state"] == "mismatch"


def test_symlink_swapped_in_after_indexing_is_unauthorized(lab: Lab, indexed: Att) -> None:
    outside = lab.tmp / "elsewhere.json"
    outside.write_text("{}")
    target = indexed.path("outputs/metrics.json")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(Unauthorized):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))


def test_output_symlinked_outside_is_refused_even_before_any_change(lab: Lab) -> None:
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    secret = lab.tmp / "secret.jsonl"
    secret.write_text(metrics_jsonl())
    link = att.path("outputs/metrics.jsonl")
    link.unlink()
    link.symlink_to(secret)
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    with pytest.raises(Unauthorized):
        artifact_read(lab.ctx, artifact_ref=ref(lab, att, "metrics"))


def test_removed_file_is_stale_and_missing_file_is_not_found(lab: Lab, indexed: Att) -> None:
    indexed.path("outputs/metrics.json").unlink()
    with pytest.raises(Stale):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    index_attempt(lab.ctx, attempt_id=indexed.attempt_id)
    with pytest.raises(NotFound):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))


def test_mutable_output_is_refreshed_and_flagged(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    att.write("outputs/metrics.jsonl", metrics_jsonl(final=False))
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    with att.path("outputs/metrics.jsonl").open("a") as fh:
        fh.write(
            json.dumps(
                {"schema": "hpclab.metrics/v1", "step": 40, "split": "train", "metrics": {"loss": 0.5}}
            )
            + "\n"
        )
    data = artifact_read(lab.ctx, artifact_ref=ref(lab, att, "metrics"))
    assert data.refreshed and data.records and data.records[-1]["step"] == 40
    assert not data.digest_verified and "still active" in (data.note or "")
    assert (
        data.artifact.integrity == "mutable"
        and data.artifact.size_bytes == att.path("outputs/metrics.jsonl").stat().st_size
    )


def test_cluster_down_is_an_error_unless_a_verified_copy_is_cached(lab: Lab, indexed: Att) -> None:
    lab.transport.down = conn_lost()
    with pytest.raises(TransportError):
        artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    lab.transport.down = None
    download_to_cache(lab.ctx, lab.artifact(indexed, "summary"))
    lab.transport.down = conn_lost()
    data = artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "summary"))
    assert data.source == "cache" and data.value["split"] == "test" and data.digest_verified
    assert data.coverage.complete is False


def test_output_bounds(lab: Lab, indexed: Att) -> None:
    data = artifact_read(lab.ctx, artifact_ref=ref(lab, indexed, "metrics"), view="head", max_bytes=10**9)
    assert data.bytes_read <= 64 * 1024
    with pytest.raises(InvalidArgument):
        artifact_read(
            lab.ctx, artifact_ref=ref(lab, indexed, "metrics"), view="tail", range=ByteRange(offset=1)
        )
