"""Checkpoint manifest validation (hpclab.checkpoint/v1) and classification of bare checkpoint dirs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hpclab.adapters.base import StatInfo
from hpclab.adapters.checkpoint import (
    CHECKPOINT_SCHEMA,
    MANIFEST_NAME,
    CheckpointCheck,
    best_resume_candidate,
    check_manifest_text,
    classify_checkpoint_dir,
    validate_checkpoint_manifest,
)
from hpclab.domain.hashing import digest

SPEC_HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64
CONFIG = {"lr": 0.1, "batch_size": 128, "world_size": 4, "method": "mixup"}
SEMANTIC = ["lr", "batch_size", "method"]
FILES = {"model.pt": b"weights" * 100, "optim.pt": b"adam-state" * 50, "extra/rng.bin": b"\x00\x01" * 8}


def write_ckpt(
    root: Path, blobs: dict[str, bytes] = FILES, **overrides: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Write files + manifest like hpclab_io.write_checkpoint; return (manifest, file_stats from disk)."""
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in blobs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(
            {"path": name, "size": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
        )
    manifest: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "step": 500,
        "epoch": 2,
        "complete": True,
        "created_at": 1_790_000_000.0,
        "run_id": "run_01ABC",
        "attempt_id": "att_01ABC",
        "run_spec_hash": SPEC_HASH,
        "config": dict(CONFIG),
        "semantic_keys": list(SEMANTIC),
        "files": entries,
        "framework": {"name": "pytorch", "version": "2.8.0"},
        "resumable": True,
    }
    manifest.update(overrides)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest, stats_from_disk(root, with_digests=True)


def stats_from_disk(root: Path, *, with_digests: bool) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != MANIFEST_NAME:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            stats[rel] = {"exists": True, "kind": "file", "size": len(data)}
            if with_digests:
                stats[rel]["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
    return stats


def check(manifest: dict[str, Any], stats: dict[str, dict[str, Any]], **kw: Any) -> CheckpointCheck:
    kw.setdefault("expected_run_spec_hash", SPEC_HASH)
    return validate_checkpoint_manifest(manifest, stats, **kw)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_valid_checkpoint_same_run_is_exact_but_needs_probe(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    c = check(manifest, stats, expected_config=dict(CONFIG))
    assert (c.status, c.continuation, c.step) == ("valid", "exact", 500)
    assert c.files_checked == 3 and c.digests_verified == 3
    assert c.needs_load_probe and c.metadata_ok_for_resume and not c.ready_for_resume
    assert c.errors == [] and c.epoch == 2.0 and c.run_id == "run_01ABC"
    assert c.manifest_digest == digest(manifest)
    passed = check(manifest, stats, probe_passed=True)
    assert passed.ready_for_resume and not passed.needs_load_probe


def test_truncated_file_is_invalid(tmp_path: Path) -> None:
    manifest, _ = write_ckpt(tmp_path)
    (tmp_path / "model.pt").write_bytes(FILES["model.pt"][:-7])
    c = check(manifest, stats_from_disk(tmp_path, with_digests=False))
    assert c.status == "invalid" and "size_mismatch" in c.issue_codes()
    assert not c.needs_load_probe  # probing a truncated file is wasted GPU time


def test_same_size_but_different_content_is_invalid(tmp_path: Path) -> None:
    manifest, _ = write_ckpt(tmp_path)
    (tmp_path / "optim.pt").write_bytes(b"X" * len(FILES["optim.pt"]))
    c = check(manifest, stats_from_disk(tmp_path, with_digests=True))
    assert c.status == "invalid" and c.issue_codes().count("digest_mismatch") == 1


def test_size_only_verification_is_valid_with_warning(tmp_path: Path) -> None:
    manifest, _ = write_ckpt(tmp_path)
    c = check(manifest, stats_from_disk(tmp_path, with_digests=False))
    assert c.status == "valid" and c.digests_verified == 0
    assert "digest_not_verified" in c.issue_codes()


def test_missing_file_is_invalid(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    del stats["extra/rng.bin"]
    c = check(manifest, stats)
    assert c.status == "invalid" and "file_missing" in c.issue_codes()


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"complete": False}, "incomplete"),
        ({"complete": "true"}, "incomplete"),
        ({"step": 0}, "step_not_positive"),
        ({"step": -3}, "step_not_positive"),
        ({"step": "500"}, "step_invalid"),
        ({"step": True}, "step_invalid"),
        ({"schema": "hpclab.checkpoint/v2"}, "unsupported_schema"),
        ({"schema": None}, "schema_missing"),
        ({"files": []}, "no_files"),
        ({"files": "model.pt"}, "no_files"),
    ],
)
def test_manifest_header_errors(tmp_path: Path, override: dict[str, Any], code: str) -> None:
    manifest, stats = write_ckpt(tmp_path, **override)
    if override.get("schema", CHECKPOINT_SCHEMA) is None:
        manifest.pop("schema")
    c = check(manifest, stats)
    assert c.status == "invalid" and code in c.issue_codes()


def test_unsupported_schema_has_unknown_continuation(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, schema="other/v9")
    assert check(manifest, stats).continuation == "unknown"


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ({"path": "../../etc/passwd", "size": 1}, "unsafe_path"),
        ({"path": "/abs/model.pt", "size": 1}, "unsafe_path"),
        ({"path": "a\\..\\b", "size": 1}, "unsafe_path"),
        ({"path": "", "size": 1}, "bad_file_entry"),
        ("model.pt", "bad_file_entry"),
        ({"path": "./model.pt", "size": len(FILES["model.pt"])}, "duplicate_path"),
    ],
)
def test_bad_file_entries(tmp_path: Path, entry: Any, code: str) -> None:
    manifest, stats = write_ckpt(tmp_path)
    manifest["files"].append(entry)
    c = check(manifest, stats)
    assert c.status == "invalid" and code in c.issue_codes()


def test_directory_in_place_of_file(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    stats["model.pt"] = {"exists": True, "kind": "dir", "size": 4096}
    assert "not_a_file" in check(manifest, stats).issue_codes()


def test_statinfo_values_and_bare_hex_digests_are_accepted(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    for entry in manifest["files"]:
        entry["sha256"] = entry["sha256"].removeprefix("sha256:").upper()
    stat_infos: dict[str, Any] = {
        p: StatInfo(path=p, exists=True, kind="file", size=s["size"]) for p, s in stats.items()
    }
    c = check(manifest, stat_infos)
    assert c.status == "valid" and c.digests_verified == 0  # StatInfo carries no digest


def test_malformed_digest_and_missing_size_are_warnings(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    manifest["files"][0]["sha256"] = "md5:abc"
    del manifest["files"][1]["size"]
    c = check(manifest, stats)
    assert c.status == "valid"
    assert {"bad_digest_format", "size_unknown"} <= set(c.issue_codes())


def test_unreadable_current_size_is_a_warning(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    stats["model.pt"] = {"exists": True}
    c = check(manifest, stats)
    assert c.status == "valid" and "size_unchecked" in c.issue_codes()


def test_extra_files_are_reported_but_harmless(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    stats["stale-from-old-run.pt"] = {"exists": True, "size": 10}
    stats[MANIFEST_NAME] = {"exists": True, "size": 999}
    c = check(manifest, stats)
    assert c.status == "valid"
    extra = next(i for i in c.issues if i.code == "extra_files")
    assert extra.severity == "info" and "1 file(s)" in extra.message


def test_non_dict_manifest_and_unreadable_text() -> None:
    c = validate_checkpoint_manifest(["not", "a", "dict"], {}, expected_run_spec_hash=None)  # type: ignore[arg-type]
    assert (c.status, c.continuation, c.needs_load_probe) == ("invalid", "unknown", False)
    t = check_manifest_text('{"schema": "hpclab.checkpoint/v1", "step": 5', {}, expected_run_spec_hash=None)
    assert t.status == "invalid" and t.issue_codes() == ["manifest_unreadable"]
    assert check_manifest_text(b"\xff\xfe", {}, expected_run_spec_hash=None).status == "invalid"


def test_check_manifest_text_round_trip(tmp_path: Path) -> None:
    _, stats = write_ckpt(tmp_path)
    text = (tmp_path / MANIFEST_NAME).read_text()
    c = check_manifest_text(text, stats, expected_run_spec_hash=SPEC_HASH)
    assert c.status == "valid" and c.continuation == "exact"
    assert CheckpointCheck.model_validate_json(c.model_dump_json()) == c


def test_nan_in_manifest_does_not_crash_digest(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    manifest["config"]["lr"] = float("nan")
    c = check(manifest, stats)
    assert c.manifest_digest is None and c.status == "valid"


# ---------------------------------------------------------------------------
# Continuation / compatibility
# ---------------------------------------------------------------------------


def test_different_hash_but_identical_semantic_config_is_exact(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH, expected_config={**CONFIG, "log_every": 5})
    assert c.continuation == "exact"
    assert "same_semantic_config" in c.issue_codes()


def test_semantic_config_diff_is_incompatible_and_lists_keys(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    c = check(
        manifest,
        stats,
        expected_run_spec_hash=OTHER_HASH,
        expected_config={**CONFIG, "lr": 0.05, "method": "base"},
    )
    assert c.continuation == "incompatible" and c.status == "valid"
    assert c.config_diffs == {
        "lr": {"checkpoint": 0.1, "expected": 0.05},
        "method": {"checkpoint": "mixup", "expected": "base"},
    }
    assert "lr, method" in next(i.message for i in c.issues if i.code == "semantic_config_mismatch")


def test_semantic_diff_wins_even_with_same_hash(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    assert check(manifest, stats, expected_config={**CONFIG, "batch_size": 64}).continuation == "incompatible"


def test_different_hash_without_config_is_weights_only(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH)
    assert c.continuation == "weights_only"
    assert "provenance_unconfirmed" in c.issue_codes()
    assert check(manifest, stats, expected_run_spec_hash=None).continuation == "weights_only"


def test_unverifiable_semantic_key_is_weights_only(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    partial = {k: v for k, v in CONFIG.items() if k != "method"}
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH, expected_config=partial)
    assert c.continuation == "weights_only"
    assert "method" in next(i.message for i in c.issues if i.code == "provenance_unconfirmed")


def test_semantic_key_absent_from_checkpoint_config(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, semantic_keys=[*SEMANTIC, "dropout"])
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH, expected_config={**CONFIG, "dropout": 0.1})
    assert "semantic_key_missing" in c.issue_codes() and c.continuation == "weights_only"


def test_world_size_change_is_not_exact(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    c = check(manifest, stats, expected_config=dict(CONFIG), expected_world_size=8)
    assert c.continuation == "weights_only"
    msg = next(i.message for i in c.issues if i.code == "world_size_changed")
    assert "4 to 8" in msg and "global batch" in msg


def test_world_size_declared_semantic_is_incompatible(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, semantic_keys=[*SEMANTIC, "world_size"])
    c = check(manifest, stats, expected_config={**CONFIG, "world_size": 2})
    assert c.continuation == "incompatible"
    assert c.config_diffs["world_size"] == {"checkpoint": 4, "expected": 2}


def test_same_world_size_from_resources_keeps_exact(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    assert check(manifest, stats, expected_world_size=4).continuation == "exact"


@pytest.mark.parametrize(("resumable", "code"), [(False, "not_resumable"), (None, "resumable_unknown")])
def test_not_resumable_is_weights_only(tmp_path: Path, resumable: bool | None, code: str) -> None:
    manifest, stats = write_ckpt(tmp_path, resumable=resumable)
    if resumable is None:
        manifest.pop("resumable")
    c = check(manifest, stats)
    assert c.continuation == "weights_only" and code in c.issue_codes()


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [(128, 128.0, True), (0.1, 0.1 + 1e-12, True), (0.1, 0.1001, False), (True, 1, False), ([1, 2], [1, 2], True),
     ({"x": 1}, {"x": 1.0}, False), ("0.1", 0.1, False), (float("nan"), float("nan"), False)],
)  # fmt: skip
def test_config_value_equality(tmp_path: Path, a: Any, b: Any, same: bool) -> None:
    manifest, stats = write_ckpt(tmp_path, config={**CONFIG, "lr": a})
    c = check(manifest, stats, expected_config={**CONFIG, "lr": b})
    assert (c.continuation == "exact") is same


def test_foreign_run_and_missing_hash_are_warned(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, run_spec_hash=None)
    c = check(manifest, stats, expected_run_id="run_OTHER")
    assert {"no_run_spec_hash", "foreign_run"} <= set(c.issue_codes())
    assert c.continuation == "weights_only"


def test_bad_config_and_semantic_keys_types(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, config="lr=0.1", semantic_keys="lr")
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH, expected_config=dict(CONFIG))
    assert {"bad_config", "bad_semantic_keys"} <= set(c.issue_codes())
    assert c.continuation == "weights_only"


def test_input_manifest_is_not_mutated(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    before = copy.deepcopy(manifest)
    check(manifest, stats, expected_config={**CONFIG, "lr": 9.0}, expected_world_size=8)
    assert manifest == before


# ---------------------------------------------------------------------------
# Directories without a manifest
# ---------------------------------------------------------------------------


def listing(entries: dict[str, int]) -> list[dict[str, Any]]:
    return [{"path": name, "size": size, "kind": "file", "exists": True} for name, size in entries.items()]


def test_hf_style_dir_is_unverified_never_valid() -> None:
    c = classify_checkpoint_dir(
        listing(
            {
                "pytorch_model.bin": 10_000,
                "optimizer.pt": 20_000,
                "scheduler.pt": 900,
                "trainer_state.json": 300,
            }
        ),
        dir_path="/scratch/run/checkpoint-500",
    )
    assert (c.status, c.continuation, c.step) == ("unverified", "unknown", 500)
    assert c.needs_load_probe and c.files_checked == 3  # .bin + two .pt files
    codes = c.issue_codes()
    assert {"no_manifest", "training_state_present", "step_from_name"} <= set(codes)
    assert ".pt" in next(i.message for i in c.issues if i.code == "no_manifest")


@pytest.mark.parametrize(
    ("names", "step"),
    [
        (["epoch=3-step=1200.ckpt"], 1200),
        (["model_0005000.pth"], 5000),
        (["ckpt_step_700.pt", "ckpt_step_900.pt"], 900),
        (["global_step42/mp_rank_00_model_states.pt"], 42),
        (["weights.safetensors"], None),
    ],
)
def test_step_inference_from_names(names: list[str], step: int | None) -> None:
    assert classify_checkpoint_dir(listing(dict.fromkeys(names, 100))).step == step


def test_zero_byte_weights_are_invalid() -> None:
    c = classify_checkpoint_dir(listing({"model.pt": 0, "optim.pt": 50}))
    assert c.status == "invalid" and "empty_file" in c.issue_codes()
    assert not c.needs_load_probe


def test_empty_dir_is_invalid() -> None:
    c = classify_checkpoint_dir([{"path": "sub", "kind": "dir"}])
    assert c.status == "invalid" and c.issue_codes() == ["no_files"]


def test_partial_and_non_weight_files_are_flagged() -> None:
    c = classify_checkpoint_dir(listing({"model.pt.tmp": 100, "notes.txt": 5}))
    assert c.status == "unverified"
    assert {"partial_write_suspected", "no_weight_files"} <= set(c.issue_codes())


def test_manifest_present_redirects_to_validation() -> None:
    c = classify_checkpoint_dir(listing({"model.pt": 100, MANIFEST_NAME: 400}))
    assert c.status == "unverified" and c.issue_codes() == ["manifest_present"]


def test_statinfo_listing_with_dir_prefix() -> None:
    base = "/scratch/ckpt"
    entries = [
        StatInfo(path=f"{base}/model.pt", exists=True, kind="file", size=100),
        StatInfo(path=f"{base}/gone.pt", exists=False, kind="missing"),
        StatInfo(path=f"{base}/sub", exists=True, kind="dir"),
    ]
    c = classify_checkpoint_dir(entries, dir_path=base + "/")
    assert c.status == "unverified" and c.files_checked == 1


# ---------------------------------------------------------------------------
# Picking a checkpoint to resume from
# ---------------------------------------------------------------------------


def test_best_resume_candidate(tmp_path: Path) -> None:
    checks: dict[str, CheckpointCheck] = {}
    for step in (100, 200, 300):
        manifest, stats = write_ckpt(tmp_path / f"ckpt-{step}", step=step)
        checks[f"ckpt-{step}"] = check(manifest, stats, probe_passed=step != 300)
    manifest, stats = write_ckpt(tmp_path / "ckpt-400", step=400, complete=False)
    checks["ckpt-400"] = check(manifest, stats, probe_passed=True)
    assert best_resume_candidate(checks) == "ckpt-200"  # 300 not probed, 400 incomplete
    assert best_resume_candidate(checks, require_probe=False) == "ckpt-300"
    assert best_resume_candidate({}) is None


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------

WS_SEMANTIC = [*SEMANTIC, "world_size"]


def test_world_size_only_semantic_key_is_never_exact_when_nothing_was_compared(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path, semantic_keys=["world_size"])
    c = check(manifest, stats, expected_run_spec_hash=OTHER_HASH, expected_config={})
    assert c.continuation == "weights_only"
    assert "same_semantic_config" not in c.issue_codes()
    assert "world_size_unverified" in c.issue_codes()


def test_semantic_world_size_must_be_known_even_with_the_same_hash(tmp_path: Path) -> None:
    # run_spec_hash does not cover resources: the same Run may resume on a different GPU count.
    manifest, stats = write_ckpt(tmp_path, semantic_keys=WS_SEMANTIC)
    c = check(manifest, stats, expected_config={k: v for k, v in CONFIG.items() if k != "world_size"})
    assert c.continuation == "weights_only"
    assert "expected_world_size" in next(i.message for i in c.issues if i.code == "world_size_unverified")


@pytest.mark.parametrize(
    "kw",
    [{"expected_world_size": 4}, {"expected_config": dict(CONFIG)}],
)
def test_semantic_world_size_compared_equal_stays_exact(tmp_path: Path, kw: dict[str, Any]) -> None:
    manifest, stats = write_ckpt(tmp_path, semantic_keys=WS_SEMANTIC)
    c = check(manifest, stats, **kw)
    assert c.continuation == "exact"
    assert "world_size_unverified" not in c.issue_codes()


def test_semantic_world_size_missing_from_checkpoint_config(tmp_path: Path) -> None:
    config = {k: v for k, v in CONFIG.items() if k != "world_size"}
    manifest, stats = write_ckpt(tmp_path, config=config, semantic_keys=WS_SEMANTIC)
    c = check(manifest, stats, expected_world_size=4)
    assert c.continuation == "weights_only"
    assert "semantic_key_missing" in c.issue_codes()


def test_deeply_nested_manifest_text_is_invalid_not_a_crash() -> None:
    c = check_manifest_text("[" * 100_000, {}, expected_run_spec_hash=None)
    assert c.status == "invalid" and c.issue_codes() == ["manifest_unreadable"]


def test_deeply_nested_config_values_do_not_crash(tmp_path: Path) -> None:
    deep: Any = 1
    for _ in range(50_000):  # deep enough that canonical JSON and repr hit the recursion limit
        deep = [deep]
    manifest, stats = write_ckpt(tmp_path)
    manifest["config"]["method"] = deep
    manifest["files"].append(deep)  # malformed entry whose preview cannot be repr()-ed either
    c = check(manifest, stats, expected_config={**CONFIG, "method": deep})
    assert c.status == "invalid" and "bad_file_entry" in c.issue_codes()
    assert c.manifest_digest is None


def test_file_stats_keyed_with_dot_slash_paths_match(tmp_path: Path) -> None:
    manifest, stats = write_ckpt(tmp_path)
    dotted = {f"./{k}": v for k, v in stats.items()}
    c = check(manifest, dotted)
    assert c.status == "valid" and "file_missing" not in c.issue_codes()
