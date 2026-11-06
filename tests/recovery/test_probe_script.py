"""The built-in checkpoint probe program and its Run spec."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ckpt_support import fake_torch, manifest_digest, run_probe, write_checkpoint
from wf_support import run_spec

from hpclab.domain.hashing import digest
from hpclab.domain.models import RUNTIME_PLACEHOLDERS, ResourceSpec
from hpclab.services.probe import (
    PROBE_SCHEMA,
    PROBE_SOURCE,
    PROBE_VERSION,
    parse_probe_result,
    probe_resources,
    probe_run_spec,
)


def _ckpt(tmp_path: Path, files: dict[str, bytes] | None = None) -> tuple[Path, str]:
    d = tmp_path / "ckpt"
    m = write_checkpoint(
        d,
        files or {"model.pt": b"weights", "optim.pt": b"state"},
        step=10,
        run_id="run_x",
        attempt_id="att_x",
        run_spec_hash="sha256:" + "1" * 64,
        config={"lr": 0.1},
        semantic_keys=["lr"],
    )
    return d, manifest_digest(m)


def _report(out: Path) -> dict:
    return json.loads((out / "probe.json").read_text())


def test_source_is_safe_as_one_argv_element() -> None:
    # the wrapper substitutes runtime placeholders in argv: the source must contain none
    assert not [p for p in RUNTIME_PLACEHOLDERS if "{" + p + "}" in PROBE_SOURCE]
    assert "\x00" not in PROBE_SOURCE
    compile(PROBE_SOURCE, "hpclab_ckpt_probe.py", "exec")


def test_probe_passes_on_an_intact_checkpoint(tmp_path: Path) -> None:
    d, dg = _ckpt(tmp_path)
    assert run_probe(str(d), dg, tmp_path / "out") == 0
    rep = _report(tmp_path / "out")
    assert rep["ok"] is True and rep["schema"] == PROBE_SCHEMA and rep["probe_version"] == PROBE_VERSION
    assert (
        rep["manifest_digest"] == dg
    )  # the node computes the same canonical digest as hpclab.domain.hashing
    assert all(f["sha256_ok"] and f["size_ok"] for f in rep["files"])
    assert rep["load_checked"] is False  # no torch here
    assert parse_probe_result(json.dumps(rep), manifest_digest=dg).ok


@pytest.mark.parametrize("damage", ["content", "size", "missing", "manifest_changed", "incomplete"])
def test_probe_fails_on_damage(tmp_path: Path, damage: str) -> None:
    d, dg = _ckpt(tmp_path)
    if damage == "content":
        (d / "model.pt").write_bytes(b"WEIGHTS")
    elif damage == "size":
        (d / "model.pt").write_bytes(b"weights+more")
    elif damage == "missing":
        (d / "optim.pt").unlink()
    elif damage == "manifest_changed":
        m = json.loads((d / "hpclab-ckpt.json").read_text())
        m["step"] = 11
        (d / "hpclab-ckpt.json").write_text(json.dumps(m))
    else:
        m = json.loads((d / "hpclab-ckpt.json").read_text())
        m["complete"] = False
        (d / "hpclab-ckpt.json").write_text(json.dumps(m))
        dg = digest(m)
    assert run_probe(str(d), dg, tmp_path / "out") == 2
    rep = _report(tmp_path / "out")
    assert rep["ok"] is False and rep["errors"]
    assert not parse_probe_result(json.dumps(rep), manifest_digest=dg).ok


def test_probe_without_a_manifest_fails(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert run_probe(str(tmp_path / "empty"), "sha256:" + "0" * 64, tmp_path / "out") == 2
    assert "manifest unreadable" in _report(tmp_path / "out")["errors"][0]


def test_probe_loads_with_torch_and_reports_load_failures(tmp_path: Path) -> None:
    lib = fake_torch(tmp_path / "lib")
    d, dg = _ckpt(tmp_path, {"model.pt": b"weights", "state.pt": b"PICKLED-optimizer"})
    assert run_probe(str(d), dg, tmp_path / "out", extra_path=lib) == 0
    rep = _report(tmp_path / "out")
    assert rep["load_checked"] and rep["torch_version"] == "2.8.0"
    by = {f["path"]: f for f in rep["files"]}
    assert by["model.pt"]["weights_only"] is True and by["state.pt"]["weights_only"] is False
    d2, dg2 = _ckpt(tmp_path / "b", {"model.pt": b"BROKEN zip"})
    assert run_probe(str(d2), dg2, tmp_path / "out2", extra_path=lib) == 2
    assert "did not load" in _report(tmp_path / "out2")["errors"][0]


def test_parse_probe_result_is_strict() -> None:
    good = {
        "schema": PROBE_SCHEMA,
        "probe_version": PROBE_VERSION,
        "manifest_digest": "sha256:a",
        "ok": True,
        "errors": [],
    }
    assert parse_probe_result(json.dumps(good), manifest_digest="sha256:a").ok
    assert not parse_probe_result(json.dumps(good), manifest_digest="sha256:b").ok  # another manifest
    assert not parse_probe_result(json.dumps({**good, "probe_version": "0"}), manifest_digest="sha256:a").ok
    assert not parse_probe_result(json.dumps({**good, "ok": False}), manifest_digest="sha256:a").ok
    assert not parse_probe_result("{not json", manifest_digest="sha256:a").ok
    assert not parse_probe_result("[1]", manifest_digest="sha256:a").ok


def test_probe_run_spec_and_resources() -> None:
    src = run_spec(cell={"method": "a"}, seed=3)
    spec = probe_run_spec(
        src,
        target_run_id="run_t",
        checkpoint_dir="/r/c",
        manifest_digest="sha256:" + "a" * 64,
        artifact_id="art_1",
        step=5,
    )
    assert (
        spec.kind == "probe" and spec.seed is None and spec.code == src.code and spec.runtime == src.runtime
    )
    assert spec.argv == [src.runtime.python, "-c", PROBE_SOURCE, "/r/c", "sha256:" + "a" * 64]
    assert [o.path for o in spec.outputs] == ["{output_dir}/probe.json"] and spec.outputs[0].required
    assert spec.params["target_run_id"] == "run_t" and spec.params["checkpoint_step"] == 5
    base = ResourceSpec(
        partition="gpu", account="lab", gpus=4, gpu_type="h100", time="10:00:00", exclude_nodes=["bad1"]
    )
    small = probe_resources(base, checkpoint_bytes=10 * 1024**2, largest_file_bytes=1024**2)
    assert (
        small.gpus == 0
        and small.partition == "gpu"
        and small.account == "lab"
        and small.exclude_nodes == ["bad1"]
    )
    assert small.mem == "8G" and small.time == "00:17:00"
    big = probe_resources(base, checkpoint_bytes=400 * 1024**3, largest_file_bytes=100 * 1024**3)
    assert big.mem == "152G" and big.time_seconds <= 4 * 3600
