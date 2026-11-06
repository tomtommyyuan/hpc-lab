"""The facade calls these services through the registry: params models must derive from the
signatures and results must become Envelopes with coverage/observed_at lifted."""

from __future__ import annotations

import json
from pathlib import Path

from artifacts_support import Att, Lab

from hpclab.services import artifacts, evaluations, vault
from hpclab.services.registry import Registry


def registry() -> Registry:
    reg = Registry()
    reg.register("artifacts_list", artifacts.artifacts_list)
    reg.register("artifact_read", artifacts.artifact_read)
    reg.register("artifact_get", artifacts.artifact_get)
    reg.register("evaluation_collect", evaluations.evaluation_collect, mutation=True)
    reg.register("evaluation_get", evaluations.evaluation_get)
    reg.register("vault_status", vault.vault_status)
    reg.register("backup", vault.backup, mutation=True)
    return reg


def test_services_round_trip_through_the_registry(lab: Lab, indexed: Att, tmp_path: Path) -> None:
    reg = registry()
    listed = reg.dispatch(
        lab.ctx, "artifacts_list", {"attempt_id": indexed.attempt_id, "limit": 2}, request_id="r1"
    )
    assert listed["ok"] and listed["next_cursor"] and listed["coverage"]["complete"] is True
    assert "coverage" not in listed["data"] and listed["observed_at"]
    art = listed["data"]["artifacts"][1]["artifact_ref"]
    read = reg.dispatch(
        lab.ctx,
        "artifact_read",
        {"artifact_ref": art, "range": {"offset": 0, "length": 10}, "max_bytes": 100},
        request_id="r2",
    )
    assert read["ok"] and read["data"]["view"] == "text" and read["data"]["bytes_read"] == 10
    bad = reg.dispatch(lab.ctx, "artifact_read", {"artifact_ref": art, "view": "nope"}, request_id="r3")
    assert not bad["ok"] and bad["error"]["code"] == "invalid_argument"
    collected = reg.dispatch(
        lab.ctx,
        "evaluation_collect",
        {"idempotency_key": "k", "attempt_id": indexed.attempt_id},
        request_id="r4",
    )
    assert collected["ok"] and collected["data"]["evaluation"]["status"] == "valid"
    ev_id = collected["data"]["evaluation"]["evaluation_id"]
    got = reg.dispatch(lab.ctx, "evaluation_get", {"evaluation_id": ev_id}, request_id="r5")
    assert got["ok"] and got["data"]["evaluation_ref"].endswith(ev_id)
    stale = reg.dispatch(
        lab.ctx, "artifact_get", {"artifact_id": "art_01J00000000000000000000000"}, request_id="r6"
    )
    assert not stale["ok"] and stale["error"]["code"] == "not_found"
    status = reg.dispatch(lab.ctx, "vault_status", {}, request_id="r7")
    assert status["ok"] and status["data"]["attempts_in_scope"] == 1
    bk = reg.dispatch(
        lab.ctx, "backup", {"dest": str(tmp_path / "bk"), "idempotency_key": "b"}, request_id="r8"
    )
    assert bk["ok"] and bk["data"]["integrity_check"] == "ok"
    for env in (listed, read, collected, got, status, bk):
        json.dumps(env)  # strict JSON over the socket


def test_params_models_expose_precise_types() -> None:
    reg = registry()
    schema = reg.get("artifact_read").params_model.model_json_schema()
    view = schema["properties"]["view"]
    assert {"head", "tail", "text", "json", "jsonl_tail"} <= set(json.dumps(view).replace('"', " ").split())
    assert "ByteRange" in json.dumps(schema)
    listing = reg.get("artifacts_list").params_model.model_json_schema()
    assert set(listing["required"]) == {"attempt_id"}
    collect = reg.get("evaluation_collect").params_model.model_json_schema()
    assert collect["required"] == ["idempotency_key"]
