"""The tool catalog as an agent sees it: 17 tools, documented inputs, valid schemas, skills that
use the real tool and parameter names, prompts and resource routes that resolve."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp_support import ALL_TOOLS, REPO

from hpclab.bundled import skill_text
from hpclab.mcp.catalog import DEF_DOCS, SERVER_INSTRUCTIONS, TOOLS, TOOLS_BY_NAME
from hpclab.mcp.server import PROMPTS, RESOURCES, build_tools, match_resource
from hpclab.services.app import build_registry
from hpclab.services.registry import Registry

SKILLS = sorted((REPO / "skills").glob("*/SKILL.md"))


@pytest.fixture(scope="module")
def registry() -> Registry:
    return build_registry(strict=True)


@pytest.fixture(scope="module")
def tools(registry: Registry) -> dict[str, Any]:
    return {t.name: t for t in build_tools(registry)}


def _walk_properties(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(dotted name, property schema) for top-level properties and those of every $defs model."""
    out = [(name, prop) for name, prop in (schema.get("properties") or {}).items()]
    for def_name, d in (schema.get("$defs") or {}).items():
        for name, prop in (d.get("properties") or {}).items():
            out.append((f"{def_name}.{name}", prop))
    return out


def _has_key(value: Any, key: str) -> bool:
    """Does a schema (walked structurally, not through property *names*) carry ``key``?"""
    if isinstance(value, dict):
        if key in value and isinstance(value[key], str):
            return True
        for k, v in value.items():
            if k in ("properties", "$defs"):
                if any(_has_key(p, key) for p in v.values()):
                    return True
            elif _has_key(v, key):
                return True
    elif isinstance(value, list):
        return any(_has_key(v, key) for v in value)
    return False


def test_the_catalog_exposes_exactly_the_17_tools(tools: dict[str, Any]) -> None:
    assert len(TOOLS) == 17
    assert set(TOOLS_BY_NAME) == ALL_TOOLS == set(tools)


def test_descriptions_are_substantial_and_bounded(tools: dict[str, Any]) -> None:
    for t in tools.values():
        assert t.title and t.annotations.title == t.title
        assert 300 <= len(t.description) <= 1300, (t.name, len(t.description))
        assert t.description == t.description.strip() and "  " not in t.description, t.name
    total = sum(len(t.description) for t in tools.values())
    assert total < 14_000  # the whole catalog lands in the agent's context every session


def test_runs_list_documents_every_cell_code(tools: dict[str, Any]) -> None:
    """Cell codes come from services/coverage.py; an agent must know E is not a failure."""
    desc = tools["runs_list"].description
    for code in ("V ", "R ", "Q ", "S ", "P ", "E ", "F ", "C! ", "U ", "X ", "? "):
        assert f" {code}" in desc, code
    assert "E completed, evaluation pending (not a failure" in desc
    for skill in ("hpclab-experiment-design", "hpclab-run-doctor"):
        text = skill_text(skill) or ""
        assert "`E`" in text and "not a failure" in text.lower(), skill


def test_every_input_property_is_documented(tools: dict[str, Any]) -> None:
    missing = [
        f"{t.name}:{name}"
        for t in tools.values()
        for name, prop in _walk_properties(t.input_schema)
        if len(str(prop.get("description") or "")) < 8
    ]
    assert not missing, missing


def test_input_schemas_are_valid_compact_and_closed(tools: dict[str, Any], registry: Registry) -> None:
    for t in tools.values():
        schema = t.input_schema
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert not _has_key(schema, "title"), f"{t.name}: generated titles are noise for the model"
        for name in schema.get("required") or []:
            assert name in schema["properties"], (t.name, name)
        # the server's schema says exactly what the controller accepts
        params = registry.get(TOOLS_BY_NAME[t.name].method).params_model.model_fields
        assert set(schema["properties"]) == set(params), t.name


def test_output_schemas_are_valid_and_accept_error_envelopes(tools: dict[str, Any]) -> None:
    error_env = {
        "schema_version": "hpclab/v1",
        "request_id": "req_1",
        "ok": False,
        "data": None,
        "error": {
            "code": "not_found",
            "message": "x",
            "retryable": False,
            "operation_ref": None,
            "details": {},
        },
        "coverage": None,
        "observed_at": None,
        "next_cursor": None,
        "warnings": [],
        "summary": "not_found: x",
    }
    for t in tools.values():
        Draft202012Validator.check_schema(t.output_schema)
        errors = list(Draft202012Validator(t.output_schema).iter_errors(error_env))
        assert not errors, (t.name, errors[0].message)
        data_schema = t.output_schema["properties"]["data"]["anyOf"][0]
        for lifted in ("coverage", "observed_at", "warnings", "next_cursor"):
            assert lifted not in data_schema.get("properties", {}), (t.name, lifted)


def test_catalog_prose_is_not_stale(tools: dict[str, Any]) -> None:
    for spec in TOOLS:
        props = tools[spec.name].input_schema["properties"]
        assert set(spec.params) <= set(props), (spec.name, set(spec.params) - set(props))
    defs: dict[str, set[str]] = {}
    for t in tools.values():
        for name, d in (t.input_schema.get("$defs") or {}).items():
            defs.setdefault(name, set()).update((d.get("properties") or {}).keys())
    for def_name, fields in DEF_DOCS.items():
        assert def_name in defs, f"DEF_DOCS documents unknown model {def_name}"
        assert set(fields) <= defs[def_name], (def_name, set(fields) - defs[def_name])


def test_annotations_agree_with_the_method_table(registry: Registry) -> None:
    for spec in TOOLS:
        mutation = registry.get(spec.method).mutation
        assert spec.read_only is not mutation, spec.name
        if spec.destructive:
            assert mutation and "cancel" in spec.name
    assert {s.name for s in TOOLS if s.destructive} == {"run_cancel", "workflow_cancel"}


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (
            "cluster_inspect",
            {"cluster_id": "hai", "checks": ["auth"], "candidate_shapes": [{"partition": "p"}]},
        ),
        (
            "experiment_prepare",
            {"idempotency_key": "k", "extends": "exp_1", "add_seeds": [4], "add_matrix": {"m": ["a", 1]}},
        ),
        (
            "experiment_prepare",
            {"idempotency_key": "k", "derive_from": {"run_id": "r", "overrides": {"params": {"lr": 0.1}}}},
        ),
        (
            "workflow_submit",
            {"idempotency_key": "k", "plan_id": "p", "failure_policy": {"circuit_breaker": {"threshold": 3}}},
        ),
        ("runs_list", {"experiment_id": "e", "view": "grid", "states": ["failed"], "metric": "acc"}),
        ("run_inspect", {"job_ref": {"cluster_id": "hai", "job_id": "1"}, "probe": "gpu"}),
        (
            "run_recover",
            {
                "idempotency_key": "k",
                "selector": {"workflow_id": "w", "cells": {"diagnosis_in": ["timeout"]}},
            },
        ),
        (
            "run_recover",
            {
                "idempotency_key": "k",
                "attempt_id": "a",
                "resource_patch": {"mem": "8G", "exclude_nodes": ["n1"]},
            },
        ),
        ("artifact_read", {"artifact_ref": "art_1", "range": {"offset": 10}, "view": "text"}),
        (
            "results_compare",
            {
                "planned_run_set_ref": "e",
                "metric_names": ["acc"],
                "comparison_contract": {"treatment": ["method"], "baseline": {"method": "b", "lr": 0.1}},
                "exclusions": [{"run_id": "r", "reason": "bad node"}],
            },
        ),
        (
            "evidence_export",
            {"purpose": "t", "idempotency_key": "k", "comparison_ref": "c", "table_spec": {"scale": 100}},
        ),
        ("usage_report", {"cluster_id": "hai", "scope": "account", "group_by": ["user"], "include": ["now"]}),
    ],
)
def test_schema_and_controller_agree_on_valid_arguments(
    tool: str, args: dict[str, Any], tools: dict[str, Any], registry: Registry
) -> None:
    errors = list(Draft202012Validator(tools[tool].input_schema).iter_errors(args))
    assert not errors, errors[0].message
    registry.get(TOOLS_BY_NAME[tool].method).validate(args)  # raises InvalidArgument if not


_CALL = re.compile(r"`([a-z][a-z0-9_]*)\(([^`]*)\)`")


@pytest.mark.parametrize("path", SKILLS, ids=[p.parent.name for p in SKILLS])
def test_skills_use_real_tool_and_parameter_names(
    path: Path, tools: dict[str, Any], registry: Registry
) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nname: ") and "\ndescription: " in text
    seen = 0
    for name, arglist in _CALL.findall(text):
        if name in registry.methods and name not in tools:
            pytest.fail(f"{path.parent.name}: `{name}(...)` is not an MCP tool (CLI/controller only)")
        if name not in tools:
            continue
        seen += 1
        props = tools[name].input_schema["properties"]
        for param in re.findall(r"(?:^|[,{(]\s*)([a-z_]+)\s*=", arglist):
            assert param in props, f"{path.parent.name}: {name}({param}=...) is not a parameter of {name}"
        for param in re.findall(r"(?:^|,)\s*([a-z_]+)\s*(?:,|$)", arglist):  # positional mentions
            assert param in props, f"{path.parent.name}: {name}({param}) is not a parameter of {name}"
    assert seen, f"{path.parent.name} shows no tool call"


def test_skills_use_real_data_fields() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in SKILLS)
    # fields that do not exist (or live elsewhere) in the tools' results
    for wrong in (
        "coverage.visibility",
        "`guard_failed`",
        "est_start` only",
        "`coverage` lists every planned run",
    ):
        assert wrong not in text, wrong


def test_prompts_carry_their_skill() -> None:
    for name, (skill, desc, args) in PROMPTS.items():
        body = skill_text(skill)
        assert body and len(body) > 500, name
        assert desc and all(a.description for a in args)


def test_server_instructions_mention_the_flow_and_rules() -> None:
    for word in ("experiment_prepare", "workflow_submit", "submission_uncertain", "idempotency_key", "Next:"):
        assert word in SERVER_INSTRUCTIONS
    assert len(SERVER_INSTRUCTIONS) < 2500


@pytest.mark.parametrize(
    ("uri", "method", "params"),
    [
        ("hpclab://operations/op_1", "operation_get", {"operation_id": "op_1"}),
        ("hpclab://experiments", "experiments_list", {"limit": 50}),
        ("hpclab://experiments/exp_1", "experiment_get", {"experiment_id": "exp_1"}),
        ("hpclab://plans/pln_1", "plan_get", {"plan_id": "pln_1"}),
        (
            "hpclab://plans/pln_1/script",
            "plan_script",
            {"plan_id": "pln_1", "view": "generated", "max_bytes": 65536},
        ),
        (
            "hpclab://plans/pln_1/script/full",
            "plan_script",
            {"plan_id": "pln_1", "view": "full", "max_bytes": 65536},
        ),
        ("hpclab://workflows/wf_1", "workflow_get", {"workflow_id": "wf_1"}),
        ("hpclab://workflows/wf_1/grid", "runs_list", {"workflow_id": "wf_1", "view": "grid"}),
        ("hpclab://runs/run_1", "run_inspect", {"run_id": "run_1"}),
        ("hpclab://artifacts/art_1", "artifact_get", {"artifact_id": "art_1"}),
        ("hpclab://evaluations/ev_1", "evaluation_get", {"evaluation_id": "ev_1"}),
        ("hpclab://comparisons/cmp_1", "comparison_get", {"comparison_id": "cmp_1"}),
        ("hpclab://evidence/evb_1", "evidence_get", {"bundle_id": "evb_1"}),
        (
            "hpclab://evidence/evb_1/table.tex",
            "evidence_get",
            {"bundle_id": "evb_1", "file": "table.tex", "max_bytes": 65536},
        ),
        ("hpclab://trace/evb_1:a/b.acc.mean", "trace", {"cell_ref": "evb_1:a/b.acc.mean"}),
    ],
)
def test_resource_routes(uri: str, method: str, params: dict[str, Any], registry: Registry) -> None:
    matched = match_resource(uri)
    assert matched is not None
    route, groups = matched
    assert route.method == method and route.params(groups) == params
    assert method in registry.methods
    registry.get(method).validate(params)


def test_every_resource_route_is_served(registry: Registry) -> None:
    assert all(r.method in registry.methods for r in RESOURCES)
    assert match_resource("hpclab://nope/1") is None


def test_run_cancel_describes_the_cancel_state_lifecycle() -> None:
    """cancel_state goes requested -> sent (identity-checked scancel issued) -> confirmed
    (terminal observed); the description must not claim it stays 'requested' until confirmed."""
    from hpclab.domain.states import CancelState

    text = " ".join(TOOLS_BY_NAME["run_cancel"].description.split())
    assert "'requested' until" not in text
    order = [text.find(f"'{s}'") for s in ("requested", "sent", "confirmed")]
    assert -1 not in order and order == sorted(order), text
    assert "refused" in text and "failed" in text
    for state in CancelState:
        if state is not CancelState.NONE:
            assert state.value in text, state


def test_run_cancel_says_what_failed_means_and_the_already_ended_shortcut() -> None:
    """'failed' is not only a scheduler refusal (services/submission.py sets it when the job never
    becomes visible, the adapter cannot cancel, or an abandon drops a pending request): an agent
    reading 'failed = scheduler refused' would retry scancel on a job nobody can see. A job found
    already ended is 'confirmed' with nothing sent (it never passes through 'sent')."""
    text = " ".join(TOOLS_BY_NAME["run_cancel"].description.split())
    failed = text[text.find("'failed'") :]
    assert "visible" in failed and "scheduler" in failed and "abandon" in failed, text
    assert "nothing was cancelled" in failed or "nothing cancelled" in failed, text
    assert "already ended" in text and "nothing sent" in text, text
