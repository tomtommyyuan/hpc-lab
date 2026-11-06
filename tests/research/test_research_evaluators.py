"""Registered evaluators: selection rules, strictness, bounds and determinism."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.services import evaluators
from hpclab.services.compare import results_compare
from hpclab.services.context import ServiceContext
from hpclab.services.evaluators import (
    EVALUATORS,
    EvalInput,
    EvalOutput,
    Evaluator,
    get_evaluator,
    list_evaluators,
    parse_path,
    register_evaluator,
    run_evaluator,
    validate_config,
)
from hpclab.services.evidence import evidence_export
from hpclab.services.verify import evidence_verify

SCHEMA = "hpclab.metrics/v1"


def rec(**fields: Any) -> dict[str, Any]:
    return {"schema": SCHEMA, **fields}


def jsonl(*records: Any) -> bytes:
    return "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records).encode()


def run(
    evaluator: str, config: dict[str, Any], data: bytes, name: str = "metrics.jsonl", **kw: Any
) -> EvalOutput:
    return run_evaluator(evaluator, "1", config, [EvalInput(name=name, data=data, **kw)])


def codes(out: EvalOutput, severity: str | None = None) -> list[str]:
    return [i.code for i in out.issues if severity is None or i.severity == severity]


def values(out: EvalOutput) -> dict[str, float | None]:
    return {m.name: m.value for m in out.metrics}


TRAIN_LOG = [
    rec(step=100, total_steps=1000, split="train", time=1.0, metrics={"loss": 0.9}),
    rec(step=500, split="val", metrics={"acc": 0.80, "loss": 0.45}, higher_is_better={"acc": True}),
    rec(step=600, split="train", metrics={"loss": 0.4}),
    rec(step=800, split="val", metrics={"acc": 0.86, "loss": 0.35}, higher_is_better={"acc": True}),
    rec(
        step=1000,
        split="val",
        final=True,
        seed=3,
        dataset="cifar10",
        metrics={"acc": 0.85, "loss": 0.31},
        units={"acc": "fraction", "loss": "nats"},
        higher_is_better={"acc": True, "loss": False},
    ),
]


# ---------------------------------------------------------------------------
# hpclab_metrics@1
# ---------------------------------------------------------------------------


def test_hpclab_metrics_selects_final_record_with_full_context() -> None:
    data = jsonl(*TRAIN_LOG)
    out = run("hpclab_metrics", {}, data)
    assert out.status == "valid" and out.issues == []
    assert values(out) == {"acc": 0.85, "loss": 0.31}
    acc = out.metrics[0]
    assert (acc.unit, acc.higher_is_better, acc.split, acc.step, acc.seed, acc.dataset) == (
        "fraction",
        True,
        "val",
        1000,
        3,
        "cifar10",
    )
    assert acc.source.locator == "L5#/metrics/acc"
    assert acc.source.digest == "sha256:" + hashlib.sha256(data).hexdigest()
    assert out.selected_record is not None and out.selected_record["final"] is True
    assert out.config == {"split": "val", "select": "final", "metrics": None, "require": []}


@pytest.mark.parametrize(
    ("records", "config", "expected", "warnings"),
    [
        pytest.param(
            [
                rec(step=900, total_steps=1000, split="val", metrics={"acc": 0.84}),
                rec(step=1000, total_steps=1000, split="val", metrics={"acc": 0.86}),
            ],
            {},
            {"acc": 0.86},
            ["no_final_record"],
            id="no-final-but-reached-total-steps",
        ),
        pytest.param(
            [*TRAIN_LOG, rec(split="val", final=True, metrics={"acc": 0.87})],
            {},
            {"acc": 0.87},
            ["multiple_final_records"],
            id="multiple-finals-last-wins",
        ),
        pytest.param(TRAIN_LOG, {"select": "last"}, {"acc": 0.85, "loss": 0.31}, [], id="last"),
        pytest.param(TRAIN_LOG, {"split": "train", "select": "last"}, {"loss": 0.4}, [], id="last-train"),
        pytest.param(
            TRAIN_LOG, {"split": None, "select": "last"}, {"acc": 0.85, "loss": 0.31}, [], id="any-split"
        ),
        pytest.param(
            TRAIN_LOG,
            {"select": "best:acc"},
            {"acc": 0.86, "loss": 0.35},
            ["selection_on_split"],
            id="best-declared",
        ),
        pytest.param(
            TRAIN_LOG,
            {"select": "best:loss:min"},
            {"acc": 0.85, "loss": 0.31},
            ["selection_on_split"],
            id="best-explicit-min",
        ),
        pytest.param(
            TRAIN_LOG,
            {"select": "best:acc:min"},
            {"acc": 0.80, "loss": 0.45},
            ["selection_on_split"],
            id="best-min",
        ),
        pytest.param(TRAIN_LOG, {"metrics": ["acc"]}, {"acc": 0.85}, [], id="filter"),
        pytest.param(
            TRAIN_LOG, {"metrics": ["acc", "f1"]}, {"acc": 0.85}, ["metric_absent"], id="filter-absent"
        ),
        pytest.param(
            [rec(split="val", final=True, metrics={"acc": 1}), {"event": "checkpoint_saved"}],
            {},
            {"acc": 1.0},
            [],
            id="int-value-and-event-lines",
        ),
        pytest.param(
            [{"schema": "other/v9", "split": "val", "final": True, "metrics": {"acc": 0.1}}, TRAIN_LOG[4]],
            {},
            {"acc": 0.85, "loss": 0.31},
            ["foreign_schema"],
            id="foreign-schema-ignored",
        ),
        pytest.param(
            [{"split": "val", "final": True, "metrics": {"acc": 0.7}}],
            {},
            {"acc": 0.7},
            [],
            id="schema-optional",
        ),
    ],
)
def test_hpclab_metrics_selection(
    records: list[Any], config: dict[str, Any], expected: dict[str, float], warnings: list[str]
) -> None:
    out = run("hpclab_metrics", config, jsonl(*records))
    assert out.status == "valid", out.issues
    assert values(out) == expected
    assert codes(out, "warning") == warnings


def test_best_ties_keep_the_first_record() -> None:
    data = jsonl(
        rec(step=1, split="val", metrics={"acc": 0.9}), rec(step=2, split="val", metrics={"acc": 0.9})
    )
    out = run("hpclab_metrics", {"select": "best:acc:max"}, data)
    assert out.metrics[0].step == 1


@pytest.mark.parametrize(
    ("records", "config", "code"),
    [
        pytest.param(
            [rec(split="test", final=True, metrics={"acc": 0.9}), rec(split="val", metrics={"acc": 0.8})],
            {},
            "split_mismatch",
            id="final-only-for-other-split",
        ),
        pytest.param([rec(split="train", metrics={"loss": 1.0})], {}, "split_mismatch", id="no-val-records"),
        pytest.param(
            [rec(final=True, metrics={"acc": 0.9})], {}, "split_mismatch", id="record-without-split"
        ),
        pytest.param([{"event": "start"}], {}, "no_record", id="no-metrics-records"),
        pytest.param(TRAIN_LOG, {"select": "best:f1"}, "direction_unknown", id="best-direction-unknown"),
        pytest.param(
            [
                rec(split="val", metrics={"m": 1.0}, higher_is_better={"m": True}),
                rec(split="val", metrics={"m": 2.0}, higher_is_better={"m": False}),
            ],
            {"select": "best:m"},
            "conflicting_direction",
            id="best-conflicting-direction",
        ),
        pytest.param(TRAIN_LOG, {"select": "best:f1:max"}, "no_record", id="best-metric-never-finite"),
    ],
)
def test_hpclab_metrics_fatal_selection_errors(records: list[Any], config: dict[str, Any], code: str) -> None:
    out = run("hpclab_metrics", config, jsonl(*records))
    assert out.status == "invalid"
    assert out.metrics == [] and out.selected_record is None
    assert codes(out, "error")[-1] == code


def test_split_mismatch_message_is_actionable() -> None:
    data = jsonl(rec(split="test", final=True, metrics={"acc": 0.9}))
    out = run("hpclab_metrics", {}, data)
    assert "['test']" in out.issues[-1].message and "select='last'" in out.issues[-1].message


@pytest.mark.parametrize(
    ("line", "code", "value"),
    [
        pytest.param(
            '{"split":"val","final":true,"metrics":{"acc":0.9,"loss":NaN}}', "non_finite", None, id="nan"
        ),
        pytest.param(
            '{"split":"val","final":true,"metrics":{"acc":0.9,"loss":Infinity}}', "non_finite", None, id="inf"
        ),
        pytest.param(
            '{"split":"val","final":true,"metrics":{"acc":0.9,"loss":1' + "0" * 400 + "}}",
            "non_finite",
            None,
            id="overflow",
        ),
    ],
)
def test_non_finite_values_are_kept_as_none_with_error(line: str, code: str, value: None) -> None:
    out = run("hpclab_metrics", {}, (line + "\n").encode())
    assert out.status == "partial"
    assert values(out) == {"acc": 0.9, "loss": value}
    assert codes(out) == [code]
    assert out.selected_record is not None
    json.dumps(out.selected_record, allow_nan=False)  # strict JSON even when the record held NaN


@pytest.mark.parametrize("bad", ["true", '"0.9"', "null", "[0.9]", '{"v":0.9}'])
def test_requested_non_numeric_values_are_dropped_with_error(bad: str) -> None:
    line = '{"split":"val","final":true,"metrics":{"acc":0.9,"loss":' + bad + "}}\n"
    out = run("hpclab_metrics", {"metrics": ["acc", "loss"]}, line.encode())
    assert out.status == "partial"
    assert values(out) == {"acc": 0.9}
    assert codes(out) == ["non_numeric"]


def test_unrequested_non_numeric_entries_do_not_spoil_the_evaluation() -> None:
    # Loggers put per-class arrays, notes or nulls next to scalar metrics; with no metrics filter
    # they are skipped with a warning instead of turning every run into a non-countable 'partial'.
    record = rec(split="val", final=True, metrics={"acc": 0.9, "per_class": [0.8, 1.0], "note": "ok"})
    out = run("hpclab_metrics", {}, jsonl(record))
    assert out.status == "valid" and values(out) == {"acc": 0.9}
    assert codes(out, "warning") == ["non_numeric", "non_numeric"]


def test_unrequested_non_finite_metric_is_still_an_error() -> None:
    # A diverged auxiliary value is a result signal (spec: NaN/inf -> non_finite error), not noise.
    line = b'{"split":"val","final":true,"metrics":{"acc":0.9,"grad_norm":NaN}}\n'
    out = run("hpclab_metrics", {}, line)
    assert out.status == "partial" and codes(out) == ["non_finite"]


def test_require_missing_or_non_finite_makes_evaluation_invalid() -> None:
    data = jsonl(rec(split="val", final=True, metrics={"acc": 0.9}))
    out = run("hpclab_metrics", {"require": ["acc", "f1"]}, data)
    assert out.status == "invalid"
    assert codes(out) == ["missing_required_metric"]  # no duplicate 'metric_absent' warning
    assert values(out) == {"acc": 0.9}

    nan = b'{"split":"val","final":true,"metrics":{"acc":0.9,"f1":NaN}}\n'
    out = run("hpclab_metrics", {"require": ["f1"]}, nan)
    assert out.status == "invalid"
    assert codes(out) == ["non_finite", "missing_required_metric"]


def test_require_extends_the_metrics_filter() -> None:
    data = jsonl(rec(split="val", final=True, metrics={"acc": 0.9, "loss": 0.3, "f1": 0.5}))
    out = run("hpclab_metrics", {"metrics": ["acc"], "require": ["loss"]}, data)
    assert values(out) == {"acc": 0.9, "loss": 0.3}


def test_empty_metrics_object_is_invalid() -> None:
    out = run("hpclab_metrics", {}, jsonl(rec(split="val", final=True, metrics={})))
    assert out.status == "invalid" and codes(out) == ["no_finite_metrics"]


def test_invalid_context_fields_are_warnings() -> None:
    data = jsonl(
        rec(split="val", final=True, step="ten", seed=1.5, dataset=7, units=[1], metrics={"acc": 0.5})
    )
    out = run("hpclab_metrics", {}, data)
    assert out.status == "valid"
    assert codes(out, "warning") == ["invalid_field"] * 4
    m = out.metrics[0]
    assert (m.step, m.seed, m.dataset, m.unit) == (None, None, None, None)


def test_integral_float_step_is_accepted() -> None:
    out = run("hpclab_metrics", {}, jsonl(rec(split="val", final=True, step=100.0, seed=2, metrics={"a": 1})))
    assert (out.metrics[0].step, out.metrics[0].seed) == (100, 2)


# ---------------------------------------------------------------------------
# Malformed and edge inputs (shared by all record-based evaluators)
# ---------------------------------------------------------------------------


def test_malformed_lines_are_errors_but_good_records_still_count() -> None:
    data = jsonl(TRAIN_LOG[0], "{broken", "[1, 2]", "42", TRAIN_LOG[4]) + b"\xff\xfe\n"
    out = run("hpclab_metrics", {}, data)
    assert out.status == "partial"
    assert values(out) == {"acc": 0.85, "loss": 0.31}
    assert [(i.code, i.locator) for i in out.issues] == [
        ("parse_error", "L2"),
        ("not_an_object", "L3"),
        ("not_an_object", "L4"),
        ("parse_error", "L6"),
    ]


def test_partially_written_last_line_is_reported() -> None:
    data = jsonl(*TRAIN_LOG) + b'{"schema":"hpclab.metrics/v1","step":1001,"spl'
    out = run("hpclab_metrics", {}, data)
    assert out.status == "partial" and codes(out) == ["parse_error"]
    assert out.issues[0].locator == "L6"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        pytest.param(b"", "empty_input", id="empty"),
        pytest.param(b" \n\n\t\n", "empty_input", id="whitespace"),
        pytest.param(b"\xef\xbb\xbf", "empty_input", id="bom-only"),
        pytest.param(b"{broken\n[1]\n", "no_record", id="nothing-parseable"),
    ],
)
def test_empty_or_unusable_input_is_invalid(data: bytes, code: str) -> None:
    out = run("hpclab_metrics", {}, data)
    assert out.status == "invalid" and codes(out, "error")[-1] == code


def test_bom_and_crlf_are_tolerated() -> None:
    data = b"\xef\xbb\xbf" + jsonl(*TRAIN_LOG).replace(b"\n", b"\r\n")
    out = run("hpclab_metrics", {}, data)
    assert out.status == "valid" and values(out) == {"acc": 0.85, "loss": 0.31}


def test_blank_lines_keep_physical_line_numbers() -> None:
    data = b"\n\n" + jsonl(TRAIN_LOG[4])
    out = run("hpclab_metrics", {}, data)
    assert out.metrics[0].source.locator == "L3#/metrics/acc"


def test_json_document_single_record_and_array() -> None:
    single = json.dumps(TRAIN_LOG[4]).encode()
    out = run("hpclab_metrics", {}, single, name="final.json")
    assert out.status == "valid" and out.metrics[0].source.locator == "#/metrics/acc"

    array = json.dumps([TRAIN_LOG[1], 7, TRAIN_LOG[4]]).encode()
    out = run("hpclab_metrics", {}, array, name="all.json")
    assert out.status == "partial" and codes(out) == ["not_an_object"]
    assert out.metrics[0].source.locator == "#/2/metrics/acc"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        pytest.param(b"42", "not_an_object", id="scalar"),
        pytest.param(b'{"a": ', "parse_error", id="truncated"),
        pytest.param(b"\xff\xfe{}", "decode_error", id="not-utf8"),
        pytest.param(b"[" * 100_000 + b"]" * 100_000, "parse_error", id="deeply-nested"),
    ],
)
def test_bad_json_documents_are_fatal(data: bytes, code: str) -> None:
    out = run("hpclab_metrics", {}, data, name="final.json")
    assert out.status == "invalid" and codes(out) == [code]


@pytest.mark.parametrize(
    ("name", "fmt", "data", "locator"),
    [
        ("metrics", None, json.dumps(TRAIN_LOG[4]).encode(), "#/metrics/acc"),  # sniffed: one JSON doc
        ("metrics", None, jsonl(TRAIN_LOG[3], TRAIN_LOG[4]), "L2#/metrics/acc"),  # sniffed: JSONL
        ("m.ndjson", None, jsonl(TRAIN_LOG[4]), "L1#/metrics/acc"),
        ("m.txt", "jsonl", jsonl(TRAIN_LOG[4]), "L1#/metrics/acc"),
        ("m.jsonl", "json", json.dumps(TRAIN_LOG[4]).encode(), "#/metrics/acc"),  # explicit format wins
    ],
)
def test_format_inference(name: str, fmt: str | None, data: bytes, locator: str) -> None:
    out = run("hpclab_metrics", {}, data, name=name, format=fmt)
    assert out.metrics[0].source.locator == locator


def test_input_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    data = jsonl(*TRAIN_LOG)
    monkeypatch.setattr(evaluators, "MAX_INPUT_BYTES", len(data) - 1)
    out = run("hpclab_metrics", {}, data)
    assert out.status == "invalid" and codes(out) == ["input_too_large"]


def test_line_limit_truncates_with_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "MAX_JSONL_LINES", 3)
    out = run("hpclab_metrics", {"select": "last"}, jsonl(*TRAIN_LOG))
    assert out.status == "partial"
    assert codes(out) == ["input_truncated"] and out.issues[0].locator == "L4"
    assert out.metrics[0].step == 500  # the last record *read*, and the issue says so


def test_line_limit_exactly_reached_is_not_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "MAX_JSONL_LINES", 5)
    out = run("hpclab_metrics", {}, jsonl(*TRAIN_LOG) + b"\n  \n")
    assert out.status == "valid"


def test_issue_list_is_bounded_but_errors_still_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "MAX_ISSUES", 5)
    data = jsonl(*(["{bad"] * 20), TRAIN_LOG[4])
    out = run("hpclab_metrics", {}, data)
    assert out.status == "partial"
    assert codes(out) == ["parse_error"] * 5 + ["issues_truncated"]
    assert "15 further" in out.issues[-1].message


def test_fatal_issue_is_listed_even_past_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "MAX_ISSUES", 2)
    out = run("hpclab_metrics", {}, jsonl(*(["{bad"] * 5)))
    assert out.status == "invalid"
    assert codes(out) == ["parse_error", "parse_error", "no_record", "issues_truncated"]


def test_digest_is_verified() -> None:
    data = jsonl(*TRAIN_LOG)
    good = "sha256:" + hashlib.sha256(data).hexdigest()
    assert run("hpclab_metrics", {}, data, digest=good).status == "valid"
    out = run("hpclab_metrics", {}, data, digest="sha256:" + "0" * 64)
    assert out.status == "invalid" and codes(out) == ["input_digest_mismatch"] and out.metrics == []


def test_large_selected_record_is_bounded() -> None:
    record = rec(split="val", final=True, metrics={"acc": 0.9}, histogram=list(range(20_000)))
    out = run("hpclab_metrics", {}, jsonl(record))
    assert out.status == "valid" and codes(out, "warning") == ["selected_record_truncated"]
    assert out.selected_record is not None and out.selected_record["_truncated"] is True
    assert "histogram" in out.selected_record["_keys"]


def test_output_is_deterministic_and_sorted() -> None:
    data = jsonl(rec(split="val", final=True, metrics={"zeta": 1, "alpha": 2, "mid": 3}))
    first, second = run("hpclab_metrics", {}, data), run("hpclab_metrics", {}, data)
    assert first == second
    assert [m.name for m in first.metrics] == ["alpha", "mid", "zeta"]


# ---------------------------------------------------------------------------
# json_path@1
# ---------------------------------------------------------------------------

RESULTS_DOC = {
    "eval": {"split": "val", "dataset": "imagenet-1k", "step": 5000},
    "seed": 7,
    "results": [{"top1": 0.71, "top5": 0.9}, {"top1": 0.76, "top5": 0.93}],
    "metrics": {"val/acc": 0.76, "loss": float("nan")},
}


def doc_bytes(doc: Any) -> bytes:
    return json.dumps(doc).encode()


def test_json_path_extracts_with_indexes_and_quoted_keys() -> None:
    config = {
        "fields": {"top1": "results[-1].top1", "top5_first": "results[0].top5", "acc": 'metrics["val/acc"]'},
        "split": "val",
        "split_path": "eval.split",
        "dataset_path": "eval.dataset",
        "step_path": "eval.step",
        "seed_path": "seed",
        "units": {"top1": "fraction"},
        "higher_is_better": {"top1": True, "acc": True},
    }
    out = run("json_path", config, doc_bytes(RESULTS_DOC), name="results.json")
    assert out.status == "valid", out.issues
    assert values(out) == {"acc": 0.76, "top1": 0.76, "top5_first": 0.9}
    by_name = {m.name: m for m in out.metrics}
    assert by_name["top1"].source.locator == "#/results/1/top1"
    assert by_name["acc"].source.locator == "#/metrics/val~1acc"
    m = by_name["top1"]
    assert (m.split, m.dataset, m.step, m.seed, m.unit, m.higher_is_better) == (
        "val",
        "imagenet-1k",
        5000,
        7,
        "fraction",
        True,
    )


@pytest.mark.parametrize(
    ("fields", "status", "issue_codes"),
    [
        pytest.param(
            {"a": "results[0].top1", "b": "results[5].top1"}, "partial", ["missing_field"], id="index-range"
        ),
        pytest.param({"a": "results[0].top1", "b": "nope.x"}, "partial", ["missing_field"], id="missing-key"),
        pytest.param(
            {"a": "results[0].top1", "b": "seed.x"}, "partial", ["missing_field"], id="not-an-object"
        ),
        pytest.param(
            {"a": "results[0].top1", "b": "eval[0]"}, "partial", ["missing_field"], id="not-an-array"
        ),
        pytest.param({"a": "results[0].top1", "b": "eval.split"}, "partial", ["non_numeric"], id="string"),
        pytest.param({"a": "results[0].top1", "b": "metrics.loss"}, "partial", ["non_finite"], id="nan"),
        pytest.param({"b": "missing"}, "invalid", ["missing_field"], id="nothing-found"),
    ],
)
def test_json_path_field_problems(fields: dict[str, str], status: str, issue_codes: list[str]) -> None:
    out = run("json_path", {"fields": fields}, doc_bytes(RESULTS_DOC), name="r.json")
    assert out.status == status
    assert codes(out) == issue_codes


def test_json_path_split_mismatch_is_fatal() -> None:
    config = {"fields": {"a": "results[0].top1"}, "split": "test", "split_path": "eval.split"}
    out = run("json_path", config, doc_bytes(RESULTS_DOC), name="r.json")
    assert out.status == "invalid" and codes(out) == ["split_mismatch"]


def test_json_path_dataset_mismatch_is_fatal() -> None:
    config = {"fields": {"a": "results[0].top1"}, "dataset": "cifar10", "dataset_path": "eval.dataset"}
    out = run("json_path", config, doc_bytes(RESULTS_DOC), name="r.json")
    assert out.status == "invalid" and codes(out) == ["dataset_mismatch"]


def test_json_path_bad_context_fields_are_errors() -> None:
    config = {"fields": {"a": "results[0].top1"}, "step_path": "eval.split", "seed_path": "nope"}
    out = run("json_path", config, doc_bytes(RESULTS_DOC), name="r.json")
    assert out.status == "partial" and codes(out) == ["invalid_field", "missing_field"]
    assert out.metrics[0].step is None and out.metrics[0].seed is None


def test_json_path_declared_split_without_path() -> None:
    out = run("json_path", {"fields": {"a": "seed"}, "split": "test"}, doc_bytes(RESULTS_DOC), name="r.json")
    assert out.metrics[0].split == "test"


def test_json_path_rejects_jsonl() -> None:
    out = run("json_path", {"fields": {"a": "x"}}, jsonl({"x": 1}, {"x": 2}), name="r.jsonl")
    assert out.status == "invalid" and codes(out) == ["unsupported_format"]


def test_json_path_on_top_level_array() -> None:
    out = run("json_path", {"fields": {"a": "[1].v"}}, b'[{"v": 1}, {"v": 2}]', name="r.json")
    assert values(out) == {"a": 2.0} and out.selected_record is None


# ---------------------------------------------------------------------------
# jsonl_record@1
# ---------------------------------------------------------------------------

EVAL_LOG = [
    {"step": 100, "phase": "val", "acc": 0.5, "final": False},
    {"step": 200, "phase": "val", "acc": 0.7},
    {"step": 200, "phase": "train", "acc": 0.9},
    {"step": 300, "phase": "val", "acc": 0.6, "final": True, "seed": 4},
    {"step": 200, "phase": "val", "acc": 0.65},  # logged again after a resume
]


@pytest.mark.parametrize(
    ("select", "extra", "acc", "step", "warnings"),
    [
        pytest.param("last", {}, 0.65, 200, [], id="last"),
        pytest.param("step:300", {}, 0.6, 300, [], id="step"),
        pytest.param("step:200", {}, 0.65, 200, ["duplicate_step"], id="duplicate-step"),
        pytest.param("best:acc:max", {}, 0.9, 200, ["selection_on_split"], id="best-max"),
        pytest.param("best:acc:min", {}, 0.5, 100, ["selection_on_split"], id="best-min"),
        pytest.param("where:final=true", {}, 0.6, 300, [], id="where-bool"),
        pytest.param("where:phase=train", {}, 0.9, 200, [], id="where-string"),
        pytest.param("where:step=100", {}, 0.5, 100, [], id="where-number"),
        pytest.param(
            "best:acc:max",
            {"split": "val", "split_path": "phase"},
            0.7,
            200,
            ["selection_on_split"],
            id="split-filter",
        ),
        pytest.param("last", {"split": "train", "split_path": "phase"}, 0.9, 200, [], id="split-filter-last"),
    ],
)
def test_jsonl_record_selection(
    select: str, extra: dict[str, Any], acc: float, step: int, warnings: list[str]
) -> None:
    config = {"fields": {"acc": "acc"}, "step_path": "step", "select": select, **extra}
    out = run("jsonl_record", config, jsonl(*EVAL_LOG))
    assert out.status == "valid", out.issues
    assert values(out) == {"acc": acc} and out.metrics[0].step == step
    assert codes(out, "warning") == warnings


def test_jsonl_record_where_is_json_typed() -> None:
    config = {"fields": {"acc": "acc"}, "select": "where:final=1"}  # 1 must not match true
    out = run("jsonl_record", config, jsonl(*EVAL_LOG))
    assert out.status == "invalid" and codes(out) == ["record_not_found"]


@pytest.mark.parametrize(
    ("config", "code"),
    [
        pytest.param({"select": "step:999", "step_path": "step"}, "record_not_found", id="step-absent"),
        pytest.param({"select": "where:phase=test"}, "record_not_found", id="where-absent"),
        pytest.param({"split": "test", "split_path": "phase"}, "split_mismatch", id="split-absent"),
        pytest.param({"fields": {"f1": "f1"}, "select": "best:f1:max"}, "no_record", id="best-never-finite"),
    ],
)
def test_jsonl_record_fatal_errors(config: dict[str, Any], code: str) -> None:
    out = run("jsonl_record", {"fields": {"acc": "acc"}, **config}, jsonl(*EVAL_LOG))
    assert out.status == "invalid" and codes(out) == [code]


def test_jsonl_record_reads_json_array_too() -> None:
    out = run("jsonl_record", {"fields": {"acc": "acc"}}, json.dumps(EVAL_LOG).encode(), name="log.json")
    assert values(out) == {"acc": 0.65} and out.metrics[0].source.locator == "#/4/acc"


EARLY_STOP_LOG = [
    {"epoch": 1, "val_loss": 0.9, "test_acc": 0.70},
    {"epoch": 2, "val_loss": 0.5, "test_acc": 0.80},
    {"epoch": 3, "val_loss": 0.6, "test_acc": 0.83},
]


def test_jsonl_record_best_selection_without_select_split_leaves_the_split_open() -> None:
    """Early stopping: pick the epoch with the lowest val_loss, report its test_acc.

    One record carries fields of several splits, so without a declared ``select_split``
    jsonl_record cannot tell which split the selection field was measured on. It must not claim
    the selection was made on the reported split, nor that it was not: it records what it
    selected by (``details``) with the split left open (``on_reported_split`` null), which
    downstream checks treat as a possible leak. (Configs written before ``select_split``
    existed keep exactly this behaviour.)
    """
    config = {
        "fields": {"val_loss": "val_loss", "test_acc": "test_acc"},
        "select": "best:val_loss:min",
        "split": "test",
    }
    out = run("jsonl_record", config, jsonl(*EARLY_STOP_LOG))
    assert out.status == "valid" and values(out) == {"test_acc": 0.80, "val_loss": 0.5}
    (issue,) = out.issues
    assert (issue.severity, issue.code, issue.locator) == ("warning", "selection_on_split", "L2#")
    assert issue.details == {
        "select_metric": "val_loss",
        "direction": "min",
        "select_split": None,
        "report_split": "test",
        "selected_step": None,
        "select_value": 0.5,
        "select_locator": "L2#",
        "on_reported_split": None,
    }
    assert out.selection == issue.details
    assert "cannot tell which split 'val_loss'" in issue.message and "select_split" in issue.message
    assert "on the split you report is optimistic" not in issue.message
    stored = json.loads(json.dumps(issue.model_dump(mode="json"), allow_nan=False))
    assert stored["code"] == "selection_on_split" and stored["details"] == issue.details


@pytest.mark.parametrize(
    ("records", "config", "details"),
    [
        pytest.param(
            TRAIN_LOG,
            {"select": "best:acc"},
            {
                "select_metric": "acc",
                "direction": "max",
                "select_split": "val",
                "report_split": "val",
                "selected_step": 800,
                "select_value": 0.86,
                "select_locator": "L4#",
                "on_reported_split": True,
            },
            id="declared-direction",
        ),
        pytest.param(
            [{"metrics": {"acc": 0.5}}, {"metrics": {"acc": 0.7}}],
            {"split": None, "select": "best:acc:max"},
            {
                "select_metric": "acc",
                "direction": "max",
                "select_split": None,
                "report_split": None,
                "selected_step": None,
                "select_value": 0.7,
                "select_locator": "L2#",
                "on_reported_split": True,
            },
            id="unlabelled-split-is-still-the-reported-one",
        ),
        pytest.param(
            TRAIN_LOG,
            {"select": "best:acc", "select_split": "val"},
            {
                "select_metric": "acc",
                "direction": "max",
                "select_split": "val",
                "report_split": "val",
                "selected_step": 800,
                "select_value": 0.86,
                "select_locator": "L4#",
                "on_reported_split": True,
            },
            id="select-split-declared-equal-to-the-reported-one",
        ),
    ],
)
def test_hpclab_metrics_best_selection_is_known_to_be_on_the_reported_split(
    records: list[Any], config: dict[str, Any], details: dict[str, Any]
) -> None:
    """An hpclab.metrics/v1 record carries one split, so the selection metric and the reported
    metrics always share it: ``on_reported_split`` is true and no contract can clear it."""
    out = run("hpclab_metrics", config, jsonl(*records))
    (issue,) = [i for i in out.issues if i.code == "selection_on_split"]
    assert issue.details == details
    assert out.selection == details


@pytest.mark.parametrize(
    ("evaluator", "config", "noise", "noise_code", "records", "status"),
    [
        pytest.param(
            "hpclab_metrics",
            {"split": "val", "select": "best:acc:max"},
            {"schema": "wandb/v1", "loss": 0.1},
            "foreign_schema",
            [rec(split="val", step=s, metrics={"acc": a}) for s, a in [(1, 0.5), (2, 0.9), (3, 0.7)]],
            "valid",
            id="hpclab_metrics-after-foreign-schema-warnings",
        ),
        pytest.param(
            "jsonl_record",
            {"fields": {"val_loss": "val_loss", "test_acc": "test_acc"}, "select": "best:val_loss:min"},
            "{bad",
            "parse_error",
            EARLY_STOP_LOG,
            "partial",
            id="jsonl_record-after-parse-errors",
        ),
    ],
)
def test_selection_flag_survives_the_issue_bound(
    monkeypatch: pytest.MonkeyPatch,
    evaluator: str,
    config: dict[str, Any],
    noise: Any,
    noise_code: str,
    records: list[Any],
    status: str,
) -> None:
    """``selection_on_split`` is the issue results_compare acts on. Dropping it past the
    issue bound (e.g. a metrics log interleaved with another logger's records, each one a
    ``foreign_schema`` warning) would report a best-of-split number as leak-free."""
    monkeypatch.setattr(evaluators, "MAX_ISSUES", 3)
    out = run(evaluator, config, jsonl(*([noise] * 10), *records))
    assert out.status == status
    (selection,) = [i for i in out.issues if i.code == "selection_on_split"]
    assert selection.details is not None
    assert selection.details["select_metric"] == config["select"].split(":")[1]
    assert codes(out) == [noise_code] * 3 + ["selection_on_split", "issues_truncated"]
    assert "7 further" in out.issues[-1].message


def test_issues_without_structured_context_carry_no_details() -> None:
    config = {"fields": {"acc": "acc"}, "step_path": "step", "select": "step:200"}
    out = run("jsonl_record", config, jsonl(*EVAL_LOG))
    assert codes(out) == ["duplicate_step"] and out.issues[0].details is None


# ---------------------------------------------------------------------------
# Early stopping: choose the step on select_split, report split's metrics at that step
# ---------------------------------------------------------------------------

EARLY_STOP_METRICS = [
    rec(step=100, split="train", metrics={"loss": 0.9}),
    rec(step=100, split="test", metrics={"acc": 0.70, "loss": 0.80}),
    rec(step=100, split="val", metrics={"acc": 0.72, "loss": 0.60}),
    rec(step=200, split="test", metrics={"acc": 0.78, "loss": 0.55}),  # logged before its val record
    rec(step=200, split="val", metrics={"acc": 0.79, "loss": 0.40}),
    rec(step=300, split="val", metrics={"acc": 0.81, "loss": 0.45}),
    rec(step=300, split="test", final=True, metrics={"acc": 0.83, "loss": 0.50}),
]
EARLY_STOP_CONFIG = {"split": "test", "select": "best:loss:min", "select_split": "val"}
EARLY_STOP_SELECTION = {
    "select_metric": "loss",
    "direction": "min",
    "select_split": "val",
    "report_split": "test",
    "selected_step": 200,
    "select_value": 0.40,
    "select_locator": "L5#",
    "on_reported_split": False,
}


def test_hpclab_metrics_early_stopping_reports_the_report_split_at_the_selected_step() -> None:
    """Select on val, report test: the step comes from the lowest *val* loss (200), the numbers
    from the *test* record at that step, not from test's own best (300) or final record.

    Selecting on one split and reporting another is the correct protocol, so it must not carry
    ``selection_on_split`` (``results_compare`` refuses every evaluation that does, and
    ``evidence_verify`` then fails the bundle with SELECTION_LEAK). What was selected, on which
    split, and which step is reported is recorded in ``selection`` and in a ``selection_protocol``
    disclosure whose details survive in the stored validation record.
    """
    out = run("hpclab_metrics", EARLY_STOP_CONFIG, jsonl(*EARLY_STOP_METRICS))
    assert out.status == "valid", out.issues
    assert values(out) == {"acc": 0.78, "loss": 0.55}
    assert {(m.split, m.step) for m in out.metrics} == {("test", 200)}
    assert out.metrics[0].source.locator == "L4#/metrics/acc"
    assert out.selected_record is not None and out.selected_record["split"] == "test"
    assert out.selected_record["step"] == 200
    assert out.selection == EARLY_STOP_SELECTION
    assert "selection_on_split" not in codes(out)
    (issue,) = out.issues
    assert (issue.severity, issue.code, issue.locator) == ("warning", "selection_protocol", "L5#")
    assert issue.details == EARLY_STOP_SELECTION
    assert "step 200" in issue.message and "'val'" in issue.message and "'test'" in issue.message
    assert out.config["select_split"] == "val"
    stored = json.loads(json.dumps(out.model_dump(mode="json"), allow_nan=False))
    assert stored["selection"] == EARLY_STOP_SELECTION


def test_hpclab_metrics_early_stopping_uses_the_declared_direction_of_the_select_split() -> None:
    records = [
        rec(step=1, split="val", metrics={"acc": 0.6}, higher_is_better={"acc": True}),
        rec(step=1, split="test", metrics={"acc": 0.5}, higher_is_better={"acc": False}),  # ignored
        rec(step=2, split="val", metrics={"acc": 0.7}, higher_is_better={"acc": True}),
        rec(step=2, split="test", metrics={"acc": 0.4}),
    ]
    config = {"split": "test", "select": "best:acc", "select_split": "val"}
    out = run("hpclab_metrics", config, jsonl(*records))
    assert out.status == "valid" and values(out) == {"acc": 0.4}
    assert out.selection is not None and out.selection["direction"] == "max"
    assert out.selection["selected_step"] == 2


def test_hpclab_metrics_selecting_on_the_reported_split_is_still_a_leak() -> None:
    config = {**EARLY_STOP_CONFIG, "select_split": "test"}
    out = run("hpclab_metrics", config, jsonl(*EARLY_STOP_METRICS))
    assert values(out) == {"acc": 0.83, "loss": 0.50}
    (issue,) = [i for i in out.issues if i.code == "selection_on_split"]
    assert issue.details is not None and issue.details["on_reported_split"] is True
    assert (issue.details["select_split"], issue.details["report_split"]) == ("test", "test")
    assert "selection_protocol" not in codes(out)


def test_hpclab_metrics_select_split_equal_to_the_default_split_says_to_set_split() -> None:
    """``split`` defaults to ``val``: declaring only ``select_split: val`` selects and reports
    val, a leak. The message must point at the setting that is missing (``split``, the split
    to report), not ask for the ``select_split`` that is already declared."""
    out = run(
        "hpclab_metrics", {"select": "best:loss:min", "select_split": "val"}, jsonl(*EARLY_STOP_METRICS)
    )
    (issue,) = [i for i in out.issues if i.code == "selection_on_split"]
    assert "select_split 'val' is also the reported split" in issue.message
    assert "set split to the split you report" in issue.message
    no_select_split = run("hpclab_metrics", {"select": "best:loss:min"}, jsonl(*EARLY_STOP_METRICS))
    (plain,) = [i for i in no_select_split.issues if i.code == "selection_on_split"]
    assert plain.message.endswith("set select_split")


@pytest.mark.parametrize(
    ("records", "code", "message"),
    [
        pytest.param(
            [r for r in EARLY_STOP_METRICS if not (r["split"] == "test" and r["step"] == 200)],
            "record_not_found",
            "no 'test' record at step 200",
            id="no-report-record-at-the-selected-step",
        ),
        pytest.param(
            [r for r in EARLY_STOP_METRICS if r["split"] != "test"],
            "record_not_found",
            "no 'test' record at step 200",
            id="report-split-absent",
        ),
        pytest.param(
            [r for r in EARLY_STOP_METRICS if r["split"] != "val"],
            "split_mismatch",
            "no record for split 'val'",
            id="select-split-absent",
        ),
        pytest.param(
            [rec(split="val", metrics={"loss": 0.1}), rec(split="test", step=1, metrics={"acc": 0.9})],
            "no_step",
            "no integer 'step'",
            id="selected-record-without-step",
        ),
    ],
)
def test_hpclab_metrics_early_stopping_fatal_errors(records: list[Any], code: str, message: str) -> None:
    out = run("hpclab_metrics", EARLY_STOP_CONFIG, jsonl(*records))
    assert out.status == "invalid" and out.metrics == [] and out.selection is None
    assert codes(out, "error")[-1] == code
    assert message in out.issues[-1].message


def test_hpclab_metrics_early_stopping_duplicate_report_step_uses_the_last_record() -> None:
    records = [
        rec(step=1, split="val", metrics={"loss": 0.5}),
        rec(step=1, split="test", metrics={"acc": 0.60}),
        rec(step=1, split="test", metrics={"acc": 0.65}),  # logged again after a resume
    ]
    out = run("hpclab_metrics", EARLY_STOP_CONFIG, jsonl(*records))
    assert out.status == "valid" and values(out) == {"acc": 0.65}
    assert codes(out) == ["duplicate_step", "selection_protocol"]
    assert out.metrics[0].source.locator == "L3#/metrics/acc"


def test_early_stopping_reads_the_input_once_for_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    """The report record is found in a second pass over the input; problems of the input are
    reported once (not once per pass) and the disclosure survives the issue bound."""
    monkeypatch.setattr(evaluators, "MAX_ISSUES", 3)
    noise = [{"schema": "wandb/v1", "loss": 0.1}] * 10
    out = run("hpclab_metrics", EARLY_STOP_CONFIG, jsonl(*noise, "{bad", *EARLY_STOP_METRICS))
    assert out.status == "partial" and values(out) == {"acc": 0.78, "loss": 0.55}
    assert codes(out) == ["foreign_schema"] * 3 + ["selection_protocol", "issues_truncated"]
    assert "8 further" in out.issues[-1].message


def test_jsonl_record_early_stopping_in_one_record_declares_the_selection_split() -> None:
    """One record holds both splits (val_loss next to test_acc): ``select_split`` declares the
    split of the selection field, so the selection is known not to be on the reported split,
    and the selection field itself is labelled with the split it was measured on."""
    config = {
        "fields": {"val_loss": "val_loss", "test_acc": "test_acc"},
        "select": "best:val_loss:min",
        "split": "test",
        "select_split": "val",
        "step_path": "epoch",
    }
    out = run("jsonl_record", config, jsonl(*EARLY_STOP_LOG))
    assert out.status == "valid", out.issues
    assert values(out) == {"test_acc": 0.80, "val_loss": 0.5}
    assert {m.name: (m.split, m.step) for m in out.metrics} == {
        "test_acc": ("test", 2),
        "val_loss": ("val", 2),
    }
    assert out.selection == {
        "select_metric": "val_loss",
        "direction": "min",
        "select_split": "val",
        "report_split": "test",
        "selected_step": 2,
        "select_value": 0.5,
        "select_locator": "L2#",
        "on_reported_split": False,
    }
    assert codes(out) == ["selection_protocol"]
    assert out.issues[0].details == out.selection


SPLIT_LOG = [
    {"step": 1, "phase": "val", "loss": 0.6},
    {"step": 1, "phase": "test", "loss": 0.7, "acc": 0.70},
    {"step": 2, "phase": "test", "loss": 0.5, "acc": 0.78},
    {"step": 2, "phase": "val", "loss": 0.4},
    {"step": 3, "phase": "val", "loss": 0.45},
    {"step": 3, "phase": "test", "loss": 0.45, "acc": 0.83},
]
SPLIT_LOG_CONFIG = {
    "fields": {"acc": "acc", "loss": "loss"},
    "split": "test",
    "split_path": "phase",
    "step_path": "step",
    "select": "best:loss:min",
    "select_split": "val",
}


def test_jsonl_record_early_stopping_joins_split_records_by_step() -> None:
    out = run("jsonl_record", SPLIT_LOG_CONFIG, jsonl(*SPLIT_LOG))
    assert out.status == "valid", out.issues
    assert values(out) == {"acc": 0.78, "loss": 0.5}
    assert {(m.split, m.step) for m in out.metrics} == {("test", 2)}
    assert out.metrics[0].source.locator == "L3#/acc"
    assert out.selection is not None
    assert (out.selection["select_locator"], out.selection["selected_step"]) == ("L4#", 2)
    assert (out.selection["select_split"], out.selection["report_split"]) == ("val", "test")
    assert out.selection["on_reported_split"] is False
    assert codes(out) == ["selection_protocol"]


def test_jsonl_record_early_stopping_join_with_split_specific_field_names() -> None:
    """Records carry their split and name their fields per split (``val_loss`` only on val
    records, ``test_acc`` only on test records). The selection field is not logged for the
    reported split, which is expected, not a missing field: it is taken from the selected
    record and labelled with ``select_split``, so the protocol stays ``valid``."""
    records = [
        {"step": 1, "phase": "val", "val_loss": 0.6},
        {"step": 1, "phase": "test", "seed": 7, "test_acc": 0.70},
        {"step": 2, "phase": "val", "val_loss": 0.4},
        {"step": 2, "phase": "test", "seed": 7, "test_acc": 0.78},
        {"step": 3, "phase": "val", "val_loss": 0.45},
        {"step": 3, "phase": "test", "seed": 7, "test_acc": 0.83},
    ]
    config = {
        **SPLIT_LOG_CONFIG,
        "fields": {"val_loss": "val_loss", "test_acc": "test_acc"},
        "select": "best:val_loss:min",
        "seed_path": "seed",  # logged on the reported (test) records only
    }
    out = run("jsonl_record", config, jsonl(*records))
    assert out.status == "valid", out.issues
    assert values(out) == {"test_acc": 0.78, "val_loss": 0.4}
    assert {m.name: (m.split, m.step, m.seed, m.source.locator) for m in out.metrics} == {
        "test_acc": ("test", 2, 7, "L4#/test_acc"),
        "val_loss": ("val", 2, 7, "L3#/val_loss"),
    }
    assert out.selected_record == records[3]
    assert out.selection is not None and out.selection["select_locator"] == "L3#"
    assert codes(out) == ["selection_protocol"]


def test_jsonl_record_early_stopping_join_reports_the_report_split_value_of_the_selection_field() -> None:
    """When the reported record does log the selection field, its own (report split) value is
    reported: the numbers come from the report split at the selected step."""
    out = run("jsonl_record", SPLIT_LOG_CONFIG, jsonl(*SPLIT_LOG))
    assert {m.name: (m.value, m.split) for m in out.metrics} == {
        "acc": (0.78, "test"),
        "loss": (0.5, "test"),
    }


WIDE_SPLIT_LOG_CONFIG = {
    **SPLIT_LOG_CONFIG,
    "fields": {"val_loss": "val_loss", "test_acc": "test_acc"},
    "select": "best:val_loss:min",
}


def test_jsonl_record_early_stopping_join_treats_a_null_selection_field_as_not_logged() -> None:
    """Wide logs (a CSV logger exported with pandas ``to_json``) carry every column on every
    row and write ``null`` where a split did not log it: the test rows hold ``val_loss: null``.
    That is the selection field not logged for the reported split, like an absent key, so it
    is read from the selected record and the correctly declared protocol stays ``valid``."""
    records = [
        {"step": 1, "phase": "val", "val_loss": 0.6, "test_acc": None},
        {"step": 1, "phase": "test", "val_loss": None, "test_acc": 0.70},
        {"step": 2, "phase": "val", "val_loss": 0.4, "test_acc": None},
        {"step": 2, "phase": "test", "val_loss": None, "test_acc": 0.78},
    ]
    out = run("jsonl_record", WIDE_SPLIT_LOG_CONFIG, jsonl(*records))
    assert out.status == "valid", out.issues
    assert {m.name: (m.value, m.split, m.step, m.source.locator) for m in out.metrics} == {
        "test_acc": (0.78, "test", 2, "L4#/test_acc"),
        "val_loss": (0.4, "val", 2, "L3#/val_loss"),
    }
    assert codes(out) == ["selection_protocol"]


def test_jsonl_record_early_stopping_join_keeps_a_non_finite_report_value() -> None:
    """A non-finite value is a logged result (a diverged loss), not an absent one: the reported
    record's own value is kept, visible as ``non_finite``, never replaced by the selected one."""
    data = (
        b'{"step":1,"phase":"val","val_loss":0.6}\n{"step":1,"phase":"test","val_loss":NaN,"test_acc":0.7}\n'
    )
    out = run("jsonl_record", WIDE_SPLIT_LOG_CONFIG, data)
    assert out.status == "partial" and codes(out, "error") == ["non_finite"]
    assert {m.name: (m.value, m.split) for m in out.metrics} == {
        "test_acc": (0.7, "test"),
        "val_loss": (None, "test"),
    }


def test_jsonl_record_early_stopping_join_needs_the_report_record() -> None:
    records = [r for r in SPLIT_LOG if not (r["phase"] == "test" and r["step"] == 2)]
    out = run("jsonl_record", SPLIT_LOG_CONFIG, jsonl(*records))
    assert out.status == "invalid" and codes(out) == ["record_not_found"]
    assert "no 'test' record at step 2" in out.issues[0].message


def test_jsonl_record_early_stopping_join_reports_parse_errors_once() -> None:
    out = run("jsonl_record", SPLIT_LOG_CONFIG, jsonl("{bad", *SPLIT_LOG))
    assert out.status == "partial" and values(out) == {"acc": 0.78, "loss": 0.5}
    assert codes(out) == ["parse_error", "selection_protocol"]


def test_jsonl_record_selecting_on_the_reported_split_is_a_leak() -> None:
    config = {**SPLIT_LOG_CONFIG, "select_split": "test"}
    out = run("jsonl_record", config, jsonl(*SPLIT_LOG))
    assert values(out) == {"acc": 0.83, "loss": 0.45}
    (issue,) = out.issues
    assert issue.code == "selection_on_split" and issue.details is not None
    assert issue.details["on_reported_split"] is True and issue.details["selected_step"] == 3
    assert "on the split you report is optimistic" in issue.message


@pytest.mark.parametrize(
    ("select_split", "leaking"),
    [
        pytest.param("val", False, id="select-on-val-report-test"),
        pytest.param("test", True, id="select-on-test"),
    ],
)
def test_comparison_refuses_only_selection_on_the_reported_split(
    paths: Paths, select_split: str, leaking: bool
) -> None:
    """End to end: ``results_compare`` refuses every evaluation carrying ``selection_on_split``
    and ``evidence_verify`` then fails the bundle with SELECTION_LEAK. A correctly declared
    early-stopping protocol (select on val, report test) must pass both checks; selecting on
    the reported split must not."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
    from lab_support import Lab

    config = {**EARLY_STOP_CONFIG, "select_split": select_split}
    out = run("hpclab_metrics", config, jsonl(*EARLY_STOP_METRICS))
    assert out.status == "valid"
    clock = FakeClock()
    ctx = ServiceContext.create(paths, clock=clock)
    try:
        lab = Lab(ctx, clock)
        exp = lab.experiment(comparison={})
        plan = lab.plan(exp)
        rids = [lab.run(exp, plan, method="erm", seed=seed) for seed in (1, 2)]
        for rid in rids:
            lab.valid(
                rid,
                metrics={m.name: m.value for m in out.metrics},
                split="test",
                issues=[i.model_dump(mode="json") for i in out.issues],
            )
        contract = {
            "selection": {
                "rule": "best_by",
                "select_metric": "loss",
                "select_split": "val",
                "mode": "min",
                "report_split": "test",
            }
        }
        data = results_compare(
            ctx,
            planned_run_set_ref=exp,
            metric_names=["acc"],
            comparison_contract=contract,  # type: ignore[arg-type]
        )
        bundle = evidence_export(
            ctx, purpose="Table 1", idempotency_key="k", comparison_ref=data.comparison_ref
        )
        report = evidence_verify(ctx, bundle_ref=bundle.bundle_ref)
    finally:
        ctx.db.close_all()
    assert data.integrity.selection.evaluation_selection_on_split == (rids if leaking else [])
    assert any(i.reason.startswith("selection_leak") for i in data.incomparable) is leaking
    assert data.incomparable == [] or leaking
    leak_findings = [f for f in report.findings if f.code == "SELECTION_LEAK"]
    assert [f.severity for f in leak_findings] == (["error"] if leaking else [])
    assert report.ok is not leaking, report.findings


@pytest.mark.parametrize(
    ("evaluator", "config", "normalized"),
    [
        pytest.param(
            "hpclab_metrics",
            {"select": "best:acc:max"},
            {"split": "val", "select": "best:acc:max", "metrics": None, "require": []},
            id="hpclab_metrics",
        ),
        pytest.param(
            "jsonl_record",
            {"fields": {"a": "x"}, "select": "best:a:max"},
            {
                "fields": {"a": "x"},
                "split": None,
                "split_path": None,
                "step_path": None,
                "seed_path": None,
                "dataset": None,
                "dataset_path": None,
                "units": {},
                "higher_is_better": {},
                "select": "best:a:max",
            },
            id="jsonl_record",
        ),
    ],
)
def test_configs_without_select_split_keep_their_normalized_form(
    evaluator: str, config: dict[str, Any], normalized: dict[str, Any]
) -> None:
    """The normalized config is hashed into the evaluation identity: configs written before
    ``select_split`` existed must normalize (and hash) exactly as before."""
    assert validate_config(evaluator, "1", config) == normalized
    assert validate_config(evaluator, "1", {**config, "select_split": None}) == normalized
    declared = validate_config(evaluator, "1", {**config, "split": "t", "select_split": "v"})
    assert declared == {**normalized, "split": "t", "select_split": "v"}
    (info,) = [i for i in list_evaluators() if i.id == evaluator]
    assert "select_split" in info.config_schema["properties"]


# ---------------------------------------------------------------------------
# Paths, configs, registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "tokens"),
    [
        ("a", ("a",)),
        ("a.b[0].c", ("a", "b", 0, "c")),
        ("h[-1]", ("h", -1)),
        ('m["val/acc"]', ("m", "val/acc")),
        ("m['a.b']", ("m", "a.b")),
        ('m["q\\"x"]', ("m", 'q"x')),
        ("[0][1]", (0, 1)),
        ('["a b"].c', ("a b", "c")),
        ("a=b", ("a=b",)),
    ],
)
def test_parse_path_valid(expr: str, tokens: tuple[Any, ...]) -> None:
    assert parse_path(expr) == tokens


@pytest.mark.parametrize(
    "expr", ["", ".a", "a.", "a..b", "a[", "a[x]", "a[0", "a]b", 'a["x"', "a[0]b", "a.[0]", "[]"]
)
def test_parse_path_invalid(expr: str) -> None:
    with pytest.raises(ValueError):
        parse_path(expr)


@pytest.mark.parametrize(
    ("evaluator", "config"),
    [
        ("hpclab_metrics", {"select": "best"}),
        ("hpclab_metrics", {"select": "median"}),
        ("hpclab_metrics", {"metrics": []}),
        ("hpclab_metrics", {"unknown": 1}),
        ("json_path", {}),
        ("json_path", {"fields": {}}),
        ("json_path", {"fields": {"a": "x..y"}}),
        ("json_path", {"fields": {"a": "x"}, "units": {"b": "s"}}),
        ("json_path", {"fields": {"a": "x"}, "step_path": "["}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "step:3"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "step:x", "step_path": "s"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "best:a"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "best:b:max"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "where:=1"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "where:k"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "first"}),
        ("hpclab_metrics", {"select_split": "train"}),
        ("hpclab_metrics", {"select": "last", "select_split": "train"}),
        ("hpclab_metrics", {"split": None, "select": "best:acc:max", "select_split": "val"}),
        ("hpclab_metrics", {"split": "test", "select": "best:acc:max", "select_split": ""}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "last", "select_split": "val"}),
        ("jsonl_record", {"fields": {"a": "x"}, "select": "best:a:max", "select_split": "val"}),
        (
            "jsonl_record",
            {
                "fields": {"a": "x"},
                "select": "best:a:max",
                "select_split": "val",
                "split": "t",
                "split_path": "p",
            },
        ),
    ],
)
def test_invalid_configs_raise(evaluator: str, config: dict[str, Any]) -> None:
    with pytest.raises(InvalidArgument) as err:
        run(evaluator, config, b"{}")
    assert err.value.details["errors"]


def test_validate_config_normalizes_defaults() -> None:
    assert validate_config("hpclab_metrics", "1", {}) == validate_config(
        "hpclab_metrics", "1", {"split": "val"}
    )
    assert validate_config("hpclab_metrics", "1", None)["select"] == "final"


def test_unknown_evaluator_lists_available() -> None:
    with pytest.raises(NotFound) as err:
        run_evaluator("hpclab_metrics", "9", {}, [])
    assert "hpclab_metrics@1" in err.value.details["available"]


@pytest.mark.parametrize("count", [0, 2])
def test_wrong_input_count_is_a_request_error(count: int) -> None:
    inputs = [EvalInput(name=f"m{i}.jsonl", data=jsonl(TRAIN_LOG[4])) for i in range(count)]
    with pytest.raises(InvalidArgument):
        run_evaluator("hpclab_metrics", "1", {}, inputs)


def test_list_evaluators_is_sorted_and_self_describing() -> None:
    infos = list_evaluators()
    assert [(i.id, i.version) for i in infos] == [
        ("hpclab_metrics", "1"),
        ("json_path", "1"),
        ("jsonl_record", "1"),
    ]
    assert all(i.description and i.config_schema.get("properties") for i in infos)
    assert all(isinstance(e, Evaluator) for e in EVALUATORS.values())


def test_register_evaluator_refuses_silent_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "EVALUATORS", dict(EVALUATORS))
    existing = get_evaluator("json_path", "1")
    with pytest.raises(InvalidArgument):
        register_evaluator(existing)
    register_evaluator(existing, replace=True)
    assert evaluators.EVALUATORS[("json_path", "1")] is existing


# ---------------------------------------------------------------------------
# Review regressions: unknown must never become valid; data problems never raise
# ---------------------------------------------------------------------------


def test_fallback_without_any_final_evidence_is_not_a_valid_result() -> None:
    # A run killed by TIMEOUT mid-training leaves intermediate val records and no final one.
    # Reporting its last val number as a *valid* result would put a mid-training value into a
    # paper table, so the fallback is shown for inspection but is an error (not countable).
    out = run("hpclab_metrics", {}, jsonl(*TRAIN_LOG[:4]))
    assert out.status == "partial"
    assert values(out) == {"acc": 0.86, "loss": 0.35}
    assert codes(out, "error") == ["no_final_record"]
    assert "select='last'" in out.issues[0].message


def test_fallback_short_of_total_steps_is_not_a_valid_result() -> None:
    data = jsonl(rec(step=400, total_steps=1000, split="val", metrics={"acc": 0.7}))
    out = run("hpclab_metrics", {}, data)
    assert out.status == "partial" and codes(out, "error") == ["no_final_record"]
    assert "step 400 of 1000" in out.issues[0].message


def test_single_json_summary_without_final_marker_is_not_valid_either() -> None:
    # A summary file overwritten every epoch looks exactly like a final one.
    data = json.dumps(rec(split="val", metrics={"acc": 0.9})).encode()
    assert run("hpclab_metrics", {}, data, name="summary.json").status == "partial"
    assert run("hpclab_metrics", {"select": "last"}, data, name="summary.json").status == "valid"


def test_explicit_last_is_the_opt_in_for_loggers_without_final_markers() -> None:
    out = run("hpclab_metrics", {"select": "last"}, jsonl(*TRAIN_LOG[:4]))
    assert out.status == "valid" and out.issues == []
    assert values(out) == {"acc": 0.86, "loss": 0.35}


def _deep(depth: int) -> str:
    return "[" * depth + "]" * depth


@pytest.mark.parametrize("depth", [1_200, 3_000])
@pytest.mark.parametrize(
    ("evaluator", "config", "name"),
    [
        ("hpclab_metrics", {}, "m.jsonl"),
        ("hpclab_metrics", {}, "m.json"),
        ("json_path", {"fields": {"acc": "metrics.acc"}}, "m.json"),
        ("jsonl_record", {"fields": {"acc": "metrics.acc"}}, "m.jsonl"),
    ],
)
def test_deeply_nested_record_never_raises(
    depth: int, evaluator: str, config: dict[str, Any], name: str
) -> None:
    # Deeper than Python's recursion limit but within the C JSON parser's: parsing succeeds, so
    # anything that walks the record recursively afterwards must not blow the stack.
    line = '{"split":"val","final":true,"metrics":{"acc":0.9},"trace":' + _deep(depth) + "}\n"
    out = run(evaluator, config, line.encode(), name=name)
    if out.status == "invalid":  # a Python whose JSON parser refuses this depth: a parse error
        assert "parse_error" in codes(out, "error")
        return
    assert out.status == "valid" and values(out) == {"acc": 0.9}
    assert codes(out, "warning") == ["selected_record_truncated"]
    assert out.selected_record is not None and out.selected_record["_truncated"] is True
    json.dumps(out.model_dump(mode="json"), allow_nan=False)  # storable as strict JSON


def test_deeply_nested_split_value_never_raises() -> None:
    line = '{"split":' + _deep(3_000) + ',"final":true,"metrics":{"acc":0.9}}\n'
    out = run("hpclab_metrics", {}, line.encode())
    assert out.status == "invalid"
    assert codes(out, "error")[-1] in ("split_mismatch", "parse_error")


@pytest.mark.parametrize("config", [["split", "val"], "split=val", 3])
def test_non_mapping_config_is_a_request_error(config: Any) -> None:
    with pytest.raises(InvalidArgument):
        run_evaluator("hpclab_metrics", "1", config, [EvalInput(name="m.jsonl", data=b"{}")])
    with pytest.raises(InvalidArgument):
        validate_config("hpclab_metrics", "1", config)


def test_digest_spelling_is_normalized_not_reported_as_corruption() -> None:
    data = jsonl(TRAIN_LOG[4])
    hexd = hashlib.sha256(data).hexdigest()
    for spelling in (hexd, hexd.upper(), "SHA256:" + hexd.upper(), " sha256:" + hexd + " "):
        out = run("hpclab_metrics", {}, data, digest=spelling)
        assert out.status == "valid", spelling
        assert out.metrics[0].source.digest == "sha256:" + hexd


@pytest.mark.parametrize(
    "bad", ["md5:" + "0" * 32, "sha256:xyz", "", "sha256:" + "0" * 63, "sha1:" + "0" * 64]
)
def test_malformed_digest_is_a_caller_bug_not_a_data_mismatch(bad: str) -> None:
    with pytest.raises(ValueError):
        EvalInput(name="m.jsonl", data=b"{}", digest=bad)


def test_all_metrics_read_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluators, "MAX_METRICS", 3)
    record = rec(split="val", final=True, metrics={f"m{i}": float(i) for i in range(5)})
    out = run("hpclab_metrics", {}, jsonl(record))
    assert out.status == "partial"
    assert [m.name for m in out.metrics] == ["m0", "m1", "m2"]
    assert codes(out, "error") == ["too_many_metrics"]
    assert "metrics filter" in out.issues[0].message
    # an explicit filter is the way out, and is not capped by the "all metrics" bound
    out = run("hpclab_metrics", {"metrics": ["m4", "m3", "m1", "m0"]}, jsonl(record))
    assert out.status == "valid" and len(out.metrics) == 4
