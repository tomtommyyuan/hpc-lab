"""Design scenario 3: methods x seeds -> paper table with a missing seed -> recovery -> stale TeX.

One cell of the 2x3 matrix dies with its node (sim fault ``node_fail_next``). The comparison
must keep the planned denominator (n=2/3, the failed seed named with its reason), the LaTeX
table must footnote the incomplete cell, a batch recovery plan must fill exactly the missing
cell, and after the table is complete ``evidence_verify`` must flag the numbers of the OLD
table (still in the manuscript) as stale.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from e2e_support import Lab, grid_codes

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]

DAGGER = r"\textsuperscript{\dag}"


def _group(cmp: dict[str, Any], label: str) -> dict[str, Any]:
    return next(g for g in cmp["groups"] if g["label"] == label)


def test_missing_seed_table_recovery_and_stale_tex(lab: Lab) -> None:
    prep = lab.prepare(lab.spec(smoke=None))
    exp = prep["experiment_id"]
    lab.sim.set_faults(node_fail_next=1)
    wf = lab.call(
        "workflow_submit",
        idempotency_key=lab.key("wf"),
        plan_id=prep["plan_id"],
        plan_hash=prep["plan_hash"],
        max_parallel=2,
    )
    assert wf["planned_run_count"] == 6 and wf["gate"] is None
    done = lab.wait_workflow(wf["workflow_id"], timeout=75)
    assert done["counts"].get("failed") == 1 and done["counts"].get("succeeded") == 5, done["counts"]

    failed = [a for a in lab.attempts() if a["scheduler_state"] == "NODE_FAIL"]
    assert len(failed) == 1, lab.snapshot()
    bad = failed[0]
    bad_run = lab.call("run_inspect", run_id=bad["run_id"])
    assert bad_run["diagnosis"]["primary_code"] == "node_fail", bad_run["diagnosis"]
    method, seed = bad_run["run"]["cell"]["method"], bad_run["run"]["seed"]
    # evaluations: collected automatically when outputs validate; an explicit batch collect is
    # idempotent (a user or agent runs it before comparing) and may hand the rest to the background
    col = lab.call("evaluation_collect", idempotency_key=lab.key("col"), run_set_ref=wf["workflow_id"])
    reasons = Counter(s["reason"] for s in col["skipped"])
    assert len(col["collected"]) + reasons["already_evaluated"] + col["pending"] == 5, col
    assert reasons["scheduler_state_not_completed"] == 1, col  # the NODE_FAIL cell has no result to evaluate

    def five_valid() -> dict[str, dict[str, str]] | None:
        codes = grid_codes(lab.grid(workflow_id=wf["workflow_id"], metric="acc"))
        return codes if sum(c == "V" for row in codes.values() for c in row.values()) == 5 else None

    codes = lab.wait(five_valid, timeout=30, what="5 valid cells")
    assert codes[f"method={method}"][f"seed={seed}"] == "F", codes

    # the planned denominator: n=2 of 3, the failed seed named with its reason
    cmp1 = lab.call(
        "results_compare", idempotency_key=lab.key("cmp"), planned_run_set_ref=exp, metric_names=["acc"]
    )
    stats = _group(cmp1, method)["metrics"]["acc"]
    assert (stats["n"], stats["expected_n"]) == (2, 3), stats
    assert stats["failed_seeds"] == [seed] and str(seed) not in stats["values_by_seed"], stats
    cov = next(r for r in cmp1["runs"] if r["run_id"] == bad["run_id"])
    assert cov["status"] == "failed" and "node_fail" in (cov["reason"] or ""), cov
    assert cmp1["counts"]["included"] == 5 and cmp1["counts"]["failed"] == 1

    ev1_env = lab.ok(
        "evidence_export",
        purpose="Table 1 draft",
        idempotency_key=lab.key("ev"),
        comparison_ref=cmp1["comparison_ref"],
    )
    ev1 = ev1_env["data"]
    old_tex = lab.call("evidence_get", bundle_id=ev1["bundle_id"], file="table.tex")["content"]
    old_row = next(line for line in old_tex.splitlines() if line.startswith(f"{method} &"))
    assert DAGGER in old_row, old_row
    assert "\\dag" in old_tex.split("\\bottomrule", 1)[1], "incomplete cell is not footnoted"
    assert ev1_env["coverage"]["complete"] is False and ev1["counts"]["failed"] == 1
    assert ev1_env["warnings"], "an incomplete table is exported with a warning"

    # batch recovery of the missing cell, dispatched as a workflow of its own
    rec = lab.call(
        "run_recover",
        idempotency_key=lab.key("rec"),
        selector={"workflow_id": wf["workflow_id"], "cells": "missing_valid"},
    )
    assert rec["mode"] == "batch" and rec["plan_id"], rec
    eligible = [c for c in rec["cells"] if c["eligible"]]
    assert [c["run_id"] for c in eligible] == [bad["run_id"]] and eligible[0]["strategy"] == "retry", rec[
        "cells"
    ]
    wf2 = lab.call("workflow_submit", idempotency_key=lab.key("wf"), plan_id=rec["plan_id"])
    lab.wait_workflow(wf2["workflow_id"], timeout=45)
    retried = lab.wait(
        lambda: [e for e in lab.evaluated(bad["run_id"]) if e["attempt_id"] != bad["attempt_id"]],
        timeout=30,
        what="evaluation of the retried cell",
    )
    assert retried[0]["selected"]

    cmp2_env = lab.ok(
        "results_compare", idempotency_key=lab.key("cmp"), planned_run_set_ref=exp, metric_names=["acc"]
    )
    cmp2 = cmp2_env["data"]
    for g in cmp2["groups"]:
        assert (g["metrics"]["acc"]["n"], g["metrics"]["acc"]["expected_n"]) == (3, 3), g
    assert cmp2["counts"]["included"] == 6 and cmp2_env["coverage"]["complete"] is True
    ev2 = lab.call(
        "evidence_export",
        purpose="Table 1",
        idempotency_key=lab.key("ev"),
        comparison_ref=cmp2["comparison_ref"],
    )
    new_tex = lab.call("evidence_get", bundle_id=ev2["bundle_id"], file="table.tex")["content"]
    assert DAGGER not in new_tex

    # the manuscript still carries the OLD table: its numbers are stale now
    paper = lab.root / "paper" / "main.tex"
    paper.parent.mkdir()
    paper.write_text("\\section{Results}\n" + old_tex, encoding="utf-8")
    ver = lab.call("evidence_verify", bundle_ref=ev1["bundle_ref"], tex_paths=[str(paper)])
    stale = [f for f in ver["findings"] if f["code"] == "STALE_TEX_NUMBER"]
    assert stale and ver["ok"] is False, ver["findings"]
    assert any(
        f"{ev1['bundle_id']}:{method}.acc.mean" in " ".join(f["refs"]) + f["message"] for f in stale
    ), stale
    assert ver["newer_comparison_ref"] == cmp2["comparison_ref"], ver
    # the new table verifies clean
    ok = lab.call("evidence_verify", bundle_ref=ev2["bundle_ref"], tex_content=new_tex)
    assert not [f for f in ok["findings"] if f["severity"] == "error"], ok["findings"]
    # ... and a PI who finds a hand-"improved" number in the student's draft sees which one
    tag = f"\\hpcval{{{ev2['bundle_id']}:{method}.acc.mean}}{{"
    value = new_tex.split(tag, 1)[1].split("}", 1)[0]
    edited = new_tex.replace(tag + value + "}", tag + "0.999}")
    bad_ver = lab.call("evidence_verify", bundle_ref=ev2["bundle_ref"], tex_content=edited)
    mismatch = [f for f in bad_ver["findings"] if f["code"] == "VALUE_MISMATCH"]
    assert bad_ver["ok"] is False and mismatch, bad_ver["findings"]
    assert f"{method}.acc.mean" in " ".join(mismatch[0]["refs"]) + mismatch[0]["message"], mismatch

    # ground truth: 6 cells + 1 retry, nothing submitted twice
    assert len(lab.sim.jobs()) == 7 and not lab.duplicate_jobs()
