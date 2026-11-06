"""End to end through MCP, exactly the tool calls an agent makes, against a real controller
subprocess and the Slurm simulator:

1. "run the tiny-classifier ablation, 2 methods x 3 seeds, smoke first" (+ "give me the paper
   table and check my main.tex" in a fresh session afterwards);
2. "one seed failed - what happened and fix it" (an injected host OOM), including a stale
   manuscript number after the fix;
3. stopping work: cancel a running attempt, cancel a sweep, plan the recovery of the cells.

Every call's structuredContent is validated against the tool's outputSchema and every text
block is kept under the agent budget (see ``mcp_support.Agent``).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import anyio
import pytest
import yaml
from mcp.shared.exceptions import MCPError
from mcp_support import (
    ALL_TOOLS,
    CALLED_TOOLS,
    EXAMPLE_SPEC,
    REPO,
    Agent,
    Call,
    Live,
    agent_session,
    grid_counts,
    grid_settled,
)

from mcp import Client, StdioServerParameters

pytestmark = [pytest.mark.anyio, pytest.mark.sim]
SCENARIOS_DONE: set[str] = set()


async def _with_current_version(agent: Agent, tool: str, args: dict[str, Any], version: int) -> Call:
    """Mutations of live objects take expected_version; while jobs run it moves. Retry on
    version_conflict with details.current_version - exactly what the error text tells an agent."""
    for _ in range(5):
        call = await agent.call(tool, {**args, "expected_version": version}, expect_error=None)
        if call.ok:
            return call
        assert call.env["error"]["code"] == "version_conflict", call.text
        assert f"expected_version={call.env['error']['details']['current_version']}" in call.text
        version = call.env["error"]["details"]["current_version"]
    raise AssertionError(f"{tool}: version kept moving")


async def _prepare(agent: Agent, name: str, **extra: Any) -> Call:
    prep = await agent.call(
        "experiment_prepare",
        {
            "spec_path": str(EXAMPLE_SPEC.relative_to(EXAMPLE_SPEC.parents[2])),
            "experiment": name,
            "idempotency_key": f"prep-{name}",
            **extra,
        },
    )
    d = prep.data
    assert d["status"] == "ready" and d["plan_hash"], prep.text
    # the text keeps what the next call needs and says what to do next
    assert d["plan_id"] in prep.text and d["plan_hash"] in prep.text and d["experiment_id"] in prep.text
    assert f'workflow_submit(plan_id="{d["plan_id"]}", plan_hash="{d["plan_hash"]}"' in prep.next_steps
    assert "confirm" in prep.next_steps  # GPU spend is the user's decision
    for r in d["runs"]:
        assert r["run_id"] in prep.text
    return prep


async def _settled_grid(agent: Agent, experiment_id: str) -> Call:
    """Wait until no main cell is queued/running. Under load a finished run can briefly show
    ``E`` (completed, evaluation pending); the text must say it is not a failure and point at
    evaluation_collect - follow that hint, as an agent would."""
    args = {"experiment_id": experiment_id, "view": "grid"}
    grid = await agent.wait("runs_list", args, grid_settled)
    cells = [c for r in grid.data["grid"]["rows"] for c in r["cells"]]
    pending = [c["run_id"] for c in cells if c["code"] == "E"]
    if pending:
        assert "evaluation pending (not a failure)" in grid.text, grid.text
        assert "evaluation_collect(" in grid.next_steps, grid.text
        failed_hint = next((ln for ln in grid.next_steps.splitlines() if "why it failed" in ln), "")
        assert not any(rid in failed_hint for rid in pending), grid.text
        await agent.call(
            "evaluation_collect", {"run_set_ref": experiment_id, "idempotency_key": f"c-{grid.size}"}
        )
        grid = await agent.wait("runs_list", args, lambda c: grid_settled(c) and not grid_counts(c).get("E"))
    return grid


async def _wait_for_new_job(live: Live, before: int, timeout: float = 30.0) -> None:
    """Until Slurm (the simulator) accepted one more job than ``before``."""
    with anyio.fail_after(timeout):
        while len(live.sim.jobs()) <= before:
            await anyio.sleep(0.05)


async def test_ablation_with_smoke_gate_then_paper_table_in_a_new_session(live: Live) -> None:
    async with agent_session(live) as agent:
        assert set(agent.tools) == ALL_TOOLS
        insp = await agent.call("cluster_inspect", {"cluster_id": "sim"})
        assert insp.data["status"] in ("ok", "warn")
        assert re.search(r"^- auth: ok", insp.text, re.M) and "experiment_prepare(" in insp.next_steps

        prep = await _prepare(agent, "ablation")
        d = prep.data
        script, mime = await agent.resource(f"hpclab://plans/{d['plan_id']}/script")
        assert mime == "text/x-shellscript" and "#SBATCH --gres=gpu:a100:1" in script and "train.py" in script
        assert len(script.encode()) < 16 * 1024  # the reviewable part, not the static wrapper
        full, _ = await agent.resource(f"hpclab://plans/{d['plan_id']}/script/full")
        assert len(full) > len(script)

        wf = await agent.call(
            "workflow_submit",
            {"plan_id": d["plan_id"], "plan_hash": d["plan_hash"], "idempotency_key": "dispatch-1"},
        )
        wf_id = wf.data["workflow_id"]
        assert wf.data["gate"]["run_id"] == d["smoke_run_id"] and "smoke gate" in wf.text
        assert f'runs_list(experiment_id="{d["experiment_id"]}", view="grid")' in wf.next_steps
        # the same key is a safe retry; a new key for the same plan points at the existing workflow
        again = await agent.call(
            "workflow_submit",
            {"plan_id": d["plan_id"], "plan_hash": d["plan_hash"], "idempotency_key": "dispatch-1"},
        )
        assert again.data["workflow_id"] == wf_id
        dup = await agent.call(
            "workflow_submit", {"plan_id": d["plan_id"], "idempotency_key": "dispatch-2"}, expect_error=True
        )
        assert wf_id in dup.text and "runs_list(workflow_id=" in dup.text

        grid = await _settled_grid(agent, d["experiment_id"])
        assert grid_counts(grid).get("V") == 6, grid.text
        assert re.search(r"^baseline\s+V\s+V\s+V$", grid.text, re.M), grid.text
        assert "results_compare(" in grid.next_steps
        assert grid.size < 2048

        lst = await agent.call("runs_list", {"experiment_id": d["experiment_id"]})
        smoke_row = next(r for r in lst.data["runs"] if r["kind"] == "smoke")
        assert smoke_row["code"] == "V" and smoke_row["run_id"] in lst.text
        run = next(r for r in lst.data["runs"] if r["kind"] == "main")
        att = run["latest_attempt"]["attempt_id"]
        assert att in lst.text

        ins = await agent.call("run_inspect", {"run_id": run["run_id"]})
        assert "results valid" in ins.text and att in ins.text
        logs = await agent.call("run_logs", {"attempt_id": att, "tail_lines": 3})
        assert "done: method=" in logs.text
        arts = await agent.call("artifacts_list", {"attempt_id": att})
        metrics = next(a for a in arts.data["artifacts"] if a["name"] == "metrics")
        assert metrics["artifact_id"] in arts.text and "artifact_read(" in arts.next_steps
        read = await agent.call("artifact_read", {"artifact_ref": metrics["artifact_id"]})
        assert '"final":true' in read.text.replace(" ", "")
        art_json, _ = await agent.resource(f"hpclab://artifacts/{metrics['artifact_id']}")
        assert json.loads(art_json)["data"]["integrity"] == "finalized"

        coll = await agent.call(
            "evaluation_collect", {"run_set_ref": d["experiment_id"], "idempotency_key": "collect-1"}
        )
        assert "results_compare(" in coll.next_steps
        cmp_ = await agent.call(
            "results_compare",
            {
                "planned_run_set_ref": d["experiment_id"],
                "metric_names": ["acc", "loss"],
                "expected_seeds": [1, 2, 3],
            },
        )
        assert cmp_.data["counts"]["included"] == 6
        assert "mixup − baseline" in cmp_.text and "descriptive" in cmp_.text
        assert f'evidence_export(comparison_ref="{cmp_.data["comparison_id"]}"' in cmp_.next_steps
        comp_json, _ = await agent.resource(f"hpclab://comparisons/{cmp_.data['comparison_id']}")
        assert json.loads(comp_json)["ok"] is True
        exp_json, _ = await agent.resource(f"hpclab://experiments/{d['experiment_id']}")
        assert json.loads(exp_json)["data"]["experiment_id"] == d["experiment_id"]
        wf_json, _ = await agent.resource(f"hpclab://workflows/{wf_id}")
        assert json.loads(wf_json)["data"]["state"] == "completed"
        wgrid, _ = await agent.resource(f"hpclab://workflows/{wf_id}/grid")
        assert json.loads(wgrid)["data"]["view"] == "grid"
        run_json, _ = await agent.resource(f"hpclab://runs/{run['run_id']}")
        evs = json.loads(run_json)["data"]["evaluations"]
        ev_json, _ = await agent.resource(f"hpclab://evaluations/{evs[0]['evaluation_id']}")
        assert json.loads(ev_json)["data"]["status"] == "valid"
        op_json, _ = await agent.resource(f"hpclab://operations/{wf.data['operation_id']}")
        assert json.loads(op_json)["data"]["kind"]

        usage = await agent.call("usage_report", {"cluster_id": "sim"})
        assert "visibility" in usage.text and "GPU-h" in usage.text

    # ---- a new chat: "give me the paper table and check my main.tex"
    paper = live.root / "paper"
    paper.mkdir()
    async with agent_session(live, cwd=paper) as agent:
        listed = await agent.client.list_resources()
        uris = [str(r.uri) for r in listed.resources]
        assert "hpclab://experiments" in uris and f"hpclab://experiments/{d['experiment_id']}" in uris
        index, _ = await agent.resource("hpclab://experiments")
        exp = next(e for e in json.loads(index)["data"]["experiments"] if e["name"] == "ablation")
        cmp2 = await agent.call(
            "results_compare", {"planned_run_set_ref": exp["experiment_id"], "metric_names": ["acc"]}
        )
        ev = await agent.call(
            "evidence_export",
            {
                "comparison_ref": cmp2.data["comparison_id"],
                "purpose": "Table 1",
                "idempotency_key": "export-t1",
            },
        )
        bid = ev.data["bundle_id"]
        assert f"hpclab://evidence/{bid}/table.tex" in ev.text and "evidence_verify(" in ev.next_steps
        assert "| baseline" in ev.text  # the results table itself, not just file names
        tex, mime = await agent.resource(f"hpclab://evidence/{bid}/table.tex")
        assert mime == "application/x-tex" and "\\hpcval{" in tex
        csv, mime = await agent.resource(f"hpclab://evidence/{bid}/table.csv")
        assert mime == "text/csv" and csv.strip()
        meta, _ = await agent.resource(f"hpclab://evidence/{bid}")
        assert json.loads(meta)["data"]["bundle_id"] == bid
        cell = re.search(r"\\hpcval\{(evb_\w+:[^}<]+)\}", tex).group(1)  # type: ignore[union-attr]
        trace, _ = await agent.resource(f"hpclab://trace/{cell}")
        chain = json.loads(trace)["data"]
        assert chain["complete"] is True and len(chain["seeds"]) == 3

        # the student pastes the table and "fixes" one number by hand
        edited = re.sub(
            r"(\\hpcval\{[^}]+\.acc\.mean\})\{0\.(\d)",
            lambda m: f"{m.group(1)}{{0.{(int(m.group(2)) + 1) % 10}",
            tex,
            count=1,
        )
        assert edited != tex
        (paper / "main.tex").write_text(
            "\\documentclass{article}\n\\begin{document}\n" + edited + "\n\\end{document}\n"
        )
        ok = await agent.call("evidence_verify", {"bundle_ref": bid, "tex_content": tex})
        assert ok.data["ok"] is True and "OK" in ok.text.splitlines()[0]
        bad = await agent.call(
            "evidence_verify", {"bundle_ref": bid, "tex_paths": ["main.tex"]}
        )  # relative to cwd
        assert bad.data["ok"] is False and "VALUE_MISMATCH" in bad.text and "main.tex:" in bad.text
        assert "restore hand-edited numbers" in bad.next_steps

        # "check my main.tex" where the table is pulled in with \input: the included file is
        # scanned (and listed); one that cannot be read is called out, never silently skipped
        (paper / "table.tex").write_text(tex)
        (paper / "wrapper.tex").write_text("Mixup reaches 84.2\\%.\n\\input{table}\n")
        wrapped = await agent.call("evidence_verify", {"bundle_ref": bid, "tex_paths": ["wrapper.tex"]})
        assert wrapped.data["checked"] is True and wrapped.data["ok"] is True, wrapped.text
        assert "files scanned (2)" in wrapped.text and str(paper / "table.tex") in wrapped.text
        (paper / "broken.tex").write_text("\\input{table}\n\\input{tables/missing}\n")
        broken = await agent.call("evidence_verify", {"bundle_ref": bid, "tex_paths": ["broken.tex"]})
        assert "UNRESOLVED_INPUT" in broken.text and "UNRESOLVED_INPUT" in broken.next_steps, broken.text
        assert ": OK —" not in broken.text.splitlines()[0], broken.text
        # no tag anywhere: nothing in the manuscript was checked, so the answer must not be "OK"
        (paper / "prose.tex").write_text("Mixup reaches 84.2\\%.\n")
        prose = await agent.call("evidence_verify", {"bundle_ref": bid, "tex_paths": ["prose.tex"]})
        first = prose.text.splitlines()[0]
        assert ": OK —" not in first and "NO_TAGS_FOUND" in prose.text, prose.text
        assert "manuscript NOT checked" in prose.text and "\\input" in prose.next_steps
        typo = await agent.call("evidence_verify", {"bundle_ref": bid, "tex_paths": ["mian.tex"]})
        first = typo.text.splitlines()[0]
        assert ": OK —" not in first and "INCOMPLETE" in typo.text and "mian.tex" in typo.text, typo.text
        assert "tex_paths" in typo.next_steps
        assert "no \\hpcval-tagged number was found" not in typo.next_steps  # it was never read

        # a metric name nobody reports is called out instead of becoming an all-missing table
        wrong = await agent.call(
            "results_compare",
            {"planned_run_set_ref": exp["experiment_id"], "metric_names": ["accuracy"]},
            expect_error=True,
        )
        assert "accuracy" in wrong.text and "evidence_export(" not in wrong.text
        assert 'metric_names=["acc", "loss"]' in wrong.text, wrong.text
    SCENARIOS_DONE.add("ablation")


async def test_failed_seed_is_diagnosed_recovered_and_stale_paper_numbers_are_caught(live: Live) -> None:
    async with agent_session(live) as agent:
        prep = await _prepare(agent, "oom")
        d = prep.data
        jobs_before = len(live.sim.jobs())
        await agent.call(
            "workflow_submit",
            {"plan_id": d["plan_id"], "plan_hash": d["plan_hash"], "idempotency_key": "dispatch-oom"},
        )
        await _wait_for_new_job(
            live, jobs_before
        )  # the smoke job is in Slurm; the next sbatch is a main cell
        live.sim.set_faults(oom_next=1)
        try:
            grid = await _settled_grid(agent, d["experiment_id"])
        finally:
            live.sim.clear_faults()
        assert grid_counts(grid) == {**grid_counts(grid), "V": 5, "F": 1}, grid.text
        failed = next(c for r in grid.data["grid"]["rows"] for c in r["cells"] if c["code"] == "F")
        assert "needs attention" in grid.text and failed["run_id"] in grid.text and "host_oom" in grid.text
        assert f'run_inspect(run_id="{failed["run_id"]}")' in grid.next_steps

        # what happened?
        ins = await agent.call("run_inspect", {"run_id": failed["run_id"]})
        assert "host_oom" in ins.text.splitlines()[0] and "evidence:" in ins.text and "mem=6G" in ins.text
        attempt = ins.data["attempts"][0]
        assert "run_recover(" in ins.next_steps and attempt["attempt_id"] in ins.next_steps
        logs = await agent.call(
            "run_logs", {"attempt_id": attempt["attempt_id"], "stream": "stderr", "tail_lines": 30}
        )
        assert "oom-kill" in logs.text

        # the gap stays visible before the fix
        gap = await agent.call(
            "results_compare", {"planned_run_set_ref": d["experiment_id"], "metric_names": ["acc"]}
        )
        assert gap.data["counts"]["failed"] == 1 and "not included" in gap.text and "INCOMPLETE" in gap.text
        assert "run_recover(" in gap.next_steps
        draft = await agent.call(
            "evidence_export",
            {
                "comparison_ref": gap.data["comparison_id"],
                "purpose": "draft",
                "idempotency_key": "export-draft",
            },
        )
        old_tex, _ = await agent.resource(f"hpclab://evidence/{draft.data['bundle_id']}/table.tex")

        # fix it: the recovery plan raises memory; nothing is submitted until run_submit
        rec = await agent.call(
            "run_recover", {"attempt_id": attempt["attempt_id"], "idempotency_key": "recover-1"}
        )
        rd = rec.data
        assert (
            rd["eligible"] and rd["strategy"] == "adjust_resources" and rd["resource_patch"] == {"mem": "6G"}
        )
        assert "plan ready" in rec.text.splitlines()[0] and "nothing submitted" in rec.text.splitlines()[0]
        assert (
            f'run_submit(run_id="{failed["run_id"]}", execution_plan_hash="{rd["exec_plan_hash"]}"'
            in rec.next_steps
        )
        sub = await agent.call(
            "run_submit",
            {
                "run_id": failed["run_id"],
                "execution_plan_hash": rd["exec_plan_hash"],
                "idempotency_key": "resubmit-1",
            },
        )
        assert "not yet accepted by Slurm" in sub.text and "do not submit this run again" in sub.next_steps
        refused = await agent.call(
            "run_submit",
            {
                "run_id": failed["run_id"],
                "execution_plan_hash": rd["exec_plan_hash"],
                "idempotency_key": "resubmit-2",
            },
            expect_error=True,
        )
        assert refused.text.startswith("ERROR ")
        op_json, _ = await agent.resource(sub.data["operation_ref"])
        assert json.loads(op_json)["data"]["kind"] == "submit"
        grid = await agent.wait(
            "runs_list",
            {"experiment_id": d["experiment_id"], "view": "grid"},
            lambda c: grid_counts(c).get("V") == 6,
        )
        ins = await agent.call("run_inspect", {"run_id": failed["run_id"]})
        assert ins.data["attempts_total"] == 2 and "mem=6G" in ins.text

        # the manuscript still has the draft numbers
        full = await agent.call(
            "results_compare", {"planned_run_set_ref": d["experiment_id"], "metric_names": ["acc"]}
        )
        assert full.data["counts"]["included"] == 6
        stale = await agent.call(
            "evidence_verify", {"bundle_ref": draft.data["bundle_id"], "tex_content": old_tex}
        )
        assert stale.data["ok"] is False and "STALE_TEX_NUMBER" in stale.text
        assert f'evidence_export(comparison_ref="{full.data["comparison_id"]}"' in stale.next_steps
    SCENARIOS_DONE.add("oom")


async def test_stop_work_cancel_attempt_and_sweep_then_plan_recovery(live: Live) -> None:
    spec = yaml.safe_load(EXAMPLE_SPEC.read_text())
    spec.update(name="slow", seeds=[1], smoke=None)
    spec["code"]["project_root"] = str(EXAMPLE_SPEC.parent)
    spec["entrypoint"]["argv"] += ["--sleep-per-step", "0.1"]  # ~30 s per job
    async with agent_session(live) as agent:
        prep = await agent.call("experiment_prepare", {"spec": spec, "idempotency_key": "prep-slow"})
        d = prep.data
        wf = await agent.call(
            "workflow_submit",
            {
                "plan_id": d["plan_id"],
                "plan_hash": d["plan_hash"],
                "max_parallel": 1,
                "idempotency_key": "dispatch-slow",
            },
        )
        wf_id = wf.data["workflow_id"]
        first = d["runs"][0]["run_id"]
        ins = await agent.wait(
            "run_inspect",
            {"run_id": first, "refresh": True},
            lambda c: bool(c.data["attempts"]) and c.data["attempts"][0]["scheduler_state"] == "RUNNING",
            timeout=40,
        )
        assert "run_logs(" in ins.next_steps
        probe = await agent.call("run_inspect", {"run_id": first, "probe": "gpu"})
        assert "gpu probe:" in probe.text
        att = ins.data["attempts"][0]["attempt_id"]
        listed = await agent.call(
            "artifacts_list", {"attempt_id": att, "refresh": True, "include_members": True}
        )
        assert "mutable" in listed.text
        waiting = await agent.call("runs_list", {"workflow_id": wf_id})
        assert "queued/running" in waiting.next_steps

        # cancel needs the attempt's current version: an old one is refused with the fix spelled out
        cancel = await _with_current_version(
            agent, "run_cancel", {"attempt_id": att, "idempotency_key": "cancel-1", "reason": "user asked"}, 0
        )
        assert any(c.tool == "run_cancel" and not c.ok for c in agent.calls)  # version 0 was refused first
        assert (
            cancel.data["cancel_state"] in ("requested", "confirmed") and "run_inspect(" in cancel.next_steps
        )

        wf_json, _ = await agent.resource(f"hpclab://workflows/{wf_id}")
        wc = await _with_current_version(
            agent,
            "workflow_cancel",
            {"workflow_id": wf_id, "scope": "all_managed", "idempotency_key": "wf-cancel-1"},
            json.loads(wf_json)["data"]["version"],
        )
        assert wc.data["dispatch_state"] == "closed"

        def nothing_active(c: Call) -> bool:
            counts = grid_counts(c)
            return bool(counts) and not any(counts.get(k) for k in ("R", "Q", "S", "?"))

        grid = await agent.wait(
            "runs_list", {"workflow_id": wf_id, "view": "grid"}, nothing_active, timeout=40
        )
        assert grid_counts(grid).get("X"), grid.text
        # never-dispatched cells stay 'P'; the grid says why they will not start
        if grid_counts(grid).get("P"):
            assert f"workflow {wf_id}: state cancel" in grid.text, grid.text

        batch = await agent.call(
            "run_recover", {"selector": {"workflow_id": wf_id}, "idempotency_key": "recover-batch"}
        )
        assert batch.data["mode"] == "batch" and "workflow_submit(plan_id=" in batch.next_steps

        # comparing a run set with nothing valid is honest about it (and still schema-valid)
        empty = await agent.call(
            "results_compare", {"planned_run_set_ref": d["experiment_id"], "metric_names": ["acc"]}
        )
        assert empty.data["counts"]["included"] == 0 and "INCOMPLETE" in empty.text
        cap = await agent.call(
            "cluster_inspect",
            {
                "cluster_id": "sim",
                "checks": ["capacity", "queue"],
                "candidate_shapes": [{"partition": "gpu", "gpu_type": "a100", "gpus": 2, "time": "02:00:00"}],
            },
        )
        assert "candidate gpu" in cap.text
        lab = await agent.call(
            "usage_report", {"cluster_id": "sim", "scope": "account", "group_by": ["user", "state"]}
        )
        if lab.data["visibility"] != "full":
            assert "no lab total" in lab.text
    SCENARIOS_DONE.add("stop")


async def test_errors_are_actionable_and_resources_fail_cleanly(live: Live) -> None:
    async with agent_session(live) as agent:
        e = await agent.call("run_inspect", {}, expect_error=True)
        assert "exactly one of run_id, attempt_id or job_ref" in e.text
        e = await agent.call("runs_list", {"view": "table"}, expect_error=True)
        assert (
            "- view: Input should be 'list' or 'grid'" in e.text
        )  # the actual validation error, not just a count
        e = await agent.call("run_submit", {"run_id": "run_x"}, expect_error=True)
        assert (
            "- execution_plan_hash: Field required" in e.text
            and "- idempotency_key: Field required" in e.text
        )
        e = await agent.call(
            "experiment_prepare", {"spec_path": "nope.yaml", "idempotency_key": "x"}, expect_error=True
        )
        assert "check the path" in e.text
        e = await agent.call("cluster_inspect", {"cluster_id": "hai"}, expect_error=True)
        assert "registered clusters: sim" in e.text
        e = await agent.call("run_inspect", {"run_id": "run_nope"}, expect_error=True)
        assert e.env["error"]["code"] == "not_found"
        # explicit nulls mean "not set" (agents send them)
        ok = await agent.call("runs_list", {"experiment_id": None, "view": None, "limit": 5})
        assert ok.data["view"] == "list"

        with pytest.raises(MCPError, match="not found"):
            await agent.resource("hpclab://runs/run_nope")
        with pytest.raises(MCPError, match="known: hpclab://operations"):
            await agent.resource("hpclab://nothing/1")

        prompts = {p.name: p for p in (await agent.client.list_prompts()).prompts}
        assert {"plan_experiment", "diagnose_run", "write_results", "onboard_cluster", "lab_overview"} <= set(
            prompts
        )
        got = await agent.client.get_prompt("diagnose_run", {"run_id": "run_x"})
        text = got.messages[0].content.text  # type: ignore[union-attr]
        assert text.startswith("run_id: run_x") and "run_recover" in text
        with pytest.raises(MCPError, match="needs: run_id"):
            await agent.client.get_prompt("diagnose_run", {})
    SCENARIOS_DONE.add("errors")


async def test_an_expired_ssh_login_is_named_instead_of_check_again_later(live: Live) -> None:
    """The user's SSH master connection expired (MFA): the smoke run's submission waits in
    'staging' and the grid only shows S. Found by hand: every text said "check again later" and
    nothing named the blocked operation, so an agent would have waited forever."""
    spec = yaml.safe_load(EXAMPLE_SPEC.read_text())
    spec.update(name="authwait", seeds=[1], matrix={"method": ["baseline"]})
    spec["code"]["project_root"] = str(EXAMPLE_SPEC.parent)
    async with agent_session(live) as agent:
        prep = await agent.call("experiment_prepare", {"spec": spec, "idempotency_key": "prep-authwait"})
        d = prep.data
        # prepared but never dispatched: waiting cannot help, dispatching does
        idle = await agent.call("runs_list", {"experiment_id": d["experiment_id"], "view": "grid"})
        assert "check again later" not in idle.next_steps, idle.text
        assert f'workflow_submit(prepared_experiment_id="{d["experiment_id"]}"' in idle.next_steps

        live.sim.set_faults(auth_required=True)
        try:
            wf = await agent.call(
                "workflow_submit", {"plan_id": d["plan_id"], "idempotency_key": "dispatch-authwait"}
            )
            grid = await agent.call("runs_list", {"experiment_id": d["experiment_id"], "view": "grid"})
            assert f'run_inspect(run_id="{d["smoke_run_id"]}")' in grid.next_steps, grid.text
            ins = await agent.wait(
                "run_inspect",
                {"run_id": d["smoke_run_id"]},
                lambda c: any(op["state"] == "blocked" for op in c.data["operations"]),
                timeout=30,
            )
            assert "blocked" in ins.text.splitlines()[0] and "hpclab auth sim" in ins.text.splitlines()[0]
            assert "at auth_required" in ins.text
            assert "`hpclab auth sim`" in ins.next_steps and "check again later" not in ins.next_steps
        finally:
            live.sim.clear_faults()
        # stop it: the staging attempt is cancelled before it reaches Slurm
        wf_json, _ = await agent.resource(f"hpclab://workflows/{wf.data['workflow_id']}")
        await _with_current_version(
            agent,
            "workflow_cancel",
            {
                "workflow_id": wf.data["workflow_id"],
                "scope": "all_managed",
                "idempotency_key": "wf-cancel-auth",
            },
            json.loads(wf_json)["data"]["version"],
        )
        after = await agent.call("run_inspect", {"run_id": d["smoke_run_id"]})
        assert after.data["attempts"][0]["phase"] == "cancelled_before_submit", after.text


async def test_the_stdio_server_process_talks_only_protocol_on_stdout(live: Live) -> None:
    """What `claude mcp add hpclab -- hpclab mcp serve` launches: a separate process that finds
    the running controller through HPCLAB_HOME. Any stray print would corrupt the protocol."""
    env = {**os.environ, "HPCLAB_HOME": str(live.paths.home), "HPCLAB_MCP_LOG_LEVEL": "INFO"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "hpclab.mcp"], env=env, cwd=str(REPO))
    with anyio.fail_after(60):
        async with Client(params) as client:
            assert "HPC Lab" in (client.instructions or "")
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == ALL_TOOLS
            r = await client.call_tool("runs_list", {"limit": 3})
            assert r.is_error is False and r.structured_content["ok"] is True
            listed = await client.list_resources()
            assert any(str(x.uri) == "hpclab://experiments" for x in listed.resources)


def test_zz_every_tool_was_exercised_by_the_scenarios() -> None:
    """Runs last in this module: the scenarios above called every tool successfully."""
    if {"ablation", "oom", "stop", "errors"} != SCENARIOS_DONE:
        pytest.skip(f"only these live scenarios ran: {sorted(SCENARIOS_DONE)}")
    assert CALLED_TOOLS == ALL_TOOLS, sorted(ALL_TOOLS - CALLED_TOOLS)
