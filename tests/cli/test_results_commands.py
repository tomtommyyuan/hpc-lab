"""artifacts / read / collect / compare / export / verify / trace / paper against a scripted controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cli_support import Cli, env, err, ok

from hpclab.cli.commands import strip_hpcval

EXP = "exp_01M3K9Q1CC33H1R96NRPJEDAVW"
GOOD_RUN = "run_01M3K9Q1D83ZECZ5F8ZVGB3VS7"
GOOD_ATT = "att_01M3K9Q7QY2C92358BRYJ7C2R7"
FAILED_RUN = "run_01M3K9Q1D8QH3MCQSDR0S9N2WS"
CMP = "cmp_01M3K9QS021VRHCTHP20BAHNMW"
BUNDLE = "evb_01M3K9QS049T7A49DZQY7J7WJF"


# ---------------------------------------------------------------------------
# artifacts / read / collect
# ---------------------------------------------------------------------------


def test_artifacts_table_and_run_resolution(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.succeeded")
    cli.client.responses["artifacts_list"] = env("artifacts_list")
    r = cli.run("artifacts", GOOD_RUN, "--role", "metrics", "--members")
    assert r.exit_code == 0, r.text
    assert "finalized" in r.stdout and "summary" in r.stdout and "checkpoints" in r.stdout
    assert cli.client.params("artifacts_list") == {
        "attempt_id": GOOD_ATT,
        "kind": "metrics",
        "include_members": True,
    }
    r = cli.run("--json", "artifacts", GOOD_ATT)
    assert r.json == env("artifacts_list")


def test_read_json_view_prints_parseable_json_and_meta_on_stderr(cli: Cli) -> None:
    e = env("artifact_read.json")
    cli.client.responses["artifact_read"] = e
    art = e["data"]["artifact"]["artifact_id"]
    r = cli.run("read", art, "--view", "json")
    assert r.exit_code == 0
    assert json.loads(r.stdout) == e["data"]["value"]
    assert "digest verified" in r.stderr
    assert cli.client.params("artifact_read") == {"artifact_ref": art, "view": "json"}


def test_read_jsonl_tail_and_head(cli: Cli) -> None:
    e = env("artifact_read.jsonl_tail")
    cli.client.responses["artifact_read"] = e
    r = cli.run("read", e["data"]["artifact"]["artifact_id"], "--view", "jsonl_tail", "--max-bytes", "600")
    records = [json.loads(line) for line in r.stdout.splitlines()]
    assert records == e["data"]["records"]
    assert cli.client.params("artifact_read")["max_bytes"] == 600
    h = env("artifact_read.head")
    cli.client.responses["artifact_read"] = h
    r = cli.run("read", h["data"]["artifact"]["artifact_id"], "--view", "head")
    assert r.stdout.startswith(h["data"]["text"][:50])
    assert "truncated" in r.stderr
    assert cli.run("read", "art_x", "--view", "binary").exit_code == 2


def test_read_stale_artifact_is_an_error(cli: Cli) -> None:
    cli.client.responses["artifact_read"] = err(
        "stale", "the file changed since it was indexed", digest_state="mismatch"
    )
    r = cli.run("read", "art_01M3K9Q7QZ00000000000000000")
    assert r.exit_code == 1 and "stale" in r.stderr and r.stdout == ""


def test_collect_single_batch_and_config(cli: Cli) -> None:
    cli.client.responses["evaluation_collect"] = env("evaluation_collect.single")
    r = cli.run(
        "collect",
        GOOD_ATT,
        "--evaluator",
        "hpclab_metrics",
        "--config",
        "split=test",
        "--config",
        "select=last",
    )
    assert r.exit_code == 0, r.text
    assert "valid" in r.stdout and "already existed" in r.stdout and "acc" in r.stdout
    p = cli.client.params("evaluation_collect")
    assert p["attempt_id"] == GOOD_ATT and p["config"] == {"split": "test", "select": "last"}
    assert p["evaluator_id"] == "hpclab_metrics" and p["idempotency_key"].startswith("collect-")

    cli.client.responses["evaluation_collect"] = env("evaluation_collect.batch")
    cli.client.responses["experiments_list"] = env("experiments_list")
    r = cli.run("collect", "--experiment", "tiny-mixup-ablation")
    assert r.exit_code == 0
    assert cli.client.params("evaluation_collect")["run_set_ref"] == EXP
    assert "already had an evaluation" in r.stdout
    assert "scheduler_state_not_completed" in r.stdout  # real skips are listed, idempotent ones summarized
    cli.run("collect", "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA")
    assert cli.client.params("evaluation_collect")["run_set_ref"] == "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA"
    assert cli.run("collect").exit_code == 2
    assert cli.run("collect", GOOD_ATT, "--experiment", EXP).exit_code == 2


def test_collect_wait_polls_the_background_operation(cli: Cli) -> None:
    batch = env("evaluation_collect.batch")
    batch["data"].update(pending=3, operation_id="op_01M3K9Q1MHKZ8HWQR03AZ2GP5A")
    cli.client.responses["evaluation_collect"] = batch
    running = env("operation_get.succeeded")
    running["data"]["state"] = "running"
    cli.client.responses["operation_get"] = [running, env("operation_get.succeeded")]
    r = cli.run("collect", "--experiment", EXP, "--wait")
    assert r.exit_code == 0, r.text
    assert cli.client.methods().count("operation_get") == 2


# ---------------------------------------------------------------------------
# compare / export / verify / trace
# ---------------------------------------------------------------------------


def test_compare_defaults_to_contract_metrics_and_shows_coverage(cli: Cli) -> None:
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["results_compare"] = env("results_compare")
    r = cli.run("compare", EXP)
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert cli.client.params("results_compare")["metric_names"] == ["acc", "loss"]
    assert "acc (fraction) ↑" in out and "loss (nats) ↓" in out
    assert "(2/3)" in out and "(3/3)" in out  # planned denominator stays visible
    assert "baseline (reference)" in out
    assert "differences vs baseline (descriptive; not a significance test)" in out
    assert "mixup − baseline" in out and "CI [" in out
    assert "coverage: 5/6 planned runs included" in out and "failed 1" in out
    assert "failed: mixup seed 1" in out and "node_fail" in out and FAILED_RUN in out
    assert f"hpclab export {CMP} --out <dir>" in out


def test_compare_with_runs_awaiting_evaluation_says_to_collect_before_exporting(cli: Cli) -> None:
    """E runs (completed, evaluation pending) count as missing: exporting now would publish a
    table with seeds that are merely not evaluated yet — collect first, then compare again."""
    cmp_env = env("results_compare")
    runs = cmp_env["data"]["runs"]
    pending = next(r for r in runs if r["status"] == "included")
    pending.update(code="E", status="missing", reason="completed; evaluation pending")
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["results_compare"] = cmp_env
    r = cli.run("compare", EXP)
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert f"hpclab collect {EXP}" in out and "then compare again" in out
    assert out.index(f"hpclab collect {EXP}") < out.index(f"hpclab export {CMP}")


def test_compare_parameters_and_usage_errors(cli: Cli) -> None:
    cli.client.responses["results_compare"] = env("results_compare")
    r = cli.run("compare", EXP, "-m", "acc", "--seeds", "1,2,3", "--exclude", f"{FAILED_RUN}=node died twice")
    assert r.exit_code == 0, r.text
    p = cli.client.params("results_compare")
    assert p["planned_run_set_ref"] == EXP and p["metric_names"] == ["acc"]
    assert p["expected_seeds"] == [1, 2, 3]
    assert p["exclusions"] == [{"run_id": FAILED_RUN, "reason": "node died twice"}]
    assert p["idempotency_key"].startswith("compare-")
    assert cli.run("compare", EXP, "-m", "acc", "--exclude", FAILED_RUN).exit_code == 2
    assert cli.run("compare", EXP, "-m", "acc", "--seeds", "one").exit_code == 2
    no_contract = env("experiment_get")
    no_contract["data"]["spec"].pop("comparison")
    cli.client.responses["experiment_get"] = no_contract
    assert cli.run("compare", EXP).exit_code == 2
    r = cli.run("--json", "compare", EXP, "-m", "acc")
    assert r.json == env("results_compare")


def test_export_writes_files_and_prints_next_steps(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["evidence_export"] = env("evidence_export")
    written = env("evidence_write")
    cli.client.responses["evidence_write"] = written
    out_dir = tmp_path / "paper" / "evidence"
    r = cli.run("export", CMP, "--out", str(out_dir), "--format", "md,tex,csv", "--purpose", "Table 2")
    assert r.exit_code == 0, r.text
    p = cli.client.params("evidence_export")
    assert p["comparison_ref"] == CMP and p["formats"] == ["md", "latex", "csv"] and p["purpose"] == "Table 2"
    w = cli.client.params("evidence_write")
    assert w == {"bundle_id": BUNDLE, "out_dir": str(out_dir.resolve())}
    assert f"written to: {written['data']['out_dir']}" in r.stdout
    for f in written["data"]["files"]:
        assert f["name"] in r.stdout
    assert "paper check" in r.stdout and f"hpclab verify {BUNDLE}" in r.stdout


def test_export_experiment_mode_json_and_errors(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["evidence_export"] = env("evidence_export")
    cli.client.responses["evidence_write"] = env("evidence_write")
    r = cli.run("--json", "export", "--experiment", EXP, "-m", "acc", "--out", str(tmp_path))
    assert r.exit_code == 0
    doc = r.json
    assert doc["ok"] is True
    assert doc["evidence_export"] == env("evidence_export") and doc["evidence_write"] == env("evidence_write")
    p = cli.client.params("evidence_export")
    assert p["planned_run_set_ref"] == EXP and p["metric_names"] == ["acc"] and "comparison_ref" not in p

    cli.client.responses["evidence_write"] = err(
        "precondition_failed", "files with different content already exist", conflicts=["x"]
    )
    r = cli.run("export", CMP, "--out", str(tmp_path))
    assert r.exit_code == 1 and "--overwrite" not in r.stdout and "precondition_failed" in r.stderr
    cli.run("export", CMP, "--out", str(tmp_path), "--overwrite")
    assert cli.client.params("evidence_write")["overwrite"] is True
    assert cli.run("export", "--out", str(tmp_path)).exit_code == 2
    assert cli.run("export", CMP, "--experiment", EXP).exit_code == 2
    assert cli.run("export", CMP, "-m", "acc").exit_code == 2
    assert cli.run("export", CMP, "--format", "docx").exit_code == 2
    assert cli.run("export", "run_01M3K9Q1D83ZECZ5F8ZVGB3VS7").exit_code == 2


def test_export_bundle_rewrites_an_existing_bundle_without_a_new_one(cli: Cli, tmp_path: Path) -> None:
    written = env("evidence_write")
    cli.client.responses["evidence_write"] = written
    out_dir = tmp_path / "paper" / "evidence"
    r = cli.run("export", "--bundle", BUNDLE, "--out", str(out_dir), "--overwrite")
    assert r.exit_code == 0, r.text
    assert cli.client.methods() == ["evidence_write"]  # no evidence_export: no new bundle
    assert cli.client.params("evidence_write") == {
        "bundle_id": BUNDLE,
        "out_dir": str(out_dir.resolve()),
        "overwrite": True,
    }
    assert f"written to: {written['data']['out_dir']}" in r.stdout
    assert f"hpclab verify {BUNDLE}" in r.stdout
    cli.run("export", "--bundle", f"hpclab://evidence/{BUNDLE}", "--out", str(out_dir))
    assert cli.client.params("evidence_write") == {"bundle_id": BUNDLE, "out_dir": str(out_dir.resolve())}
    r = cli.run("--json", "export", "--bundle", BUNDLE, "--out", str(out_dir))
    assert r.json == {"ok": True, "evidence_write": written}
    cli.client.responses["evidence_write"] = err(
        "precondition_failed", "files with different content already exist", conflicts=["x"]
    )
    r = cli.run("export", "--bundle", BUNDLE, "--out", str(out_dir))
    assert r.exit_code == 1 and "--overwrite" in r.stderr
    for bad in (
        ("export", "--bundle", BUNDLE, CMP),
        ("export", "--bundle", BUNDLE, "--experiment", EXP),
        ("export", "--bundle", BUNDLE, "--format", "md"),
        ("export", "--bundle", BUNDLE, "--purpose", "x"),
        ("export", "--bundle", CMP),
    ):
        assert cli.run(*bad).exit_code == 2, bad
    assert "evidence_export" not in cli.client.methods()


def test_verify_clean_and_mismatch_exit_codes(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["evidence_verify"] = env("evidence_verify.clean")
    r = cli.run("verify", BUNDLE)
    assert r.exit_code == 0 and "no errors" in r.stdout and "scientific correctness" in r.stdout
    assert "NOT checked" not in r.stdout  # no manuscript was given: nothing to check there
    tex = tmp_path / "main.tex"
    tex.write_text("x")
    cli.client.responses["evidence_verify"] = env("evidence_verify.mismatch")
    r = cli.run("verify", BUNDLE, str(tex))
    assert r.exit_code == 1
    assert "VALUE_MISMATCH" in r.stdout and "0.999" in r.stdout
    assert cli.client.params("evidence_verify") == {"bundle_ref": BUNDLE, "tex_paths": [str(tex.resolve())]}
    assert cli.run("verify", BUNDLE, str(tmp_path / "nope.tex")).exit_code == 2
    r = cli.run("evidence", "verify", BUNDLE)
    assert r.exit_code == 1  # alias of `hpclab verify`


def test_verify_prints_the_files_scanned_and_fails_when_nothing_was_checked(cli: Cli, tmp_path: Path) -> None:
    main = tmp_path / "main.tex"
    main.write_text("\\input{table}\n")
    table = tmp_path / "table.tex"
    e = env("evidence_verify.clean")
    e["data"].update(checked=True, files_scanned=[str(main), str(table)], tags_checked=3)
    cli.client.responses["evidence_verify"] = e
    r = cli.run("verify", BUNDLE, str(main))
    assert r.exit_code == 0, r.text
    assert f"scanned 2 file(s):\n  {main}\n  {table}\n" in r.stdout
    no_tags = {
        "code": "NO_TAGS_FOUND",
        "severity": "error",
        "message": "no \\hpcval tag in the manuscript: it was NOT checked",
        "refs": [str(main)],
    }
    e = env("evidence_verify.clean")
    e["data"].update(checked=False, files_scanned=[str(main)], findings=[no_tags], ok=False)
    cli.client.responses["evidence_verify"] = e
    r = cli.run("verify", BUNDLE, str(main))
    assert r.exit_code == 1
    assert "NO_TAGS_FOUND" in r.stdout and "manuscript NOT checked" in r.stdout
    e["data"].update(findings=[], ok=True)  # an ok bundle does not make an unchecked manuscript pass
    cli.client.responses["evidence_verify"] = e
    assert cli.run("verify", BUNDLE, str(main)).exit_code == 1


def test_trace_renders_the_chain(cli: Cli) -> None:
    t = env("trace")
    cli.client.responses["trace"] = t
    r = cli.run("trace", t["data"]["cell_ref"])
    assert r.exit_code == 0, r.text
    for word in ("evaluation", "artifact", "attempt", "job", "run", "code", "env"):
        assert f"   {word}" in r.stdout
    assert "chain complete" in r.stdout and "seed 1: failed" in r.stdout
    assert cli.client.params("trace") == {"cell_ref": t["data"]["cell_ref"]}
    r = cli.run("evidence", "trace", t["data"]["cell_ref"])
    assert r.exit_code == 0


# ---------------------------------------------------------------------------
# paper check / strip
# ---------------------------------------------------------------------------

TEX = (
    "\\providecommand{\\hpcval}[2]{#2}\n"
    "% Numbers are tagged \\hpcval{evb:<cell>}{<value>} for verification\n"
    "\\begin{tabular}{lc}\n"
    "baseline & $\\hpcval{"
    + BUNDLE
    + ":baseline.acc.mean}{0.856} \\pm \\hpcval{"
    + BUNDLE
    + ":baseline.acc.std}{0.013}$ \\\\\n"
    "mixup & $\\hpcval{" + BUNDLE + ":mixup.acc.mean}{\\textbf{0.845}}$ \\\\\n"
    "% old: \\hpcval{" + BUNDLE + ":mixup.acc.mean}{0.9}\n"
    "\\end{tabular}\n"
)


def test_strip_hpcval_replaces_tags_and_drops_the_macro() -> None:
    text, n = strip_hpcval(TEX)
    assert n == 3
    assert "\\hpcval" not in text
    assert "$0.856 \\pm 0.013$" in text
    assert "\\textbf{0.845}" in text  # nested braces in the value survive
    assert text.startswith("\\begin{tabular}")
    assert strip_hpcval("no tags here\n") == ("no tags here\n", 0)


def test_paper_strip_stdout_file_and_refuses_to_overwrite_source(cli: Cli, tmp_path: Path) -> None:
    src = tmp_path / "table.tex"
    src.write_text(TEX)
    r = cli.run("paper", "strip", str(src))
    assert r.exit_code == 0
    assert r.stdout == strip_hpcval(TEX)[0]
    assert "stripped 3" in r.stderr
    out = tmp_path / "clean.tex"
    r = cli.run("paper", "strip", str(src), "-o", str(out))
    assert out.read_text() == strip_hpcval(TEX)[0] and r.stdout == ""
    assert cli.run("paper", "strip", str(src), "-o", str(src)).exit_code == 2
    assert src.read_text() == TEX
    assert cli.run("paper", "strip", str(tmp_path / "missing.tex")).exit_code == 2
    assert cli.client.calls == []  # purely local


def _verify_answer(
    findings: list[dict[str, Any]],
    *,
    tags: int = 3,
    extra_files: tuple[str, ...] = (),
    bundles: tuple[str, ...] = (),
) -> Any:
    """evidence_verify as the service answers: it checks every tag of every file it is given and of
    the files they \\input (``files_scanned``), whatever the bundle."""

    def answer(p: dict[str, Any]) -> dict[str, Any]:
        e = env("evidence_verify.clean")
        files = [*p["tex_paths"], *extra_files]
        e["data"]["bundle_id"] = p["bundle_ref"]
        e["data"]["findings"] = findings
        e["data"]["ok"] = not any(f["severity"] == "error" for f in findings)
        e["data"]["checked"] = tags > 0
        e["data"]["tags_checked"] = tags
        e["data"]["files_scanned"] = files
        e["data"]["tex"] = [
            {
                "source": f,
                "tags": tags if i == len(files) - 1 else 0,
                "untagged_numbers": 0,
                "bundles": list(bundles),
            }
            for i, f in enumerate(files)
        ]
        return e

    return answer


def test_paper_check_verifies_each_bundle_against_every_file(cli: Cli, tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    tex.write_text(TEX)
    cli.client.responses["evidence_verify"] = _verify_answer([])
    r = cli.run("paper", "check", str(tex))
    assert r.exit_code == 0, r.text
    assert cli.client.params("evidence_verify") == {"bundle_ref": BUNDLE, "tex_paths": [str(tex.resolve())]}
    assert cli.client.methods().count("evidence_verify") == 1
    assert f"scanned 1 file(s):\n  {tex.resolve()}\n" in r.stdout
    assert "3 tag(s) in 1 scanned file(s), 1 bundle(s): no errors" in r.stdout


def test_paper_check_follows_input_through_the_service(cli: Cli, tmp_path: Path) -> None:
    """main.tex only \\inputs the table: the service follows it; bundles it finds there are
    verified too. The CLI does not scan includes itself."""
    other = "evb_01M3K9QS049T7A49DZQY7J7AAA"
    main = tmp_path / "main.tex"
    main.write_text("\\documentclass{article}\n\\input{tables/t1}\n")
    t1 = str(tmp_path / "tables" / "t1.tex")
    cli.client.responses["evidence_verify"] = _verify_answer([], extra_files=(t1,), bundles=(BUNDLE, other))
    r = cli.run("paper", "check", str(main))
    assert r.exit_code == 1  # nothing tells the CLI which bundle to verify: no guessing
    assert "--bundle" in r.stderr and "evidence_verify" not in cli.client.methods()
    r = cli.run("paper", "check", str(main), "--bundle", BUNDLE)
    assert r.exit_code == 0, r.text
    calls = [p for m, p in cli.client.calls if m == "evidence_verify"]
    assert [c["bundle_ref"] for c in calls] == [BUNDLE, other]
    assert all(c["tex_paths"] == [str(main.resolve())] for c in calls)
    assert f"scanned 2 file(s):\n  {main.resolve()}\n  {t1}\n" in r.stdout
    assert "2 bundle(s): no errors" in r.stdout


def test_paper_check_dedupes_findings_and_fails_on_errors(cli: Cli, tmp_path: Path) -> None:
    other = "evb_01M3K9QS049T7A49DZQY7J7AAA"
    a = tmp_path / "a.tex"
    b = tmp_path / "b.tex"
    a.write_text(TEX)
    b.write_text(
        "x & \\hpcval{" + other + ":mixup.acc.mean}{0.5} \\\\\n\\hpcval{not-a-bundle:x.acc.mean}{1}\n"
    )
    mismatch = {
        "code": "VALUE_MISMATCH",
        "severity": "error",
        "message": "mixup/acc mean: tex shows 0.5",
        "refs": [f"{b}:1"],
    }
    info = {
        "code": "UNTAGGED_TEX_NUMBER",
        "severity": "info",
        "message": "number 3.14 is not tagged",
        "refs": [f"{a}:9"],
    }
    # the service checks every tag of each file it is given: the bad reference in b.tex is
    # reported by the verify calls themselves (the CLI does not report it a second time)
    unresolved = {
        "code": "UNRESOLVED_REF",
        "severity": "error",
        "message": "\\hpcval references unknown bundle 'not-a-bundle'",
        "refs": [f"{b}:2"],
    }
    unresolved_input = {
        "code": "UNRESOLVED_INPUT",
        "severity": "warn",
        "message": "\\input{missing} at a.tex:3: not found; the numbers in it were NOT checked",
        "refs": [f"{a}:3"],
    }
    cli.client.responses["evidence_verify"] = _verify_answer(
        [mismatch, info, unresolved, unresolved_input], tags=5
    )
    r = cli.run("paper", "check", str(a), str(b))
    assert r.exit_code == 1
    assert r.stdout.count("VALUE_MISMATCH") == 1  # reported by both bundle checks, printed once
    assert r.stdout.count("UNRESOLVED_REF") == 1
    assert r.stdout.count("UNRESOLVED_INPUT") == 1
    assert "unknown bundle 'not-a-bundle'" in r.stdout and "not an evidence bundle id" not in r.stdout
    assert "2 error(s)" in r.stdout
    calls = [p for m, p in cli.client.calls if m == "evidence_verify"]
    assert sorted(c["bundle_ref"] for c in calls) == sorted([BUNDLE, other])
    assert all(c["tex_paths"] == [str(a.resolve()), str(b.resolve())] for c in calls)
    r = cli.run("--json", "paper", "check", str(a), str(b))
    doc = r.json
    assert doc["ok"] is False and doc["tags"] == 5 and set(doc["bundles"]) == {BUNDLE, other}
    assert doc["checked"] is True and doc["files_scanned"] == [str(a.resolve()), str(b.resolve())]


def test_paper_check_unknown_bundle_and_no_tags(cli: Cli, tmp_path: Path) -> None:
    tex = tmp_path / "main.tex"
    tex.write_text(TEX)
    cli.client.responses["evidence_verify"] = err("not_found", f"evidence bundle {BUNDLE} not found")
    r = cli.run("paper", "check", str(tex))
    assert r.exit_code == 1 and "UNRESOLVED_REF" in r.stdout and "not_found" in r.stdout
    # tags exist but the service found none it could check (e.g. an older file): NOT checked
    no_tags = {
        "code": "NO_TAGS_FOUND",
        "severity": "error",
        "message": "no \\hpcval tag in the manuscript: it was NOT checked",
        "refs": [str(tex.resolve())],
    }
    cli.client.responses["evidence_verify"] = _verify_answer([no_tags], tags=0)
    r = cli.run("paper", "check", str(tex))
    assert r.exit_code == 1 and "NO_TAGS_FOUND" in r.stdout and "manuscript NOT checked" in r.stdout
    plain = tmp_path / "plain.tex"
    plain.write_text("Accuracy was 0.91.\n")
    r = cli.run("paper", "check", str(plain))
    assert r.exit_code == 1 and "no \\hpcval tag" in r.stderr and "--bundle" in r.stderr
    r = cli.run("--json", "paper", "check", str(plain))
    assert r.exit_code == 1 and r.json["ok"] is False and r.json["checked"] is False
    assert r.json["findings"][0]["code"] == "NO_TAGS_FOUND"
    assert cli.run("paper", "check", str(tmp_path / "missing.tex")).exit_code == 2
    assert cli.run("paper", "check", str(tex), "--bundle", "cmp_x").exit_code == 2


@pytest.mark.parametrize("name", ["evidence_verify.clean", "evidence_verify.mismatch"])
def test_captured_verify_envelopes_render(cli: Cli, name: str) -> None:
    cli.client.responses["evidence_verify"] = env(name)
    r = cli.run("verify", BUNDLE)
    assert ("ERRORS" in r.stdout) == (name.endswith("mismatch"))


def test_ok_helper_shape() -> None:
    e = ok({"a": 1}, "s", warnings=["w"])
    assert e["ok"] and e["warnings"] == ["w"] and e["summary"] == "s"
