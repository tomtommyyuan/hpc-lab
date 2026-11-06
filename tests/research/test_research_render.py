"""Evidence renderers: golden outputs, escaping, gap reporting and bundle validation."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from research_bundle_support import complete_bundle, make_cell, provenance_row

from hpclab.domain.errors import InvalidArgument
from hpclab.services.render import (
    CSV_COLUMNS,
    VALIDITY_LIMITATION,
    EvidenceBundle,
    latex_escape,
    parse_bundle,
    render_csv,
    render_latex,
    render_markdown,
)

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("HPCLAB_UPDATE_GOLDEN") == "1"

Factory = Callable[[], dict[str, Any]]


def check_golden(name: str, text: str) -> None:
    """Compare byte-for-byte (CRLF included); HPCLAB_UPDATE_GOLDEN=1 rewrites the file."""
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    assert path.exists(), f"missing golden file {path}; run with HPCLAB_UPDATE_GOLDEN=1"
    assert text == path.read_bytes().decode("utf-8")


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


def test_markdown_golden(bundle: dict[str, Any]) -> None:
    check_golden("evidence.md", render_markdown(bundle))


def test_markdown_complete_golden() -> None:
    check_golden("evidence_complete.md", render_markdown(complete_bundle()))


def test_latex_golden(bundle: dict[str, Any]) -> None:
    check_golden("evidence.tex", render_latex(bundle, label="tab:mixup"))


def test_latex_options_golden(bundle: dict[str, Any]) -> None:
    text = render_latex(
        bundle, digits=2, bold_best=True, wrap_table=False, metrics=["loss"], group_header="Run"
    )
    check_golden("evidence_bare.tex", text)


def test_csv_golden(bundle: dict[str, Any]) -> None:
    check_golden("evidence.csv", render_csv(bundle))


def test_rendering_is_deterministic_under_key_reordering(bundle_factory: Factory) -> None:
    a = bundle_factory()
    b = json.loads(json.dumps(a, sort_keys=True))  # same content, different key order
    for group in b["groups"]:
        for cell in group["metrics"].values():
            cell["values_by_seed"] = dict(reversed(list(cell["values_by_seed"].items())))
    for render in (render_markdown, render_latex, render_csv):
        assert render(a) == render(b)


def test_bundle_model_and_dict_render_identically(bundle: dict[str, Any]) -> None:
    model = parse_bundle(bundle)
    assert isinstance(model, EvidenceBundle)
    assert render_markdown(model) == render_markdown(bundle)


# ---------------------------------------------------------------------------
# Markdown guarantees
# ---------------------------------------------------------------------------


def test_markdown_headline_cells_and_directions(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert "| Group | acc (fraction) ↑ | loss (nats) ↓ |" in md
    assert "| erm (baseline) | 0.9117 ± 0.0065 (3/3) |" in md
    assert "| mixup | 0.9190 ± 0.0028 (2/3) |" in md


def test_single_seed_std_is_a_dash_never_zero(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert r"| lr=1e-3 & wd\_0.1 (50%) | 0.8740 ± — (1/3) | 0.4100 ± — (1/3) |" in md
    assert "± 0.0000" not in md


def test_missing_section_is_never_omitted(bundle_factory: Factory) -> None:
    for b in (bundle_factory(), complete_bundle()):
        assert "## Missing, failed and excluded runs" in render_markdown(b)
    md = render_markdown(complete_bundle())
    assert "No gaps: every planned run contributed a value to every reported metric." in md
    assert "> **Coverage: complete.** All 4 planned runs are included." in md


def test_markdown_lists_every_gap(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert "> **Coverage: incomplete.** 6 of 9 planned runs included (1 missing, 1 failed, 1 excluded)." in md
    assert "| mixup | acc | 2/3 | — | 3 | — |" in md
    assert "| 1/3 | 2 | — | 3 (diverged: loss=nan at step 400) |" in md
    assert "| mixup | 0.9210 | 0.9170 | failed |" in md
    assert r"| lr=1e-3 & wd\_0.1 (50%) | 0.8740 | missing | excluded |" in md
    assert "OUT\\_OF\\_MEMORY on gpu-07" in md


def test_metric_absent_from_a_group_is_called_out(bundle: dict[str, Any]) -> None:
    del bundle["groups"][1]["metrics"]["loss"]
    bundle["differences"] = [d for d in bundle["differences"] if d["metric"] == "acc"]
    bundle["provenance"] = [
        p for p in bundle["provenance"] if (p["group_label"], p["metric"]) != ("mixup", "loss")
    ]
    md = render_markdown(bundle)
    assert "| mixup | 0.9190 ± 0.0028 (2/3) | not reported |" in md
    assert "Metrics not reported at all: mixup/loss." in md


def test_differences_are_labelled_descriptive(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert "## Differences vs baseline (descriptive)" in md
    assert "not hypothesis tests" in md
    assert "| mixup | erm | acc ↑ | paired | 2 pairs | +0.0105 | [-0.0086, 0.0296] |" in md
    assert r"| lr=1e-3 & wd\_0.1 (50%) | erm | acc ↑ | unpaired | 1 vs 3 | -0.0377 | — |" in md


def test_definitions_state_the_formulas(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert "denominator n − 1" in md
    assert "mean ± t(q=0.975, df=n − 1) × SEM" in md
    assert "for each seed s present in both groups" in md


def test_ci_level_is_propagated(bundle_factory: Factory) -> None:
    b = bundle_factory()
    b["ci_level"] = 0.9
    for group in b["groups"]:
        for name, cell in group["metrics"].items():
            values = {int(k): v for k, v in cell["values_by_seed"].items()}
            gaps = {k: cell[k] for k in ("missing_seeds", "failed_seeds", "excluded")}
            group["metrics"][name] = make_cell(values, expected=cell["expected_n"], ci_level=0.9, **gaps)
    b["differences"] = []
    md = render_markdown(b)
    assert "| 90% CI |" in md and "t(q=0.95, df=n − 1)" in md


def test_incomparable_warning_is_prominent(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle)
    assert "> **Warning:** 1 incomparability issue(s) flagged" in md
    assert '- **environment differs**: env\\_torch=\\["2.7.1", "2.8.0+cu128"\\]; group=mixup' in md
    assert "None flagged by the comparison contract checks." in render_markdown(complete_bundle())


def test_limitations_always_include_validity_disclaimer(bundle: dict[str, Any]) -> None:
    bundle["limitations"] = [VALIDITY_LIMITATION, "Small n."]
    md = render_markdown(bundle)
    assert md.count("Record consistency is not proof of scientific validity") == 1
    assert "- Small n." in md
    assert "Record consistency is not proof of scientific validity" in render_markdown(complete_bundle())


def test_provenance_table_is_capped_with_a_count_note(bundle: dict[str, Any]) -> None:
    md = render_markdown(bundle, max_provenance_rows=5)
    section = md.split("## Provenance")[1].split("## Limitations")[0]
    assert section.count("| `ev_") == 5
    assert "Showing 5 of 13 provenance rows" in section
    assert "Showing" not in render_markdown(bundle).split("## Provenance")[1]


def test_untraced_values_are_flagged(bundle: dict[str, Any]) -> None:
    bundle["provenance"] = [
        p for p in bundle["provenance"] if not (p["group_label"] == "erm" and p["seed"] == 2)
    ]
    md = render_markdown(bundle)
    assert (
        "**Warning:** 2 included value(s) have no provenance row and cannot be traced: erm/acc/seed 2, erm/loss/seed 2."
        in md
    )


def test_markdown_escapes_table_breaking_labels(bundle: dict[str, Any]) -> None:
    rename(bundle, "mixup", "a|b *c* `d`\nnext")
    md = render_markdown(bundle)
    assert "| a\\|b \\*c\\* \\`d\\` next | 0.9190" in md


def test_no_significance_language_anywhere(bundle: dict[str, Any]) -> None:
    for text in (render_markdown(bundle), render_latex(bundle), render_csv(bundle)):
        lowered = text.lower()
        assert "significan" not in lowered and "p-value" not in lowered and "p =" not in lowered


# ---------------------------------------------------------------------------
# LaTeX guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("a_b", r"a\_b"),
        ("50%", r"50\%"),
        ("x & y", r"x \& y"),
        ("$5 #1", r"\$5 \#1"),
        ("{x}", r"\{x\}"),
        ("~^", r"\textasciitilde{}\textasciicircum{}"),
        ("a\\b", r"a\textbackslash{}b"),
        ("<|>", r"\textless{}\textbar{}\textgreater{}"),
        ("line\nbreak", "line break"),
        ("\\_", r"\textbackslash{}\_"),  # one pass: no double escaping
        ("plain ümlaut", "plain ümlaut"),
    ],
)
def test_latex_escape(raw: str, escaped: str) -> None:
    assert latex_escape(raw) == escaped


def test_latex_escapes_labels_and_uses_booktabs(bundle: dict[str, Any]) -> None:
    tex = render_latex(bundle)
    assert r"lr=1e-3 \& wd\_0.1 (50\%)" in tex
    for rule in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{tabular}{lcc}"):
        assert rule in tex
    assert "siunitx" not in tex and r"\sisetup" not in tex
    assert r"Method & acc (fraction) $\uparrow$ & loss (nats) $\downarrow$ \\" in tex


def test_latex_dagger_footnote_lists_missing_seeds(bundle: dict[str, Any]) -> None:
    tex = render_latex(bundle)
    assert r"mixup & $0.919 \pm 0.003$\textsuperscript{\dag}" in tex
    assert r"erm & $0.912 \pm 0.007$ &" in tex  # complete cells carry no dagger
    footnote = tex.split(r"\footnotesize")[1]
    assert r"\textsuperscript{\dag}Fewer runs than planned:" in footnote
    assert "mixup, acc: 2 of 3 runs, failed seeds 3" in footnote
    assert "missing seeds 2, seed 3 excluded (diverged: loss=nan at step 400)" in footnote


def test_latex_single_seed_renders_dash(bundle: dict[str, Any]) -> None:
    tex = render_latex(bundle)
    assert r"$0.874 \pm$ ---\textsuperscript{\dag}" in tex
    assert "---: standard deviation undefined with a single seed." in tex
    assert r"\pm 0.000" not in tex


def test_latex_complete_bundle_has_no_footnote() -> None:
    tex = render_latex(complete_bundle())
    assert r"\dag" not in tex and r"\footnotesize" not in tex
    assert "(complete)" in tex


def test_latex_bold_best_respects_direction_and_ties(bundle_factory: Factory) -> None:
    b = bundle_factory()
    tex = render_latex(b, bold_best=True)
    assert r"mixup & $\mathbf{0.919 \pm 0.003}$\textsuperscript{\dag} & $\mathbf{0.292 \pm 0.006}$" in tex
    assert render_latex(b).count(r"\mathbf") == 0  # off by default

    b["metric_decls"][0]["higher_is_better"] = None  # no direction → never bolded
    assert r"\mathbf{0.919" not in render_latex(b, bold_best=True)


def test_latex_bold_best_marks_display_ties() -> None:
    b = complete_bundle()
    b["groups"][1]["metrics"]["acc"] = copy.deepcopy(b["groups"][0]["metrics"]["acc"])
    b["provenance"] = [
        provenance_row(g["label"], s, "acc", v) for g in b["groups"] for s, v in [(1, 0.93), (2, 0.94)]
    ]
    tex = render_latex(b, bold_best=True, digits=2)
    assert tex.count(r"\mathbf") == 2


def test_latex_caption_label_and_metric_selection(bundle: dict[str, Any]) -> None:
    tex = render_latex(bundle, caption="Top-1 (%) & loss_", label="tab:main", metrics=["loss", "acc"])
    assert r"\caption{Top-1 (\%) \& loss\_}" in tex and r"\label{tab:main}" in tex
    assert r"Method & loss (nats) $\downarrow$ & acc (fraction) $\uparrow$ \\" in tex
    with pytest.raises(InvalidArgument):
        render_latex(bundle, label="bad label{}")
    with pytest.raises(InvalidArgument):
        render_latex(bundle, metrics=["f1"])


@pytest.mark.parametrize("digits", [-1, 11, 2.5, True])
def test_digits_are_validated(bundle: dict[str, Any], digits: Any) -> None:
    with pytest.raises(InvalidArgument):
        render_latex(bundle, digits=digits)
    with pytest.raises(InvalidArgument):
        render_markdown(bundle, digits=digits)


def test_tiny_and_huge_values_use_scientific_notation() -> None:
    b = complete_bundle()
    values = {1: 1.5e-6, 2: 2.5e-6}
    b["groups"][0]["metrics"]["acc"] = make_cell(values, expected=2)
    b["provenance"] = [p for p in b["provenance"] if p["group_label"] != "erm"]
    b["provenance"] += [provenance_row("erm", s, "acc", v) for s, v in values.items()]
    tex = render_latex(b, digits=3)
    assert r"$2.00\times10^{-6} \pm 7.07\times10^{-7}$" in tex
    assert "2.00e-06 ± 7.07e-07" in render_markdown(b, digits=3)


# ---------------------------------------------------------------------------
# CSV guarantees
# ---------------------------------------------------------------------------


def test_csv_is_rfc4180_long_format(bundle: dict[str, Any]) -> None:
    rename(bundle, "mixup", 'mix,"up"\nv2')
    text = render_csv(bundle)
    assert text.endswith("\r\n") and "\r\n" in text
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 3 * 2 * 3  # group × metric × expected seed, gaps included
    mixup = [r for r in rows if r["group_label"] == 'mix,"up"\nv2']
    assert [r["status"] for r in mixup if r["metric"] == "acc"] == ["included", "included", "failed"]
    assert mixup[0]["group_key"] == '{"method":"mixup"}'


def test_csv_values_round_trip_exactly(bundle: dict[str, Any]) -> None:
    rows = list(csv.DictReader(io.StringIO(render_csv(bundle), newline="")))
    for group in bundle["groups"]:
        for metric, cell in group["metrics"].items():
            for seed, value in cell["values_by_seed"].items():
                (row,) = [
                    r
                    for r in rows
                    if (r["group_label"], r["metric"], r["seed"]) == (group["label"], metric, seed)
                ]
                assert float(row["value"]) == value
                assert row["evaluation_id"].startswith("ev_") and row["artifact_digest"].startswith("sha256:")


def test_csv_excluded_row_keeps_reason_and_provenance(bundle: dict[str, Any]) -> None:
    rows = list(csv.DictReader(io.StringIO(render_csv(bundle), newline="")))
    (row,) = [r for r in rows if r["status"] == "excluded" and r["metric"] == "loss"]
    assert row["exclusion_reason"] == "diverged: loss=nan at step 400"
    assert row["value"] == "" and row["attempt_id"].startswith("att_")


# ---------------------------------------------------------------------------
# Bundle validation: inconsistent bundles are refused, not rendered
# ---------------------------------------------------------------------------


def rename(bundle: dict[str, Any], old: str, new: str) -> None:
    for group in bundle["groups"]:
        if group["label"] == old:
            group["label"] = new
    for item in bundle["differences"] + bundle["provenance"]:
        for key in ("group_label", "baseline_label"):
            if item.get(key) == old:
                item[key] = new


def cell(bundle: dict[str, Any], label: str, metric: str) -> dict[str, Any]:
    return next(g for g in bundle["groups"] if g["label"] == label)["metrics"][metric]


MUTATIONS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "schema": (lambda b: b.update(schema="hpclab.evidence/v2"), "unsupported evidence schema"),
    "n-vs-values": (lambda b: cell(b, "erm", "acc").update(n=2), "values_by_seed has 3 values"),
    "denominator": (lambda b: cell(b, "mixup", "acc").update(failed_seeds=[]), "expected_n=3"),
    "seed-overlap": (lambda b: cell(b, "mixup", "acc").update(missing_seeds=[1]), "seed 1 appears in both"),
    "mean": (lambda b: cell(b, "erm", "acc").update(mean=0.95), "mean=0.95 but values_by_seed give"),
    "fake-zero-std": (lambda b: cell(b, "lr=1e-3 & wd_0.1 (50%)", "acc").update(std=0.0), "std=0.0"),
    "ci-level": (lambda b: b.update(ci_level=0.9), "ci_low="),
    "nan-value": (lambda b: cell(b, "erm", "acc")["values_by_seed"].update({"1": float("nan")}), "finite"),
    "dup-label": (lambda b: b["groups"][2].update(label="erm"), "unique"),
    "diff-group": (lambda b: b["differences"][0].update(group_label="nope"), "unknown group"),
    "diff-metric": (lambda b: b["differences"][0].update(metric="f1"), "has no metric"),
    "diff-mean": (lambda b: b["differences"][0].update(mean_diff=0.5), "mean_diff=0.5"),
    "diff-n": (lambda b: b["differences"][2].update(n_baseline=2), "n_baseline=2"),
    "prov-value": (lambda b: b["provenance"][0].update(value=0.5), "differs from values_by_seed"),
    "prov-group": (lambda b: b["provenance"][0].update(group_label="nope"), "unknown group"),
    "coverage-sum": (lambda b: b["coverage"].update(planned_runs=10), "planned_runs = 10"),
    "coverage-complete": (lambda b: b["coverage"].update(complete=True), "complete is true"),
    "prov-duplicate": (
        lambda b: b["provenance"].append(copy.deepcopy(b["provenance"][0])),
        "more than one provenance row",
    ),
    "prov-unplanned-seed": (lambda b: b["provenance"][0].update(seed=9), "seed 9 is not one of"),
    "dup-metric-decl": (
        lambda b: b["metric_decls"].append({"name": "acc", "higher_is_better": False}),
        "declared more than once",
    ),
    "missing-groups": (lambda b: b.pop("groups"), "groups"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_inconsistent_bundles_are_rejected(bundle_factory: Factory, name: str) -> None:
    mutate, message = MUTATIONS[name]
    b = bundle_factory()
    mutate(b)
    for render in (render_markdown, render_latex, render_csv):
        with pytest.raises(InvalidArgument) as err:
            render(b)
        assert message in json.dumps(err.value.details) + err.value.message


def test_int_seed_keys_and_unknown_fields_are_accepted(bundle: dict[str, Any]) -> None:
    c = cell(bundle, "erm", "acc")
    c["values_by_seed"] = {int(k): v for k, v in c["values_by_seed"].items()}
    bundle["future_field"] = {"x": 1}
    assert "0.9117 ± 0.0065 (3/3)" in render_markdown(bundle)


def test_string_seeds_sort_after_numeric_ones() -> None:
    b = complete_bundle()
    values = {"10": 0.93, "2": 0.92}
    b["groups"][0]["metrics"]["acc"] = {
        **make_cell({10: 0.93, 2: 0.92}, expected=3),
        "missing_seeds": ["none"],
    }
    b["provenance"] = [p for p in b["provenance"] if p["group_label"] != "erm"]
    b["provenance"] += [provenance_row("erm", int(s), "acc", v) for s, v in values.items()]
    b["coverage"] = {
        "planned_runs": 5,
        "included": 4,
        "missing": 1,
        "failed": 0,
        "excluded": 0,
        "complete": False,
    }
    md = render_markdown(b)
    assert re.search(r"\| Group \| seed 1 \| seed 2 \| seed 10 \| seed none \|", md)


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["[ablation] mixup", "*mixup"])
def test_latex_row_start_is_not_parsed_as_row_syntax(bundle: dict[str, Any], label: str) -> None:
    # After \\ or \midrule LaTeX reads a leading '[' as an optional length (compile error)
    # and a leading '*' as the \\* variant (the star silently vanishes from the label).
    rename(bundle, "erm", label)  # the first body row, right after \midrule
    rename(bundle, "mixup", label + " 2")  # a later row, right after \\
    tex = render_latex(bundle, group_header=label)
    assert "\n{}" + label + " & $0.912" in tex
    assert "\n{}" + label + " 2 & $0.919" in tex
    assert "\\toprule\n{}" + label + " & acc" in tex


def test_parse_bundle_rejects_non_mappings() -> None:
    values: list[Any] = [[], "hpclab.evidence/v1", None]
    for value in values:
        with pytest.raises(InvalidArgument):
            parse_bundle(value)


PDFLATEX = shutil.which("pdflatex")


@pytest.mark.skipif(PDFLATEX is None, reason="pdflatex is not installed")
def test_latex_output_compiles_with_hostile_labels(bundle: dict[str, Any], tmp_path: Path) -> None:
    rename(bundle, "erm", "[base] erm_1 & 50% #1 $x$ ~^ \\ {y} <a|b>")
    rename(bundle, "mixup", "*mixup")
    bundle["title"] = "Title % with } specials"
    table = render_latex(bundle, label="tab:x", caption="Top-1 [draft] 50% & more_", group_header="[m]")
    bare = render_latex(bundle, wrap_table=False, bold_best=True, digits=2)
    doc = "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{booktabs}",
            "\\begin{document}",
            table,
            bare,
            "\\end{document}",
            "",
        ]
    )
    (tmp_path / "t.tex").write_text(doc, encoding="utf-8")
    assert PDFLATEX is not None
    proc = subprocess.run(
        [PDFLATEX, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    errors = [line for line in proc.stdout.splitlines() if line.startswith("!")]
    assert proc.returncode == 0, errors or proc.stdout[-2000:]


def _tiny_bundle() -> dict[str, Any]:
    # A memorizing toy run: training loss around 1e-12. Verification must scale with the data.
    b = complete_bundle()
    values = {1: 1e-12, 2: 3e-12}
    b["groups"][0]["metrics"]["acc"] = make_cell(values, expected=2)
    b["provenance"] = [p for p in b["provenance"] if p["group_label"] != "erm"]
    b["provenance"] += [provenance_row("erm", s, "acc", v) for s, v in values.items()]
    return b


def test_verification_tolerance_scales_with_tiny_metrics() -> None:
    assert "2.000e-12" in render_markdown(_tiny_bundle())  # the honest bundle renders
    tampered = _tiny_bundle()
    tampered["groups"][0]["metrics"]["acc"]["mean"] = 1.5e-12  # true mean is 2e-12
    with pytest.raises(InvalidArgument, match=re.escape("mean=1.5e-12")):
        render_markdown(tampered)
    tampered = _tiny_bundle()
    tampered["provenance"][-1]["value"] = 2.5e-12  # true value 3e-12
    with pytest.raises(InvalidArgument, match="differs from values_by_seed"):
        render_csv(tampered)


def test_complete_run_coverage_does_not_hide_per_metric_gaps() -> None:
    # Every run finished and was evaluated, but one evaluation lacks the metric for seed 2.
    b = complete_bundle()
    values = {1: 0.93}
    b["groups"][0]["metrics"]["acc"] = make_cell(values, expected=2, missing_seeds=[2])
    b["provenance"] = [p for p in b["provenance"] if p["group_label"] != "erm"]
    b["provenance"] += [provenance_row("erm", 1, "acc", 0.93)]
    md = render_markdown(b)
    assert "All 4 planned runs are included" in md
    assert "1 metric cell(s) have fewer values than planned" in md
    assert "| erm | acc | 1/2 | 2 | — | — |" in md
    tex = render_latex(b)
    assert "(complete)" not in tex and "1 metric cell(s) with fewer values than planned" in tex
