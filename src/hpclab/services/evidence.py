"""Evidence bundles (tools §14 ``evidence_export``): a frozen comparison, rendered for papers.

A bundle is the ``hpclab.evidence/v1`` document of :mod:`hpclab.services.render` built from one
archived comparison, plus evidence-only keys the renderers ignore:

* ``provenance`` — one row per (group, metric, seed) that has an execution record: evaluation →
  artifact digest → attempt → Slurm job → run spec hash → code digest/commit → env GPU/torch.
  Included values carry the value; failed/excluded seeds carry what was observed (or null).
* ``cell_ids`` — the stable ids of every table cell (``render.cell_ids``), so ``\\hpcval`` tags,
  ``trace`` and ``evidence_verify`` resolve the same cell forever, even if slugging changes.
* ``integrity`` — post-hoc seeds, exclusions decided after the result was observed, environment
  heterogeneity and attempt selection (rendered as disclosures and LaTeX footnotes); runs that
  had COMPLETED but were not evaluated yet (cell code ``E``) are disclosed as making the numbers
  provisional. Evaluations pinned by the caller (``evaluation_refs``) bypass the contract's
  ``attempt_selection``: ``evaluations_pinned`` says so, each run with several valid attempts
  carries ``rule: "evaluation_refs"``, the attempt the contract's rule would have picked
  (``rule_attempt_id``) and whether the pin overrode it (``overrides_rule``); the pinned ids are
  the bundle's top-level ``evaluation_refs``.
* ``coverage.notes`` — besides the comparison's notes, one line per missing/failed run with its
  cell code and reason (``E`` → missing: completed, evaluation pending / outputs being checked;
  ``C!`` → failed: completed with incomplete/invalid outputs or a partial/invalid evaluation),
  so every footnoted seed can be explained.
* ``table`` — the LaTeX display options (decimals, scale, spread) the verifier needs for rounding.

Files (``summary.md``, ``table.tex``, ``table.csv``, ``manifest.json``) are stored in the ledger
(``evidence_bundles``) and served as resources ``hpclab://evidence/<id>/<file>``; the CLI writes
them to a caller-chosen directory with :func:`evidence_write`. An incomplete comparison still
exports — with ``coverage.complete=False``, warnings, and every gap listed in every format.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hpclab import __version__
from hpclab.api.envelope import MAX_TEXT_BYTES, Coverage, SourceStatus, clamp_bytes
from hpclab.domain.clock import iso
from hpclab.domain.errors import InvalidArgument, NotFound, PreconditionFailed
from hpclab.domain.hashing import digest, sha256_hex
from hpclab.domain.ids import kind_of, new_id
from hpclab.services.compare import (
    MAX_METRICS,
    ComparisonData,
    MultiValidAttempt,
    Reader,
    RunCoverage,
    ValueChain,
    awaiting_evaluation,
    load_archive,
    resolve_run_set,
    results_compare,
    seed_key,
    strip_ref,
)
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import select_evaluations
from hpclab.services.render import (
    EVIDENCE_SCHEMA,
    cell_ids,
    render_csv,
    render_latex,
    render_markdown,
)

# The facade's method table (services/app.py) resolves ``trace`` from this module.
from hpclab.services.verify import TraceData, trace
from hpclab.storage.db import jdump, jload

log = logging.getLogger("hpclab.evidence")

__all__ = [
    "FORMAT_FILES",
    "EvidenceData",
    "EvidenceGetData",
    "EvidenceWriteData",
    "TableSpec",
    "TraceData",
    "build_bundle",
    "evidence_export",
    "evidence_get",
    "evidence_write",
    "load_bundle",
    "outbox_handlers",
    "periodic_tasks",
    "trace",
]

TOOL = "evidence_export"
PINNED_RULE = "evaluation_refs"  # the attempt "rule" when the caller pinned the evaluations
FORMAT_FILES: dict[str, str] = {
    "md": "summary.md",
    "latex": "table.tex",
    "csv": "table.csv",
    "json": "manifest.json",
}
MEDIA_TYPES = {
    "summary.md": "text/markdown",
    "table.tex": "application/x-tex",
    "table.csv": "text/csv",
    "manifest.json": "application/json",
}
SUMMARY_MAX_BYTES = 16 * 1024
MAX_GAP_NOTES = 50  # per-run gap lines in the bundle's coverage notes
Format = Literal["md", "json", "csv", "latex"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TableSpec(BaseModel):
    """How the LaTeX table displays the comparison (the bundle itself always has everything)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[str] | None = Field(default=None, description="Group labels to show, in order (default all).")
    cols: list[str] | None = Field(default=None, description="Metrics to show, in order (default all).")
    stat: Literal["mean_std", "mean_sem"] = "mean_std"
    decimals: int = Field(default=3, ge=0, le=6)
    scale: float = Field(default=1.0, gt=0, le=1e6, allow_inf_nan=False)
    min_n: int | None = Field(default=None, ge=1, le=64)
    caption: str | None = Field(default=None, max_length=500)
    label: str | None = Field(default=None, max_length=100)
    bold_best: bool = False

    @field_validator("rows", "cols")
    @classmethod
    def _non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not v:
            raise ValueError("must be non-empty (omit it to show everything)")
        return v


class EvidenceFile(_Model):
    name: str
    uri: str
    media_type: str
    bytes: int
    sha256: str


class EvidenceCounts(_Model):
    planned_runs: int
    included: int
    missing: int
    failed: int
    excluded: int


class EvidenceData(_Model):
    bundle_id: str
    bundle_ref: str
    comparison_id: str
    comparison_ref: str
    run_set_ref: str
    purpose: str
    created_at: str
    files: list[EvidenceFile]
    counts: EvidenceCounts
    incomparable: int
    integrity_flags: list[str] = Field(default_factory=list)
    summary_markdown: str
    summary_truncated: bool = False
    summary: str
    coverage: Coverage
    warnings: list[str] = Field(default_factory=list)


class EvidenceGetData(_Model):
    bundle_id: str
    bundle_ref: str
    comparison_id: str | None
    comparison_ref: str | None
    run_set_ref: str | None
    purpose: str
    created_at: str
    files: list[EvidenceFile]
    file: str | None = None
    content: str | None = None
    offset: int = 0
    size: int | None = None
    truncated: bool = False
    next_offset: int | None = None
    summary: str
    coverage: Coverage


class WrittenFile(_Model):
    name: str
    path: str
    bytes: int
    sha256: str
    written: bool


class EvidenceWriteData(_Model):
    bundle_id: str
    out_dir: str
    files: list[WrittenFile]
    summary: str


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def evidence_export(
    ctx: ServiceContext,
    *,
    purpose: str,
    idempotency_key: str,
    comparison_ref: str | None = None,
    planned_run_set_ref: str | None = None,
    evaluation_refs: list[str] | None = None,
    formats: list[Format] | None = None,
    table_spec: TableSpec | None = None,
    metric_names: list[str] | None = None,
) -> EvidenceData:
    """Freeze a comparison into an evidence bundle (md / json / csv / LaTeX with ``\\hpcval`` tags).

    Pass ``comparison_ref`` (an archived comparison) or ``planned_run_set_ref`` (compare with the
    experiment's contract, then export; ``metric_names`` default to the contract's metrics, else
    every metric of the valid evaluations). The same ``idempotency_key`` + payload returns the same
    bundle. Incomplete coverage still exports, flagged in ``coverage`` and ``warnings``.
    """
    purpose = " ".join(str(purpose).split())
    if not purpose or len(purpose) > 500:
        raise InvalidArgument("purpose must be 1..500 characters (e.g. 'Table 2, NeurIPS draft')")
    if not idempotency_key:
        raise InvalidArgument("idempotency_key is required")
    if (comparison_ref is None) == (planned_run_set_ref is None):
        raise InvalidArgument("pass exactly one of comparison_ref or planned_run_set_ref")
    if comparison_ref is not None and (evaluation_refs is not None or metric_names is not None):
        raise InvalidArgument(
            "evaluation_refs / metric_names only apply with planned_run_set_ref; a comparison already fixed them"
        )
    fmt = _formats(formats)
    spec = _coerce_table_spec(table_spec)
    payload = {
        "purpose": purpose,
        "comparison_ref": strip_ref(comparison_ref) if comparison_ref is not None else None,
        "planned_run_set_ref": planned_run_set_ref,
        "evaluation_refs": sorted(evaluation_refs) if evaluation_refs is not None else None,
        "metric_names": list(metric_names) if metric_names is not None else None,
        "formats": fmt,
        "table_spec": spec.model_dump(mode="json"),
    }
    payload_hash = digest(payload)
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
    if stored is not None:
        return EvidenceData.model_validate(stored)

    if planned_run_set_ref is not None:
        names = metric_names or _default_metric_names(ctx.db, planned_run_set_ref)
        try:
            comparison_id = results_compare(
                ctx,
                planned_run_set_ref=planned_run_set_ref,
                metric_names=names,
                evaluation_refs=evaluation_refs,
            ).comparison_id
        except InvalidArgument as exc:
            unknown = exc.details.get("unknown_metrics")
            if metric_names or not unknown:
                raise
            # default names that no evaluation reports can only be the contract's declarations
            # (names taken from the evaluations are reported by construction)
            known = exc.details.get("known_metrics") or []
            raise InvalidArgument(
                f"the experiment's comparison contract declares metric(s) {unknown} but none of the run "
                f"set's evaluations reports them; pass metric_names (known: {', '.join(known[:20]) or 'none'}) "
                "or fix the evaluator / the contract",
                details={**exc.details, "declared_by": "comparison_contract"},
            ) from None
    else:
        assert comparison_ref is not None
        comparison_id = strip_ref(comparison_ref)
    data, chains = load_archive(ctx.db, comparison_id)
    bundle_id = new_id("evidence")
    bundle = build_bundle(
        data,
        chains,
        bundle_id=bundle_id,
        purpose=purpose,
        generated_at=iso(ctx.clock.now()) or "",
        table=spec,
        rule_attempts=_rule_attempts(ctx.db, data),
    )
    files = render_files(bundle, fmt, spec)
    manifest_json = jdump(bundle)
    coverage, warnings, flags = _coverage(data, bundle)

    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
        if stored is not None:  # a concurrent request with the same key finished first
            return EvidenceData.model_validate(stored)
        tx.insert(
            "evidence_bundles",
            {
                "bundle_id": bundle_id,
                "comparison_id": data.comparison_id,
                "run_set_ref": data.run_set.ref,
                "purpose": purpose,
                "manifest_json": manifest_json,
                "files_json": jdump(files),
                "coverage_json": jdump(bundle["coverage"]),
                "created_at": tx.now,
            },
        )
        md = files.get("summary.md") or render_markdown(bundle)
        summary_md, truncated = _clip_utf8(md, SUMMARY_MAX_BYTES)
        if truncated:
            summary_md += f"\n\n…(truncated; read hpclab://evidence/{bundle_id}/summary.md)\n"
        result = EvidenceData(
            bundle_id=bundle_id,
            bundle_ref=f"hpclab://evidence/{bundle_id}",
            comparison_id=data.comparison_id,
            comparison_ref=data.comparison_ref,
            run_set_ref=data.run_set.ref,
            purpose=purpose,
            created_at=tx.now,
            files=_file_list(bundle_id, files),
            counts=EvidenceCounts(**{k: bundle["coverage"][k] for k in EvidenceCounts.model_fields}),
            incomparable=len(data.incomparable),
            integrity_flags=flags,
            summary_markdown=summary_md,
            summary_truncated=truncated,
            summary=_summary_line(bundle_id, data),
            coverage=coverage,
            warnings=warnings,
        )
        tx.event(
            "evidence",
            bundle_id,
            "evidence.created",
            {
                "comparison_id": data.comparison_id,
                "run_set_ref": data.run_set.ref,
                "purpose": purpose,
                "files": sorted(files),
                "complete": coverage.complete,
            },
        )
        tx.idem_store(ctx.principal, TOOL, idempotency_key, payload_hash, result.model_dump(mode="json"))
    return result


def evidence_get(
    ctx: ServiceContext,
    *,
    bundle_id: str,
    file: str | None = None,
    offset: int = 0,
    max_bytes: int | None = None,
) -> EvidenceGetData:
    """Bundle metadata, or one file's text (``file``; at most 64 KiB per call, continue at
    ``next_offset``). Resources: ``hpclab://evidence/<id>`` and ``hpclab://evidence/<id>/<file>``."""
    row = _bundle_row(ctx.db, bundle_id)
    files: dict[str, str] = jload(row["files_json"], {}) or {}
    bid = row["bundle_id"]
    cov = _stored_coverage(row)
    base = {
        "bundle_id": bid,
        "bundle_ref": f"hpclab://evidence/{bid}",
        "comparison_id": row["comparison_id"],
        "comparison_ref": f"hpclab://comparisons/{row['comparison_id']}" if row["comparison_id"] else None,
        "run_set_ref": row["run_set_ref"],
        "purpose": row["purpose"],
        "created_at": row["created_at"],
        "files": _file_list(bid, files),
        "coverage": cov,
    }
    if file is None:
        return EvidenceGetData(**base, summary=f"evidence bundle {bid}: {len(files)} file(s)")
    if file not in files:
        raise NotFound(
            f"bundle {bid} has no file {file!r}", details={"bundle_id": bid, "available": sorted(files)}
        )
    raw = files[file].encode("utf-8")
    if offset < 0 or offset > len(raw):
        raise InvalidArgument(f"offset must be within [0, {len(raw)}]")
    if offset < len(raw) and (raw[offset] & 0xC0) == 0x80:
        raise InvalidArgument("offset points inside a UTF-8 character; continue from next_offset")
    limit = MAX_TEXT_BYTES if max_bytes is None else clamp_bytes(max_bytes)
    end = min(len(raw), offset + limit)
    while offset < end < len(raw) and (raw[end] & 0xC0) == 0x80:
        end -= 1  # never split a code point
    if end == offset < len(raw):
        # the page is smaller than the character at ``offset``: return that one character whole
        # (at most 3 bytes over) rather than an empty page that would never advance
        end += 1
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end += 1
    return EvidenceGetData(
        **base,
        file=file,
        content=raw[offset:end].decode("utf-8"),
        offset=offset,
        size=len(raw),
        truncated=end < len(raw),
        next_offset=end if end < len(raw) else None,
        summary=f"{file} of evidence bundle {bid} (bytes {offset}-{end} of {len(raw)})",
    )


def evidence_write(
    ctx: ServiceContext, *, bundle_id: str, out_dir: str, overwrite: bool = False
) -> EvidenceWriteData:
    """Write every file of a bundle into ``out_dir`` (absolute; created if needed).

    Files are written atomically; an existing file with identical content is left alone; one
    with different content is refused unless ``overwrite`` (no file is written in that case).
    """
    row = _bundle_row(ctx.db, bundle_id)
    files: dict[str, str] = jload(row["files_json"], {}) or {}
    target = Path(out_dir).expanduser()
    if not target.is_absolute():
        raise InvalidArgument(
            "out_dir must be an absolute path (the controller's working directory is not the caller's)"
        )
    if target.exists() and not target.is_dir():
        raise PreconditionFailed(f"{target} exists and is not a directory")
    plan: list[tuple[str, Path, bytes, bool]] = []
    conflicts: list[str] = []
    for name in sorted(files):
        data = files[name].encode("utf-8")
        path = target / name
        same = path.is_file() and path.read_bytes() == data
        if path.exists() and not same and not overwrite:
            conflicts.append(str(path))
        plan.append((name, path, data, not same))
    if conflicts:
        raise PreconditionFailed(
            "files with different content already exist; pass overwrite=true to replace them",
            details={"conflicts": conflicts},
        )
    target.mkdir(parents=True, exist_ok=True)
    written: list[WrittenFile] = []
    for name, path, data, needed in plan:
        if needed:
            _atomic_write(path, data)
        written.append(
            WrittenFile(
                name=name,
                path=str(path),
                bytes=len(data),
                sha256="sha256:" + sha256_hex(data),
                written=needed,
            )
        )
    with ctx.db.tx(source="api") as tx:
        tx.event(
            "evidence",
            row["bundle_id"],
            "evidence.written",
            {"out_dir": str(target), "files": [w.name for w in written if w.written]},
        )
    return EvidenceWriteData(
        bundle_id=row["bundle_id"],
        out_dir=str(target),
        files=written,
        summary=f"wrote {sum(w.written for w in written)} of {len(written)} file(s) to {target}",
    )


def load_bundle(reader: Reader, bundle_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(bundle manifest dict, evidence_bundles row) for a bundle id or ``hpclab://evidence/<id>``."""
    row = _bundle_row(reader, bundle_ref)
    manifest = jload(row["manifest_json"], {}) or {}
    if not isinstance(manifest, dict):
        raise PreconditionFailed(f"bundle {row['bundle_id']} has an unreadable manifest")
    return manifest, row


def outbox_handlers() -> dict[str, Any]:
    """Exports are bounded local renders done in the request; no background work."""
    return {}


def periodic_tasks() -> list[Any]:
    return []


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def _seed_value(key: str) -> int | str:
    try:
        return int(key)
    except ValueError:
        return key


def build_bundle(
    data: ComparisonData,
    chains: Mapping[str, ValueChain],
    *,
    bundle_id: str,
    purpose: str,
    generated_at: str,
    table: TableSpec,
    rule_attempts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The ``hpclab.evidence/v1`` dict for an archived comparison (validated by the renderers).

    ``rule_attempts`` (run id → attempt) is what the contract's ``attempt_selection`` would have
    picked for the runs with several valid attempts when the comparison pinned
    ``evaluation_refs`` (:func:`_rule_attempts`); unknown runs are disclosed as pinned without
    saying whether the pin overrode the rule.
    """
    runs = {r.run_id: r for r in data.runs}
    groups = []
    provenance = []
    for g in data.groups:
        cells: dict[str, Any] = {}
        for m, c in g.metrics.items():
            cells[m] = {
                "n": c.n,
                "expected_n": c.expected_n,
                "mean": c.mean,
                "std": c.std,
                "sem": c.sem,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "min": c.min,
                "max": c.max,
                "values_by_seed": dict(c.values_by_seed),
                "missing_seeds": list(c.missing_seeds),
                "failed_seeds": list(c.failed_seeds),
                "excluded": [e.model_dump(mode="json") for e in c.excluded],
            }
            for s, run_id in c.run_by_seed.items():
                chain = chains.get(run_id)
                if chain is None:
                    continue  # never executed: nothing to trace
                value = c.values_by_seed.get(s)
                if value is None:
                    run = runs.get(run_id)
                    value = run.values.get(m) if run is not None else None
                provenance.append(_provenance_row(g.label, s, m, value, chain))
        groups.append({"key": g.key, "label": g.label, "is_baseline": g.is_baseline, "metrics": cells})
    counts = data.counts
    bundle: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "title": _title(data),
        "purpose": purpose,
        "generated_at": generated_at,
        "hpclab_version": __version__,
        "ci_level": data.contract.ci_level,
        "experiment": {
            "id": data.experiment.id,
            "name": data.experiment.name,
            "question": data.experiment.question,
        },
        "contract": {
            "treatment": list(data.treatment_axes),
            "invariants": list(data.contract.invariants),
            "baseline": dict(data.contract.baseline) if data.contract.baseline else None,
        },
        "metric_decls": [
            {"name": i.name, "higher_is_better": i.higher_is_better, "unit": i.unit} for i in data.metric_info
        ],
        "groups": groups,
        "differences": [
            {
                "group_label": d.group_label,
                "baseline_label": d.baseline_label,
                "metric": d.metric,
                "kind": d.kind,
                "n": d.n,
                "n_baseline": d.n_baseline,
                "mean_diff": d.mean_diff,
                "ci_low": d.ci_low,
                "ci_high": d.ci_high,
            }
            for d in data.differences
        ],
        "coverage": {
            "planned_runs": counts.planned_runs,
            "included": counts.included,
            "missing": counts.missing,
            "failed": counts.failed,
            "excluded": counts.excluded,
            "complete": counts.included == counts.planned_runs,
            "notes": [*data.coverage.notes, *_gap_notes(data.runs)],
        },
        "provenance": provenance,
        "incomparable": [
            {
                "reason": i.reason,
                "details": {**i.details, **({"metric": i.metric} if i.metric else {})} or None,
            }
            for i in data.incomparable
        ],
        "limitations": list(data.warnings),
        "integrity": _integrity(data, rule_attempts or {}),
        # evidence-only keys (ignored by the renderers)
        "bundle_id": bundle_id,
        "comparison_id": data.comparison_id,
        "comparison_ref": data.comparison_ref,
        "comparison_content_hash": data.content_hash,
        "run_set_ref": data.run_set.ref,
        # hand-picked evaluations (results_compare evaluation_refs) or null: the contract's
        # attempt_selection did not choose them
        "evaluation_refs": list(data.evaluation_refs) if data.evaluation_refs is not None else None,
        "table": table.model_dump(mode="json"),
        "runs": [r.model_dump(mode="json") for r in data.runs],
    }
    bundle["cell_ids"] = {
        base: {"group_label": label, "metric": metric} for (label, metric), base in cell_ids(bundle).items()
    }
    return bundle


def _provenance_row(
    label: str, seed: str, metric: str, value: float | None, chain: ValueChain
) -> dict[str, Any]:
    return {
        "group_label": label,
        "seed": _seed_value(seed),
        "metric": metric,
        "value": value,
        "evaluation_id": chain.evaluation_id,
        "evaluator": chain.evaluator,
        "artifact_id": chain.artifact_id,
        "artifact_digest": chain.artifact_digest,
        "attempt_id": chain.attempt_id,
        "job_id": chain.job_id,
        "cluster_id": chain.cluster_id,
        "run_id": chain.run_id,
        "run_spec_hash": chain.run_spec_hash,
        "code_digest": chain.code_digest,
        "git_commit": chain.git_commit,
        "env_gpu": chain.env_gpu,
        "env_torch": chain.env_torch,
        # extra chain fields (ignored by the renderers, used by trace/verify)
        "attempt_seq": chain.attempt_seq,
        "exec_plan_hash": chain.exec_plan_hash,
        "job_owner": chain.job_owner,
        "job_submit_time": chain.job_submit_time,
        "config_hash": chain.config_hash,
        "artifact_link": chain.artifact_link,
        "plan_id": chain.plan_id,
    }


def _gap_notes(runs: Sequence[RunCoverage]) -> list[str]:
    """One line per missing/failed run: ``<group> seed <s>: <status> (<code>): <reason>``."""
    lines = [
        f"{r.group_label} seed {seed_key(r.seed)}: {r.status} ({r.code}): {r.reason or r.code}"
        for r in runs
        if r.status in ("missing", "failed")
    ]
    if len(lines) > MAX_GAP_NOTES:
        more = len(lines) - MAX_GAP_NOTES
        lines = [*lines[:MAX_GAP_NOTES], f"{more} more missing/failed run(s): see runs in manifest.json"]
    return lines


def _title(data: ComparisonData) -> str:
    name = data.experiment.name or data.run_set.id
    axes = ", ".join(data.treatment_axes) or "group"
    return f"{name}: {', '.join(data.metric_names)} by {axes}"


def _integrity(data: ComparisonData, rule_attempts: Mapping[str, str]) -> dict[str, Any]:
    integ = data.integrity
    pinned = data.evaluation_refs is not None
    notes = []
    if integ.selection.evaluation_selection_on_split:
        notes.append(
            f"{len(integ.selection.evaluation_selection_on_split)} evaluation(s) selected the best record on "
            "the reported split (selection_on_split)"
        )
    notes += [f"{b.group_label}/{b.metric}: n={b.n} is below min_n={b.min_n}" for b in integ.below_min_n]
    early = [e for e in integ.exclusions if not e.observed_before_exclusion]
    if early:
        notes.append(f"{len(early)} exclusion(s) were declared before the result was observed")
    awaiting = [r for r in data.runs if awaiting_evaluation(r)]
    if awaiting:
        seeds = "; ".join(f"{r.group_label} seed {seed_key(r.seed)}" for r in awaiting[:20])
        notes.append(
            f"{len(awaiting)} planned run(s) had completed but were not evaluated yet when compared "
            f"({seeds}{'; ...' if len(awaiting) > 20 else ''}): they are counted as missing, not failed, "
            "and the numbers are provisional until they are evaluated"
        )
    return {
        "attempt_selection": integ.attempt_selection,
        "post_hoc": [
            {
                "group_label": p.group_label,
                "seed": seed_key(p.seed) if p.seed is None else p.seed,
                "run_id": p.run_id,
            }
            for p in integ.post_hoc
        ],
        "late_exclusions": [
            {
                "group_label": e.group_label,
                "seed": seed_key(e.seed) if e.seed is None else e.seed,
                "run_id": e.run_id,
                "reason": e.reason,
            }
            for e in integ.exclusions
            if e.observed_before_exclusion
        ],
        "heterogeneity": [
            {
                "dimension": h.dimension,
                "values": h.values,
                "unknown": h.unknown,
                "within_groups": h.within_groups,
            }
            for h in integ.heterogeneity
        ],
        # pinned evaluation_refs bypass attempt_selection: the pin chose, not the contract's rule
        "evaluations_pinned": pinned,
        "multiple_valid_attempts": [
            _multi_attempt(m, pinned, rule_attempts) for m in integ.multiple_valid_attempts
        ],
        "notes": notes,
    }


def _multi_attempt(m: MultiValidAttempt, pinned: bool, rule_attempts: Mapping[str, str]) -> dict[str, Any]:
    """One run with several valid attempts: which attempt is reported and what chose it."""
    rule_attempt = rule_attempts.get(m.run_id) if pinned else m.selected_attempt_id
    return {
        "group_label": m.group_label,
        "seed": seed_key(m.seed) if m.seed is None else m.seed,
        "run_id": m.run_id,
        "valid_attempts": m.valid_attempts,
        "selected_attempt_id": m.selected_attempt_id,
        "rule": PINNED_RULE if pinned else m.rule,
        "rule_attempt_id": rule_attempt,
        "overrides_rule": rule_attempt is not None and rule_attempt != m.selected_attempt_id,
    }


def _rule_attempts(reader: Reader, data: ComparisonData) -> dict[str, str]:
    """For a comparison with pinned ``evaluation_refs``: run id → the attempt the contract's
    ``attempt_selection`` would have picked, for each run with several valid attempts.

    The contract's own selection (:func:`select_evaluations`, any evaluator) over the evaluations
    that existed when the comparison was computed, so a later attempt does not change what the
    frozen table discloses.
    """
    if data.evaluation_refs is None:
        return {}
    run_ids = sorted({m.run_id for m in data.integrity.multiple_valid_attempts})
    if not run_ids:
        return {}
    cutoff = data.observed_at or data.created_at or None
    existing: set[str] | None = None
    if cutoff is not None:
        existing = set()
        for i in range(0, len(run_ids), 500):
            chunk = run_ids[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            existing.update(
                r["evaluation_id"]
                for r in reader.all(
                    f"SELECT evaluation_id FROM evaluations WHERE run_id IN ({marks}) AND created_at <= ?",
                    [*chunk, cutoff],
                )
            )
    chosen = select_evaluations(
        reader, run_ids, attempt_selection=data.contract.attempt_selection, evaluation_ids=existing
    )
    return {rid: str(row["attempt_id"]) for rid, row in chosen.items()}


def render_files(bundle: Mapping[str, Any], formats: Sequence[str], table: TableSpec) -> dict[str, str]:
    """Render the requested formats (``manifest.json`` is always included: it is the bundle).

    Renderer ``InvalidArgument`` (an inconsistent bundle) propagates: a wrong table is never
    rendered."""
    files: dict[str, str] = {}
    if "md" in formats:
        files["summary.md"] = render_markdown(bundle)
    if "latex" in formats:
        treatment = bundle["contract"]["treatment"]
        header = treatment[0][:1].upper() + treatment[0][1:] if len(treatment) == 1 else "Group"
        files["table.tex"] = render_latex(
            bundle,
            digits=table.decimals,
            bold_best=table.bold_best,
            caption=table.caption,
            label=table.label,
            metrics=table.cols,
            group_header=header,
            hpcval_bundle_id=bundle["bundle_id"],
            groups=table.rows,
            spread="std" if table.stat == "mean_std" else "sem",
            scale=table.scale,
            min_n=table.min_n,
        )
    if "csv" in formats:
        files["table.csv"] = render_csv(bundle)
    files["manifest.json"] = (
        json.dumps(bundle, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _formats(formats: Sequence[str] | None) -> list[str]:
    if formats is None:
        return sorted(FORMAT_FILES)
    unknown = sorted(set(formats) - set(FORMAT_FILES))
    if unknown:
        raise InvalidArgument(f"unknown formats {unknown}; choose from {sorted(FORMAT_FILES)}")
    if not formats:
        raise InvalidArgument("formats is empty; omit it to export every format")
    return sorted(set(formats))


def _coerce_table_spec(spec: TableSpec | Mapping[str, Any] | None) -> TableSpec:
    if spec is None:
        return TableSpec()
    if isinstance(spec, TableSpec):
        return spec
    try:
        return TableSpec.model_validate(dict(spec))
    except ValueError as exc:
        raise InvalidArgument(f"invalid table_spec: {exc}") from None


def _default_metric_names(reader: Reader, planned_run_set_ref: str) -> list[str]:
    rs = resolve_run_set(reader, planned_run_set_ref)
    exp = reader.one("SELECT spec_json FROM experiments WHERE experiment_id=?", (rs.experiment_id,))
    spec = jload(exp["spec_json"], {}) if exp else {}
    comparison = spec.get("comparison") if isinstance(spec, dict) else None
    declared = [
        m["name"]
        for m in (comparison or {}).get("metrics") or []
        if isinstance(m, dict) and isinstance(m.get("name"), str)
    ]
    if declared:
        return declared[:MAX_METRICS]
    names: set[str] = set()
    ids = rs.run_ids
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        for row in reader.all(
            f"SELECT metrics_json FROM evaluations WHERE status='valid' AND run_id IN ({marks})", chunk
        ):
            for m in jload(row["metrics_json"], []) or []:
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    names.add(m["name"])
    if not names:
        raise PreconditionFailed(
            "no metric_names given, the contract declares none and no valid evaluation exists yet",
            details={"remediation": "collect evaluations first, or pass metric_names"},
        )
    if len(names) > MAX_METRICS:
        raise InvalidArgument(
            f"the evaluations report {len(names)} metrics; pass metric_names (at most {MAX_METRICS})",
            details={"metrics": sorted(names)[:100]},
        )
    return sorted(names)


def _coverage(data: ComparisonData, bundle: Mapping[str, Any]) -> tuple[Coverage, list[str], list[str]]:
    counts = data.counts
    complete = counts.included == counts.planned_runs
    warnings: list[str] = []
    flags: list[str] = []
    if not complete:
        warnings.append(
            f"coverage incomplete: {counts.included} of {counts.planned_runs} planned runs included "
            f"({counts.missing} missing, {counts.failed} failed, {counts.excluded} excluded); every gap is "
            "listed and footnoted in the bundle"
        )
    gap_cells = [
        f"{g['label']}/{m}"
        for g in bundle["groups"]
        for m, c in g["metrics"].items()
        if c["n"] < c["expected_n"]
    ]
    if complete and gap_cells:
        warnings.append(
            f"all runs included but {len(gap_cells)} metric cell(s) lack values: {gap_cells[:10]}"
        )
    if data.incomparable:
        flags.append("incomparable")
        warnings.append(
            f"{len(data.incomparable)} incomparability issue(s): "
            + "; ".join(i.reason for i in data.incomparable[:5])
        )
    integ = bundle["integrity"]
    if integ["post_hoc"]:
        flags.append("post_hoc_seeds")
        warnings.append(f"{len(integ['post_hoc'])} run(s) were added post hoc (after results existed)")
    if integ["late_exclusions"]:
        flags.append("late_exclusions")
        warnings.append(
            f"{len(integ['late_exclusions'])} exclusion(s) were decided after the result was observed"
        )
    mixed = [h["dimension"] for h in integ["heterogeneity"] if len(h["values"]) > 1]
    if mixed:
        flags.append("heterogeneous_environment")
        warnings.append(f"included runs differ in {', '.join(mixed)}")
    if integ["multiple_valid_attempts"]:
        flags.append("multiple_valid_attempts")
    if integ.get("evaluations_pinned"):
        flags.append("evaluations_pinned")
        warnings.append(_pinned_warning(data, integ))
    awaiting = [r for r in data.runs if awaiting_evaluation(r)]
    if awaiting:
        flags.append("awaiting_evaluation")
        warnings.append(
            f"{len(awaiting)} planned run(s) completed but had no valid evaluation yet (E) when compared: "
            "counted as missing, not failed, so the table is provisional; run evaluation_collect, then "
            "results_compare and evidence_export again"
        )
    for w in data.warnings:
        if w not in warnings:
            warnings.append(w)
    cov = Coverage(
        complete=complete and not gap_cells,
        sources=[SourceStatus(name="ledger", status="ok")],
        missing=list(data.coverage.missing),
        notes=list(data.coverage.notes),
    )
    return cov, warnings, flags


def _pinned_warning(data: ComparisonData, integ: Mapping[str, Any]) -> str:
    rule = integ.get("attempt_selection") or "attempt_selection"
    text = (
        f"the evaluations were pinned by the caller (evaluation_refs: {len(data.evaluation_refs or [])} "
        f"evaluation(s)), not chosen by the contract's attempt_selection {rule}; newer attempts awaiting "
        "evaluation were not checked"
    )
    over = [m for m in integ["multiple_valid_attempts"] if m.get("overrides_rule")]
    if over:
        who = "; ".join(f"{m['group_label']} seed {m['seed']}" for m in over[:20])
        text += (
            f"; {len(over)} run(s) report an attempt other than the one {rule} picks "
            f"({who}{'; ...' if len(over) > 20 else ''})"
        )
    return text


def _stored_coverage(row: Mapping[str, Any]) -> Coverage:
    """Same completeness as ``evidence_export`` reported: run level AND every metric cell full."""
    raw = jload(row["coverage_json"], {}) or {}
    complete = bool(raw.get("complete"))
    notes = [str(n) for n in raw.get("notes") or []]
    if not complete:
        notes.insert(
            0,
            f"{raw.get('included')} of {raw.get('planned_runs')} planned runs included "
            f"({raw.get('missing')} missing, {raw.get('failed')} failed, {raw.get('excluded')} excluded)",
        )
    manifest = jload(row.get("manifest_json"), {}) or {}
    gaps = [
        f"{g.get('label')}/{m}"
        for g in (manifest.get("groups") or [] if isinstance(manifest, dict) else [])
        if isinstance(g, dict)
        for m, c in (g.get("metrics") or {}).items()
        if isinstance(c, dict) and isinstance(c.get("n"), int) and c["n"] < (c.get("expected_n") or 0)
    ]
    if gaps:
        complete = False
        notes.append(f"{len(gaps)} metric cell(s) have fewer values than planned: {', '.join(gaps[:10])}")
    return Coverage(complete=complete, sources=[SourceStatus(name="ledger", status="ok")], notes=notes)


def _bundle_row(reader: Reader, bundle_ref: str) -> dict[str, Any]:
    raw = (bundle_ref or "").strip()
    if raw.startswith("hpclab://evidence/"):
        raw = raw[len("hpclab://evidence/") :].split("/", 1)[0]
    bid = strip_ref(raw)
    if kind_of(bid) != "evidence":
        raise InvalidArgument(f"not an evidence bundle id: {bundle_ref!r}")
    row = reader.one("SELECT * FROM evidence_bundles WHERE bundle_id=?", (bid,))
    if row is None:
        raise NotFound(f"evidence bundle {bid} not found", details={"bundle_id": bid})
    return row


def _file_list(bundle_id: str, files: Mapping[str, str]) -> list[EvidenceFile]:
    out = []
    for name in sorted(files):
        data = files[name].encode("utf-8")
        out.append(
            EvidenceFile(
                name=name,
                uri=f"hpclab://evidence/{bundle_id}/{name}",
                media_type=MEDIA_TYPES.get(name, "text/plain"),
                bytes=len(data),
                sha256="sha256:" + sha256_hex(data),
            )
        )
    return out


def _clip_utf8(text: str, limit: int) -> tuple[str, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False
    end = limit
    while end > 0 and (raw[end] & 0xC0) == 0x80:
        end -= 1
    return raw[:end].decode("utf-8"), True


def _summary_line(bundle_id: str, data: ComparisonData) -> str:
    c = data.counts
    text = (
        f"evidence bundle {bundle_id}: {len(data.groups)} group(s) × {len(data.metric_names)} metric(s), "
        f"{c.included}/{c.planned_runs} planned runs included"
    )
    if c.included < c.planned_runs:
        text += f" ({c.missing} missing, {c.failed} failed, {c.excluded} excluded)"
    if data.incomparable:
        text += f"; {len(data.incomparable)} incomparability issue(s)"
    return text


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
