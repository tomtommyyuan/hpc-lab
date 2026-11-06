"""Pure matrix expansion for ``experiment_prepare`` (build spec §7.2 prepare, "Matrix").

A matrix is a finite product: ``cells = product(non-seed axes) x seeds``. Everything here is
deterministic and side-effect free so the same spec always yields the same cells, labels,
cell keys and argv (the plan hash depends on it):

* axes are iterated in **sorted axis-name order** (a spec reloaded from canonical JSON has its
  keys sorted, so insertion order cannot be relied on), values in the order given, and seeds
  innermost: ``method=a seed=1, method=a seed=2, method=b seed=1, ...``;
* a cell's identity is ``cell_key`` = canonical JSON of its coordinates *including* the seed
  (``runs.cell_key``); ``1`` and ``1.0`` are different JSON values and different cells;
* labels look like ``method-mixup.seed-1`` (values sanitized, bounded, made unique with a short
  hash suffix only when sanitizing/truncating makes two labels collide);
* argv placeholders ``{name}`` are rendered from params (fixed ``parameters`` overlaid by the
  cell's axis values) and ``{seed}``; runtime placeholders (``{output_dir}`` ... ``{job_id}``)
  are kept for the compute node; ``${...}`` (shell/Hydra interpolation) is never a
  placeholder; any other ``{identifier}`` is *unknown* and makes the plan invalid;
* hard limit: at most ``MAX_MATRIX_CELLS`` (256) main cells.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hpclab.domain.errors import InvalidArgument, LimitExceeded
from hpclab.domain.hashing import canonical_json, sha256_hex
from hpclab.domain.ids import sanitize_label
from hpclab.domain.models import (
    MAX_MATRIX_CELLS,
    MAX_SEEDS,
    RUNTIME_PLACEHOLDERS,
    ExperimentSpec,
    Scalar,
    SmokeSpec,
)

SEED_AXIS = "seed"
MAX_LABEL_LEN = 120
_LABEL_PART_LEN = 40

# ``{identifier}`` not preceded by ``$`` (``${VAR}`` belongs to the shell / Hydra).
_PLACEHOLDER_RE = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def render_value(value: Scalar) -> str:
    """How a parameter value appears in argv.

    ``bool`` -> ``true``/``false`` (JSON spelling, not Python's ``True``), ``float`` -> the
    shortest round-tripping repr (``0.001``, ``1e-05``), ``None`` -> empty string (reported as
    a warning by the placeholder check: an empty argv element is rarely intended).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidArgument(f"parameter value {value!r} is not a finite number")
        return repr(value)
    return str(value)


def cell_key(cell: Mapping[str, Scalar]) -> str:
    """``runs.cell_key``: canonical JSON of the cell coordinates (incl. the seed)."""
    return canonical_json(dict(cell))


def as_int_seed(value: Any, *, what: str = "seed") -> int:
    """An integer seed from ``3`` / ``3.0``; anything else (bool, str, fractional, NaN/inf)
    raises ``InvalidArgument`` (never ``ValueError``/``OverflowError``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidArgument(f"{what} values must be integers, got {value!r}")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise InvalidArgument(f"{what} value {value!r} is not a finite integer")
    return int(value)


def seed_values(spec: ExperimentSpec) -> list[int | None]:
    """Seeds from ``spec.seeds`` or ``matrix.seed``; ``[None]`` when the spec has no seeds.

    ``matrix.seed`` values must be integers (``3`` or ``3.0``); anything else (strings, bools,
    fractional numbers) raises ``InvalidArgument`` - a seed that is not an integer cannot be
    exported as ``HPCLAB_SEED`` or paired across treatments.
    """
    if spec.seeds:
        return list(spec.seeds)
    if SEED_AXIS in spec.matrix:
        out: list[int | None] = []
        for raw in spec.matrix[SEED_AXIS]:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise InvalidArgument(
                    f"matrix.seed values must be integers, got {raw!r}",
                    details={"matrix.seed": list(spec.matrix[SEED_AXIS])},
                )
            if isinstance(raw, float) and not raw.is_integer():
                raise InvalidArgument(f"matrix.seed value {raw!r} is not an integer")
            out.append(int(raw))
        if len(set(out)) != len(out):
            raise InvalidArgument("matrix.seed has duplicate values", details={"seeds": out})
        return out
    return [None]


def axis_names(matrix: Mapping[str, Sequence[Scalar]]) -> list[str]:
    """Non-seed axes in deterministic (sorted) order."""
    return sorted(k for k in matrix if k != SEED_AXIS)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One matrix cell: coordinates of the non-seed axes + the seed (or None)."""

    coords: dict[str, Scalar]
    seed: int | None
    label: str = ""

    @property
    def cell(self) -> dict[str, Scalar]:
        """Matrix coordinates including the seed (what ``RunSpec.cell`` stores)."""
        out: dict[str, Scalar] = dict(self.coords)
        if self.seed is not None:
            out[SEED_AXIS] = self.seed
        return out

    @property
    def key(self) -> str:
        return cell_key(self.cell)


def _label_part(value: Scalar) -> str:
    return sanitize_label(render_value(value) or "none", _LABEL_PART_LEN)


def raw_label(coords: Mapping[str, Scalar], seed: int | None, *, default: str = "run") -> str:
    """``axis-value`` parts joined by ``.``; axes sorted, seed last; bounded length."""
    parts = [f"{sanitize_label(k, _LABEL_PART_LEN)}-{_label_part(coords[k])}" for k in sorted(coords)]
    if seed is not None:
        parts.append(f"seed-{seed}")
    label = ".".join(parts) if parts else sanitize_label(default, _LABEL_PART_LEN)
    return label


def _unique_labels(cells: Sequence[Cell]) -> list[Cell]:
    """Assign labels; disambiguate collisions/truncations with a short cell-key hash."""
    raws = [raw_label(c.coords, c.seed, default=c.label or "run") for c in cells]
    counts: dict[str, int] = {}
    for r in raws:
        counts[r] = counts.get(r, 0) + 1
    out: list[Cell] = []
    for c, raw in zip(cells, raws, strict=True):
        label = raw
        if counts[raw] > 1 or len(raw) > MAX_LABEL_LEN:
            suffix = sha256_hex(c.key)[:6]
            label = f"{raw[: MAX_LABEL_LEN - 7]}~{suffix}"
        out.append(Cell(coords=c.coords, seed=c.seed, label=label))
    return out


def expand_cells(
    matrix: Mapping[str, Sequence[Scalar]],
    seeds: Sequence[int | None],
    *,
    default_label: str = "run",
    max_cells: int = MAX_MATRIX_CELLS,
) -> list[Cell]:
    """``product(non-seed axes) x seeds`` in deterministic order, labelled.

    Raises ``LimitExceeded`` above ``max_cells`` (checked *before* expanding) and
    ``InvalidArgument`` for an empty axis or duplicate values within an axis (compared by
    canonical JSON, so ``1`` and ``1.0`` are distinct but ``"a"`` twice is not).
    """
    names = axis_names(matrix)
    seeds = list(seeds) or [None]
    if len(seeds) > MAX_SEEDS:
        raise LimitExceeded(f"at most {MAX_SEEDS} seeds", details={"seeds": len(seeds)})
    count = len(seeds)
    for name in names:
        values = list(matrix[name])
        if not values:
            raise InvalidArgument(f"matrix axis {name!r} is empty")
        keys = [canonical_json(v) for v in values]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise InvalidArgument(f"matrix axis {name!r} has duplicate values", details={"duplicates": dupes})
        count *= len(values)
    if count > max_cells:
        raise LimitExceeded(
            f"matrix expands to {count} cells; hard limit is {max_cells}",
            details={"cells": count, "limit": max_cells, "axes": {n: len(matrix[n]) for n in names}},
        )
    cells: list[Cell] = []
    for combo in itertools.product(*(list(matrix[n]) for n in names)):
        coords = dict(zip(names, combo, strict=True))
        for seed in seeds:
            cells.append(Cell(coords=coords, seed=seed, label=default_label))
    return _unique_labels(cells)


def expand_spec(spec: ExperimentSpec) -> list[Cell]:
    """All main cells of a spec (see ``expand_cells``)."""
    matrix = {k: v for k, v in spec.matrix.items() if k != SEED_AXIS}
    return expand_cells(matrix, seed_values(spec), default_label=spec.name)


def cell_params(spec: ExperimentSpec, cell: Cell) -> dict[str, Scalar]:
    """Params of a cell: fixed ``parameters`` overlaid by the cell's axis values (seed excluded:
    it is ``RunSpec.seed`` / ``HPCLAB_SEED``)."""
    params: dict[str, Scalar] = dict(spec.parameters)
    params.update(cell.coords)
    return params


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def find_placeholders(text: str) -> list[str]:
    """``{name}`` placeholders in ``text`` (``${...}`` excluded), in order of appearance."""
    return [m.group(1) for m in _PLACEHOLDER_RE.finditer(text)]


@dataclass
class RenderResult:
    argv: list[str]
    used: set[str] = field(default_factory=set)  # param names (and "seed") that were substituted
    unknown: list[str] = field(default_factory=list)  # placeholders with no value (plan invalid)
    nulls: list[str] = field(default_factory=list)  # params rendered from None (empty string)

    @property
    def ok(self) -> bool:
        return not self.unknown


def render_argv(template: Iterable[str], params: Mapping[str, Scalar], seed: int | None) -> RenderResult:
    """Substitute ``{param}``/``{seed}``; keep runtime placeholders; collect unknown ones.

    A param named like a runtime placeholder is *not* substituted (runtime wins); the spec
    check reports that collision separately.
    """
    result = RenderResult(argv=[])
    unknown: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in RUNTIME_PLACEHOLDERS:
            return m.group(0)
        if name == SEED_AXIS:
            if seed is None:
                unknown.append(name)
                return m.group(0)
            result.used.add(name)
            return str(seed)
        if name in params:
            result.used.add(name)
            value = params[name]
            if value is None and name not in result.nulls:
                result.nulls.append(name)
            return render_value(value)
        unknown.append(name)
        return m.group(0)

    for element in template:
        result.argv.append(_PLACEHOLDER_RE.sub(sub, element))
    result.unknown = list(dict.fromkeys(unknown))
    return result


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def smoke_enabled(spec: ExperimentSpec) -> bool:
    return spec.smoke is not None and spec.smoke.enabled


def smoke_cell(cells: Sequence[Cell], smoke: SmokeSpec) -> Cell:
    """The cell the smoke run exercises: ``smoke.cell`` (partial coordinates allowed; the
    first matching cell in matrix order wins) or the first cell.

    Raises ``InvalidArgument`` when ``smoke.cell`` matches no cell.
    """
    if not cells:
        raise InvalidArgument("the matrix has no cells to smoke-test")
    if not smoke.cell:
        return cells[0]
    wanted = {k: canonical_json(v) for k, v in smoke.cell.items()}
    for c in cells:
        coords = c.cell
        if all(k in coords and canonical_json(coords[k]) == v for k, v in wanted.items()):
            return c
    raise InvalidArgument(
        f"smoke.cell {dict(smoke.cell)!r} matches no matrix cell",
        details={"smoke_cell": dict(smoke.cell), "example_cell": cells[0].cell},
    )


def smoke_label(label: str) -> str:
    return f"smoke.{label}"


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def effective_identity(argv: Sequence[str], params: Mapping[str, Scalar], seed: int | None) -> str:
    """What the *program* can observe of a cell: argv, the string forms of the params and the
    seed. Two cells with the same identity are scientifically the same run (e.g. an axis with
    ``1`` and ``"1"``, which are different JSON values but identical on the command line)."""
    return canonical_json(
        {"argv": list(argv), "params": {k: render_value(v) for k, v in params.items()}, "seed": seed}
    )


def find_duplicates(identities: Sequence[tuple[str, str]]) -> list[list[str]]:
    """Groups (>= 2) of labels sharing one effective identity; input ``(label, identity)``."""
    groups: dict[str, list[str]] = {}
    for label, ident in identities:
        groups.setdefault(ident, []).append(label)
    return [g for g in groups.values() if len(g) > 1]


def extend_matrix(
    spec: ExperimentSpec,
    *,
    add_seeds: Sequence[int] | None,
    add_matrix: Mapping[str, Sequence[Scalar]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spec dict extended with new seeds / axis values, plus a record of what was added.

    New values are appended (existing ones are kept in place so existing cells keep their
    identity); values already present are ignored. A *new* axis is refused: it would change
    the coordinates (and the meaning) of every existing cell. For the same reason seeds cannot
    be added to an experiment whose cells have no seed.
    """
    data = spec.model_dump(mode="json")
    added: dict[str, Any] = {"seeds": [], "matrix": {}}
    seeds_new = [as_int_seed(s, what="add_seeds") for s in (add_seeds or [])]
    for axis, values in (add_matrix or {}).items():
        if axis == SEED_AXIS:
            seeds_new.extend(as_int_seed(v, what="add_matrix.seed") for v in values)
            continue
        if axis not in spec.matrix:
            raise InvalidArgument(
                f"add_matrix axis {axis!r} is not an axis of the experiment",
                details={
                    "axes": axis_names(spec.matrix),
                    "hint": "a new axis changes every existing cell: prepare a new experiment "
                    "(or derive_from a run) instead",
                },
            )
        existing = [canonical_json(v) for v in data["matrix"][axis]]
        for v in values:
            if canonical_json(v) not in existing:
                data["matrix"][axis].append(v)
                existing.append(canonical_json(v))
                added["matrix"].setdefault(axis, []).append(v)
    if seeds_new and seed_values(spec) == [None]:
        # The existing runs have no seed (HPCLAB_SEED empty; the program used its own default).
        # Seeded cells next to them would pool a seedless run with seeded ones in one treatment
        # group and break every expected-n count.
        raise InvalidArgument(
            "the experiment has no seeds: its existing runs are unseeded, so seeds cannot be added to it",
            details={
                "add_seeds": seeds_new,
                "hint": "prepare a new experiment with seeds (or matrix.seed) for a seeded comparison",
            },
        )
    if seeds_new:
        if SEED_AXIS in spec.matrix:
            target = data["matrix"][SEED_AXIS]
        else:
            target = data["seeds"]
        present = {int(s) for s in target}
        for s in seeds_new:
            if s not in present:
                target.append(s)
                present.add(s)
                added["seeds"].append(s)
    return data, added
