"""Pure matrix expansion: ordering, labels, cell keys, seeds, placeholders, limits, smoke."""

from __future__ import annotations

from typing import Any

import pytest

from hpclab.domain.errors import InvalidArgument, LimitExceeded
from hpclab.domain.models import EntrypointSpec, ExperimentSpec, SmokeSpec
from hpclab.services import matrix as mx


def spec(**kw: Any) -> ExperimentSpec:
    data: dict[str, Any] = {"name": "exp", "cluster": "c", "entrypoint": {"argv": ["python", "t.py"]}}
    data.update(kw)
    return ExperimentSpec.model_validate(data)


def test_no_matrix_no_seeds_is_one_unlabelled_cell() -> None:
    cells = mx.expand_spec(spec())
    assert len(cells) == 1
    assert cells[0].seed is None and cells[0].coords == {} and cells[0].cell == {}
    assert cells[0].label == "exp" and cells[0].key == "{}"


def test_seeds_only_innermost_and_labels() -> None:
    cells = mx.expand_spec(spec(matrix={"method": ["b", "a"]}, seeds=[3, 1]))
    assert [(c.coords["method"], c.seed) for c in cells] == [("b", 3), ("b", 1), ("a", 3), ("a", 1)]
    assert cells[0].label == "method-b.seed-3"
    assert cells[0].key == '{"method":"b","seed":3}'


def test_axes_sorted_by_name_regardless_of_insertion_order() -> None:
    a = mx.expand_spec(spec(matrix={"z": [1, 2], "a": ["x", "y"]}))
    b = mx.expand_spec(spec(matrix={"a": ["x", "y"], "z": [1, 2]}))
    assert [c.key for c in a] == [c.key for c in b]
    assert [c.label for c in a] == ["a-x.z-1", "a-x.z-2", "a-y.z-1", "a-y.z-2"]


def test_seeds_via_matrix_seed_axis() -> None:
    s = spec(matrix={"method": ["m"], "seed": [7, 8.0]})
    assert mx.seed_values(s) == [7, 8]
    cells = mx.expand_spec(s)
    assert [c.cell for c in cells] == [{"method": "m", "seed": 7}, {"method": "m", "seed": 8}]
    assert "seed" not in cells[0].coords


@pytest.mark.parametrize("bad", [["a"], [1.5], [True]])
def test_matrix_seed_values_must_be_integers(bad: list[Any]) -> None:
    s = ExperimentSpec.model_construct(
        name="e", cluster="c", matrix={"seed": bad}, seeds=[], entrypoint=EntrypointSpec(argv=["x"])
    )
    with pytest.raises(InvalidArgument):
        mx.seed_values(s)


def test_one_and_one_point_zero_are_distinct_cells() -> None:
    cells = mx.expand_cells({"lr": [1, 1.0]}, [None])
    assert [c.key for c in cells] == ['{"lr":1}', '{"lr":1.0}']
    # labels collide after rendering ("lr-1" vs "lr-1.0" differ here) and stay unique
    assert len({c.label for c in cells}) == 2


def test_label_collisions_after_sanitizing_get_hash_suffix() -> None:
    cells = mx.expand_cells({"name": ["a b", "a-b"]}, [None])
    labels = [c.label for c in cells]
    assert len(set(labels)) == 2
    assert all(label.startswith("name-a-b~") for label in labels)


def test_long_labels_are_bounded() -> None:
    cells = mx.expand_cells({f"axis{i}": ["v" * 40] for i in range(6)}, [1])
    assert len(cells[0].label) <= mx.MAX_LABEL_LEN


def test_duplicate_values_rejected() -> None:
    with pytest.raises(InvalidArgument):
        mx.expand_cells({"m": ["a", "a"]}, [None])
    with pytest.raises(InvalidArgument):
        mx.expand_cells({"m": []}, [None])


def test_hard_limit_256() -> None:
    assert len(mx.expand_cells({"a": list(range(16)), "b": list(range(16))}, [None])) == 256
    with pytest.raises(LimitExceeded) as ei:
        mx.expand_cells({"a": list(range(16)), "b": list(range(16))}, [1, 2])
    assert ei.value.details["cells"] == 512
    with pytest.raises(ValueError):
        spec(matrix={"a": list(range(16)), "b": list(range(17))})


def test_render_argv_substitutes_params_and_keeps_runtime_placeholders() -> None:
    r = mx.render_argv(
        [
            "python",
            "t.py",
            "--lr={lr}",
            "--flag",
            "{flag}",
            "--out",
            "{output_dir}/x",
            "{seed}",
            "${HOME}/{job_id}",
        ],
        {"lr": 1e-05, "flag": True},
        3,
    )
    assert r.ok and r.used == {"lr", "flag", "seed"}
    assert r.argv == [
        "python",
        "t.py",
        "--lr=1e-05",
        "--flag",
        "true",
        "--out",
        "{output_dir}/x",
        "3",
        "${HOME}/{job_id}",
    ]


def test_render_argv_reports_unknown_placeholders_and_nulls() -> None:
    r = mx.render_argv(["{nope}", "{seed}", "{x}", "{nope}", "{json:1}"], {"x": None}, None)
    assert r.unknown == ["nope", "seed"]
    assert r.nulls == ["x"]
    assert r.argv[2] == "" and r.argv[4] == "{json:1}"  # not an identifier: literal


def test_render_value_forms() -> None:
    assert mx.render_value(False) == "false"
    assert mx.render_value(0.1) == "0.1"
    assert mx.render_value(3) == "3"
    assert mx.render_value(None) == ""
    with pytest.raises(InvalidArgument):
        mx.render_value(float("nan"))


def test_cell_params_overlay_parameters() -> None:
    s = spec(parameters={"lr": 0.1, "bs": 32}, matrix={"lr": [0.01]}, seeds=[1])
    cell = mx.expand_spec(s)[0]
    assert mx.cell_params(s, cell) == {"lr": 0.01, "bs": 32}


def test_smoke_cell_selection() -> None:
    cells = mx.expand_spec(spec(matrix={"m": ["a", "b"]}, seeds=[1, 2]))
    assert mx.smoke_cell(cells, SmokeSpec()).cell == {"m": "a", "seed": 1}
    assert mx.smoke_cell(cells, SmokeSpec(cell={"m": "b"})).cell == {"m": "b", "seed": 1}
    assert mx.smoke_cell(cells, SmokeSpec(cell={"m": "b", "seed": 2})).cell == {"m": "b", "seed": 2}
    with pytest.raises(InvalidArgument):
        mx.smoke_cell(cells, SmokeSpec(cell={"m": "zzz"}))
    with pytest.raises(InvalidArgument):
        mx.smoke_cell([], SmokeSpec())


def test_effective_identity_detects_type_only_duplicates() -> None:
    a = mx.effective_identity(["--x", "1"], {"x": 1}, 1)
    b = mx.effective_identity(["--x", "1"], {"x": "1"}, 1)
    c = mx.effective_identity(["--x", "1"], {"x": 2}, 1)
    assert mx.find_duplicates([("p", a), ("q", b), ("r", c)]) == [["p", "q"]]


def test_extend_matrix_appends_new_values_only() -> None:
    s = spec(matrix={"m": ["a"]}, seeds=[1, 2])
    data, added = mx.extend_matrix(s, add_seeds=[2, 3], add_matrix={"m": ["a", "b"]})
    assert data["seeds"] == [1, 2, 3] and data["matrix"]["m"] == ["a", "b"]
    assert added == {"seeds": [3], "matrix": {"m": ["b"]}}


def test_extend_matrix_seed_axis_and_errors() -> None:
    s = spec(matrix={"m": ["a"], "seed": [1]})
    data, added = mx.extend_matrix(s, add_seeds=None, add_matrix={"seed": [2]})
    assert data["matrix"]["seed"] == [1, 2] and added["seeds"] == [2]
    with pytest.raises(InvalidArgument):
        mx.extend_matrix(s, add_seeds=None, add_matrix={"new_axis": [1]})
    with pytest.raises(InvalidArgument):
        mx.extend_matrix(s, add_seeds=None, add_matrix={"seed": ["x"]})


def test_expansion_is_deterministic() -> None:
    s = spec(matrix={"b": [2, 1], "a": ["y", "x"]}, seeds=[5, 4])
    assert [c.key for c in mx.expand_spec(s)] == [c.key for c in mx.expand_spec(s)]
    assert [c.label for c in mx.expand_spec(s)][:2] == ["a-y.b-2.seed-5", "a-y.b-2.seed-4"]
