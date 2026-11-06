"""hpclab examples list / copy: the bundled example projects, usable from an installed wheel."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli_support import Cli

from hpclab.cli import main as cli_main


@pytest.fixture
def examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "bundled" / "examples"
    ex = root / "demo-proj"
    (ex / "__pycache__").mkdir(parents=True)
    (ex / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"\0")
    (ex / "pkg").mkdir()
    (ex / "pkg" / "mod.pyc").write_bytes(b"\0")
    (ex / "pkg" / "mod.py").write_text("X = 1\n")
    (ex / "train.py").write_text("print('train')\n")
    (ex / "README.md").write_text("# demo-proj: a tiny research project\n\nMore text.\n")
    (ex / "hpclab.yaml").write_text("name: demo-ablation\ncluster: sim\n")
    (ex / "hpclab.hai.yaml").write_text("name: demo-ablation-hai\ncluster: hai\n")
    (root / "not-an-example.txt").write_text("x")
    monkeypatch.setattr(cli_main, "bundled_path", lambda kind: root if kind == "examples" else None)
    return root


def test_examples_list_names_description_and_specs(cli: Cli, examples: Path) -> None:
    r = cli.run("examples", "list")
    assert r.exit_code == 0, r.text
    assert "demo-proj" in r.stdout and "a tiny research project" in r.stdout
    assert "hpclab.yaml (cluster sim)" in r.stdout and "hpclab.hai.yaml (cluster hai)" in r.stdout
    assert "not-an-example" not in r.stdout
    assert "hpclab examples copy demo-proj" in r.stdout
    r = cli.run("--json", "examples", "list")
    (item,) = r.json["examples"]
    assert item["name"] == "demo-proj" and item["path"] == str(examples / "demo-proj")
    assert {s["file"]: s["cluster"] for s in item["specs"]} == {
        "hpclab.yaml": "sim",
        "hpclab.hai.yaml": "hai",
    }


def test_examples_copy_skips_bytecode_and_prints_next_steps(
    cli: Cli, examples: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    r = cli.run("examples", "copy", "demo-proj")  # DEST defaults to ./NAME
    assert r.exit_code == 0, r.text
    dest = work / "demo-proj"
    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*"))
    assert files == ["README.md", "hpclab.hai.yaml", "hpclab.yaml", "pkg", "pkg/mod.py", "train.py"]
    assert "hpclab prepare demo-proj/hpclab.yaml" in r.stdout
    assert "hpclab submit <plan id>" in r.stdout
    assert "hpclab init --sim" in r.stdout  # the default spec targets the simulator
    assert "hpclab prepare demo-proj/hpclab.hai.yaml" in r.stdout


def test_examples_copy_refuses_a_non_empty_destination_without_force(
    cli: Cli, examples: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "mine"
    dest.mkdir()
    (dest / "train.py").write_text("my own edits\n")
    (dest / "notes.txt").write_text("keep me\n")
    r = cli.run("examples", "copy", "demo-proj", str(dest))
    assert r.exit_code == 1
    assert "not empty" in r.stderr and "--force" in r.stderr
    assert (dest / "train.py").read_text() == "my own edits\n" and not (dest / "hpclab.yaml").exists()
    r = cli.run("examples", "copy", "demo-proj", str(dest), "--force")
    assert r.exit_code == 0, r.text
    assert (dest / "train.py").read_text() == "print('train')\n"
    assert (dest / "notes.txt").read_text() == "keep me\n"  # --force overwrites, never deletes
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.run("examples", "copy", "demo-proj", str(empty)).exit_code == 0
    assert (empty / "hpclab.yaml").is_file()


def test_examples_copy_errors(cli: Cli, examples: Path, tmp_path: Path) -> None:
    r = cli.run("examples", "copy", "nope", str(tmp_path / "x"))
    assert r.exit_code == 2 and "unknown example 'nope'" in r.stderr and "demo-proj" in r.stderr
    assert cli.run("examples", "copy", "../demo-proj", str(tmp_path / "y")).exit_code == 2
    f = tmp_path / "a-file"
    f.write_text("x")
    r = cli.run("examples", "copy", "demo-proj", str(f), "--force")
    assert r.exit_code == 1 and "not a directory" in r.stderr


def test_examples_copy_json(cli: Cli, examples: Path, tmp_path: Path) -> None:
    r = cli.run("--json", "examples", "copy", "demo-proj", str(tmp_path / "j"))
    assert r.exit_code == 0, r.text
    assert r.json["ok"] is True and r.json["dest"] == str(tmp_path / "j") and r.json["files"] == 5
    r = cli.run("--json", "examples", "copy", "demo-proj", str(tmp_path / "j"))
    assert r.exit_code == 1 and r.json["ok"] is False


def test_examples_not_bundled(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "bundled_path", lambda kind: None)
    r = cli.run("examples", "list")
    assert r.exit_code == 1 and "not bundled" in r.stderr


def test_the_real_bundled_example_copies(cli: Cli, tmp_path: Path) -> None:
    r = cli.run("examples", "copy", "tiny-classifier", str(tmp_path / "tc"))
    assert r.exit_code == 0, r.text
    for name in ("hpclab.yaml", "train.py", "hpclab_io.py", "README.md"):
        assert (tmp_path / "tc" / name).is_file(), name
    assert not list((tmp_path / "tc").rglob("__pycache__"))


def test_init_sim_points_at_examples_copy(cli: Cli) -> None:
    """The service's next steps name a repo path (examples/...) that an installed wheel does not
    have; the CLI points at `hpclab examples copy` instead."""
    from cli_support import env

    cli.client.responses["init_sim"] = env("init_sim")
    r = cli.run("init", "--sim")
    assert r.exit_code == 0, r.text
    assert "next: run the guided demo: hpclab demo" in r.stdout
    assert "hpclab examples copy tiny-classifier && hpclab prepare tiny-classifier/hpclab.yaml" in r.stdout
    assert "examples/tiny-classifier" not in r.stdout
    # a controller whose init_sim already suggests `hpclab examples copy`: not twice
    e = env("init_sim")
    e["data"]["next_steps"] = [
        "copy the example project: hpclab examples copy tiny-classifier ./tiny-classifier",
        "then freeze it into a validated plan: hpclab prepare tiny-classifier/hpclab.yaml",
    ]
    cli.client.responses["init_sim"] = e
    r = cli.run("init", "--sim")
    assert r.stdout.count("hpclab examples copy") == 1, r.stdout


def test_examples_copy_into_a_symlinked_destination(cli: Cli, examples: Path, tmp_path: Path) -> None:
    """DEST itself may be a symlink the user made (e.g. ~/work -> /scratch/work on a cluster):
    the copy goes where it points. Symlinks *inside* DEST are still never written through."""
    scratch = tmp_path / "scratch" / "work"
    scratch.mkdir(parents=True)
    link = tmp_path / "work"
    link.symlink_to(scratch, target_is_directory=True)
    r = cli.run("examples", "copy", "demo-proj", str(link))
    assert r.exit_code == 0, r.text
    assert (scratch / "hpclab.yaml").is_file() and (scratch / "pkg" / "mod.py").is_file()

    outside = tmp_path / "outside.py"
    outside.write_text("not yours to overwrite\n")
    (scratch / "train.py").unlink()
    (scratch / "train.py").symlink_to(outside)
    r = cli.run("examples", "copy", "demo-proj", str(link), "--force")
    assert r.exit_code == 1 and "symlink" in r.stderr
    assert outside.read_text() == "not yours to overwrite\n"


def test_examples_list_survives_a_malformed_spec(cli: Cli, examples: Path) -> None:
    (examples / "demo-proj" / "hpclab.hai.yaml").write_text("name: [unclosed\n  cluster: : :\n")
    r = cli.run("examples", "list")
    assert r.exit_code == 0, r.text
    assert "Traceback" not in r.text and "hpclab.hai.yaml" in r.stdout
