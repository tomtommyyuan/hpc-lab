"""Locate bundled data (profiles, skills, examples) in a dev checkout or an installed wheel."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def bundled_path(kind: str) -> Path | None:
    """Directory for ``kind`` in {"profiles", "skills", "examples"}, or None if not shipped.

    Installed wheels carry them under ``hpclab/_bundled/<kind>`` (pyproject force-include);
    a source checkout has them at the repository root.
    """
    try:
        pkg = resources.files("hpclab").joinpath("_bundled", kind)
        if pkg.is_dir():
            return Path(str(pkg))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    repo = Path(__file__).resolve().parents[2] / kind
    return repo if repo.is_dir() else None


def skill_text(name: str) -> str | None:
    root = bundled_path("skills")
    if root is None:
        return None
    path = root / name / "SKILL.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    return text
