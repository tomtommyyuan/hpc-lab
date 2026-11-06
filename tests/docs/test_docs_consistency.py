"""The hand-written docs must agree with the repository and the CLI.

* Every relative link in README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, docs/**, the
  bundled agent skills (skills/*/SKILL.md) and the example READMEs (examples/*/README.md) points
  to a file that exists, and every ``#anchor`` to a heading that exists.
* Every ``hpclab ...`` invocation in those files (commands in ``bash``/``console`` blocks,
  inline code spans) names a real command of the Typer tree, and every ``--option`` it uses is
  an option of that command. Positional arguments and placeholders are not checked.
* No doc contains a personal identifier from the private HAI operator notes (the same hashed
  list the bundled-profile test uses) or a real email address: site pages use placeholders.

When this fails after a CLI change, fix the docs (or the command): a guide that tells a
researcher to run a command that does not exist is a bug.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOP_LEVEL = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"]
GLOBAL_OPTIONS_WITH_VALUE = {"--home", "--timeout"}
GLOBAL_FLAGS = {"--json", "--version", "--help"}


def doc_files() -> list[Path]:
    files = [REPO / name for name in TOP_LEVEL if (REPO / name).is_file()]
    files += sorted((REPO / "docs").rglob("*.md"))
    # what agents and new users read first: a skill or an example telling them to run a command
    # that does not exist is the same bug as a guide doing so
    files += sorted(REPO.glob("skills/*/SKILL.md"))
    files += sorted(REPO.glob("examples/*/README.md"))
    files += [p for p in (REPO / "benchmarks" / "README.md",) if p.is_file()]
    return files


def test_benchmark_readme_is_scanned() -> None:
    # it tells reviewers which `hpclab` commands resolve a flagged attempt and links the results
    assert REPO / "benchmarks" / "README.md" in doc_files()


def test_skills_and_example_readmes_are_scanned() -> None:
    scanned = {p.relative_to(REPO).as_posix() for p in doc_files()}
    skills = {p.relative_to(REPO).as_posix() for p in REPO.glob("skills/*/SKILL.md")}
    examples = {p.relative_to(REPO).as_posix() for p in REPO.glob("examples/*/README.md")}
    assert skills and examples
    assert skills | examples <= scanned


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)")


def split_blocks(text: str) -> tuple[list[tuple[int, str]], list[tuple[str, int, list[str]]]]:
    """(prose lines, fenced blocks) with 1-based line numbers; blocks are (lang, start, lines)."""
    prose: list[tuple[int, str]] = []
    blocks: list[tuple[str, int, list[str]]] = []
    fence: str | None = None
    lang = ""
    start = 0
    body: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence, lang, start, body = m.group(2), m.group(3).lower(), n, []
            else:
                prose.append((n, line))
        elif m and m.group(2).startswith(fence[0]) and len(m.group(2)) >= len(fence) and not m.group(3):
            blocks.append((lang, start, body))
            fence = None
        else:
            body.append(line)
    return prose, blocks


def slug(title: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    title = re.sub(r"<[^>]+>", "", title)  # inline HTML
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)  # links keep their text
    keep = []
    for ch in title.strip().lower():
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
        elif ch == " ":
            keep.append("-")
    return "".join(keep)


def anchors(path: Path) -> set[str]:
    prose, _ = split_blocks(path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    out: set[str] = set()
    for _, line in prose:
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        base = slug(m.group(2))
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.add(base if k == 0 else f"{base}-{k}")
    return out


_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_LINK = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def links(path: Path) -> Iterator[tuple[int, str]]:
    prose, _ = split_blocks(path.read_text(encoding="utf-8"))
    for n, line in prose:
        for m in _LINK.finditer(_INLINE_CODE.sub("", line)):
            yield n, m.group(1)


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc", doc_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_and_anchors_resolve(doc: Path) -> None:
    problems: list[str] = []
    cache: dict[Path, set[str]] = {}
    for n, target in links(doc):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:, hpclab: ...
            continue
        file_part, _, frag = target.partition("#")
        dest = doc if not file_part else (doc.parent / file_part).resolve()
        where = f"{doc.relative_to(REPO)}:{n}: {target}"
        if not dest.exists():
            problems.append(f"{where}: {file_part} does not exist")
            continue
        if frag and dest.suffix == ".md":
            if dest not in cache:
                cache[dest] = anchors(dest)
            if frag not in cache[dest]:
                problems.append(f"{where}: no heading with anchor #{frag} in {dest.name}")
    assert not problems, "broken links:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# hpclab invocations
# ---------------------------------------------------------------------------

_SHELL_STOP = re.compile(r"\s(?:&&|\|\||;|\||>|2>|<<|#)\s|\s#")
_INVOCATION = re.compile(r"(?:^|(?<=[\s;&|(]))hpclab\s+(.*)$")


def invocations(path: Path) -> Iterator[tuple[int, str]]:
    """``hpclab`` command lines: commands in shell blocks, prompts in console blocks, inline code."""
    prose, blocks = split_blocks(path.read_text(encoding="utf-8"))
    for n, line in prose:
        for m in _INLINE_CODE.finditer(line):
            code = m.group(2).strip()
            if code.startswith("hpclab "):
                yield n, code
    for lang, start, body in blocks:
        for i, line in enumerate(body, start + 1):
            if lang in ("bash", "sh", "shell", "zsh"):
                cmd = line
            elif lang == "console" and line.lstrip().startswith("$ "):
                cmd = line.lstrip()[2:]
            else:
                continue
            cmd = re.sub(r"(?:^|\s)#.*$", "", cmd)  # shell comment
            for m in _INVOCATION.finditer(cmd):
                yield i, "hpclab " + m.group(1)


def _tokens(invocation: str) -> list[str]:
    rest = invocation.split(None, 1)[1] if " " in invocation else ""
    cut = _SHELL_STOP.search(" " + rest)
    if cut:
        rest = (" " + rest)[: cut.start()]
    return rest.split()


def _is_placeholder(tok: str) -> bool:
    return tok[:1] in "<[{$'\"" or tok.startswith("...") or tok.isupper()


def check_invocation(root: Any, invocation: str) -> str | None:
    """None when ``invocation`` names a real command and only options that command has."""
    toks = _tokens(invocation)
    i = 0
    while i < len(toks) and toks[i].startswith("-"):  # global options before the command
        name = toks[i].split("=", 1)[0]
        if name in GLOBAL_OPTIONS_WITH_VALUE:
            i += 1 if "=" in toks[i] else 2
        elif name in GLOBAL_FLAGS:
            i += 1
        else:
            return None if name == "--" else f"unknown global option {name}"
    cmd = root
    path = ["hpclab"]
    while isinstance(getattr(cmd, "commands", None), dict):
        if i >= len(toks) or _is_placeholder(toks[i]) or toks[i] == "--help":
            return None
        sub = cmd.commands.get(toks[i])
        if sub is None or getattr(sub, "hidden", False):
            return f"{' '.join(path)} has no command {toks[i]!r}"
        cmd, path, i = sub, [*path, toks[i]], i + 1
    known = {"--help"}
    for p in cmd.params:
        known.update(getattr(p, "opts", []))
        known.update(getattr(p, "secondary_opts", []))
    for tok in toks[i:]:
        if tok.startswith("-") and tok not in ("-", "--") and not re.match(r"^-\d", tok):
            name = tok.split("=", 1)[0]
            if name not in known:
                return f"{' '.join(path)} has no option {name}"
    return None


@pytest.fixture(scope="module")
def cli_root() -> Any:
    from typer.main import get_command

    from hpclab.cli.main import app

    return get_command(app)


@pytest.mark.parametrize("doc", doc_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_documented_commands_exist(doc: Path, cli_root: Any) -> None:
    problems = []
    for n, inv in invocations(doc):
        err = check_invocation(cli_root, inv)
        if err:
            problems.append(f"{doc.relative_to(REPO)}:{n}: `{inv}`: {err}")
    assert not problems, "documented hpclab commands that do not exist:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# Personal data
# ---------------------------------------------------------------------------

_EXAMPLE_EMAIL_DOMAINS = ("example.org", "example.com", "example.net")


def _profile_privacy_module() -> ModuleType:
    """``tests/profiles/test_bundled_profiles.py`` owns the hashed private-token list."""
    path = REPO / "tests" / "profiles" / "test_bundled_profiles.py"
    name = "hpclab_docs_profile_privacy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("doc", doc_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_docs_contain_no_personal_identifiers(doc: Path) -> None:
    privacy = _profile_privacy_module()
    text = doc.read_text(encoding="utf-8")
    assert privacy.private_tokens(text) == set(), f"{doc.name} contains private identifiers"
    emails = [e for e in privacy.EMAIL_RE.findall(text) if not e.lower().endswith(_EXAMPLE_EMAIL_DOMAINS)]
    assert emails == [], f"{doc.name} contains email addresses: {emails}"


def test_the_checker_catches_drift(cli_root: Any) -> None:
    """Guards the checker itself (a checker that accepts everything proves nothing)."""
    assert check_invocation(cli_root, "hpclab prepare spec.yaml --all-findings") is None
    assert check_invocation(cli_root, "hpclab --json runs --experiment x") is None
    assert check_invocation(cli_root, "hpclab cluster add hai --profile hai --user <you>") is None
    assert check_invocation(cli_root, "hpclab grid <workflow id> --watch") is None
    assert check_invocation(cli_root, "hpclab frobnicate") is not None
    assert check_invocation(cli_root, "hpclab cluster frobnicate") is not None
    assert check_invocation(cli_root, "hpclab prepare spec.yaml --no-such-flag") is not None
    assert slug("Where did the GPU-hours go? `usage_report`") == "where-did-the-gpu-hours-go-usage_report"
    assert slug("7. Known issues are expiring observations") == "7-known-issues-are-expiring-observations"
