"""Discover, load and lint cluster profiles; read and write private user overlays.

Two kinds of files, with very different trust and privacy rules:

* **Profiles** (``ClusterProfile``) are public site rules. Bundled profiles ship with hpclab
  (``<repo>/profiles`` in a dev checkout, ``hpclab/_bundled/profiles`` in an installed wheel).
  A user may drop ``<name>.yaml`` into ``$HPCLAB_HOME/profiles/``; a local file *replaces* the
  bundled profile of the same name wholesale (no deep merge, so what you read is what runs).
  Profiles are looked up by file name, and the file name must equal the ``cluster_id`` inside,
  so ``load_profile("hai")`` can never silently return some other cluster.
* **Overlays** (``UserOverlay``) are private: remote user, Slurm account, QoS, SSH alias. They
  live in ``$HPCLAB_HOME/clusters/<cluster_id>.yaml`` with mode 0600 inside a 0700 directory
  and are never bundled or published.

Failure semantics: a missing profile raises ``NotFound`` (with the available names); an
unreadable, malformed or schema-invalid file raises ``InvalidArgument`` whose ``details`` carry
the path and the validation errors. A missing overlay is not an error (``None``) because a
cluster can be registered from a profile alone; a malformed overlay is.

``validate_profile_is_public`` and ``lint_profile`` return problem lists instead of raising so
the CLI can print every problem at once and tests can assert on them.

Files are parsed with ``yaml_load``: YAML 1.1 dates and base-60 numbers (``12:00:00``) stay
strings, because profiles are written by hand and those types silently corrupt time fields.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from hpclab.config import Paths
from hpclab.domain.clock import parse_iso, parse_slurm_duration
from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.profiles.schema import ClusterProfile, ResolvedCluster, SiteObservation, UserOverlay

log = logging.getLogger("hpclab.profiles")

ProfileOrigin = Literal["bundled", "local"]

PROFILE_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")
MAX_PROFILE_BYTES = 1024 * 1024
"""Profiles and overlays are small hand-written files; anything bigger is a mistake."""

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TEMPLATE_RE = re.compile(r"<[A-Za-z][^<>\s]*>")
_DEV_PROFILES_DIR = Path(__file__).resolve().parents[3] / "profiles"
_OVERLAY_HEADER = (
    "# hpclab private cluster overlay. Holds YOUR identity for this cluster (remote user,\n"
    "# Slurm account, QoS, SSH alias). Never commit or share this file; it is written 0600.\n"
)


class _HandWrittenYamlLoader(yaml.SafeLoader):
    """``SafeLoader`` without the YAML 1.1 implicit types that corrupt hand-written profiles.

    Unquoted ``2026-09-01`` would become a ``date`` and ``12:00:00`` the base-60 integer 43200,
    both rejected by string fields (and the second is unrecoverable: ``60`` means minutes to
    Slurm). Here timestamps and base-60 numbers stay strings; everything else resolves as usual.
    """


def _without_base60(pattern: re.Pattern[str]) -> re.Pattern[str]:
    return re.compile(r"(?!.*:)" + pattern.pattern, pattern.flags)


_HandWrittenYamlLoader.yaml_implicit_resolvers = {
    first: [
        (tag, _without_base60(rx) if tag in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float") else rx)
        for tag, rx in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def yaml_load(text: str) -> Any:
    """Parse hand-written YAML (profiles, overlays, experiment specs) safely.

    Like ``yaml.safe_load`` except that dates/timestamps and ``HH:MM:SS`` durations are kept
    as strings. Raises ``yaml.YAMLError`` on malformed input.
    """
    return yaml.load(text, Loader=_HandWrittenYamlLoader)  # noqa: S506 - a SafeLoader subclass


class ProfileInfo(BaseModel):
    """One entry of the effective profile set (local files shadow bundled ones).

    ``error`` is set, instead of dropping the entry, when the file exists but does not load:
    a broken local override must be visible, not silently replaced by the bundled profile.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    cluster_id: str
    display_name: str
    source: ProfileOrigin
    path: str
    shadows_bundled: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def bundled_profiles_dir() -> Path:
    """Directory holding the profiles shipped with hpclab.

    An installed wheel carries them as package data (``hpclab/_bundled/profiles``, see the
    ``force-include`` table in pyproject). A source checkout (including editable installs,
    which do not materialize force-included data) uses ``<repo>/profiles``. The returned path
    may not exist in a broken install; callers then simply see no bundled profiles.
    """
    packaged = resources.files("hpclab").joinpath("_bundled", "profiles")
    if isinstance(packaged, Path) and packaged.is_dir():
        return packaged
    return _DEV_PROFILES_DIR


def list_profiles(paths: Paths | None) -> list[ProfileInfo]:
    """The effective profile set, sorted by name.

    ``paths=None`` lists bundled profiles only. Every entry is loaded and validated so the
    listing can show ``cluster_id``/``display_name``; invalid files are listed with ``error``.
    """
    bundled = _profile_files(bundled_profiles_dir())
    local = _profile_files(paths.profiles) if paths is not None else {}
    infos: list[ProfileInfo] = []
    for name in sorted(bundled.keys() | local.keys()):
        if name in local:
            infos.append(_describe(name, local[name], "local", shadows_bundled=name in bundled))
        else:
            infos.append(_describe(name, bundled[name], "bundled", shadows_bundled=False))
    return infos


def resolve_profile_path(name_or_path: str | Path, paths: Paths | None) -> tuple[Path, ProfileOrigin | None]:
    """Locate a profile file without loading it.

    A value containing a path separator or ending in ``.yaml``/``.yml`` (or any ``Path``) is an
    explicit file and is returned with origin ``None``. Anything else is a profile *name*:
    ``paths.profiles/<name>.yaml`` wins over the bundled file of the same name.
    """
    if _looks_like_path(name_or_path):
        path = Path(name_or_path).expanduser()
        if not path.is_file():
            raise NotFound(f"profile file {str(path)!r} does not exist", details={"path": str(path)})
        return path.resolve(), None

    name = _check_name(str(name_or_path), "profile name")
    local_dir = paths.profiles if paths is not None else None
    if local_dir is not None:
        local = _profile_files(local_dir)
        if name in local:
            return local[name], "local"
    bundled_dir = bundled_profiles_dir()
    bundled = _profile_files(bundled_dir)
    if name in bundled:
        return bundled[name], "bundled"
    available = sorted(bundled.keys() | (_profile_files(local_dir).keys() if local_dir else set()))
    raise NotFound(
        f"no cluster profile named {name!r}",
        details={
            "available": available,
            "searched": [str(d) for d in (local_dir, bundled_dir) if d is not None],
            "hint": "pass a file path, or copy profiles/generic-slurm.yaml to $HPCLAB_HOME/profiles/<name>.yaml",
        },
    )


def load_profile(name_or_path: str | Path, paths: Paths | None) -> ClusterProfile:
    """Load and validate a profile by name (local override first, then bundled) or by file path.

    Raises ``NotFound`` if nothing matches and ``InvalidArgument`` if the file is malformed or,
    for name lookups, declares a ``cluster_id`` different from its file name.
    """
    path, origin = resolve_profile_path(name_or_path, paths)
    profile = _read_profile(path)
    if origin is not None and profile.cluster_id != path.stem:
        raise InvalidArgument(
            f"profile file {path.name!r} declares cluster_id {profile.cluster_id!r}; "
            "the file name must match the cluster_id",
            details={"path": str(path), "cluster_id": profile.cluster_id},
        )
    return profile


def resolve_cluster(
    name_or_path: str | Path, paths: Paths, *, overlay: UserOverlay | None = None
) -> ResolvedCluster:
    """Profile + overlay as services consume them.

    Uses ``overlay`` if given, else the saved overlay, else an empty one (registration from a
    profile alone is allowed; missing account/user are reported by admission checks later).
    """
    profile = load_profile(name_or_path, paths)
    chosen = overlay if overlay is not None else load_overlay(profile.cluster_id, paths)
    if chosen is None:
        chosen = UserOverlay(cluster_id=profile.cluster_id)
    if chosen.cluster_id != profile.cluster_id:
        raise InvalidArgument(
            f"overlay is for cluster {chosen.cluster_id!r}, profile is {profile.cluster_id!r}",
            details={"overlay_cluster_id": chosen.cluster_id, "profile_cluster_id": profile.cluster_id},
        )
    return ResolvedCluster(profile=profile, overlay=chosen)


# ---------------------------------------------------------------------------
# Private overlays
# ---------------------------------------------------------------------------


def overlay_path(cluster_id: str, paths: Paths) -> Path:
    return paths.overlays / f"{_check_name(cluster_id, 'cluster_id')}.yaml"


def load_overlay(cluster_id: str, paths: Paths) -> UserOverlay | None:
    """Read ``$HPCLAB_HOME/clusters/<cluster_id>.yaml``; ``None`` if there is none.

    ``cluster_id`` may be omitted in the file (it is implied by the file name) but must match
    if present. Values still holding template placeholders such as ``<your-slurm-account>``
    are rejected: submitting with a literal placeholder account would fail far later and
    less clearly.
    """
    path = overlay_path(cluster_id, paths)
    if not path.is_file():
        return None
    _warn_if_exposed(path)
    data = _read_yaml_mapping(path, what="overlay")
    declared = data.setdefault("cluster_id", cluster_id)
    if declared != cluster_id:
        raise InvalidArgument(
            f"overlay {path.name!r} declares cluster_id {declared!r}",
            details={"path": str(path), "expected": cluster_id, "found": declared},
        )
    leftovers = _template_placeholders(data)
    if leftovers:
        raise InvalidArgument(
            f"overlay {path.name!r} still contains template placeholders; edit them first",
            details={"path": str(path), "placeholders": leftovers},
        )
    try:
        return UserOverlay.model_validate(data)
    except ValidationError as exc:
        raise InvalidArgument(
            f"overlay {path.name!r} is invalid",
            details={"path": str(path), "errors": _format_errors(exc)},
        ) from exc


def save_overlay(overlay: UserOverlay, paths: Paths) -> Path:
    """Atomically write the overlay (file 0600, directory 0700) and return its path.

    Only non-default fields are written so the file stays readable; ``load_overlay`` restores
    an equal model.
    """
    target = overlay_path(overlay.cluster_id, paths)
    data = overlay.model_dump(mode="json", exclude_defaults=True)
    data["cluster_id"] = overlay.cluster_id
    leftovers = _template_placeholders(data)
    if leftovers:
        raise InvalidArgument(
            "refusing to save an overlay with template placeholders", details={"placeholders": leftovers}
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _atomic_write_private(target, _OVERLAY_HEADER + body)
    return target


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def active_observations(profile: ClusterProfile, now: datetime) -> list[SiteObservation]:
    """Known issues that have not expired at ``now`` (naive datetimes are taken as UTC).

    An observation expires *at* ``expires_at``. One whose expiry cannot be parsed is dropped
    (and logged): a TTL that cannot be evaluated must not turn into a permanent rule. An
    observation without ``expires_at`` is kept; ``lint_profile`` flags it.
    """
    return _unexpired(profile.known_issues, now, f"{profile.cluster_id}.known_issues")


def active_facts(profile: ClusterProfile, now: datetime) -> list[SiteObservation]:
    """``profile.facts`` (e.g. ``driver_cuda``) not expired at ``now``; same rules as issues.

    An expired fact is not "the current value": callers should treat it as unknown (and say
    it needs a live probe) rather than use it.
    """
    return _unexpired(profile.facts, now, f"{profile.cluster_id}.facts")


def _unexpired(items: list[SiteObservation], now: datetime, where: str) -> list[SiteObservation]:
    now_utc = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    active: list[SiteObservation] = []
    for obs in items:
        if obs.expires_at is None:
            active.append(obs)
            continue
        expires = _parse_time(obs.expires_at)
        if expires is None:
            log.warning("dropping %s %r: unparsable expires_at %r", where, obs.key, obs.expires_at)
            continue
        if expires > now_utc:
            active.append(obs)
    return active


# ---------------------------------------------------------------------------
# Lint: internal consistency
# ---------------------------------------------------------------------------

# What ResolvedCluster actually substitutes where: expand() knows all four; storage_path()
# (and therefore remote_root()'s {scratch}) only {user}; remote_root() only {scratch}/{user}.
_KNOWN_EXPAND = frozenset({"user", "scratch", "home", "remote_root"})
_STORAGE_EXPAND = frozenset({"user"})
_REMOTE_ROOT_EXPAND = frozenset({"user", "scratch"})
_SIM_ONLY = frozenset({"sim_fs"})
# ``{name}`` placeholders, but not shell ``${NAME}`` expansions (legitimate in setup lines).
_BRACE_RE = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def lint_profile(profile: ClusterProfile) -> list[str]:
    """Consistency problems that the schema alone cannot catch; ``[]`` means clean.

    Checks partitions/default, Slurm time strings, GPU architecture coverage, placeholders the
    resolver can actually expand, timestamps/TTL of ``known_issues`` and ``facts``, provenance.
    """
    problems: list[str] = []
    problems += _lint_partitions(profile)
    problems += _lint_placeholders(profile)
    problems += _lint_observations(profile)
    problems += _lint_policy(profile)
    if not profile.sources:
        problems.append("sources: a profile needs at least one source (provenance) entry")
    return problems


def _lint_partitions(profile: ClusterProfile) -> list[str]:
    problems: list[str] = []
    flagged = sorted(name for name, rule in profile.partitions.items() if rule.default)
    if len(flagged) > 1:
        problems.append(f"partitions: more than one partition marked default: {flagged}")
    if profile.default_partition is not None:
        if profile.default_partition not in profile.partitions:
            problems.append(f"default_partition {profile.default_partition!r} is not a declared partition")
        elif flagged and flagged != [profile.default_partition]:
            problems.append(
                f"default_partition {profile.default_partition!r} disagrees with default: true on {flagged}"
            )
    for name, rule in profile.partitions.items():
        if rule.max_time is not None and not parse_slurm_duration(rule.max_time):
            problems.append(f"partitions.{name}.max_time: {rule.max_time!r} is not a Slurm duration")
        for gpu in rule.gpu_types:
            if gpu not in profile.gpu_arch:
                problems.append(f"partitions.{name}.gpu_types: {gpu!r} has no gpu_arch entry")
        per_job, per_user = rule.max_gpus_per_job, rule.max_gpus_per_user
        if per_job is not None and per_user is not None and per_job > per_user:
            problems.append(
                f"partitions.{name}: max_gpus_per_job {per_job} exceeds max_gpus_per_user {per_user}"
            )
    return problems


def _lint_placeholders(profile: ClusterProfile) -> list[str]:
    storage_ok = _STORAGE_EXPAND | (_SIM_ONLY if profile.transport == "sim" else frozenset())
    checks: list[tuple[str, str, frozenset[str]]] = [
        (f"storage.{role}.path", sr.path, storage_ok) for role, sr in profile.storage.items()
    ]
    checks.append(("remote_root_template", profile.remote_root_template, _REMOTE_ROOT_EXPAND))
    for rname, recipe in profile.runtime_recipes.items():
        checks += [(f"runtime_recipes.{rname}.env.{k}", v, _KNOWN_EXPAND) for k, v in recipe.env.items()]
        checks += [
            (f"runtime_recipes.{rname}.setup[{i}]", line, _KNOWN_EXPAND)
            for i, line in enumerate(recipe.setup)
        ]
    problems = [
        f"{loc}: placeholder {{{name}}} cannot be expanded (known: {sorted(allowed)})"
        for loc, text, allowed in checks
        for name in _BRACE_RE.findall(text)
        if name not in allowed
    ]
    if "{scratch}" in profile.remote_root_template and "scratch" not in profile.storage:
        problems.append("remote_root_template uses {scratch} but no 'scratch' storage role is declared")
    return problems


def _lint_observations(profile: ClusterProfile) -> list[str]:
    return _lint_observation_list("known_issues", profile.known_issues) + _lint_observation_list(
        "facts", profile.facts
    )


def _lint_observation_list(field: str, items: list[SiteObservation]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, obs in enumerate(items):
        loc = f"{field}[{i}] ({obs.key})"
        if obs.key in seen:
            problems.append(f"{loc}: duplicate key")
        seen.add(obs.key)
        observed = _parse_time(obs.observed_at)
        if observed is None:
            problems.append(f"{loc}: observed_at {obs.observed_at!r} is not an ISO-8601 time")
        if obs.expires_at is None:
            problems.append(f"{loc}: observations need expires_at (a TTL), not permanent status")
            continue
        expires = _parse_time(obs.expires_at)
        if expires is None:
            problems.append(f"{loc}: expires_at {obs.expires_at!r} is not an ISO-8601 time")
        elif observed is not None and expires <= observed:
            problems.append(f"{loc}: expires_at is not after observed_at")
    return problems


def _lint_policy(profile: ClusterProfile) -> list[str]:
    pol = profile.policy
    positive = {
        "poll_min_interval_s": pol.poll_min_interval_s,
        "sacct_min_interval_s": pol.sacct_min_interval_s,
        "reconcile_window_s": pol.reconcile_window_s,
        "max_parallel_default": pol.max_parallel_default,
        "submit_rate_per_min": pol.submit_rate_per_min,
    }
    problems = [f"policy.{k} must be > 0 (got {v})" for k, v in positive.items() if v <= 0]
    if pol.accounting_lag_s < 0:
        problems.append(f"policy.accounting_lag_s must be >= 0 (got {pol.accounting_lag_s})")
    return problems


# ---------------------------------------------------------------------------
# Public-safety check: nothing personal in a shareable profile
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_PATH_RE = re.compile(r"(?<![\w$}])/[^\s\"'`;,|&<>()]+")
_TILDE_USER_RE = re.compile(r"(?<![\w/])~([A-Za-z_][A-Za-z0-9_.-]*)")
_ACCOUNT_FLAG_RE = re.compile(r"--account[= ]([^\s\"']+)")
# ``-A`` is only an account flag on Slurm command lines (``ls -A`` is not), including
# ``#SBATCH`` directives in batch scripts.
_SHORT_ACCOUNT_RE = re.compile(r"(?<![\w-])-A[= ]?([A-Za-z0-9_][^\s\"']*)")
_SLURM_CMD_RE = re.compile(r"(?i)\b(?:sbatch|srun|salloc)\b")
# ``export SLURM_ACCOUNT=x`` / ``SBATCH_ACCOUNT='x'`` lines, as copied from a cluster.env file.
_ACCOUNT_ASSIGN_RE = re.compile(r"\b(?:SLURM|SBATCH|SALLOC|SRUN)_ACCOUNT\s*=\s*[\"']?([^\s\"';]+)")
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(token|secret|password|passwd|api[_-]?key|access[_-]?key)\b\s*[:=]\s*([^\s\"',;]+)"
)
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "GitHub token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
    ),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("W&B API key", re.compile(r"(?i)wandb[^\n]{0,20}\b[0-9a-f]{40}\b")),
)
_SECRET_ENV_KEY_RE = re.compile(r"(?i)(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)")
_ACCOUNT_ENV_KEYS = frozenset({"SLURM_ACCOUNT", "SBATCH_ACCOUNT", "SALLOC_ACCOUNT", "SRUN_ACCOUNT"})
# Directory names under which the next path segment is normally a user name.
_USER_PARENT_DIRS = frozenset({"users", "user", "home", "homes", "scratch", "people", "u"})
# Next-segment values that are clearly not a person.
_SHARED_SEGMENTS = frozenset(
    {
        "shared",
        "share",
        "public",
        "common",
        "group",
        "groups",
        "datasets",
        "data",
        "tmp",
        "projects",
        "software",
        "apps",
    }
)
# Fields that may legitimately hold a public role address (e.g. a support mailbox).
_EMAIL_OK_FIELDS = frozenset({"support_contact"})


def validate_profile_is_public(
    profile: ClusterProfile, *, known_identifiers: Iterable[str] = ()
) -> list[str]:
    """Things in a profile that look personal or secret; ``[]`` means safe to share.

    Heuristics (deliberately conservative, false positives are cheap): user-specific paths not
    using ``{user}``/``$USER``, ``~name`` homes, Slurm account flags or ``SLURM_ACCOUNT``-style
    env values, email addresses and ``user@host`` (except in ``support_contact``), credential
    patterns and secret-looking assignments. ``known_identifiers`` (e.g. the caller's remote
    user and account taken from their overlay) are matched case-insensitively everywhere.
    """
    idents = sorted({i.strip() for i in known_identifiers if len(i.strip()) >= 2})
    problems: list[str] = []
    data = profile.model_dump(mode="json", by_alias=True, exclude_none=True)
    for loc, key, text in _walk_strings(data):
        problems += _scan_string(loc, key, text, idents)
    return list(dict.fromkeys(problems))


def _scan_string(loc: str, key: str | None, text: str, idents: list[str]) -> list[str]:
    found: list[str] = []
    top_field = loc.split(".", 1)[0].split("[", 1)[0]
    if top_field not in _EMAIL_OK_FIELDS:
        found += [f"{loc}: contains an email address or user@host ({m!r})" for m in _EMAIL_RE.findall(text)]
    found += [f"{loc}: user-specific path {p!r}; use {{user}}" for p in _user_specific_paths(text)]
    found += [f"{loc}: references the home of user {u!r}" for u in _TILDE_USER_RE.findall(text)]
    accounts = _ACCOUNT_FLAG_RE.findall(text) + _ACCOUNT_ASSIGN_RE.findall(text)
    if _SLURM_CMD_RE.search(text):
        accounts += _SHORT_ACCOUNT_RE.findall(text)
    found += [
        f"{loc}: hard-codes Slurm account {acct!r}; accounts belong in the private overlay"
        for acct in accounts
        if not _is_placeholder(acct)
    ]
    if key is not None and key.upper() in _ACCOUNT_ENV_KEYS and text and not _is_placeholder(text):
        found.append(f"{loc}: hard-codes a Slurm account; accounts belong in the private overlay")
    if key is not None and _SECRET_ENV_KEY_RE.search(key) and text and not _is_placeholder(text):
        found.append(f"{loc}: secret-looking variable {key!r} has a literal value")
    found += [f"{loc}: looks like a {label}" for label, pat in _TOKEN_PATTERNS if pat.search(text)]
    found += [
        f"{loc}: literal {name} assignment"
        for name, value in _SECRET_ASSIGN_RE.findall(text)
        if not _is_placeholder(value)
    ]
    for ident in idents:
        if re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(ident)}(?![A-Za-z0-9])", text):
            found.append(f"{loc}: contains personal identifier {ident!r}")
    return found


def _user_specific_paths(text: str) -> Iterator[str]:
    for match in _PATH_RE.finditer(text):
        segments = match.group(0).split("/")
        for parent, child in itertools.pairwise(segments):
            if parent.lower() in _USER_PARENT_DIRS and _looks_like_username(child):
                yield match.group(0)
                break


def _looks_like_username(segment: str) -> bool:
    return bool(segment) and not _is_placeholder(segment) and segment.lower() not in _SHARED_SEGMENTS


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'")
    return v.startswith(("{", "$", "<", "%")) or v in {"", "~"}


def _walk_strings(value: Any, loc: str = "") -> Iterator[tuple[str, str | None, str]]:
    """Yield ``(location, dict key or None, text)`` for every string, including dict keys."""
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{loc}.{k}" if loc else str(k)
            yield (child, None, str(k))
            if isinstance(v, str):
                yield (child, str(k), v)
            else:
                yield from _walk_strings(v, child)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_strings(item, f"{loc}[{i}]")
    elif isinstance(value, str):
        yield (loc, None, value)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _check_name(value: str, what: str) -> str:
    if not _NAME_RE.match(value):
        raise InvalidArgument(
            f"invalid {what} {value!r} (letters, digits, '.', '_', '-'; max 64 chars)", details={what: value}
        )
    return value


def _looks_like_path(value: str | Path) -> bool:
    if isinstance(value, Path):
        return True
    return "/" in value or os.sep in value or value.endswith(PROFILE_SUFFIXES) or value.startswith("~")


def _profile_files(directory: Path) -> dict[str, Path]:
    """``name -> file`` for top-level profile YAMLs (``.yaml`` wins over ``.yml``).

    Hidden files and ``*.example.yaml`` templates are skipped; sub-directories (such as
    ``overlays/``) are not profiles.
    """
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix not in PROFILE_SUFFIXES:
            continue
        if path.stem.endswith(".example"):
            continue
        if path.stem in found:
            log.warning("ignoring %s: %s has the same profile name", path, found[path.stem])
            continue
        found[path.stem] = path
    return found


def _describe(name: str, path: Path, source: ProfileOrigin, *, shadows_bundled: bool) -> ProfileInfo:
    try:
        profile = _read_profile(path)
    except InvalidArgument as exc:
        return ProfileInfo(
            name=name,
            cluster_id=name,
            display_name=name,
            source=source,
            path=str(path),
            shadows_bundled=shadows_bundled,
            error=exc.message,
        )
    error = None
    if profile.cluster_id != name:
        error = f"file name {name!r} does not match cluster_id {profile.cluster_id!r}"
    return ProfileInfo(
        name=name,
        cluster_id=profile.cluster_id,
        display_name=profile.display_name,
        source=source,
        path=str(path),
        shadows_bundled=shadows_bundled,
        error=error,
    )


def _read_profile(path: Path) -> ClusterProfile:
    data = _read_yaml_mapping(path, what="profile")
    try:
        return ClusterProfile.model_validate(data)
    except ValidationError as exc:
        raise InvalidArgument(
            f"profile {path.name!r} is invalid",
            details={"path": str(path), "errors": _format_errors(exc)},
        ) from exc


def _read_yaml_mapping(path: Path, *, what: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > MAX_PROFILE_BYTES:
            raise InvalidArgument(
                f"{what} {path.name!r} is {size} bytes; limit is {MAX_PROFILE_BYTES}",
                details={"path": str(path)},
            )
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InvalidArgument(
            f"cannot read {what} {str(path)!r}: {exc}", details={"path": str(path)}
        ) from exc
    try:
        data = yaml_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = {"line": mark.line + 1, "column": mark.column + 1} if mark is not None else {}
        raise InvalidArgument(
            f"{what} {path.name!r} is not valid YAML", details={"path": str(path), **where}
        ) from exc
    if not isinstance(data, dict):
        raise InvalidArgument(f"{what} {path.name!r} must be a YAML mapping", details={"path": str(path)})
    return data


def _format_errors(exc: ValidationError, limit: int = 20) -> list[dict[str, str]]:
    return [
        {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]}
        for err in exc.errors(include_url=False)[:limit]
    ]


def _template_placeholders(data: Any) -> list[str]:
    """``loc=value`` for every string still holding a ``<fill-me-in>`` template marker."""
    return [f"{loc}={text}" for loc, _key, text in _walk_strings(data) if _TEMPLATE_RE.search(text)]


def _warn_if_exposed(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning(
            "overlay %s is accessible by group/others (mode %o); run chmod 600", path, stat.S_IMODE(mode)
        )


def _atomic_write_private(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` via a 0600 temp file in the same directory + rename."""
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _parse_time(value: str) -> datetime | None:
    try:
        return parse_iso(value)
    except ValueError:
        return None
