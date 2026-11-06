"""Profile discovery/precedence, private overlays, observation TTLs, lint and the public check."""

from __future__ import annotations

import logging
import os
import stat
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from hpclab.config import Paths
from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.profiles import loader
from hpclab.profiles.loader import (
    active_facts,
    active_observations,
    bundled_profiles_dir,
    lint_profile,
    list_profiles,
    load_overlay,
    load_profile,
    overlay_path,
    resolve_cluster,
    save_overlay,
    validate_profile_is_public,
)
from hpclab.profiles.schema import ClusterProfile, NotifyChannel, SiteObservation, UserOverlay

REPO = Path(__file__).resolve().parents[2]


def profile_data(cluster_id: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": "hpclab.profile/v1",
        "cluster_id": cluster_id,
        "display_name": f"{cluster_id} cluster",
        "partitions": {"gpu": {"default": True, "max_time": "1-00:00:00", "gpu_types": ["a100"]}},
        "default_partition": "gpu",
        "storage": {"scratch": {"path": "/scratch/{user}"}},
        "gpu_arch": {"a100": "sm_80"},
        "sources": [{"title": "site docs", "url": "https://hpc.example.edu", "retrieved_at": "2026-09-01"}],
    }
    data.update(overrides)
    return data


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def make_profile(**overrides: Any) -> ClusterProfile:
    return ClusterProfile.model_validate(profile_data("site", **overrides))


@pytest.fixture
def bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated 'bundled' directory so precedence tests do not depend on shipped files."""
    d = tmp_path / "bundled"
    write_yaml(d / "alpha.yaml", profile_data("alpha", display_name="Alpha (bundled)"))
    write_yaml(d / "beta.yaml", profile_data("beta"))
    write_yaml(d / "overlays" / "alpha.example.yaml", {"cluster_id": "alpha"})
    write_yaml(d / "gamma.example.yaml", profile_data("gamma"))
    (d / ".hidden.yaml").write_text("not: [valid")
    (d / "README.md").write_text("not a profile")
    monkeypatch.setattr(loader, "bundled_profiles_dir", lambda: d)
    return d


# ---------------------------------------------------------------------------
# Discovery and precedence
# ---------------------------------------------------------------------------


def test_bundled_dir_in_a_checkout_is_repo_profiles() -> None:
    assert bundled_profiles_dir() == REPO / "profiles"
    assert (bundled_profiles_dir() / "hai.yaml").is_file()


def test_bundled_dir_prefers_installed_package_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "site-packages" / "hpclab"
    (pkg / "_bundled" / "profiles").mkdir(parents=True)
    monkeypatch.setattr(loader.resources, "files", lambda name: pkg)
    assert bundled_profiles_dir() == pkg / "_bundled" / "profiles"


def test_list_profiles_bundled_only(bundled: Path) -> None:
    infos = list_profiles(None)
    assert [(i.name, i.source, i.error) for i in infos] == [
        ("alpha", "bundled", None),
        ("beta", "bundled", None),
    ]
    assert infos[0].display_name == "Alpha (bundled)"
    assert infos[0].path == str(bundled / "alpha.yaml")


def test_local_profiles_shadow_bundled_and_errors_are_listed(bundled: Path, paths: Paths) -> None:
    write_yaml(paths.profiles / "alpha.yaml", profile_data("alpha", display_name="Alpha (mine)"))
    write_yaml(paths.profiles / "delta.yml", profile_data("delta"))
    (paths.profiles / "broken.yaml").write_text("cluster_id: broken\n")  # schema-invalid
    write_yaml(paths.profiles / "renamed.yaml", profile_data("other-id"))
    infos = {i.name: i for i in list_profiles(paths)}

    assert set(infos) == {"alpha", "beta", "broken", "delta", "renamed"}
    alpha = infos["alpha"]
    assert (alpha.source, alpha.shadows_bundled, alpha.display_name) == ("local", True, "Alpha (mine)")
    assert infos["beta"].source == "bundled" and not infos["beta"].shadows_bundled
    assert infos["delta"].source == "local" and infos["delta"].error is None
    assert infos["broken"].error is not None and infos["broken"].cluster_id == "broken"
    assert infos["renamed"].error is not None and "does not match" in infos["renamed"].error


def test_load_profile_local_override_wins(bundled: Path, paths: Paths) -> None:
    assert load_profile("alpha", paths).display_name == "Alpha (bundled)"
    write_yaml(paths.profiles / "alpha.yaml", profile_data("alpha", display_name="Alpha (mine)"))
    assert load_profile("alpha", paths).display_name == "Alpha (mine)"
    assert load_profile("alpha", None).display_name == "Alpha (bundled)"  # None = bundled only


def test_yaml_suffix_wins_over_yml(bundled: Path, paths: Paths) -> None:
    write_yaml(paths.profiles / "beta.yml", profile_data("beta", display_name="yml"))
    write_yaml(paths.profiles / "beta.yaml", profile_data("beta", display_name="yaml"))
    assert load_profile("beta", paths).display_name == "yaml"


def test_load_profile_by_explicit_path(bundled: Path, tmp_path: Path, paths: Paths) -> None:
    f = write_yaml(tmp_path / "anywhere" / "custom-name.yaml", profile_data("zeta"))
    assert load_profile(str(f), paths).cluster_id == "zeta"  # explicit files may be named freely
    assert load_profile(f, paths).cluster_id == "zeta"
    with pytest.raises(NotFound):
        load_profile(str(tmp_path / "missing.yaml"), paths)


def test_load_profile_not_found_lists_available(bundled: Path, paths: Paths) -> None:
    write_yaml(paths.profiles / "mine.yaml", profile_data("mine"))
    with pytest.raises(NotFound) as err:
        load_profile("nope", paths)
    assert err.value.details["available"] == ["alpha", "beta", "mine"]
    assert str(paths.profiles) in err.value.details["searched"]


@pytest.mark.parametrize("name", ["bad name", ".hidden", "-x", "a" * 65, ""])
def test_load_profile_rejects_invalid_names(bundled: Path, paths: Paths, name: str) -> None:
    with pytest.raises(InvalidArgument):
        load_profile(name, paths)


def test_load_profile_requires_file_name_to_match_cluster_id(bundled: Path, paths: Paths) -> None:
    write_yaml(paths.profiles / "sherlock.yaml", profile_data("generic-slurm"))
    with pytest.raises(InvalidArgument, match="must match"):
        load_profile("sherlock", paths)


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("cluster_id: [unclosed\n", "not valid YAML"),
        ("- a\n- list\n", "must be a YAML mapping"),
        ("", "must be a YAML mapping"),
        ("cluster_id: x\ndisplay_name: X\nunknown_field: 1\n", "is invalid"),
        ("cluster_id: x\ndisplay_name: X\npartitions: {gpu: {max_gpus_per_job: many}}\n", "is invalid"),
    ],
)
def test_malformed_profiles_raise_invalid_argument(tmp_path: Path, content: str, fragment: str) -> None:
    f = tmp_path / "x.yaml"
    f.write_text(content)
    with pytest.raises(InvalidArgument, match=fragment) as err:
        load_profile(str(f), None)
    assert err.value.details["path"] == str(f.resolve())


def test_schema_errors_are_itemized(tmp_path: Path) -> None:
    f = write_yaml(
        tmp_path / "x.yaml", {"cluster_id": "x", "display_name": "X", "transport": "carrier-pigeon"}
    )
    with pytest.raises(InvalidArgument) as err:
        load_profile(str(f), None)
    assert [e["loc"] for e in err.value.details["errors"]] == ["transport"]


def test_yaml_syntax_error_reports_line(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("cluster_id: x\ndisplay_name: [broken\n")
    with pytest.raises(InvalidArgument) as err:
        load_profile(str(f), None)
    assert err.value.details["line"] >= 2


def test_yaml_1_1_scalars_in_hand_written_profiles_stay_strings(tmp_path: Path) -> None:
    """Unquoted dates and HH:MM:SS are strings in a profile, not YAML 1.1 dates/base-60 ints.

    Otherwise ``max_time: 12:00:00`` silently becomes 43200 (then fails as "not a string") and
    ``retrieved_at: 2026-09-01`` becomes a ``date``: both reject a profile a human wrote correctly.
    """
    f = tmp_path / "x.yaml"
    f.write_text(
        "cluster_id: x\n"
        "display_name: X\n"
        "partitions:\n"
        "  gpu: {max_time: 12:00:00}\n"
        "  debug: {max_time: 30:00}\n"
        "known_issues:\n"
        "  - {key: k, value: 1, source: s, observed_at: 2026-08-01T00:00:00Z, expires_at: 2026-10-01}\n"
        "sources:\n"
        "  - {title: docs, retrieved_at: 2026-09-01}\n"
    )
    profile = load_profile(str(f), None)
    assert profile.partitions["gpu"].max_time == "12:00:00"
    assert profile.partitions["debug"].max_time == "30:00"
    assert profile.known_issues[0].observed_at == "2026-08-01T00:00:00Z"
    assert profile.known_issues[0].expires_at == "2026-10-01"
    assert profile.sources[0].retrieved_at == "2026-09-01"
    # Ordinary numbers and booleans still resolve as usual.
    assert loader.yaml_load("a: 30\nb: 1.5\nc: true\nd: 0x10") == {"a": 30, "b": 1.5, "c": True, "d": 16}


def test_oversized_profile_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "MAX_PROFILE_BYTES", 10)
    f = write_yaml(tmp_path / "x.yaml", profile_data("x"))
    with pytest.raises(InvalidArgument, match="limit"):
        load_profile(str(f), None)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


def full_overlay() -> UserOverlay:
    return UserOverlay(
        cluster_id="alpha",
        ssh_host="alpha-login",
        user="alice",
        account="lab-acct",
        qos="normal",
        remote_root="/scratch/alice/hpclab",
        ssh_options=["ServerAliveInterval=30"],
        lab_users=["alice", "bob"],
        notify=[NotifyChannel(kind="ntfy", target="https://ntfy.sh/topic")],
    )


def test_overlay_round_trip_with_private_permissions(paths: Paths) -> None:
    os.chmod(paths.overlays, 0o755)  # even a pre-existing, too-open directory gets tightened
    target = save_overlay(full_overlay(), paths)

    assert target == overlay_path("alpha", paths) == paths.overlays / "alpha.yaml"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.overlays.stat().st_mode) == 0o700
    assert target.read_text().startswith("# hpclab private cluster overlay")
    assert [p.name for p in paths.overlays.iterdir()] == ["alpha.yaml"]  # no temp files left
    assert load_overlay("alpha", paths) == full_overlay()


def test_save_overlay_overwrites_atomically_and_minimally(paths: Paths) -> None:
    save_overlay(full_overlay(), paths)
    save_overlay(UserOverlay(cluster_id="alpha", user="alice"), paths)
    text = (paths.overlays / "alpha.yaml").read_text()
    assert yaml.safe_load(text) == {"cluster_id": "alpha", "user": "alice"}
    assert stat.S_IMODE((paths.overlays / "alpha.yaml").stat().st_mode) == 0o600


def test_save_overlay_creates_missing_directory(tmp_path: Path) -> None:
    fresh = Paths(tmp_path / "fresh-home")
    target = save_overlay(UserOverlay(cluster_id="beta", user="bob"), fresh)
    assert target.is_file() and stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_missing_overlay_is_none(paths: Paths) -> None:
    assert load_overlay("alpha", paths) is None


def test_overlay_cluster_id_is_implied_by_file_name(paths: Paths) -> None:
    write_yaml(paths.overlays / "alpha.yaml", {"user": "alice", "account": "acct"})
    overlay = load_overlay("alpha", paths)
    assert overlay is not None and overlay.cluster_id == "alpha" and overlay.account == "acct"


def test_overlay_cluster_id_mismatch_is_rejected(paths: Paths) -> None:
    write_yaml(paths.overlays / "alpha.yaml", {"cluster_id": "beta"})
    with pytest.raises(InvalidArgument, match="declares cluster_id"):
        load_overlay("alpha", paths)


def test_overlay_with_template_placeholders_is_rejected(paths: Paths) -> None:
    write_yaml(paths.overlays / "alpha.yaml", {"user": "<your-sunet-id>", "account": "<your-slurm-account>"})
    with pytest.raises(InvalidArgument, match="placeholders") as err:
        load_overlay("alpha", paths)
    assert err.value.details["placeholders"] == ["user=<your-sunet-id>", "account=<your-slurm-account>"]
    with pytest.raises(InvalidArgument, match="placeholders"):
        save_overlay(UserOverlay(cluster_id="alpha", account="<acct>"), paths)


def test_overlay_schema_errors_are_invalid_argument(paths: Paths) -> None:
    write_yaml(paths.overlays / "alpha.yaml", {"user": "alice", "password": "nope"})
    with pytest.raises(InvalidArgument) as err:
        load_overlay("alpha", paths)
    assert err.value.details["errors"][0]["loc"] == "password"


@pytest.mark.parametrize("cluster_id", ["../escape", "a/b", "", ".hidden"])
def test_overlay_paths_reject_traversal(paths: Paths, cluster_id: str) -> None:
    with pytest.raises(InvalidArgument):
        load_overlay(cluster_id, paths)
    with pytest.raises(InvalidArgument):
        overlay_path(cluster_id, paths)


def test_group_readable_overlay_is_loaded_with_warning(
    paths: Paths, caplog: pytest.LogCaptureFixture
) -> None:
    f = write_yaml(paths.overlays / "alpha.yaml", {"user": "alice"})
    os.chmod(f, 0o644)
    with caplog.at_level(logging.WARNING, logger="hpclab.profiles"):
        assert load_overlay("alpha", paths) is not None
    assert "chmod 600" in caplog.text


def test_resolve_cluster_combines_profile_and_overlay(bundled: Path, paths: Paths) -> None:
    bare = resolve_cluster("alpha", paths)
    assert bare.overlay == UserOverlay(cluster_id="alpha")
    save_overlay(UserOverlay(cluster_id="alpha", user="alice"), paths)
    resolved = resolve_cluster("alpha", paths)
    assert resolved.remote_root() == "/scratch/alice/hpclab"
    explicit = resolve_cluster("alpha", paths, overlay=UserOverlay(cluster_id="alpha", user="bob"))
    assert explicit.remote_root() == "/scratch/bob/hpclab"
    with pytest.raises(InvalidArgument):
        resolve_cluster("alpha", paths, overlay=UserOverlay(cluster_id="beta"))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def obs(key: str, expires_at: str | None, observed_at: str = "2026-08-01T00:00:00Z") -> dict[str, Any]:
    return {"key": key, "value": 1, "source": "test", "observed_at": observed_at, "expires_at": expires_at}


def test_active_observations_drop_expired() -> None:
    profile = make_profile(
        known_issues=[
            obs("future", "2026-10-01T00:00:00Z"),
            obs("past", "2026-09-01T00:00:00Z"),
            obs("boundary", "2026-09-27T12:00:00Z"),  # expires *at* the instant: gone
            obs("no-expiry", None),
            obs("garbage", "next tuesday"),
            obs("date-only", "2026-09-28"),
        ]
    )
    assert [o.key for o in active_observations(profile, NOW)] == ["future", "no-expiry", "date-only"]
    later = NOW + timedelta(days=30)
    assert [o.key for o in active_observations(profile, later)] == ["no-expiry"]


def test_active_facts_use_the_same_ttl_rules() -> None:
    profile = make_profile(
        facts=[obs("driver_cuda", "2026-10-01T00:00:00Z"), obs("old", "2026-09-01T00:00:00Z")],
        known_issues=[obs("issue", "2026-10-01T00:00:00Z")],
    )
    assert [o.key for o in active_facts(profile, NOW)] == ["driver_cuda"]
    assert [o.key for o in active_observations(profile, NOW)] == ["issue"]  # facts are not issues


def test_active_observations_timezone_handling() -> None:
    profile = make_profile(known_issues=[obs("x", "2026-09-27T12:30:00Z")])
    assert active_observations(profile, datetime(2026, 9, 27, 12, 0))  # naive = UTC
    pacific = timezone(timedelta(hours=-7))
    assert not active_observations(profile, datetime(2026, 9, 27, 6, 0, tzinfo=pacific))  # 13:00 UTC
    assert all(isinstance(o, SiteObservation) for o in active_observations(profile, NOW))


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


def test_lint_clean_profile() -> None:
    assert lint_profile(make_profile(known_issues=[obs("ok", "2026-10-01T00:00:00Z")])) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"default_partition": "nope"}, "is not a declared partition"),
        (
            {"partitions": {"a": {"default": True}, "b": {"default": True}}, "default_partition": None},
            "more than one partition marked default",
        ),
        (
            {"partitions": {"gpu": {"default": True}, "cpu": {}}, "default_partition": "cpu"},
            "disagrees with default: true",
        ),
        ({"partitions": {"gpu": {"max_time": "forever"}}}, "is not a Slurm duration"),
        ({"partitions": {"gpu": {"gpu_types": ["h100"]}}}, "has no gpu_arch entry"),
        (
            {"partitions": {"gpu": {"max_gpus_per_job": 8, "max_gpus_per_user": 4}}},
            "exceeds max_gpus_per_user",
        ),
        ({"storage": {"scratch": {"path": "/scratch/{username}"}}}, "placeholder {username}"),
        ({"storage": {"scratch": {"path": "{sim_fs}/scratch/{user}"}}}, "placeholder {sim_fs}"),
        # storage_path() only substitutes {user}: anything else stays literal in remote_root().
        (
            {"storage": {"home": {"path": "/home/{user}"}, "scratch": {"path": "{home}/scratch"}}},
            "storage.scratch.path: placeholder {home}",
        ),
        ({"storage": {"scratch": {"path": "{remote_root}/x"}}}, "placeholder {remote_root}"),
        ({"facts": [obs("driver_cuda", None)]}, "facts[0] (driver_cuda): observations need expires_at"),
        ({"facts": [obs("f", "2026-10-01"), obs("f", "2026-10-02")]}, "facts[1] (f): duplicate key"),
        ({"remote_root_template": "{home}/hpclab"}, "placeholder {home}"),
        ({"storage": {}}, "no 'scratch' storage role"),
        (
            {"runtime_recipes": {"r": {"description": "d", "env": {"X": "{project}/x"}}}},
            "placeholder {project}",
        ),
        ({"known_issues": [obs("k", None)]}, "need expires_at"),
        ({"known_issues": [obs("k", "2026-07-01T00:00:00Z")]}, "not after observed_at"),
        ({"known_issues": [obs("k", "soon")]}, "expires_at 'soon'"),
        ({"known_issues": [obs("k", "2026-10-01", observed_at="yesterday")]}, "observed_at 'yesterday'"),
        ({"known_issues": [obs("k", "2026-10-01"), obs("k", "2026-10-02")]}, "duplicate key"),
        ({"sources": []}, "at least one source"),
        ({"policy": {"poll_min_interval_s": 0}}, "poll_min_interval_s must be > 0"),
        ({"policy": {"accounting_lag_s": -1}}, "accounting_lag_s must be >= 0"),
    ],
)
def test_lint_detects(overrides: dict[str, Any], fragment: str) -> None:
    problems = lint_profile(make_profile(**overrides))
    assert any(fragment in p for p in problems), problems


def test_lint_allows_sim_fs_only_for_sim_transport() -> None:
    sim = make_profile(transport="sim", storage={"scratch": {"path": "{sim_fs}/scratch/{user}"}})
    assert lint_profile(sim) == []


# ---------------------------------------------------------------------------
# Public-safety check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"storage": {"scratch": {"path": "/hai/scratch/jdoe"}}}, "user-specific path"),
        ({"storage": {"scratch": {"path": "/home/jdoe/work"}}}, "user-specific path"),
        ({"notes": ["the repo lives in /hai/scratch/jdoe/project"]}, "user-specific path"),
        ({"notes": ["see ~jdoe/notes.txt"]}, "home of user 'jdoe'"),
        ({"notes": ["sbatch --account=acct42 job.sh"]}, "hard-codes Slurm account 'acct42'"),
        ({"notes": ["sbatch -A acct42 job.sh"]}, "hard-codes Slurm account 'acct42'"),
        # Batch-script directives and exported env assignments, as found in cluster.env files.
        ({"notes": ["#SBATCH -A acct42"]}, "hard-codes Slurm account 'acct42'"),
        (
            {"runtime_recipes": {"r": {"description": "d", "setup": ["export SLURM_ACCOUNT=acct42"]}}},
            "hard-codes Slurm account 'acct42'",
        ),
        ({"notes": ["cluster.env: SLURM_PARTITION=hai SBATCH_ACCOUNT='acct42'"]}, "Slurm account 'acct42'"),
        (
            {"runtime_recipes": {"r": {"description": "d", "env": {"SLURM_ACCOUNT": "acct42"}}}},
            "hard-codes a Slurm account",
        ),
        ({"notes": ["mail me at jane.doe@example.edu"]}, "email address"),
        ({"login_host": "jdoe@login.example.edu"}, "email address or user@host"),
        ({"notes": ["token ghp_abcdefghijklmnopqrstuvwxyz0123456789"]}, "GitHub token"),
        ({"notes": ["hf_abcdefghijklmnopqrstuvwxyz0123"]}, "Hugging Face token"),
        ({"notes": ["-----BEGIN OPENSSH PRIVATE KEY-----"]}, "private key"),
        ({"notes": ["password=hunter2"]}, "literal password assignment"),
        (
            {"runtime_recipes": {"r": {"description": "d", "env": {"WANDB_API_KEY": "abc123"}}}},
            "secret-looking variable",
        ),
    ],
)
def test_public_check_flags_personal_data(overrides: dict[str, Any], fragment: str) -> None:
    problems = validate_profile_is_public(make_profile(**overrides))
    assert any(fragment in p for p in problems), problems


def test_public_check_scans_mapping_keys() -> None:
    profile = make_profile(runtime_recipes={"jdoe-env": {"description": "d"}})
    problems = validate_profile_is_public(profile, known_identifiers=["jdoe"])
    assert problems == ["runtime_recipes.jdoe-env: contains personal identifier 'jdoe'"]


def test_public_check_known_identifiers_are_case_insensitive_whole_words() -> None:
    profile = make_profile(notes=["Ask JDoe for access", "jdoesmith is someone else"])
    problems = validate_profile_is_public(profile, known_identifiers=["jdoe", " ", "x"])
    assert len(problems) == 1 and "notes[0]" in problems[0]


def test_public_check_accepts_placeholders_and_shared_paths() -> None:
    profile = make_profile(
        storage={
            "home": {"path": "/home/{user}"},
            "scratch": {"path": "/hai/scratch/$USER"},
            "shared": {"path": "/hai/scratch/shared/datasets"},
        },
        runtime_recipes={
            "r": {
                "description": "d",
                "setup": [
                    "sbatch --account=<your-account> x",
                    "cd ${USER}",
                    "ls /home/%u",
                    "export SLURM_ACCOUNT=${HPCLAB_ACCOUNT}",
                    "#SBATCH -A {account}",
                    "srun -N1 hostname",
                ],
                "env": {
                    "HF_TOKEN": "${HF_TOKEN}",
                    "SLURM_ACCOUNT": "{account}",
                    "HF_HOME": "{scratch}/.cache",
                },
            }
        },
        support_contact="research-computing-support@example.edu",
        notes=["docs at https://hpc.example.edu/guide", "use ~/.ssh/config", "ls -A /hai/scratch/$USER"],
    )
    assert validate_profile_is_public(profile) == []


def test_lint_ignores_shell_expansions_in_recipes() -> None:
    recipe = {
        "description": "d",
        "setup": ["export PATH=${HOME}/bin:$PATH"],
        "env": {"TMP": "${SLURM_TMPDIR}"},
    }
    assert lint_profile(make_profile(runtime_recipes={"r": recipe})) == []
