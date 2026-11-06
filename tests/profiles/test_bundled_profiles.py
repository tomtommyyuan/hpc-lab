"""The profiles hpclab ships: valid, lint-clean, free of personal data, and saying what they should."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from hpclab.config import Paths
from hpclab.domain.clock import parse_iso, parse_slurm_duration
from hpclab.domain.errors import InvalidArgument
from hpclab.profiles.loader import (
    active_facts,
    active_observations,
    bundled_profiles_dir,
    lint_profile,
    load_overlay,
    load_profile,
    validate_profile_is_public,
)
from hpclab.profiles.schema import ClusterProfile, ResolvedCluster, UserOverlay

PROFILES_DIR = bundled_profiles_dir()
PROFILE_FILES = sorted(p for p in PROFILES_DIR.glob("*.yaml") if not p.stem.endswith(".example"))
OVERLAY_EXAMPLES = sorted((PROFILES_DIR / "overlays").glob("*.example.yaml"))
EXAMPLE_DIR = PROFILES_DIR.parent / "examples" / "tiny-classifier"
EXAMPLE_FILES = sorted(
    p for p in EXAMPLE_DIR.iterdir() if p.is_file() and p.suffix in {".py", ".md", ".yaml"}
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# Truncated sha256 of lower-case tokens that appear in the private HAI operator notes (user name,
# Slurm account, GitHub handle, project and group names, real names). Stored hashed so this test
# does not itself publish them; it is obfuscation, not secrecy.
PRIVATE_TOKEN_HASHES = frozenset(
    {
        "af52016cf6e6d273",
        "6b5b58a715fb6456",
        "d683beb9a3a3aae9",
        "7063d51d1b2da165",
        "950de5d291c6708f",
        "8025b3da6a6bd75b",
        "2de8cec8e5a2ab99",
        "886f552c12a55d4b",
    }
)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def private_tokens(text: str, hashes: frozenset[str] = PRIVATE_TOKEN_HASHES) -> set[str]:
    return {t for t in set(re.findall(r"[a-z0-9]+", text.lower())) if token_hash(t) in hashes}


def test_private_token_detector_works() -> None:
    hashes = frozenset({token_hash("jdoe"), token_hash("acct42")})
    assert private_tokens("cd /hai/scratch/JDoe/x && sbatch --account=acct42", hashes) == {"jdoe", "acct42"}
    assert private_tokens("sbatch --account=<your-slurm-account> /hai/scratch/{user}", hashes) == set()
    assert len(PRIVATE_TOKEN_HASHES) == 8


@pytest.mark.parametrize("path", PROFILE_FILES, ids=lambda p: p.stem)
def test_bundled_profile_is_valid_clean_and_public(path: Path) -> None:
    profile = ClusterProfile.model_validate(yaml.safe_load(path.read_text()))
    assert profile.cluster_id == path.stem
    assert profile == load_profile(path.stem, None)
    assert lint_profile(profile) == []
    assert validate_profile_is_public(profile) == []
    assert all(s.retrieved_at is None or parse_iso(s.retrieved_at) for s in profile.sources)


@pytest.mark.parametrize("path", [*PROFILE_FILES, *OVERLAY_EXAMPLES, *EXAMPLE_FILES], ids=lambda p: p.name)
def test_shipped_files_contain_no_private_identifiers_even_in_comments(path: Path) -> None:
    text = path.read_text()
    assert private_tokens(text) == set(), f"{path.name} contains private identifiers"
    assert EMAIL_RE.findall(text) == []


# ---------------------------------------------------------------------------
# HAI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hai() -> ClusterProfile:
    return load_profile("hai", None)


def test_hai_identity_and_policy(hai: ClusterProfile) -> None:
    assert (hai.cluster_id, hai.transport, hai.scheduler) == ("hai", "ssh", "slurm")
    assert hai.login_host == "haic.stanford.edu"
    assert hai.support_contact == "hai-support"
    pol = hai.policy
    assert pol.login_node_compute == "forbidden"
    assert pol.require_account is True and pol.require_qos is False
    assert (pol.poll_min_interval_s, pol.sacct_min_interval_s) == (30, 60)
    assert any("sacctmgr" in n and "qos" in n.lower() for n in hai.notes)  # QoS must be checked, not assumed


def test_hai_partitions(hai: ClusterProfile) -> None:
    assert hai.default_partition == "hai"
    p = hai.partitions
    assert set(p) == {"hai", "hai-interactive", "hai-lo"}
    assert p["hai"].default and p["hai"].max_time == "7-00:00:00"
    assert p["hai"].gpu_types == ["h100", "b200"] and not p["hai"].preemptible
    assert p["hai-interactive"].max_time == "1-00:00:00"
    assert p["hai-interactive"].interactive_only and p["hai-interactive"].max_gpus_per_user == 1
    lo = p["hai-lo"]
    assert lo.max_time == "14-00:00:00" and lo.preemptible
    assert set(lo.gpu_types) == {"h100", "h200", "b200"}
    assert lo.notes is not None and "preempt" in lo.notes and "not for long" in lo.notes
    # Disputed per-user caps (documented 8 vs. observed looser) are not hard-coded as limits.
    assert p["hai"].max_gpus_per_user is None


def test_hai_storage_and_remote_root(hai: ClusterProfile) -> None:
    home, scratch = hai.storage["home"], hai.storage["scratch"]
    assert home.path == "/hai/users/{user}" and home.quota_hint == "50G"
    assert {"outputs", "venv", "checkpoints", "caches"} <= set(home.avoid_for)
    assert scratch.path == "/hai/scratch/{user}" and scratch.quota_hint == "5T"
    assert hai.remote_root_template == "{scratch}/hpclab"
    resolved = ResolvedCluster(profile=hai, overlay=UserOverlay(cluster_id="hai", user="alice"))
    assert resolved.remote_root() == "/hai/scratch/alice/hpclab"
    assert resolved.storage_path("home") == "/hai/users/alice"


def test_hai_runtime_recipes_keep_caches_off_home(hai: ClusterProfile) -> None:
    uv = hai.runtime_recipes["uv-scratch"]
    assert {"UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HOME"} <= set(uv.env)
    resolved = ResolvedCluster(profile=hai, overlay=UserOverlay(cluster_id="hai", user="alice"))
    for value in uv.env.values():
        assert resolved.expand(value).startswith("/hai/scratch/alice/")
    torch = hai.runtime_recipes["torch-cu128"]
    notes = torch.notes or ""
    assert "whl/cu128" in notes and "12080" in notes and "sm_100" in notes
    assert hai.gpu_arch == {"h100": "sm_90", "h200": "sm_90", "b200": "sm_100"}


def test_hai_nfs_blackhole_is_a_ttl_observation_not_a_rule(hai: ClusterProfile) -> None:
    nfs = next(o for o in hai.known_issues if "haic-hgx-2" in o.key)
    assert nfs.severity == "warning"
    assert nfs.observed_at.startswith("2026-08-23") and nfs.expires_at is not None
    assert nfs.expires_at.startswith("2026-10-23")
    assert nfs.value["node"] == "haic-hgx-2"
    assert nfs in active_observations(hai, datetime(2026, 9, 27, tzinfo=UTC))
    assert nfs not in active_observations(hai, datetime(2026, 10, 23, tzinfo=UTC))
    # Never a permanent exclude: no partition rule or policy field names the node.
    rules = yaml.safe_dump([r.model_dump() for r in hai.partitions.values()]) + hai.policy.model_dump_json()
    assert "haic-hgx-2" not in rules


def test_hai_driver_cuda_is_a_profile_fact_prepare_can_read(hai: ClusterProfile) -> None:
    """Spec §7.2 env_guard_static compares torch's CUDA build with *fact* ``driver_cuda``.

    It must be in ``facts`` under exactly that key with a plain "major.minor" string, or the
    check that would have caught HAIC's #1 failure (cu130 wheels on a 12.8 driver) is skipped.
    """
    facts = {f.key: f for f in active_facts(hai, datetime(2026, 9, 27, tzinfo=UTC))}
    assert facts["driver_cuda"].value == "12.8"
    assert re.fullmatch(r"\d+\.\d+", facts["driver_cuda"].value)
    assert not any("driver" in o.key for o in hai.known_issues)  # a fact, not an issue


def test_hai_time_varying_facts_all_expire(hai: ClusterProfile) -> None:
    assert {f.key for f in hai.facts} == {"driver_cuda", "node_inventory", "gpu_limits"}
    assert {o.key for o in hai.known_issues} == {"node.haic-hgx-2.nfs_client_blackhole"}
    assert all(o.expires_at for o in [*hai.facts, *hai.known_issues])
    far_future = datetime(2027, 6, 1, tzinfo=UTC)
    assert active_observations(hai, far_future) == []
    assert active_facts(hai, far_future) == []


def test_hai_sources_record_provenance_and_the_disagreement(hai: ClusterProfile) -> None:
    assert len(hai.sources) >= 2
    assert all(s.retrieved_at for s in hai.sources)
    public = next(s for s in hai.sources if s.url and "haic" in s.url)
    assert public.notes is not None and "disagree" in public.notes and "probe" in public.notes


def test_hai_known_identifiers_would_be_caught() -> None:
    """If the private playbook's lines were pasted in, the public check must object."""
    leaked = load_profile("hai", None).model_copy(
        update={"notes": ["repo at /hai/scratch/jdoe/proj", "sbatch --account=acct42 run.sbatch"]}
    )
    problems = validate_profile_is_public(leaked)
    assert any("user-specific path" in p for p in problems)
    assert any("'acct42'" in p for p in problems)


# ---------------------------------------------------------------------------
# Simulator and generic template
# ---------------------------------------------------------------------------


def test_sim_profile_mirrors_simulator_defaults() -> None:
    sim = load_profile("sim", None)
    assert sim.transport == "sim" and sim.login_host is None
    assert sim.default_partition == "gpu"
    expected = {
        "gpu": ("2-00:00:00", ["a100"], False, None),
        "interactive": ("04:00:00", ["a100"], False, 1),
        "scavenge": ("1-00:00:00", ["h100"], True, None),
    }
    actual = {
        name: (r.max_time, r.gpu_types, r.preemptible, r.max_gpus_per_user)
        for name, r in sim.partitions.items()
    }
    assert actual == expected
    assert sim.policy.poll_min_interval_s == 1 and sim.policy.require_account
    assert sim.storage["scratch"].path == "{sim_fs}/scratch/{user}"


def test_sim_profile_matches_simulator_config() -> None:
    """The profile promises what the simulator enforces; drift would make admission lie."""
    sim_config = pytest.importorskip("hpclab.sim.config")
    config = sim_config.SimConfig()
    profile = load_profile("sim", None)
    assert {p.name for p in config.partitions} == set(profile.partitions)
    for part in config.partitions:
        rule = profile.partitions[part.name]
        assert rule.default == part.default
        assert parse_slurm_duration(rule.max_time) == part.max_time_s
        assert rule.preemptible == part.preemptible
        assert rule.max_gpus_per_user == part.max_gpus_per_user
        node_types = {n.gpu_type for n in config.nodes if n.name in part.nodes and n.gpu_type}
        assert set(rule.gpu_types) == node_types
    assert set(profile.gpu_arch) >= config.gpu_types()
    assert "lab" in config.accounts  # the account the overlay template and README use


def test_sim_remote_root_comes_from_the_overlay() -> None:
    sim = load_profile("sim", None)
    overlay = UserOverlay(
        cluster_id="sim", user="alice", account="lab", remote_root="/tmp/sim/fs/scratch/alice/hpclab"
    )
    assert ResolvedCluster(profile=sim, overlay=overlay).remote_root() == "/tmp/sim/fs/scratch/alice/hpclab"


def test_generic_template_is_usable_as_a_starting_point(tmp_path: Path) -> None:
    generic = load_profile("generic-slurm", None)
    assert generic.transport == "ssh" and generic.login_host is not None
    assert generic.login_host.endswith("example.edu")
    for rule in generic.partitions.values():
        assert rule.max_time is None or parse_slurm_duration(rule.max_time)
    # The documented workflow: copy, rename, set cluster_id.
    home = Paths(tmp_path / "home")
    copied = home.profiles / "mycluster.yaml"
    copied.parent.mkdir(parents=True)
    copied.write_text(
        (PROFILES_DIR / "generic-slurm.yaml")
        .read_text()
        .replace("cluster_id: generic-slurm", "cluster_id: mycluster")
    )
    assert load_profile("mycluster", home).cluster_id == "mycluster"


def test_generic_template_commented_examples_are_valid_once_uncommented(tmp_path: Path) -> None:
    text = (PROFILES_DIR / "generic-slurm.yaml").read_text()
    for section in ("facts", "known_issues"):
        commented = re.search(rf"^{section}: \[\]\n((?:#  .*\n)+)", text, flags=re.MULTILINE)
        assert commented is not None, f"commented {section} example not found"
        uncommented = re.sub(r"^#", "", commented.group(1), flags=re.MULTILINE)
        text = text.replace(commented.group(0), f"{section}:\n{uncommented}")
    f = tmp_path / "site.yaml"
    f.write_text(text)
    profile = load_profile(str(f), None)
    assert [o.key for o in profile.facts] == ["driver_cuda"] and profile.facts[0].value == "12.4"
    assert len(profile.known_issues) == 1
    assert lint_profile(profile) == []


# ---------------------------------------------------------------------------
# Overlay templates
# ---------------------------------------------------------------------------


def test_every_bundled_profile_with_identity_has_an_overlay_template() -> None:
    assert {p.name for p in OVERLAY_EXAMPLES} == {"hai.example.yaml", "sim.example.yaml"}


@pytest.mark.parametrize("path", OVERLAY_EXAMPLES, ids=lambda p: p.name)
def test_overlay_template_validates_once_filled_in(path: Path, tmp_path: Path) -> None:
    raw = path.read_text()
    placeholders = set(re.findall(r"<[a-z][a-z0-9-]*>", raw))
    assert placeholders, "templates must mark what the user has to fill in"
    filled = raw
    for ph in placeholders:
        filled = filled.replace(ph, "/abs" if "path" in ph else "alice")
    overlay = UserOverlay.model_validate(yaml.safe_load(filled))
    cluster_id = path.name.removesuffix(".example.yaml")
    assert overlay.cluster_id == cluster_id
    load_profile(cluster_id, None)  # the matching public profile exists


@pytest.mark.parametrize("path", OVERLAY_EXAMPLES, ids=lambda p: p.name)
def test_overlay_template_copied_verbatim_is_refused(path: Path, tmp_path: Path) -> None:
    home = Paths(tmp_path / "home")
    cluster_id = path.name.removesuffix(".example.yaml")
    home.overlays.mkdir(parents=True)
    shutil.copy(path, home.overlays / f"{cluster_id}.yaml")
    with pytest.raises(InvalidArgument, match="placeholders"):
        load_overlay(cluster_id, home)


def test_hai_overlay_template_mentions_account_lookup() -> None:
    text = (PROFILES_DIR / "overlays" / "hai.example.yaml").read_text()
    assert "<your-sunet-id>" in text and "<your-slurm-account>" in text
    assert "sacctmgr" in text
