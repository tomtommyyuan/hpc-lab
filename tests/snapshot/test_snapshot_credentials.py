"""Credential stores never reach a code snapshot.

A snapshot is uploaded to shared cluster storage and anything in it can be printed by the
job, so (1) a project root may not be the home directory, one of its ancestors, or a
credential store such as ``~/.config/gh`` (whatever frontend supplied the path), and (2)
well-known token files are excluded by name and by content even inside a legitimate project.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from snapshot_support import GitRepo

from hpclab.domain.errors import InvalidArgument
from hpclab.domain.models import CodeOptions
from hpclab.services.snapshot import SnapshotStore, capture_snapshot, is_secret_path, scan_project

GH_TOKEN = "gho_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
HF_TOKEN = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"
ANTHROPIC_KEY = "sk-ant-api03-" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe" * 2
KUBE_KEY = "LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFb3dJQkFBS0NBUUVB"


def write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# project_root containment
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home" / "alice"
    write(home, ".config/gh/hosts.yml", f"github.com:\n    oauth_token: {GH_TOKEN}\n")
    write(home, ".config/nvim/init.lua", "vim.opt.number = true\n")
    write(home, ".ssh/id_ed25519", "not really a key\n")
    write(home, ".cache/huggingface/token", HF_TOKEN)
    write(home, ".claude/.credentials.json", json.dumps({"claudeAiOauth": {"accessToken": ANTHROPIC_KEY}}))
    write(home, ".codex/worktrees/abc/proj/train.py", "print('agent worktree')\n")
    write(home, "research/proj/train.py", "print('train')\n")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize(
    "rel",
    [
        ".",  # the agent was started in $HOME and the spec left project_root at "."
        ".config",
        ".config/gh",
        ".ssh",
        ".cache",
        ".cache/huggingface",
        ".claude",
        "..",  # an ancestor of $HOME
        "../..",
    ],
)
def test_home_ancestors_and_credential_stores_are_refused_as_project_roots(fake_home: Path, rel: str) -> None:
    root = fake_home / rel
    with pytest.raises(InvalidArgument, match="credential") as err:
        scan_project(root, CodeOptions(project_root=str(root)))
    assert err.value.details["project_root"] == str(root.resolve())


def test_filesystem_root_is_refused_before_anything_is_walked(fake_home: Path) -> None:
    with pytest.raises(InvalidArgument, match="credential"):
        scan_project(Path("/"), CodeOptions(project_root="/"))


def test_tilde_project_root_is_refused(fake_home: Path) -> None:
    with pytest.raises(InvalidArgument, match="credential"):
        scan_project(Path("~/.config"), CodeOptions(project_root="~/.config"))


def test_symlink_into_a_credential_store_is_refused(fake_home: Path, tmp_path: Path) -> None:
    link = tmp_path / "innocent-project"
    link.symlink_to(fake_home / ".config")
    with pytest.raises(InvalidArgument, match="credential"):
        scan_project(link, CodeOptions(project_root=str(link)))


def _alias_is_same_dir(alias: Path, real: Path) -> bool:
    try:
        return str(alias) != str(real) and os.path.samefile(alias, real)
    except OSError:
        return False


@pytest.mark.parametrize("rel", [".CONFIG", ".Config/GH", ".SSH", ".Cache/HuggingFace", ""])
def test_case_aliases_of_credential_stores_are_refused(fake_home: Path, rel: str) -> None:
    """On a case-insensitive filesystem (macOS's default) ``~/.CONFIG`` *is* ``~/.config`` but
    ``resolve()`` keeps the spelling: the guard must compare directories, not strings."""
    real = fake_home / rel.lower() if rel else fake_home
    alias = fake_home / rel if rel else Path(str(fake_home).upper())
    if not _alias_is_same_dir(alias, real):
        pytest.skip("case-sensitive filesystem: the alias is a different directory")
    with pytest.raises(InvalidArgument, match="credential"):
        scan_project(alias, CodeOptions(project_root=str(alias)))


def test_firmlink_alias_of_the_home_directory_is_refused(fake_home: Path) -> None:
    """macOS reaches the data volume under ``/System/Volumes/Data`` too; ``resolve()`` leaves
    that spelling alone, so a path comparison alone would let the whole home through."""
    for real in (fake_home, fake_home / ".config"):
        alias = Path("/System/Volumes/Data") / str(real.resolve()).lstrip("/")
        if not _alias_is_same_dir(alias, real):
            pytest.skip("no /System/Volumes/Data alias for the temp dir on this platform")
        with pytest.raises(InvalidArgument, match="credential"):
            scan_project(alias, CodeOptions(project_root=str(alias)))


@pytest.mark.parametrize("rel", ["research/proj", ".codex/worktrees/abc/proj", ".config/nvim"])
def test_ordinary_directories_under_home_remain_valid_roots(fake_home: Path, rel: str) -> None:
    scan = scan_project(fake_home / rel, CodeOptions())
    assert scan.file_count == 1


@pytest.mark.parametrize("rel", ["Research/Proj", ".CONFIG/nvim"])
def test_case_aliases_of_ordinary_directories_remain_valid_roots(fake_home: Path, rel: str) -> None:
    alias = fake_home / rel
    if not _alias_is_same_dir(alias, fake_home / rel.lower()):
        pytest.skip("case-sensitive filesystem: the alias is a different directory")
    assert scan_project(alias, CodeOptions()).file_count == 1


# ---------------------------------------------------------------------------
# name- and content-based exclusion inside a project
# ---------------------------------------------------------------------------

CREDENTIAL_FILES: dict[str, str] = {
    # the credential stores named in the finding, as they appear under $HOME ...
    ".config/gh/hosts.yml": f"github.com:\n    user: alice\n    oauth_token: {GH_TOKEN}\n",
    ".docker/config.json": json.dumps(
        {"auths": {"https://index.docker.io/v1/": {"auth": "YWxpY2U6aHVudGVyMg=="}}}
    ),
    ".kube/config": f"users:\n- name: alice\n  user:\n    client-key-data: {KUBE_KEY}\n",
    ".config/gcloud/application_default_credentials.json": json.dumps(
        {
            "client_id": "x.apps.googleusercontent.com",
            "refresh_token": "1//0gFAKEFAKEFAKEFAKEFAKEFAKE",
            "type": "authorized_user",
        }
    ),
    ".config/gcloud/access_tokens.db": "SQLite format 3\x00",
    ".config/gcloud/credentials.db": "SQLite format 3\x00",
    ".cache/huggingface/token": HF_TOKEN,
    ".huggingface/token": HF_TOKEN,
    ".claude/.credentials.json": json.dumps({"claudeAiOauth": {"accessToken": ANTHROPIC_KEY}}),
    ".npmrc": "//registry.npmjs.org/:_authToken=npm_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE\n",
    "infra/prod.tfvars": 'db_password = "hunter2"\n',
    # ... and the same stores copied under names that no name rule can know about
    "gh/hosts.yml": f"github.com:\n    oauth_token: {GH_TOKEN}\n",
    "docker/config.json": json.dumps({"auths": {"ghcr.io": {"auth": "YWxpY2U6Z2hwX3NlY3JldHRva2Vu"}}}),
    "kube/config": f"users:\n- user:\n    client-key-data: {KUBE_KEY}\n",
    "notes/hf.txt": f"my token is {HF_TOKEN}\n",
    # assembled at runtime so the literal never trips GitHub push protection; same string at test time
    "notes/slack.txt": "bot token "
    + "-".join(["xoxb", "123456789012", "123456789012", "AbCdEfGhIjKlMnOpQrStUvWx"])
    + "\n",
    "notes/anthropic.env.txt": f"key={ANTHROPIC_KEY}\n",
    "notes/aws.txt": "[default]\naws_access_key_id = AKIAZ3XAMPLEKEY12345\n",
    "notes/oauth.json": json.dumps(
        {"access_token": "ya29.a0AfH6SMBFAKEFAKEFAKEFAKEFAKE", "expires_in": 3599}
    ),
    "notes/client.json": json.dumps({"installed": {"client_secret": "GOCSPX-FAKEFAKEFAKEFAKEFAKE12"}}),
}

INNOCENT_FILES: dict[str, str] = {
    "train.py": "from transformers import AutoTokenizer\ntok = AutoTokenizer.from_pretrained('gpt2')\n",
    "tokenizer.py": "def tokenize(text):\n    return text.split()\n",
    "tokenization_utils.py": "MASK_TOKEN = '[MASK]'\n",
    "tokenizer_config.json": json.dumps(
        {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "model_max_length": 1024,
        }
    ),
    "hub.py": "from huggingface_hub import hf_hub_download\npath = hf_hub_download('gpt2', 'config.json')\n",
    "README.md": (
        "Uses a mask-token-embedding-projection-layer-with-weight-tying head.\n"
        "Set HF_TOKEN (an `hf_...` string) and GITHUB_TOKEN (`ghp_...`) in your shell.\n"
        "Compare with sk-learn baselines. AWS keys look like AKIAIOSFODNN7EXAMPLE.\n"
    ),
    "configs/api.json": json.dumps({"access_token": "${ACCESS_TOKEN}", "refresh_token": None}),
    "configs/kube-template.yaml": "users:\n- user:\n    client-key-data: <base64 key here>\n",
    "deploy/hosts.yml": "all:\n  hosts:\n    node1:\n      ansible_host: 10.0.0.1\n",
    "docker/app.json": json.dumps({"auths": {}, "credsStore": "desktop"}),
}


def test_credential_store_files_are_never_captured(
    tmp_path: Path, blob_exists: Callable[[bytes], bool], store: SnapshotStore
) -> None:
    project = tmp_path / "walked"  # non-git: walked like the finding's reproduction
    for rel, content in CREDENTIAL_FILES.items():
        write(project, rel, content)
    for rel, content in INNOCENT_FILES.items():
        write(project, rel, content)

    res = capture_snapshot(project, CodeOptions(), store)
    captured = {e["path"] for e in store.manifest(res.ref.digest)["files"]}

    leaked = sorted(rel for rel in CREDENTIAL_FILES if rel in captured)
    assert leaked == []
    for rel, content in CREDENTIAL_FILES.items():
        assert not blob_exists(content.encode()), rel
    assert any(w.startswith("SECRET:") for w in res.warnings)
    assert set(INNOCENT_FILES) <= captured, sorted(set(INNOCENT_FILES) - captured)


def test_token_content_in_a_tracked_file_is_excluded_loudly(
    repo: GitRepo, store: SnapshotStore, blob_exists: Callable[[bytes], bool]
) -> None:
    content = f"github.com:\n    oauth_token: {GH_TOKEN}\n"
    repo.write("config/hosts.yml", content)
    repo.commit("oops")
    res = capture_snapshot(repo.root, CodeOptions(), store)
    assert "config/hosts.yml" in res.excluded
    assert any("config/hosts.yml" in w and "GitHub token" in w for w in res.warnings)
    assert not blob_exists(content.encode())


@pytest.mark.parametrize(
    "rel",
    [
        ".npmrc",
        "a/.credentials.json",
        "gcloud/application_default_credentials.json",
        "gcloud/access_tokens.db",
        "x/.config/gh/hosts.yml",
        ".docker/config.json",
        ".kube/config",
        ".cache/huggingface/token",
        ".cache/huggingface/stored_tokens",
        ".huggingface/token",
        "infra/terraform.tfvars",
        "infra/terraform.tfstate",
        ".vault-token",
        ".azure/accessTokens.json",
        ".password-store/github.gpg",
    ],
)
def test_is_secret_path_knows_common_credential_stores(rel: str) -> None:
    assert is_secret_path(rel)


@pytest.mark.parametrize(
    "rel",
    [
        "tokenizer.py",
        "src/tokenization_utils.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "deploy/hosts.yml",
        "docker/Dockerfile",
        "src/kube/config.py",
        "configs/config.json",
        "token_utils.py",
    ],
)
def test_is_secret_path_leaves_ml_code_alone(rel: str) -> None:
    assert not is_secret_path(rel)
