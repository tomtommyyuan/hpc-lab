"""The background artifact download gets a deadline that grows with the file's size.

A fixed 600 s deadline means a 1 GiB checkpoint needs > 1.8 MB/s to ever finish; on a slow
link every retry timed out and restarted from zero, for days.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from artifacts_support import Att, Lab

from hpclab.adapters.base import TransportTimeout
from hpclab.services import artifacts
from hpclab.services import snapshot as snap
from hpclab.services.artifacts import download_to_cache


def test_download_deadline_scales_with_the_artifact_size(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []
    real = lab.transport.download

    def spy(remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        seen.append(timeout)
        real(remote_path, local_path, timeout=timeout)

    monkeypatch.setattr(lab.transport, "download", spy)
    monkeypatch.setattr(snap, "TRANSFER_MIN_RATE", 1)  # a painfully slow link: 1 byte/s
    row = lab.artifact(indexed, "summary")

    download_to_cache(lab.ctx, row)

    size = int(row["size_bytes"])
    assert seen == [pytest.approx(snap.transfer_timeout_s(size))]
    assert seen[0] >= 600.0 + size


def test_a_killed_transfer_leaves_nothing_behind_in_the_cache(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rsync writes into ``.<target>.XXXXXX`` next to the target; killed at the deadline it
    cannot remove that file, and every retry of a large file used to leak another one."""

    def killed(remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        target = Path(local_path)
        (target.parent / f".{target.name}.Ab12Cd").write_bytes(b"x" * 4096)  # rsync's temp file
        raise TransportTimeout("rsync timed out")

    monkeypatch.setattr(lab.transport, "download", killed)
    row = lab.artifact(indexed, "summary")
    shard = artifacts.cache_path_for(lab.ctx.paths, row["digest"]).parent

    with pytest.raises(TransportTimeout):
        download_to_cache(lab.ctx, row)

    assert sorted(p.name for p in shard.iterdir()) == []


def test_an_explicit_deadline_still_wins(lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    real = lab.transport.download

    def spy(remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        seen.append(timeout)
        real(remote_path, local_path, timeout=timeout)

    monkeypatch.setattr(lab.transport, "download", spy)
    artifacts.read_verified_bytes(lab.ctx, lab.artifact(indexed, "summary"), max_bytes=1 << 20, timeout=42.0)
    assert seen == [42.0]
