"""decode_pending (table-driven) and health_from (advisory telemetry flags)."""

from __future__ import annotations

import pytest

from hpclab.services.health import Health, flag_char, health_from
from hpclab.services.pending import decode_pending


@pytest.mark.parametrize(
    ("reason", "code", "category", "actionable"),
    [
        ("Priority", "Priority", "cluster_busy", False),
        ("Resources", "Resources", "cluster_busy", False),
        ("(Resources)", "Resources", "cluster_busy", False),
        ("QOSMaxGRESPerUser", "QOSMaxGRESPerUser", "self_limit", False),
        ("QOSMaxJobsPerUserLimit", "QOSMaxJobsPerUserLimit", "self_limit", False),
        ("QOSMaxSubmitJobPerUserLimit", "QOSMaxSubmitJobPerUserLimit", "self_limit", False),
        ("QOSMaxCpuPerJobLimit", "QOSMaxCpuPerJobLimit", "never_starts", True),
        ("QOSMaxWallDurationPerJobLimit", "QOSMaxWallDurationPerJobLimit", "never_starts", True),
        ("QOSGrpGRES", "QOSGrpGRES", "account_limit", False),
        ("AssocGrpGRES", "AssocGrpGRES", "account_limit", False),
        ("AssocGrpGRESMinutes", "AssocGrpGRESMinutes", "account_limit", False),
        ("AssocMaxJobsLimit", "AssocMaxJobsLimit", "self_limit", False),
        ("AssocMaxWallDurationPerJobLimit", "AssocMaxWallDurationPerJobLimit", "never_starts", True),
        ("PartitionTimeLimit", "PartitionTimeLimit", "never_starts", True),
        ("DependencyNeverSatisfied", "DependencyNeverSatisfied", "never_starts", True),
        ("BadConstraints", "BadConstraints", "never_starts", True),
        ("InvalidAccount", "InvalidAccount", "never_starts", True),
        ("JobHeldUser", "JobHeldUser", "held", True),
        ("JobHeldAdmin", "JobHeldAdmin", "held", True),
        ("BeginTime", "BeginTime", "begin_time", False),
        ("Dependency", "Dependency", "dependency", False),
        ("PartitionDown", "PartitionDown", "cluster_busy", False),
        ("ReqNodeNotAvail, UnavailableNodes:gpu-[01-02]", "ReqNodeNotAvail", "cluster_busy", False),
        ("ReqNodeNotAvail, Reserved for maintenance", "ReqNodeNotAvail", "never_starts", True),
        ("SomethingNew", "SomethingNew", "unknown", False),
    ],
)
def test_decode_pending_table(reason: str, code: str, category: str, actionable: bool) -> None:
    info = decode_pending(reason)
    assert (info.reason_code, info.category, info.actionable) == (code, category, actionable)
    assert info.explanation and info.raw == reason


@pytest.mark.parametrize("reason", [None, "", "None", "(null)"])
def test_no_reason_yet_is_unknown_not_a_guess(reason: str | None) -> None:
    info = decode_pending(reason)
    assert info.reason_code is None and info.category == "unknown"


def test_req_node_not_avail_keeps_the_node_detail() -> None:
    info = decode_pending("ReqNodeNotAvail, UnavailableNodes:gpu-[01-02]")
    assert info.detail == "UnavailableNodes:gpu-[01-02]"


NOW = 1_790_000_000.0


def csv_rows(util_by_gpu: dict[int, float], n: int, *, used: float = 1000, total: float = 81559) -> str:
    return "".join(
        f"2026/09/27 12:{i:02d}:00.000, {g}, GPU-{g}, {u}, {used}, {total}, 300.5\n"
        for i in range(n)
        for g, u in util_by_gpu.items()
    )


def test_missing_telemetry_is_source_none_never_zero_util() -> None:
    h = health_from("", "", now=NOW, gpus_allocated=2, log_mtime=None, metrics_mtime=None)
    assert h.source == "none" and h.per_gpu_util_mean == {} and h.idle_gpus is None
    assert h.flags == [] and flag_char(h) == "-"


def test_busy_gpus_are_ok() -> None:
    h = health_from(
        csv_rows({0: 95, 1: 90}, 5),
        f"{int(NOW) - 30}\n",
        now=NOW,
        gpus_allocated=2,
        log_mtime=NOW - 5,
        metrics_mtime=NOW - 5,
    )
    assert h.source == "telemetry" and h.per_gpu_util_mean == {"0": 95.0, "1": 90.0}
    assert h.idle_gpus == 0 and h.flags == [] and flag_char(h) == "."
    assert h.heartbeat_age_s == 30 and h.peak_gpu_mem_gib == pytest.approx(1000 / 1024, abs=0.01)


def test_idle_and_partial_gpu_use() -> None:
    idle = health_from(
        csv_rows({0: 0, 1: 1}, 5), "", now=NOW, gpus_allocated=2, log_mtime=None, metrics_mtime=None
    )
    assert idle.flags == ["idle_allocation"] and idle.idle_gpus == 2 and flag_char(idle) == "I"
    partial = health_from(
        csv_rows({0: 97, 1: 0}, 5), "", now=NOW, gpus_allocated=2, log_mtime=None, metrics_mtime=None
    )
    assert partial.flags == ["partial_gpu_use"] and flag_char(partial) == "P"


def test_too_few_samples_are_not_flagged_yet() -> None:
    h = health_from(csv_rows({0: 0}, 2), "", now=NOW, gpus_allocated=1, log_mtime=None, metrics_mtime=None)
    assert h.flags == [] and any("fewer than 3" in n for n in h.notes)


def test_only_the_recent_window_counts() -> None:
    text = csv_rows({0: 0}, 20) + csv_rows({0: 99}, 10)
    h = health_from(text, "", now=NOW, gpus_allocated=1, log_mtime=None, metrics_mtime=None)
    assert h.per_gpu_util_mean == {"0": 99.0} and h.flags == []


def test_near_oom_heartbeat_frozen_and_stalled() -> None:
    oom = health_from(
        csv_rows({0: 99}, 3, used=80000), "", now=NOW, gpus_allocated=1, log_mtime=None, metrics_mtime=None
    )
    assert "near_gpu_oom" in oom.flags and flag_char(oom) == "M"
    frozen = health_from(
        "", f"{int(NOW) - 900}\n", now=NOW, gpus_allocated=0, log_mtime=None, metrics_mtime=None
    )
    assert frozen.flags == ["heartbeat_frozen"] and flag_char(frozen) == "H"
    stalled = health_from(
        csv_rows({0: 99}, 5),
        f"{int(NOW) - 20}\n",
        now=NOW,
        gpus_allocated=1,
        log_mtime=NOW - 4000,
        metrics_mtime=NOW - 3000,
    )
    assert stalled.flags == ["stalled_candidate"] and flag_char(stalled) == "S"


def test_malformed_and_cut_lines_are_skipped() -> None:
    text = (
        "12:00.000, 0, GPU-0, 5\n"
        + "garbage\n"
        + "2026/09/27, x, GPU, [N/A], [N/A], 1, 1\n"
        + csv_rows({0: 50}, 3)
    )
    h = health_from(text, "not-a-number\n", now=NOW, gpus_allocated=1, log_mtime=None, metrics_mtime=None)
    assert h.per_gpu_util_mean == {"0": 50.0} and h.samples == 3 and h.heartbeat_age_s is None


def test_health_roundtrips_as_json() -> None:
    h = health_from(csv_rows({0: 0}, 5), "", now=NOW, gpus_allocated=1, log_mtime=None, metrics_mtime=None)
    assert Health.model_validate_json(h.model_dump_json()) == h
    assert "flags: idle_allocation" in h.summary()
