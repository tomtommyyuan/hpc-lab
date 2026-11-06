"""Remote prepare checks: env_guard_static / snapshot_bypass / scheduler_test.

The "cluster" filesystem is local (LocalTransport); failures are injected with transport
subclasses. Every transport problem must yield ``unchecked`` - never pass, never a crash.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from prepare_support import (
    BASE_SPEC,
    FakeScheduler,
    Lab,
    lab_overlay,
    lab_profile,
    make_venv,
    plan_context,
)

from hpclab.adapters.base import AuthRequiredError, CommandResult, StatInfo, TransportTimeout
from hpclab.adapters.local import LocalTransport
from hpclab.profiles.schema import ResolvedCluster, SiteObservation
from hpclab.services import experiments as ex
from hpclab.services import validation as vd

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def cluster(lab: Lab, **profile: Any) -> ResolvedCluster:
    return ResolvedCluster(profile=lab_profile(lab.root, **profile), overlay=lab_overlay())


def env_check(
    lab: Lab,
    venv: str | None,
    *,
    transport: Any = None,
    gpus: int = 1,
    gpu_types: tuple[str, ...] = ("a100",),
    project: set[str] | None = None,
    now: datetime = NOW,
    budget: float = 10.0,
    **profile: Any,
) -> dict[str, vd.Finding]:
    out = vd.check_env_guard_static(
        transport=transport or LocalTransport("lab", user="alice"),
        cluster=cluster(lab, **profile),
        runtime_venv=venv,
        gpus=gpus,
        gpu_types=list(gpu_types),
        project=project if project is not None else {"mypkg"},
        now=now,
        deadline=vd.Deadline(budget),
    )
    assert [f.check for f in out] == ["env_guard_static", "snapshot_bypass"]
    return {f.check: f for f in out}


def sub(f: vd.Finding, name: str) -> dict[str, Any]:
    return next(c for c in f.evidence["checks"] if c["name"] == name)


# ---------------------------------------------------------------------------
# env_guard_static
# ---------------------------------------------------------------------------


def test_no_venv_is_unchecked(lab: Lab) -> None:
    f = env_check(lab, None)
    assert f["env_guard_static"].status == "unchecked" and f["snapshot_bypass"].status == "unchecked"


def test_missing_venv_blocks(lab: Lab) -> None:
    f = env_check(lab, "{scratch}/venvs/nope")
    assert f["env_guard_static"].status == "fail" and f["env_guard_static"].severity == "block"
    assert f["env_guard_static"].evidence["venv"] == str(lab.scratch / "venvs" / "nope")
    assert f["snapshot_bypass"].status == "unchecked"


def test_not_a_virtualenv_blocks(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x", cfg=None)
    f = env_check(lab, "{scratch}/venvs/x")
    assert f["env_guard_static"].status == "fail" and "pyvenv.cfg" in f["env_guard_static"].message


def test_pyvenv_without_version_is_unchecked(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x", cfg="home = /usr/bin\n")
    f = env_check(lab, "{scratch}/venvs/x")
    assert f["env_guard_static"].status == "unchecked"


def test_venv_without_torch_passes(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x")
    f = env_check(lab, "{scratch}/venvs/x")
    assert f["env_guard_static"].status == "pass", f["env_guard_static"]
    assert sub(f["env_guard_static"], "python")["site_packages"].endswith("lib/python3.12/site-packages")
    assert f["snapshot_bypass"].status == "pass"


@pytest.mark.parametrize(
    ("torch_cuda", "gpu_types", "status", "which"),
    [
        ("12.8", ("a100", "b200"), "pass", None),
        ("13.0", ("a100",), "fail", "driver"),  # cu130 wheel on a 12.8 driver
        ("12.9", ("a100",), "warn", "driver"),  # minor-version newer than the driver
        ("12.4", ("b200",), "fail", "arch:b200"),  # no sm_100 kernels before CUDA 12.8
        ("12.4", ("h100",), "pass", None),
        ("12.4", ("mystery",), "unchecked", "arch:mystery"),
    ],
)
def test_torch_cuda_vs_driver_and_arch(
    lab: Lab, torch_cuda: str, gpu_types: tuple[str, ...], status: str, which: str | None
) -> None:
    make_venv(lab.scratch / "venvs" / "t", torch_cuda=torch_cuda)
    f = env_check(lab, "{scratch}/venvs/t", gpu_types=gpu_types)["env_guard_static"]
    assert f.status == status, f
    if which:
        assert sub(f, which)["status"] == status
    if status == "fail":
        assert f.severity == "block" and f.remediation and "cu128" in f.remediation


def test_missing_site_packages_is_unchecked_not_torch_absent(lab: Lab) -> None:
    import shutil

    venv = make_venv(lab.scratch / "venvs" / "odd")
    shutil.rmtree(venv / "lib")
    f = env_check(lab, "{scratch}/venvs/odd")
    assert f["env_guard_static"].status == "unchecked"
    assert sub(f["env_guard_static"], "site_packages")["status"] == "unchecked"
    assert f["snapshot_bypass"].status == "unchecked"


def test_cpu_torch_build(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "cpu", torch_cuda=None)
    assert env_check(lab, "{scratch}/venvs/cpu", gpus=1)["env_guard_static"].status == "fail"
    assert env_check(lab, "{scratch}/venvs/cpu", gpus=0)["env_guard_static"].status == "pass"


def test_expired_or_missing_driver_fact_is_unchecked(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "t", torch_cuda="12.8")
    later = datetime(2027, 1, 1, tzinfo=UTC)  # the fact expires 2026-12-01
    f = env_check(lab, "{scratch}/venvs/t", now=later)["env_guard_static"]
    assert f.status == "unchecked" and sub(f, "driver")["status"] == "unchecked"
    f2 = env_check(lab, "{scratch}/venvs/t", facts=[])["env_guard_static"]
    assert f2.status == "unchecked"
    f3 = env_check(
        lab,
        "{scratch}/venvs/t",
        facts=[
            SiteObservation(key="driver_cuda", value="12.8", source="x", observed_at="2026-01-01T00:00:00Z")
        ],
    )["env_guard_static"]
    assert f3.status == "pass"


def test_venv_path_forms(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "p")
    make_venv(lab.root / "home" / "alice" / "venvs" / "h")
    assert env_check(lab, f"{lab.root}/scratch/$USER/venvs/p")["env_guard_static"].status == "pass"
    assert env_check(lab, f"{lab.root}/scratch/{{user}}/venvs/p")["env_guard_static"].status == "pass"
    assert env_check(lab, "~/venvs/h")["env_guard_static"].status == "pass"
    assert env_check(lab, "$HOME/venvs/h")["env_guard_static"].status == "pass"
    assert env_check(lab, "{nope}/venv")["env_guard_static"].status == "unchecked"
    assert env_check(lab, "relative/venv")["env_guard_static"].status == "unchecked"


# ---------------------------------------------------------------------------
# snapshot_bypass
# ---------------------------------------------------------------------------


def test_editable_install_of_the_project_blocks(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "e", editable_dist={"mypkg": True, "requests": False})
    f = env_check(lab, "{scratch}/venvs/e")["snapshot_bypass"]
    assert f.status == "fail" and f.severity == "block" and "mypkg" in f.evidence["editable"]
    assert "not the frozen code snapshot" in f.message


def test_other_editable_installs_warn(lab: Lab) -> None:
    make_venv(
        lab.scratch / "venvs" / "e", editable_dist={"otherlib": True}, pth=["__editable__.helper-0.2.pth"]
    )
    f = env_check(lab, "{scratch}/venvs/e")["snapshot_bypass"]
    assert f.status == "warn" and set(f.evidence["editable"]) == {"otherlib", "helper"}


def test_pth_editable_of_project_and_egg_link(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "p", pth=["__editable__.MyPkg-0.1.pth"])
    assert env_check(lab, "{scratch}/venvs/p")["snapshot_bypass"].status == "fail"
    make_venv(lab.scratch / "venvs" / "g", pth=["mypkg.egg-link"])
    assert env_check(lab, "{scratch}/venvs/g")["snapshot_bypass"].status == "fail"


def test_project_names_from_manifest_and_pyproject() -> None:
    manifest = {
        "files": [
            {"path": "src/My_Pkg/__init__.py"},
            {"path": "tool/__init__.py"},
            {"path": "a/b/c/__init__.py"},
        ]
    }
    names = vd.project_names(manifest, '[project]\nname = "Fancy.Project"\n')
    assert names == {"my-pkg", "tool", "fancy-project"}
    assert vd.project_names(None, "not toml [[[") == set()


class FailingFind(LocalTransport):
    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        if argv[0] == "find":
            return CommandResult(list(argv), 127, "", "find: command not found", 0.0)
        return super().run(argv, timeout=timeout, stdin=stdin)


def test_listing_failure_is_unchecked(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "e")
    f = env_check(lab, "{scratch}/venvs/e", transport=FailingFind("lab", user="alice"))
    assert f["snapshot_bypass"].status == "unchecked" and f["env_guard_static"].status == "pass"


# ---------------------------------------------------------------------------
# Transport failures -> unchecked
# ---------------------------------------------------------------------------


class AuthTransport(LocalTransport):
    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        raise AuthRequiredError("ssh needs MFA", details={"login_command_str": "ssh -M lab"})

    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        raise AuthRequiredError("ssh needs MFA")


class SlowTransport(LocalTransport):
    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        raise TransportTimeout(f"stat timed out after {timeout}s")

    def read_range(self, *a: Any, **k: Any) -> Any:
        raise TransportTimeout("read timed out")


class TorchReadFails(LocalTransport):
    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> Any:
        if remote_path.endswith("torch/version.py"):
            raise TransportTimeout("read timed out")
        return super().read_range(remote_path, offset, length, timeout=timeout)


def test_auth_required_is_unchecked_with_remediation(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x")
    f = env_check(lab, "{scratch}/venvs/x", transport=AuthTransport("lab", user="alice"))
    for name in ("env_guard_static", "snapshot_bypass"):
        assert f[name].status == "unchecked" and f[name].remediation == "hpclab auth lab"


def test_timeouts_are_unchecked(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x", torch_cuda="12.8")
    f = env_check(lab, "{scratch}/venvs/x", transport=SlowTransport("lab", user="alice"))
    assert f["env_guard_static"].status == "unchecked" and f["env_guard_static"].remediation is None
    g = env_check(lab, "{scratch}/venvs/x", transport=TorchReadFails("lab", user="alice"))
    assert (
        g["env_guard_static"].status == "unchecked"
        and sub(g["env_guard_static"], "torch")["status"] == "unchecked"
    )


def test_exhausted_deadline_is_unchecked(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "x")
    f = env_check(lab, "{scratch}/venvs/x", budget=0.0)
    assert f["env_guard_static"].status == "unchecked"


def test_bounded_call_times_out() -> None:
    import time

    start = time.monotonic()
    with pytest.raises(TransportTimeout):
        vd.bounded_call(lambda: time.sleep(2), 0.2, "slow")
    assert time.monotonic() - start < 1.5
    with pytest.raises(ValueError):
        vd.bounded_call(lambda: int("x"), 1.0, "err")
    assert vd.bounded_call(lambda: 5, 1.0, "ok") == 5


# ---------------------------------------------------------------------------
# scheduler_test
# ---------------------------------------------------------------------------


def sched_check(
    lab: Lab, scheduler: FakeScheduler, *, transport: Any = None, budget: float = 10.0
) -> vd.Finding:
    spec = copy.deepcopy(BASE_SPEC)
    spec["code"] = {"project_root": str(lab.project)}
    cl = cluster(lab)
    pc = plan_context(spec, cl)
    run = next(r for r in pc.runs if r.kind == "main")
    return vd.check_scheduler_test(
        transport=transport or LocalTransport("lab", user="alice"),
        scheduler=scheduler,  # type: ignore[arg-type]
        cluster=cl,
        run=run,
        remote_root=str(lab.remote_root),
        plan_id="pln_TEST",
        deadline=vd.Deadline(budget),
    )


def test_scheduler_test_pass_uploads_renders_and_cleans_up(lab: Lab) -> None:
    sched = FakeScheduler()
    f = sched_check(lab, sched)
    assert f.status == "pass" and f.evidence["partition"] == "gpu" and f.evidence["estimated_start"]
    call = sched.calls[0]
    assert call["test_only"] is True and call["workdir"] == f"{lab.remote_root}/.prepare/pln_TEST"
    assert "#SBATCH --job-name=hl.method-baseline.seed-1.previewt" in call["script"]
    assert not Path(call["script_path"]).exists() and not Path(call["workdir"]).exists()


@pytest.mark.parametrize(
    ("mode", "status"), [("reject", "fail"), ("reject_later", "warn"), ("error", "unchecked")]
)
def test_scheduler_test_outcomes(lab: Lab, mode: str, status: str) -> None:
    f = sched_check(lab, FakeScheduler(mode=mode))
    assert f.status == status
    if status == "fail":
        assert f.severity == "block" and f.evidence["reason_code"] == "invalid_account"


def test_scheduler_test_is_bounded(lab: Lab) -> None:
    import time

    start = time.monotonic()
    f = sched_check(lab, FakeScheduler(mode="slow"), budget=1.5)
    assert f.status == "unchecked" and "did not finish" in f.message
    assert time.monotonic() - start < 2.9


def test_scheduler_test_upload_auth_failure(lab: Lab) -> None:
    f = sched_check(lab, FakeScheduler(), transport=AuthTransport("lab", user="alice"))
    assert f.status == "unchecked" and f.remediation == "hpclab auth lab"


# ---------------------------------------------------------------------------
# Through experiment_prepare
# ---------------------------------------------------------------------------


def test_prepare_with_remote_checks(lab: Lab) -> None:
    make_venv(lab.scratch / "venvs" / "proj", torch_cuda="13.0", editable_dist={"mypkg": True})
    spec = lab.spec(runtime={"python": "python3", "venv": "{scratch}/venvs/proj"})
    data = ex.experiment_prepare(lab.ctx, idempotency_key="r1", spec=spec)
    f = {x.check: x for x in data.validation.findings}
    assert f["scheduler_test"].status == "pass"
    assert f["env_guard_static"].status == "fail" and f["snapshot_bypass"].status == "fail"
    assert {x.name: x.status for x in data.coverage.sources}["cluster:lab"] == "ok"
    assert data.coverage.complete is True
    assert data.status == "invalid" and set(data.validation.blocking) == {
        "env_guard_static",
        "snapshot_bypass",
    }
    assert lab.scheduler.calls and all(c["test_only"] for c in lab.scheduler.calls)
    over = ex.experiment_prepare(
        lab.ctx,
        idempotency_key="r2",
        spec=spec,
        overrides=[
            vd.ValidationOverride(
                check="env_guard_static", reason="we pin cu130 via module load cuda/13 in setup"
            ),
            vd.ValidationOverride(check="snapshot_bypass", reason="reviewed"),
        ],
    )
    assert over.status == "ready" and set(over.validation.overridden) == {
        "env_guard_static",
        "snapshot_bypass",
    }


def test_prepare_remote_checks_with_auth_down(lab: Lab) -> None:
    lab.ctx.clusters.transport_factory = lambda cluster, paths: AuthTransport("lab", user="alice")
    lab.ctx.clusters.invalidate("lab")
    spec = lab.spec(runtime={"python": "python3", "venv": "{scratch}/venvs/proj"})
    data = ex.experiment_prepare(lab.ctx, idempotency_key="r1", spec=spec)
    f = {x.check: x for x in data.validation.findings}
    for name in ("env_guard_static", "snapshot_bypass", "scheduler_test"):
        assert f[name].status == "unchecked" and f[name].remediation == "hpclab auth lab", f[name]
    assert data.status == "ready"  # unknown never blocks; it is reported
    assert data.validation.status == "warn"
    cov = {x.name: x for x in data.coverage.sources}
    assert data.coverage.complete is False and cov["cluster:lab"].status == "unavailable"
    assert "authentication" in (cov["cluster:lab"].detail or "")


def test_prepare_skips_scheduler_test_when_script_does_not_render(lab: Lab) -> None:
    spec = lab.spec(runtime={"python": "python3", "env": {"IFS": "x"}})
    data = ex.experiment_prepare(lab.ctx, idempotency_key="r1", spec=spec)
    f = {x.check: x for x in data.validation.findings}
    assert f["scheduler_test"].status == "unchecked" and not lab.scheduler.calls


def test_prepare_coverage_partial_when_one_remote_check_fails(lab: Lab) -> None:
    lab.scheduler.mode = "error"
    make_venv(lab.scratch / "venvs" / "ok", torch_cuda="12.8")
    spec = lab.spec(runtime={"python": "python3", "venv": "{scratch}/venvs/ok"})
    data = ex.experiment_prepare(lab.ctx, idempotency_key="r1", spec=spec)
    f = {x.check: x for x in data.validation.findings}
    assert f["env_guard_static"].status == "pass"
    assert f["scheduler_test"].status == "unchecked" and f["scheduler_test"].evidence["error_type"]
    assert {x.name: x.status for x in data.coverage.sources}["cluster:lab"] == "partial"
    assert data.coverage.complete is False
    # only the (failing) scheduler test touched the cluster -> unavailable
    only = ex.experiment_prepare(lab.ctx, idempotency_key="r2", spec=lab.spec())
    assert {x.name: x.status for x in only.coverage.sources}["cluster:lab"] == "unavailable"


def test_prepare_coverage_skipped_when_nothing_remote_applies(lab: Lab) -> None:
    spec = lab.spec(runtime={"python": "python3", "env": {"IFS": "x"}})  # no venv, script cannot render
    data = ex.experiment_prepare(lab.ctx, idempotency_key="r1", spec=spec)
    assert {x.name: x.status for x in data.coverage.sources}["cluster:lab"] == "skipped"
