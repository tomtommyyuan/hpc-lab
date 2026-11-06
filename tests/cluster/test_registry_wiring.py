"""The cluster-area services behind the method registry: parameter models derived from the
signatures, Envelope lifting (coverage/observed_at/warnings), summaries, typed errors."""

from __future__ import annotations

from cluster_fakes import Env

from hpclab.services import cluster, digest, notify, usage
from hpclab.services.app import build_registry
from hpclab.services.registry import Registry


def _registry() -> Registry:
    reg = Registry()
    reg.register("cluster_inspect", cluster.cluster_inspect)
    reg.register("usage_report", usage.usage_report)
    reg.register("notify_test", notify.notify_test, mutation=True)
    reg.register("lab_digest", digest.lab_digest)
    return reg


def test_methods_dispatch_through_the_registry(env: Env) -> None:
    reg = _registry()
    out = reg.dispatch(
        env.ctx,
        "cluster_inspect",
        {
            "cluster_id": "fake",
            "checks": ["auth", "capacity"],
            "candidate_shapes": [{"partition": "gpu", "gpus": 2}],
        },
        request_id="r1",
    )
    assert out["ok"], out
    assert out["coverage"]["sources"] and "coverage" not in out["data"] and out["observed_at"]
    # the profile lists a partition sinfo does not report -> warn, with the next step in the summary
    assert out["summary"].startswith("cluster fake: warn (1 ok, 1 warn); next: update the site profile")
    assert out["data"]["candidates"][0]["est_source"] == "test_only"

    bad = reg.dispatch(
        env.ctx, "cluster_inspect", {"cluster_id": "fake", "checks": ["bogus"]}, request_id="r2"
    )
    assert not bad["ok"] and bad["error"]["code"] == "invalid_argument"
    too_many = reg.dispatch(
        env.ctx,
        "cluster_inspect",
        {"cluster_id": "fake", "candidate_shapes": [{"partition": "gpu"}] * 4},
        request_id="r3",
    )
    assert not too_many["ok"] and too_many["error"]["code"] == "invalid_argument"
    missing = reg.dispatch(env.ctx, "cluster_inspect", {"cluster_id": "nope"}, request_id="r4")
    assert not missing["ok"] and missing["error"]["code"] == "not_found"

    # this fake scheduler cannot read usage records: an honest typed error, never an empty report
    no_usage = reg.dispatch(env.ctx, "usage_report", {"cluster_id": "fake", "since": "1d"}, request_id="r5")
    assert not no_usage["ok"] and no_usage["error"]["code"] == "unsupported"
    bad_window = reg.dispatch(
        env.ctx, "usage_report", {"cluster_id": "fake", "since": "soon"}, request_id="r6"
    )
    assert not bad_window["ok"] and bad_window["error"]["code"] == "invalid_argument"

    dg = reg.dispatch(env.ctx, "lab_digest", {"since": "7d"}, request_id="r7")
    assert (
        dg["ok"]
        and dg["data"]["markdown"].startswith("# HPC Lab digest")
        and dg["summary"].startswith("digest")
    )

    nt = reg.dispatch(env.ctx, "notify_test", {}, request_id="r8")
    assert nt["ok"] and nt["summary"] == "no notification channels configured"


def test_app_registry_and_worker_wiring() -> None:
    reg = build_registry()
    assert {"cluster_inspect", "usage_report", "notify_test", "lab_digest"} <= set(reg.methods)
    assert reg.methods["notify_test"].mutation
    assert set(notify.outbox_handlers()) == {"notify.send"}
    assert cluster.outbox_handlers() == {} and usage.outbox_handlers() == {}
    names = {t.name for t in (*notify.periodic_tasks(), *cluster.periodic_tasks(), *usage.periodic_tasks())}
    assert names == {"notify.flush", "cluster.cache_gc", "usage.cache_gc"}


def test_cache_gc_only_touches_its_own_expired_keys(env: Env) -> None:
    with env.ctx.db.tx() as tx:
        for key, expires in (
            ("test_only:v1:fake:a", "2000-01-01T00:00:00.000Z"),
            ("test_only:v1:fake:b", "2999-01-01T00:00:00.000Z"),
            ("usage:v1:fake:x", "2000-01-01T00:00:00.000Z"),
            ("someone-else:1", "2000-01-01T00:00:00.000Z"),
        ):
            tx.execute(
                "INSERT INTO report_cache(cache_key, payload_json, created_at, expires_at) VALUES (?, '{}', ?, ?)",
                (key, tx.now, expires),
            )
    cluster.gc_cache(env.ctx)
    usage.gc_cache(env.ctx)
    left = {r["cache_key"] for r in env.ctx.db.all("SELECT cache_key FROM report_cache")}
    assert left == {"test_only:v1:fake:b", "someone-else:1"}
