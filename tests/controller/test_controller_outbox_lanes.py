"""A cancel is not stuck behind long background transfers on the controller's workers."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from hpclab_ctl_toy import ToyApp, wait_for

from hpclab.config import Paths
from hpclab.controller.client import HpcLabClient
from hpclab.controller.daemon import Controller
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem

pytestmark = pytest.mark.timeout(60)

DOWNLOAD_S = 2.0


class TransferApp(ToyApp):
    """Slow ``artifact.download`` items and a ``cancel.execute`` that records when it started."""

    def __init__(self, ctx: ServiceContext, **kw: Any) -> None:
        super().__init__(ctx, periodic=False, **kw)
        self.cancel_started = threading.Event()
        self.cancel_started_at: float | None = None
        self.downloads_started: list[float] = []

    def outbox_handlers(self) -> dict[str, Handler]:
        return {
            **super().outbox_handlers(),
            "artifact.download": self._download,
            "cancel.execute": self._cancel,
        }

    def _download(self, ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
        self.downloads_started.append(time.monotonic())
        time.sleep(DOWNLOAD_S)
        return HandlerResult.done()

    def _cancel(self, ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
        self.cancel_started_at = time.monotonic()
        self.cancel_started.set()
        return HandlerResult.done()


def test_cancel_runs_while_both_workers_could_be_downloading(start_ctl: Any, home: Paths) -> None:
    ctl: Controller = start_ctl(home, TransferApp, worker_threads=2)
    app = ctl.app
    assert isinstance(app, TransferApp)
    with HpcLabClient(home, autostart=False, timeout=10.0) as c:
        for _ in range(4):
            c.call("toy.enqueue", {"kind": "artifact.download"})
        assert wait_for(lambda: len(app.downloads_started) >= 1, 5)
        time.sleep(0.3)  # let the second worker pick whatever it may pick
        enqueued = time.monotonic()
        c.call("toy.enqueue", {"kind": "cancel.execute", "target_id": "att_x"})
        assert app.cancel_started.wait(10)
    assert app.cancel_started_at is not None
    waited = app.cancel_started_at - enqueued
    # before: the cancel waited for two rounds of downloads on the two workers (~2 x DOWNLOAD_S)
    assert waited < DOWNLOAD_S / 2, f"cancel.execute waited {waited:.2f}s behind transfers"
