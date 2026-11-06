"""Controller daemon, its Unix-socket protocol and the client.

Exports are resolved lazily so ``from hpclab.controller import HpcLabClient`` does not import
the daemon (and with it the whole service layer) in client-only processes.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "App": "hpclab.controller.daemon",
    "Controller": "hpclab.controller.daemon",
    "ControllerAlreadyRunning": "hpclab.controller.daemon",
    "MinimalApp": "hpclab.controller.daemon",
    "default_app_factory": "hpclab.controller.daemon",
    "HpcLabClient": "hpclab.controller.client",
    "controller_status": "hpclab.controller.client",
    "start_controller": "hpclab.controller.client",
    "stop_controller": "hpclab.controller.client",
    "MethodNotFound": "hpclab.controller.protocol",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'hpclab.controller' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
