"""The smoke_gate finding describes the gate the workflow really applies.

``workflows._gate_verdict`` passes the smoke only when its attempt ended COMPLETED with
``result_state=valid`` - every *required* declared output finalized - plus the finalized
``smoke.require_outputs``. An empty ``require_outputs`` therefore does not reduce the gate to
the exit code; it does only when no declared output is required.
"""

from __future__ import annotations

import copy
from typing import Any

from prepare_support import BASE_SPEC, Lab, lab_overlay, lab_profile, plan_context

from hpclab.profiles.schema import ResolvedCluster
from hpclab.services import validation as vd


def _finding(lab: Lab, **smoke_and_outputs: Any) -> vd.Finding:
    spec = copy.deepcopy(BASE_SPEC)
    spec["code"] = {"project_root": str(lab.project)}
    spec.update(smoke_and_outputs)
    cl = ResolvedCluster(profile=lab_profile(lab.root), overlay=lab_overlay())
    return vd.check_smoke_gate(plan_context(spec, cl))


def test_empty_require_outputs_still_gates_on_completed_and_valid_results(lab: Lab) -> None:
    f = _finding(lab, smoke={"argv_append": ["--steps", "5"]})
    assert (f.status, f.severity) == ("warn", "info")
    assert "exit code only" not in f.message
    assert "COMPLETED" in f.message and "result_state=valid" in f.message
    assert "metrics" in f.message  # the required output the valid result_state depends on


def test_empty_require_outputs_and_no_required_output_is_completed_only(lab: Lab) -> None:
    outputs = [dict(BASE_SPEC["outputs"][0], required=False)]
    f = _finding(lab, smoke={"argv_append": []}, outputs=outputs)
    assert (f.status, f.severity) == ("warn", "info")
    assert "COMPLETED" in f.message and "no declared output is required" in f.message
