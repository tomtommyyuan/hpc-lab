"""Immutable experiment specifications and scheduler facts.

Vocabulary (design §1 / system §5):

* ``ExperimentSpec`` – what the researcher (or agent) submits to ``experiment_prepare``:
  one scientific question, code/entrypoint/runtime, a finite matrix, resources, outputs.
* ``RunSpec`` – the immutable scientific configuration of one matrix cell. Changing any
  field is a *new Run* (``derived_from`` keeps lineage).
* ``ExecutionPlan`` – how one Attempt of a Run executes: cluster, resolved resources,
  snapshot location, recovery source. Pure resource changes produce a new plan (new
  Attempt) of the same Run.
* ``SchedulerJobRef`` – the external job identity; job ids are never global identities.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hpclab.domain.clock import parse_slurm_duration
from hpclab.domain.hashing import digest
from hpclab.domain.states import Pin, SchedulerState

MAX_MATRIX_CELLS = 256
MAX_SEEDS = 64
SCRIPT_TEMPLATE_VERSION = "hpclab-wrapper/1"

Scalar = str | int | float | bool | None


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Spec building blocks
# ---------------------------------------------------------------------------


class InputRef(Frozen):
    """A dataset / pretrained model / other input the run reads.

    ``uri`` is a path on the cluster (may use ``{user}`` / ``{scratch}`` placeholders from
    the profile) or an external URL. Full hashing of large datasets is not promised: the
    ``pin`` level says how firmly it is fixed.
    """

    name: str
    uri: str
    version: str | None = None
    digest: str | None = None
    pin: Pin | None = None  # computed by prepare when None
    notes: str | None = None


class CodeOptions(Frozen):
    """How to snapshot the local project."""

    project_root: str = "."
    include_untracked: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    max_bytes: int = 256 * 1024 * 1024
    max_files: int = 50_000
    allow_dirty: bool = True


class EntrypointSpec(Frozen):
    """The program to run, relative to the snapshot root.

    ``argv`` elements may contain ``{name}`` placeholders: matrix/parameter names,
    ``{seed}``, and runtime placeholders resolved on the compute node:
    ``{output_dir} {checkpoint_dir} {attempt_dir} {code_dir} {run_id} {attempt_id}
    {resume_from} {job_id}`` (``{job_id}`` is the Slurm job id, ``$SLURM_JOB_ID``). Unknown
    placeholders are rejected at prepare time.
    """

    argv: list[str]
    workdir: str = "."

    @field_validator("workdir")
    @classmethod
    def _workdir(cls, v: str) -> str:
        if v.startswith("/") or ".." in v.replace("\\", "/").split("/"):
            raise ValueError(
                "entrypoint.workdir must be relative to the snapshot root and may not contain '..'"
            )
        return v

    @field_validator("argv")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("entrypoint.argv must not be empty")
        return v


# Wrapper tuning knobs that may be set through RuntimeSpec.env (see services/jobscript.py).
RUNTIME_KNOB_ENV = frozenset(
    {
        "HPCLAB_PROBE_TIMEOUT_S",
        "HPCLAB_TELEMETRY_INTERVAL_S",
        "HPCLAB_OUTPUT_HASH_BUDGET_S",
        "HPCLAB_HELPER_PYTHON",
    }
)

RUNTIME_PLACEHOLDERS = frozenset(
    {
        "output_dir",
        "checkpoint_dir",
        "attempt_dir",
        "code_dir",
        "run_id",
        "attempt_id",
        "resume_from",
        "job_id",
    }
)


class RuntimeSpec(Frozen):
    """Execution environment on the compute node.

    ``setup`` lines run (bash) before the entrypoint, e.g. ``source .venv/bin/activate`` or
    ``module load cuda/12.8``. ``lockfiles`` (paths inside the snapshot) make the
    environment *fingerprinted* rather than *unpinned*. The job wrapper records the
    actual interpreter, packages digest, driver and GPUs at start (``env.json``).

    ``venv`` (a cluster path) is activated by the wrapper and lets ``experiment_prepare``
    statically inspect it (torch CUDA build vs driver, editable installs that would bypass
    the code snapshot). ``guard`` enables fail-fast compute-side checks before the program
    starts (exit 3 with ``HPCLAB_GUARD_FAILED:<check>`` on e.g. CUDA unavailable).
    """

    setup: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    lockfiles: list[str] = Field(default_factory=list)
    container: str | None = None
    python: str = "python"
    venv: str | None = None
    guard: bool = True

    @field_validator("env")
    @classmethod
    def _env(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"runtime.env key {key!r} is not a valid environment variable name")
            if key.startswith("HPCLAB_") and key not in RUNTIME_KNOB_ENV:
                raise ValueError(
                    f"runtime.env key {key!r} is reserved (HPCLAB_* are set by the wrapper; "
                    f"tunable knobs: {', '.join(sorted(RUNTIME_KNOB_ENV))})"
                )
        return v

    @field_validator("lockfiles")
    @classmethod
    def _lockfiles(cls, v: list[str]) -> list[str]:
        for p in v:
            if p.startswith("/") or ".." in p.split("/"):
                raise ValueError(f"lockfile {p!r} must be a relative path inside the snapshot")
        return v


# sbatch flags a spec may add. Anything touching output paths, job name, comment,
# dependencies or the wrapper is owned by hpclab and rejected.
#
# GPUs are requested only through ``ResourceSpec.gpus`` (rendered as ``--gres=gpu:``): the
# budget reservation, the GPU-hours the user confirms, partition/limit validation, the
# compute guard and GPU telemetry all read that count. A flag that makes Slurm allocate GPUs
# by itself would leave all of them at the wrong number (e.g. 0 GPU-h admitted for a job
# holding 32 GPUs), so none may appear here. ``--exclusive`` is one of them: Slurm gives an
# exclusive job every CPU *and GRES* on its nodes.
EXTRA_SBATCH_GPU_ALLOCATING = frozenset(
    {"--gres", "--gpus", "--gpus-per-node", "--gpus-per-socket", "--gpus-per-task", "--exclusive"}
)
EXTRA_SBATCH_ALLOWED = (
    "--signal=",
    "--constraint=",
    "--gpu-bind=",
    "--mem-per-gpu=",
    "--mem-per-cpu=",
    "--cpus-per-gpu=",
    "--requeue",
    "--no-requeue",
    "--hint=",
    "--mail-type=",
    "--mail-user=",
    "--licenses=",
    "--reservation=",
    "--nice=",
    "--tmp=",
    "--switches=",
)


class ResourceSpec(Frozen):
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    gpus: int = Field(default=0, ge=0, le=64)
    gpu_type: str | None = None
    cpus: int = Field(default=1, ge=1, le=512)
    mem: str | None = None
    time: str = "01:00:00"
    nodes: int = Field(default=1, ge=1, le=64)
    ntasks_per_node: int = Field(default=1, ge=1, le=64)
    exclude_nodes: list[str] = Field(default_factory=list)
    constraint: str | None = None
    extra_sbatch: list[str] = Field(default_factory=list)

    @field_validator("time")
    @classmethod
    def _time(cls, v: str) -> str:
        if parse_slurm_duration(v) in (None, 0):
            raise ValueError(f"invalid Slurm time limit {v!r} (use HH:MM:SS or D-HH:MM:SS)")
        return v

    @field_validator("mem", mode="before")
    @classmethod
    def _mem(cls, v: object) -> object:
        """Normalize to what sbatch accepts: ``64GB``/``64gib`` -> ``64G``, ``1.5G`` -> ``1536M``."""
        if v is None:
            return v
        text = str(v).strip()
        m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?$", text, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"invalid memory {text!r}: use an integer with optional K/M/G/T suffix (e.g. 64G, 1536M)"
            )
        num, unit = m.group(1), m.group(2).upper()
        if "." not in num:
            return f"{int(num)}{unit}"
        lower = {"T": "G", "G": "M", "M": "K", "K": "K", "": "K"}[unit]
        factor = 1 if unit in ("K", "") and lower == "K" else 1024
        if unit == "":  # plain number means MB in sbatch
            lower, factor = "K", 1024
        return f"{math.ceil(float(num) * factor)}{lower}"

    @field_validator("extra_sbatch")
    @classmethod
    def _extra(cls, v: list[str]) -> list[str]:
        for flag in v:
            if flag.split("=", 1)[0] in EXTRA_SBATCH_GPU_ALLOCATING:
                raise ValueError(
                    f"sbatch flag {flag!r} is not allowed in extra_sbatch: it can allocate GPUs that "
                    "resources.gpus does not count, so the budget reservation, GPU-hour estimate, "
                    "limit checks and compute guard would all be wrong; request GPUs with "
                    "resources.gpus (per node) and resources.nodes instead"
                )
            if not any(flag == a or (a.endswith("=") and flag.startswith(a)) for a in EXTRA_SBATCH_ALLOWED):
                raise ValueError(f"sbatch flag {flag!r} is not allowed in extra_sbatch")
            if any(c in flag for c in ";&|`$<>") or any(c.isspace() for c in flag):
                raise ValueError(f"sbatch flag {flag!r} contains shell metacharacters")
        return v

    @property
    def time_seconds(self) -> int:
        return parse_slurm_duration(self.time) or 0

    @property
    def total_gpus(self) -> int:
        return self.gpus * self.nodes

    @property
    def requested_gpu_hours(self) -> float:
        """GPUs x walltime limit: what a budget must reserve (not what will be used)."""
        return self.total_gpus * self.time_seconds / 3600.0

    def mem_bytes(self) -> int | None:
        if self.mem is None:
            return None
        m = re.match(r"^(\d+)([KMGT]?)$", self.mem, re.IGNORECASE)
        if not m:
            return None
        mult = {"": 1024**2, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[m.group(2).upper()]
        return int(float(m.group(1)) * mult)


OutputRole = Literal["metrics", "checkpoint", "log", "prediction", "figure", "other"]


class OutputDecl(Frozen):
    """A declared output. Only declared outputs are indexed as artifacts.

    ``path`` is relative to the attempt work dir, or starts with ``{output_dir}`` /
    ``{checkpoint_dir}``. Globs are not allowed (declare a ``dir`` instead).
    """

    name: str
    path: str
    role: OutputRole = "other"
    kind: Literal["file", "dir"] = "file"
    format: str | None = None
    required: bool = True

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9_.-]{1,64}$", v):
            raise ValueError("output name must match [A-Za-z0-9_.-]{1,64}")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if any(ch in v for ch in "*?[]"):
            raise ValueError("output paths may not contain glob characters; declare kind='dir'")
        if v.startswith("/") or ".." in v.split("/"):
            raise ValueError(
                "output paths must be relative (or start with {output_dir}) and may not contain '..'"
            )
        return v


class SmokeSpec(Frozen):
    """A short gate run on a compute node before the main matrix is dispatched.

    The smoke Run reuses the same code snapshot and runtime; it appends ``argv_append`` and
    overrides params/resources. Main cells are only dispatched after the smoke Attempt
    COMPLETED *and* its ``require_outputs`` validated.
    """

    enabled: bool = True
    argv_append: list[str] = Field(default_factory=list)
    params_override: dict[str, Scalar] = Field(default_factory=dict)
    time: str = "00:15:00"
    gpus: int | None = None
    partition: str | None = None
    require_outputs: list[str] = Field(default_factory=list)
    cell: dict[str, Scalar] | None = None  # which matrix cell to smoke; default = first


# sbatch --signal=[{R|B}:]<sig_num>[@sig_time]: Slurm keeps sig_time in a uint16 (0..65535 s).
SLURM_MAX_SIGNAL_LEAD_S = 65535


class ResilienceSpec(Frozen):
    """Walltime/preemption protection declared by the researcher.

    With ``stop_behavior='checkpoint_and_exit'`` hpclab adds ``--signal=B:<signal>@<lead_s>``;
    the wrapper forwards the signal so the program can checkpoint before the time limit, and
    ``run_cancel(mode='request_checkpoint')`` may send it on demand. ``resumable`` means the
    program resumes from ``$HPCLAB_RESUME_FROM``; it gates preemptible partitions and
    ``resume`` recovery. ``batch_invariant_to_gpus`` says whether the *global* batch stays the
    same when the GPU count changes — if not, an OOM fix that changes GPUs/batch is a new Run.
    Slurm may deliver the signal up to ~60 s late: keep ``lead_s`` >= checkpoint time + 60 s.
    ``lead_s`` is at most 65535 s: Slurm stores ``--signal``'s ``sig_time`` as a 16-bit value and
    sbatch rejects a larger one. ``max_checkpoint_interval_min`` is recorded for disclosure; not
    enforced in v0.1.
    """

    signal: Literal["USR1", "USR2", "TERM"] = "USR1"
    lead_s: int = Field(default=600, ge=60, le=SLURM_MAX_SIGNAL_LEAD_S)
    stop_behavior: Literal["checkpoint_and_exit", "none"] = "none"
    resumable: bool = False
    max_checkpoint_interval_min: int | None = Field(
        default=None,
        ge=1,
        description="Recorded for disclosure; not enforced in v0.1 (hpclab does not check checkpoint cadence).",
    )
    requeue_on_preemption: bool | None = None
    batch_invariant_to_gpus: bool = False


class BudgetSpec(Frozen):
    gpu_hours: float | None = Field(default=None, ge=0)
    max_parallel: int | None = Field(default=None, ge=1, le=256)


class MetricDecl(Frozen):
    name: str
    higher_is_better: bool | None = None
    unit: str | None = None


class EvaluationDefaults(Frozen):
    """Default evaluator applied to each run's outputs by ``evaluation_collect``."""

    evaluator_id: str = "hpclab_metrics"
    evaluator_version: str = "1"
    output: str = "metrics"
    config: dict[str, Any] = Field(default_factory=dict)


class SelectionRule(Frozen):
    """Which record/checkpoint a reported number comes from.

    Selecting the best checkpoint on the same split that is reported is a leakage risk;
    ``results_compare`` refuses it (``selection_leak``) unless explicitly declared otherwise.
    """

    rule: Literal["final", "last", "best_by", "at_step"] = "final"
    select_metric: str | None = None
    select_split: str | None = None
    mode: Literal["max", "min"] | None = None
    step: int | None = None
    report_split: str | None = None


class ComparisonContract(Frozen):
    """What may differ (treatment) and what must match (invariants) between compared runs."""

    treatment: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=lambda: ["split", "evaluator", "dataset"])
    baseline: dict[str, Scalar] | None = None
    metrics: list[MetricDecl] = Field(default_factory=list)
    pair_by_seed: bool = True
    ci_level: float = 0.95
    selection: SelectionRule = Field(default_factory=SelectionRule)
    attempt_selection: Literal["latest_valid", "first_valid"] = "latest_valid"
    min_n: int = Field(default=1, ge=1)
    bold_rule: Literal["none", "best_mean"] = "none"


class ExperimentSpec(Frozen):
    """Input to ``experiment_prepare``."""

    name: str
    question: str | None = None
    cluster: str
    code: CodeOptions = Field(default_factory=CodeOptions)
    entrypoint: EntrypointSpec
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    inputs: list[InputRef] = Field(default_factory=list)
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    matrix: dict[str, list[Scalar]] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    outputs: list[OutputDecl] = Field(default_factory=list)
    smoke: SmokeSpec | None = None
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    evaluation: EvaluationDefaults | None = None
    comparison: ComparisonContract | None = None
    resilience: ResilienceSpec | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$", v):
            raise ValueError("experiment name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
        return v

    @model_validator(mode="after")
    def _check(self) -> ExperimentSpec:
        if "seed" in self.matrix and self.seeds:
            raise ValueError("give seeds either in matrix.seed or in seeds, not both")
        if len(self.seeds) > MAX_SEEDS:
            raise ValueError(f"at most {MAX_SEEDS} seeds")
        for k, vals in self.matrix.items():
            if not vals:
                raise ValueError(f"matrix axis {k!r} is empty")
            if len({repr(v) for v in vals}) != len(vals):
                raise ValueError(f"matrix axis {k!r} has duplicate values")
        names = [o.name for o in self.outputs]
        if len(set(names)) != len(names):
            raise ValueError("output names must be unique")
        if self.cell_count() > MAX_MATRIX_CELLS:
            raise ValueError(f"matrix expands to {self.cell_count()} cells; hard limit is {MAX_MATRIX_CELLS}")
        return self

    def seed_values(self) -> list[int | None]:
        if self.seeds:
            return list(self.seeds)
        if "seed" in self.matrix:
            return [int(s) for s in self.matrix["seed"] if isinstance(s, (int, float, str))]
        return [None]

    def cell_count(self) -> int:
        n = 1
        for k, vals in self.matrix.items():
            if k != "seed":
                n *= len(vals)
        return n * max(1, len(self.seed_values()))


# ---------------------------------------------------------------------------
# Frozen Run + execution plan
# ---------------------------------------------------------------------------


class CodeSnapshotRef(Frozen):
    digest: str  # sha256 of the snapshot manifest
    file_count: int
    total_bytes: int
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    dirty_patch_digest: str | None = None
    untracked_included: list[str] = Field(default_factory=list)
    project_root: str | None = None


RunKind = Literal["main", "smoke", "probe"]


class RunSpec(Frozen):
    """Immutable scientific configuration of one Run (one matrix cell)."""

    experiment: str
    kind: RunKind = "main"
    code: CodeSnapshotRef
    entrypoint: EntrypointSpec
    argv: list[str]  # rendered: matrix/params substituted, runtime placeholders kept
    params: dict[str, Scalar] = Field(default_factory=dict)
    cell: dict[str, Scalar] = Field(default_factory=dict)  # the matrix coordinates of this run
    seed: int | None = None
    inputs: list[InputRef] = Field(default_factory=list)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    outputs: list[OutputDecl] = Field(default_factory=list)
    evaluation: EvaluationDefaults | None = None
    derived_from: str | None = None  # run id this Run was derived from

    def spec_hash(self) -> str:
        return digest(self.model_dump(mode="json"))


RecoveryStrategy = Literal["retry", "resume", "adjust_resources"]


class RecoverySource(Frozen):
    from_attempt_id: str
    strategy: RecoveryStrategy
    checkpoint_artifact_id: str | None = None
    checkpoint_digest: str | None = None
    checkpoint_remote_path: str | None = None
    resource_patch: dict[str, Any] = Field(default_factory=dict)
    diagnosis_codes: list[str] = Field(default_factory=list)


class ExecutionPlan(Frozen):
    """Frozen description of how one Attempt of a Run executes."""

    run_id: str
    run_spec_hash: str
    cluster_id: str
    profile_revision: str
    resources: ResourceSpec
    remote_root: str
    snapshot_path: str  # remote content-addressed code path
    script_template_version: str = SCRIPT_TEMPLATE_VERSION
    recovery: RecoverySource | None = None
    gate: str | None = None  # run id of the smoke gate that must pass first (workflow-managed)
    created_reason: Literal["initial", "retry", "resume", "adjust_resources"] = "initial"
    resilience: ResilienceSpec | None = None
    requeue: bool | None = None  # explicit --requeue / --no-requeue; None = site default

    def plan_hash(self) -> str:
        return digest(self.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Scheduler facts
# ---------------------------------------------------------------------------


class SchedulerJobRef(Frozen):
    """External job identity. ``job_id`` alone is never treated as global identity."""

    cluster_id: str
    job_id: str
    owner: str
    submit_time: str | None = None  # ISO UTC; from sacct/squeue when known
    array_task_id: str | None = None
    restart_count: int = 0


class JobObservation(BaseModel):
    """One observation of a job from one source at one time."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    source: Literal["squeue", "sacct", "receipt", "scontrol", "sim"]
    observed_at: str
    state: SchedulerState = SchedulerState.UNKNOWN
    raw_state: str | None = None
    name: str | None = None
    comment: str | None = None
    user: str | None = None
    account: str | None = None
    partition: str | None = None
    reason: str | None = None
    submit_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    elapsed_s: int | None = None
    time_limit_s: int | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    node_list: str | None = None
    alloc_gpus: int | None = None
    restart_count: int | None = None
    array_task_id: str | None = None
    max_rss_bytes: int | None = None
    submit_line: str | None = None
    work_dir: str | None = None
