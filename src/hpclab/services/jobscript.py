"""Attempt layout + the Slurm job wrapper (build spec §2.4, §6.1-6.4, §7.2 jobscript extensions).

``render_job_script`` produces ``job.sbatch``: ``#SBATCH`` directives followed by a bash
wrapper around the researcher's entrypoint. The wrapper is what makes an attempt *traceable*
and *diagnosable* on a shared cluster:

* ``status.json`` - ``started`` at the very beginning, ``exited``/``killed`` at the end
  (written atomically; an EXIT trap records unexpected wrapper exits too);
* a work dir ``work/`` = hardlink farm of the read-only content-addressed snapshot (the
  program cannot edit the published code; it can create new files; reused on requeue);
* ``env.json`` - interpreter, package digest (+ ``packages.txt``), torch/CUDA build, driver,
  GPUs, lockfile and input fingerprints; every probe is time-bounded and failure-tolerant;
* a compute-side *guard* (``RuntimeSpec.guard`` and GPUs requested): nvidia-smi works, torch
  sees the GPUs, the GPU architecture is in torch's build -> otherwise the job stops in
  seconds with ``HPCLAB_GUARD_FAILED:<check>`` on stderr and exit code 3, instead of burning
  a queue slot or silently training on CPU;
* the program runs with byte-exact argv, the §6.2 environment, forwarded TERM/INT/USR1/USR2
  (``--signal=B:<sig>@<lead>`` lets it checkpoint; ``$HPCLAB_STOP_FILE`` is written first and
  names the signal, e.g. ``USR1``). TERM/INT go to the program's main process; USR1/USR2 go to
  the topmost processes of its tree that handle them, so the workers behind a launcher that
  does not (torchrun, accelerate, deepspeed, a shell script) checkpoint instead of the
  launcher dying of the default action; with no handler anywhere the main process gets it;
* a background sidecar appends ``telemetry/heartbeat`` (epoch per interval) and
  ``telemetry/gpu.csv`` (nvidia-smi samples, capped) - advisory health signals only;
* a Slurm requeue (``SLURM_RESTART_COUNT`` > 0) of a ``resilience.resumable`` program resumes
  from the attempt's latest complete checkpoint (``$HPCLAB_RESUME_FROM``); any other program
  reruns from scratch: the previous segment's ``outputs/``, ``checkpoints/`` and ``work/``
  (declared outputs may be relative to it; it is rebuilt from the snapshot) move to
  ``<attempt>/requeue-<n>/`` (``status.json``/``env.json`` ``requeue_set_aside``,
  ``$HPCLAB_REQUEUE_SET_ASIDE``) and its own checkpoint is never handed to it;
* ``outputs.json`` - declared outputs with sha256 computed ON THE COMPUTE NODE (a quick
  stat-only pass first, then the hashing pass), then the wrapper exits with the program's
  exit code.

Portability: the wrapper is bash 3.2 compatible (macOS runs the simulator's jobs with
``/bin/bash`` 3.2; clusters have 4/5). JSON writing, hashing and probes use a stdlib-only
python3 (>= 3.6) helper embedded in the script; without python3 a degraded printf/sha256sum
path still writes valid status/env/outputs files. No helper failure ever changes the
program's exit code.

Value semantics: argv elements reach the program byte-exact; runtime placeholders
(``{output_dir}`` ... ``{job_id}``) are substituted, other braces stay literal.
``RuntimeSpec.env`` values additionally get cluster placeholders (``{scratch}``, ``{user}``,
``{home}``, ``{remote_root}``) substituted at render time and ``$USER``/``$HOME``/``~/``
expanded on the node (``HF_HOME: "{scratch}/hf"`` works); nothing else is shell-interpreted.
Runtime knobs (``RUNTIME_KNOBS``, e.g. ``HPCLAB_PROBE_TIMEOUT_S``) may be set via
``RuntimeSpec.env`` or the submission environment.

Trust boundary: ``RuntimeSpec.setup`` lines are the researcher's own shell code from their
own spec; each element is executed verbatim with ``eval`` (in the code dir, ``set -u``
relaxed) - hpclab does not sandbox it. Everything hpclab embeds in the script (paths, ids,
argv, env values) is shell-quoted; ``#SBATCH`` values are validated against a safe
character set because sbatch does not parse shell quoting.

Wrapper-reserved exit codes: 3 guard failed, 95 duplicate submission (another Slurm job id
already owns this attempt dir; nothing of the attempt was touched and status.json is NOT
written - stderr carries ``HPCLAB_DUPLICATE_JOB:<owner job id>``), 96 attempt/snapshot/work-dir
preparation failed, 97 a setup line failed (``status.json.failed_phase`` says which phase).
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hpclab.domain.clock import format_slurm_duration
from hpclab.domain.errors import InvalidArgument, Unsupported
from hpclab.domain.hashing import canonical_json
from hpclab.domain.ids import sanitize_label
from hpclab.domain.models import (
    RUNTIME_KNOB_ENV,
    RUNTIME_PLACEHOLDERS,
    SCRIPT_TEMPLATE_VERSION,
    ExecutionPlan,
    OutputDecl,
    ResourceSpec,
    RunSpec,
)
from hpclab.profiles.schema import ResolvedCluster

WRAPPER_VERSION = SCRIPT_TEMPLATE_VERSION
STATUS_SCHEMA = "hpclab.status/v1"
ENV_SCHEMA = "hpclab.env/v1"
OUTPUTS_SCHEMA = "hpclab.outputs/v1"

JOB_NAME_PREFIX = "hl"
JOB_COMMENT_PREFIX = "hpclab:"
MAX_JOB_NAME = 64

EXIT_GUARD_FAILED = 3
EXIT_DUPLICATE_JOB = 95
EXIT_PREPARE_FAILED = 96
EXIT_SETUP_FAILED = 97
# stderr line printed as f"{GUARD_MARKER}{check}" when a guard check fails (diagnosis keys on it)
GUARD_MARKER = "HPCLAB_GUARD_FAILED:"
# stderr line f"{DUPLICATE_MARKER}{owner_job_id}" from a duplicate job that refused to run
DUPLICATE_MARKER = "HPCLAB_DUPLICATE_JOB:"
GUARD_CHECKS = ("venv", "nvidia_smi", "torch_cuda_available", "torch_gpu_arch")

DEFAULT_PROBE_TIMEOUT_S = 60
DEFAULT_OUTPUT_HASH_BUDGET_S = 1800
TELEMETRY_CAP_BYTES = 2 * 1024 * 1024
MAX_HASH_BYTES = 50 * 1024**3

# Runtime knobs a spec may set through RuntimeSpec.env (exported before anything else runs).
RUNTIME_KNOBS = RUNTIME_KNOB_ENV

# Placeholders whose value is only known on the compute node; all others are rendered here.
_DYNAMIC_PLACEHOLDERS = {"job_id": "SLURM_JOB_ID", "resume_from": "HPCLAB_RESUME_FROM"}
_PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(sorted(RUNTIME_PLACEHOLDERS)) + r")\}")
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_TOKEN_RE = re.compile(r"^[a-z0-9]{8,64}$")
_JOB_NAME_TOKEN_RE = re.compile(r"^" + JOB_NAME_PREFIX + r"\.(?:.+)\.([a-z0-9]{8})$")
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./@+,=:~-]*$")
_SLURM_NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_NODE_RE = re.compile(r"^[A-Za-z0-9_.,\[\]-]+$")
_CONSTRAINT_BAD_RE = re.compile(r"[\s'\"\\$`;#]")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CLUSTER_PLACEHOLDER_RE = re.compile(r"\{(?:user|scratch|home|remote_root)\}")
_SHELL_PATH_TOKEN_RE = re.compile(r"\$\{USER\}|\$USER(?![A-Za-z0-9_])|\$\{HOME\}|\$HOME(?![A-Za-z0-9_])")
_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)B?$", re.IGNORECASE)
# RuntimeSpec.env keys that would change the wrapper itself: its own HL_* state variables and
# variables that alter how bash parses or starts up.
_RESERVED_ENV_PREFIX = "HL_"
_SHELL_CONTROL_ENV = frozenset({"IFS", "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "GLOBIGNORE", "CDPATH"})
# Search-path variables: a literal "$PATH" in their value breaks every later command lookup.
_SEARCH_PATH_ENV = frozenset(
    {"PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "CPATH", "PKG_CONFIG_PATH"}
)
_VAR_REFERENCE_RE = re.compile(r"\$(?:[A-Za-z_]|\{)")
_HEREDOC_HELPER = "HPCLAB_HELPER_PY_EOF"
_HEREDOC_CONFIG = "HPCLAB_CONFIG_JSON_EOF"


# ---------------------------------------------------------------------------
# Attempt layout (§6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptLayout:
    """Remote absolute paths of one attempt (§6.1). Construct with ``attempt_layout``."""

    dir: str
    script: str
    stdout: str
    stderr: str
    status_json: str
    env_json: str
    outputs_manifest: str
    receipt: str
    work: str
    output_dir: str
    checkpoint_dir: str
    token_file: str

    @property
    def logs_dir(self) -> str:
        return f"{self.dir}/logs"

    @property
    def telemetry_dir(self) -> str:
        return f"{self.dir}/telemetry"

    @property
    def heartbeat(self) -> str:
        """Wrapper sidecar appends one epoch-seconds line per interval (default 60 s)."""
        return f"{self.telemetry_dir}/heartbeat"

    @property
    def gpu_telemetry(self) -> str:
        """nvidia-smi CSV samples: timestamp,index,uuid,util%,mem used,mem total,power."""
        return f"{self.telemetry_dir}/gpu.csv"

    @property
    def stop_file(self) -> str:
        """``$HPCLAB_STOP_FILE``: written (first stop signal's name) before a stop/checkpoint
        signal (TERM, INT or the resilience signal) is forwarded to the program."""
        return f"{self.dir}/STOP"

    @property
    def packages_file(self) -> str:
        """Sorted ``name==version`` list of the runtime environment (for env diffs)."""
        return f"{self.dir}/packages.txt"

    @property
    def internal_dir(self) -> str:
        """Wrapper internals: the extracted helper and its config."""
        return f"{self.dir}/.hpclab"

    def dirs_to_create(self) -> tuple[str, ...]:
        """Directories that must exist BEFORE sbatch: Slurm opens ``--output``/``--error``
        before the script runs, and a missing directory fails the job without any log."""
        return (self.dir, self.logs_dir)


def _check_remote_path(path: str, what: str) -> str:
    if not _REMOTE_PATH_RE.match(path) or "/../" in f"{path}/" or "/./" in f"{path}/":
        raise InvalidArgument(
            f"{what} {path!r} must be an absolute path made of [A-Za-z0-9_./@+,=:~-]",
            details={what: path},
        )
    return path


def _check_id(value: str, what: str) -> str:
    """Ids become path components: also refuse '.'/'..' (all-dot names escape their parent)."""
    if not _ID_RE.match(value) or not value.strip("."):
        raise InvalidArgument(f"{what} {value!r} must match [A-Za-z0-9_.-]{{1,128}} and not be all dots")
    return value


def attempt_layout(remote_root: str, attempt_id: str) -> AttemptLayout:
    root = _check_remote_path(remote_root.rstrip("/") or "/", "remote_root").rstrip("/")
    d = f"{root}/attempts/{_check_id(attempt_id, 'attempt_id')}"
    return AttemptLayout(
        dir=d,
        script=f"{d}/job.sbatch",
        stdout=f"{d}/logs/stdout.log",
        stderr=f"{d}/logs/stderr.log",
        status_json=f"{d}/status.json",
        env_json=f"{d}/env.json",
        outputs_manifest=f"{d}/outputs.json",
        receipt=f"{d}/submit.receipt",
        work=f"{d}/work",
        output_dir=f"{d}/outputs",
        checkpoint_dir=f"{d}/checkpoints",
        token_file=f"{d}/hpclab-plan.json",
    )


# ---------------------------------------------------------------------------
# Job identity helpers
# ---------------------------------------------------------------------------


def _check_token(token: str) -> str:
    if not _TOKEN_RE.match(token):
        raise InvalidArgument(f"submit token {token!r} must be 8-64 lowercase alphanumerics")
    return token


def job_name_for(label: str, token: str) -> str:
    """``hl.<label>.<token[:8]>`` - sanitized, <= 64 chars. The full token goes in the comment."""
    name = f"{JOB_NAME_PREFIX}.{sanitize_label(label)}.{_check_token(token)[:8]}"
    if len(name) > MAX_JOB_NAME:  # sanitize_label caps at 40, so this is defensive only
        name = f"{JOB_NAME_PREFIX}.{sanitize_label(label, MAX_JOB_NAME - 12)}.{token[:8]}"
    return name


def token_from_job_name(name: str) -> str | None:
    """The 8-char token *prefix* embedded by ``job_name_for`` (None for foreign jobs).

    Compare it with ``token[:8]``; the full token is in the job comment
    (``token_from_comment``).
    """
    m = _JOB_NAME_TOKEN_RE.match(name.strip()) if name else None
    return m.group(1) if m else None


def token_from_comment(comment: str | None) -> str | None:
    """Full token from a ``hpclab:<token>`` job comment (None if absent/foreign)."""
    if not comment:
        return None
    text = comment.strip()
    if not text.startswith(JOB_COMMENT_PREFIX):
        return None
    token = text[len(JOB_COMMENT_PREFIX) :]
    return token if _TOKEN_RE.match(token) else None


# ---------------------------------------------------------------------------
# #SBATCH directives
# ---------------------------------------------------------------------------


def _slurm_name(value: str, what: str) -> str:
    if not _SLURM_NAME_RE.match(value):
        raise InvalidArgument(f"{what} {value!r} contains characters sbatch cannot take unquoted")
    return value


def _sbatch_mem(mem: str) -> str:
    """Slurm needs an integer with an optional K/M/G/T suffix (no ``B``, no decimals)."""
    m = _MEM_RE.match(mem)
    if not m:
        raise InvalidArgument(f"invalid memory {mem!r}")
    number, unit = m.group(1), m.group(2).upper()
    if "." not in number:
        return number + unit
    factor = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    return f"{math.ceil(float(number) * factor)}M"


def _resolve_requeue(plan: ExecutionPlan, extra: Sequence[str]) -> str | None:
    wanted = None if plan.requeue is None else ("--requeue" if plan.requeue else "--no-requeue")
    given = [f for f in extra if f in ("--requeue", "--no-requeue")]
    if wanted and any(f != wanted for f in given):
        raise InvalidArgument(
            f"extra_sbatch {given} contradicts the plan's requeue={plan.requeue}",
            details={"extra_sbatch": list(extra)},
        )
    return wanted if wanted and wanted not in given else None


def resilience_signal(plan: ExecutionPlan) -> str | None:
    """The checkpoint signal the wrapper announces via ``--signal=B:<sig>@<lead>`` (or None)."""
    res = plan.resilience
    if res is None or res.stop_behavior != "checkpoint_and_exit":
        return None
    return res.signal


def sbatch_directives(
    *,
    resources: ResourceSpec,
    plan: ExecutionPlan,
    cluster: ResolvedCluster,
    job_name: str,
    token: str,
    layout: AttemptLayout,
) -> list[str]:
    """The ``--flag=value`` list for the ``#SBATCH`` header (values validated, never quoted)."""
    if not _ID_RE.match(job_name) or len(job_name) > MAX_JOB_NAME:
        raise InvalidArgument(f"job name {job_name!r} is not a sanitized hpclab job name")
    res = resources
    account = res.account or cluster.overlay.account
    qos = res.qos or cluster.overlay.qos
    partition = res.partition or cluster.overlay.default_partition or cluster.profile.default_partition
    flags = [
        f"--job-name={job_name}",
        f"--comment={JOB_COMMENT_PREFIX}{_check_token(token)}",
        f"--output={layout.stdout}",
        f"--error={layout.stderr}",
        "--open-mode=append",
        f"--chdir={layout.dir}",
    ]
    if account:
        flags.append(f"--account={_slurm_name(account, 'account')}")
    if partition:
        flags.append(f"--partition={_slurm_name(partition, 'partition')}")
    if qos:
        flags.append(f"--qos={_slurm_name(qos, 'qos')}")
    if res.gpus > 0:
        gtype = f"{_slurm_name(res.gpu_type, 'gpu_type')}:" if res.gpu_type else ""
        flags.append(f"--gres=gpu:{gtype}{res.gpus}")
    flags.append(f"--cpus-per-task={res.cpus}")
    if res.mem:
        per_unit = [f for f in res.extra_sbatch if f.startswith(("--mem-per-gpu=", "--mem-per-cpu="))]
        if per_unit:
            raise InvalidArgument(
                f"resources.mem and extra_sbatch {per_unit} are mutually exclusive in sbatch; keep one",
                details={"mem": res.mem, "extra_sbatch": list(res.extra_sbatch)},
            )
        flags.append(f"--mem={_sbatch_mem(res.mem)}")
    flags.append(f"--time={format_slurm_duration(res.time_seconds)}")
    flags.append(f"--nodes={res.nodes}")
    flags.append(f"--ntasks-per-node={res.ntasks_per_node}")
    if res.exclude_nodes:
        for node in res.exclude_nodes:
            if not _NODE_RE.match(node):
                raise InvalidArgument(f"exclude node {node!r} is not a valid Slurm node expression")
        flags.append(f"--exclude={','.join(res.exclude_nodes)}")
    if res.constraint:
        if _CONSTRAINT_BAD_RE.search(res.constraint):
            raise InvalidArgument(
                f"constraint {res.constraint!r} contains whitespace, quotes or shell characters"
            )
        flags.append(f"--constraint={res.constraint}")
    resilience = plan.resilience
    if resilience is not None and resilience.stop_behavior == "checkpoint_and_exit":
        if any(f.startswith("--signal=") for f in res.extra_sbatch):
            raise InvalidArgument(
                "extra_sbatch sets --signal but resilience.stop_behavior already manages it"
            )
        flags.append(f"--signal=B:{resilience.signal}@{resilience.lead_s}")
    requeue = _resolve_requeue(plan, res.extra_sbatch)
    if requeue:
        flags.append(requeue)
    for extra in res.extra_sbatch:
        if any(ch.isspace() for ch in extra):
            raise InvalidArgument(f"extra_sbatch flag {extra!r} contains whitespace")
        flags.append(extra)
    return flags


# ---------------------------------------------------------------------------
# Bash rendering helpers
# ---------------------------------------------------------------------------


def _check_no_nul(value: str, what: str) -> str:
    if "\0" in value:
        raise InvalidArgument(f"{what} contains a NUL byte, which cannot be passed to a program")
    return value


def bash_word(template: str, static: Mapping[str, str]) -> str:
    """Render ``template`` as ONE bash word whose value is byte-exact.

    Runtime placeholders known at render time are substituted into literal text (single-quoted
    via ``shlex.quote``); ``{job_id}``/``{resume_from}`` become ``"${VAR:-}"`` expansions.
    Braces that are not runtime placeholders are kept verbatim.
    """
    pieces: list[str] = []
    literal: list[str] = []
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(template):
        literal.append(template[pos : m.start()])
        name = m.group(1)
        if name in _DYNAMIC_PLACEHOLDERS:
            if literal:
                text = "".join(literal)
                if text:
                    pieces.append(shlex.quote(text))
                literal = []
            pieces.append('"${' + _DYNAMIC_PLACEHOLDERS[name] + ':-}"')
        else:
            literal.append(static[name])
        pos = m.end()
    literal.append(template[pos:])
    tail = "".join(literal)
    if tail or not pieces:
        pieces.append(shlex.quote(tail))
    return "".join(pieces)


def node_word(text: str, static: Mapping[str, str]) -> str:
    """Like ``bash_word`` but ``$USER``/``${USER}``/``$HOME``/``${HOME}`` and a leading ``~/``
    expand on the compute node (cluster paths such as ``/scratch/$USER/hf``). Nothing else
    is interpreted by the shell."""
    if text.startswith("~/"):
        text = "$HOME/" + text[2:]
    pieces: list[str] = []
    pos = 0
    for m in _SHELL_PATH_TOKEN_RE.finditer(text):
        if m.start() > pos:
            pieces.append(bash_word(text[pos : m.start()], static))
        pieces.append('"${' + ("USER" if "USER" in m.group(0) else "HOME") + ':-}"')
        pos = m.end()
    if pos < len(text) or not pieces:
        pieces.append(bash_word(text[pos:], static))
    return "".join(pieces)


def _expand_cluster(text: str, cluster: ResolvedCluster, what: str) -> str:
    """Substitute ``{user}``/``{scratch}``/``{home}``/``{remote_root}``; refuse leftovers.

    Only used for values the compute node's shell expands (env, venv, input fingerprints), so
    an unknown ``{user}`` may become ``$USER`` here — unlike paths sbatch itself reads.
    """
    out = cluster.expand(text)
    if cluster.overlay.user is None:
        out = out.replace("{user}", "$USER")
    left = _CLUSTER_PLACEHOLDER_RE.search(out)
    if left:
        raise InvalidArgument(
            f"{what} uses {left.group(0)} but cluster {cluster.cluster_id} defines no such storage role",
            details={"value": text, "storage_roles": sorted(cluster.profile.storage)},
        )
    return out


def _comment_safe(text: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", "?", text)


def _function(name: str, body_lines: Sequence[str]) -> str:
    body = "\n".join(f"  {line}" for line in body_lines) if body_lines else "  :"
    return f"{name}() {{\n{body}\n}}"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _static_placeholders(run_id: str, attempt_id: str, layout: AttemptLayout) -> dict[str, str]:
    return {
        "output_dir": layout.output_dir,
        "checkpoint_dir": layout.checkpoint_dir,
        "attempt_dir": layout.dir,
        "code_dir": layout.work,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }


def _output_template(decl: OutputDecl, static: Mapping[str, str], work: str) -> str:
    """Declared output path with static placeholders rendered; relative -> under the work dir.
    ``{job_id}``/``{resume_from}`` are left for the compute-node helper."""
    path = _PLACEHOLDER_RE.sub(lambda m: static.get(m.group(1), m.group(0)), decl.path)
    return path if path.startswith("/") or path.startswith("{") else f"{work}/{path}"


def _validate_relative(path: str, what: str) -> str:
    parts = path.split("/")
    if path.startswith("/") or ".." in parts or "\0" in path:
        raise InvalidArgument(f"{what} {path!r} must be relative to the code dir and may not contain '..'")
    return path


def _user_env(
    run_spec: RunSpec, static: Mapping[str, str], cluster: ResolvedCluster
) -> tuple[list[str], list[str]]:
    """(knob exports, user env exports) as bash lines.

    Values: cluster placeholders are substituted here, runtime placeholders and
    ``$USER``/``$HOME`` resolve on the node, everything else is literal.
    """
    knobs: list[str] = []
    user: list[str] = []
    for key in sorted(run_spec.runtime.env):
        value = _check_no_nul(run_spec.runtime.env[key], f"runtime.env[{key}]")
        _check_env_key(key)
        if key in RUNTIME_KNOBS:
            knobs.append(f"export {key}={shlex.quote(value)}")
            continue
        expanded = _expand_cluster(value, cluster, f"runtime.env[{key}]")
        if key in _SEARCH_PATH_ENV and _VAR_REFERENCE_RE.search(_SHELL_PATH_TOKEN_RE.sub("", expanded)):
            raise InvalidArgument(
                f"runtime.env[{key}] references a variable, but env values are literal (only $USER and "
                f"$HOME expand); extend {key} in runtime.setup instead, e.g. 'export {key}=/opt/x:${key}'",
                details={"key": key, "value": value},
            )
        user.append(f"export {key}={node_word(expanded, static)}")
    return knobs, user


def _check_env_key(key: str) -> None:
    """Refuse names that are invalid or would change the wrapper rather than the program."""
    if not _ENV_KEY_RE.match(key):
        raise InvalidArgument(f"runtime.env key {key!r} is not a valid environment variable name")
    if key.startswith("HPCLAB_") and key not in RUNTIME_KNOBS:
        raise InvalidArgument(
            f"runtime.env may not set {key}: HPCLAB_* variables are provided by the wrapper",
            details={"allowed_knobs": sorted(RUNTIME_KNOBS)},
        )
    if key.startswith(_RESERVED_ENV_PREFIX) or key in _SHELL_CONTROL_ENV:
        raise InvalidArgument(
            f"runtime.env may not set {key}: it would change the job wrapper itself "
            f"({_RESERVED_ENV_PREFIX}* are wrapper-internal; {', '.join(sorted(_SHELL_CONTROL_ENV))} "
            "change how bash runs); set it in runtime.setup if you really need it",
            details={"key": key},
        )


def _outputs_config(run_spec: RunSpec, static: Mapping[str, str], work: str) -> list[dict[str, object]]:
    return [
        {
            "name": o.name,
            "path": _output_template(o, static, work),
            "kind": o.kind,
            "role": o.role,
            "required": o.required,
        }
        for o in run_spec.outputs
    ]


def _config(
    *,
    run_spec: RunSpec,
    plan: ExecutionPlan,
    cluster: ResolvedCluster,
    run_id: str,
    attempt_id: str,
    layout: AttemptLayout,
    outputs: list[dict[str, object]],
) -> dict[str, object]:
    for lockfile in run_spec.runtime.lockfiles:
        _validate_relative(lockfile, "runtime.lockfiles entry")
    return {
        "schema": "hpclab.wrapper-config/v1",
        "wrapper_version": WRAPPER_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "python": run_spec.runtime.python,
        "code_dir": layout.work,
        "attempt_dir": layout.dir,
        "snapshot_path": plan.snapshot_path,
        "code_digest": run_spec.code.digest,
        "packages_file": layout.packages_file,
        "gpus_per_node": plan.resources.gpus,
        "guard": bool(run_spec.runtime.guard and plan.resources.gpus > 0),
        "probe_timeout_s": DEFAULT_PROBE_TIMEOUT_S,
        "output_hash_budget_s": DEFAULT_OUTPUT_HASH_BUDGET_S,
        "lockfiles": list(run_spec.runtime.lockfiles),
        "inputs": [
            {
                "name": i.name,
                "uri": _expand_cluster(i.uri, cluster, f"input {i.name}"),
                "digest": i.digest,
                "pin": i.pin,
            }
            for i in run_spec.inputs
        ],
        "outputs": outputs,
        "resumable": _resumable(plan),
    }


def _resumable(plan: ExecutionPlan) -> bool:
    """``resilience.resumable``: a requeued segment continues from the attempt's own checkpoint;
    otherwise it reruns from scratch with the previous segment's files set aside."""
    return bool(plan.resilience is not None and plan.resilience.resumable)


def _values_block(
    *,
    run_spec: RunSpec,
    plan: ExecutionPlan,
    run_id: str,
    attempt_id: str,
    token: str,
    layout: AttemptLayout,
    cluster: ResolvedCluster,
    static: Mapping[str, str],
) -> list[str]:
    q = shlex.quote
    resume = plan.recovery.checkpoint_remote_path if plan.recovery else None
    if resume:
        _check_remote_path(resume, "recovery.checkpoint_remote_path")
    workdir = _validate_relative(run_spec.entrypoint.workdir.strip() or ".", "entrypoint.workdir")
    venv = run_spec.runtime.venv
    values = {
        "HL_WRAPPER_VERSION": WRAPPER_VERSION,
        "HL_RUN_ID": run_id,
        "HL_ATTEMPT_ID": attempt_id,
        "HL_TOKEN": token,
        "HL_RUN_SPEC_HASH": plan.run_spec_hash,
        "HL_ATTEMPT_DIR": layout.dir,
        "HL_LOGS_DIR": layout.logs_dir,
        "HL_WORK_DIR": layout.work,
        "HL_OUTPUT_DIR": layout.output_dir,
        "HL_CHECKPOINT_DIR": layout.checkpoint_dir,
        "HL_TELEMETRY_DIR": layout.telemetry_dir,
        "HL_INTERNAL_DIR": layout.internal_dir,
        "HL_STATUS_JSON": layout.status_json,
        "HL_ENV_JSON": layout.env_json,
        "HL_OUTPUTS_JSON": layout.outputs_manifest,
        "HL_HEARTBEAT": layout.heartbeat,
        "HL_GPU_CSV": layout.gpu_telemetry,
        "HL_STOP_FILE": layout.stop_file,
        "HL_SNAPSHOT": _check_remote_path(plan.snapshot_path.rstrip("/"), "snapshot_path"),
        "HL_ENTRY_WORKDIR": workdir,
        "HL_RUNTIME_PYTHON": _check_no_nul(run_spec.runtime.python, "runtime.python"),
        "HL_PLAN_RESUME_FROM": resume or "",
        "HL_RESUMABLE": "1" if _resumable(plan) else "0",
        "HL_SEED": "" if run_spec.seed is None else str(run_spec.seed),
        "HL_PARAMS_JSON": canonical_json(run_spec.params),
        "HL_GPUS_PER_NODE": str(plan.resources.gpus),
        "HL_GUARD": "1" if (run_spec.runtime.guard and plan.resources.gpus > 0) else "0",
        "HL_RESILIENCE_SIGNAL": resilience_signal(plan) or "",
        "HL_EXIT_GUARD": str(EXIT_GUARD_FAILED),
        "HL_EXIT_DUPLICATE": str(EXIT_DUPLICATE_JOB),
        "HL_DUPLICATE_MARKER": DUPLICATE_MARKER,
        "HL_EXIT_PREPARE": str(EXIT_PREPARE_FAILED),
        "HL_EXIT_SETUP": str(EXIT_SETUP_FAILED),
        "HL_TELEMETRY_CAP": str(TELEMETRY_CAP_BYTES),
        "HL_MAX_HASH_BYTES": str(MAX_HASH_BYTES),
    }
    # Paths inside the attempt dir are written relative to it: the long prefix once, not 16
    # times, keeps the whole script within one 64 KiB plan_script page.
    prefix = layout.dir + "/"
    lines = [
        f'{k}="$HL_ATTEMPT_DIR"/{q(v[len(prefix) :])}'
        if k != "HL_ATTEMPT_DIR" and v.startswith(prefix)
        else f"{k}={q(v)}"
        for k, v in values.items()
    ]
    venv_word = node_word(_expand_cluster(venv, cluster, "runtime.venv"), static) if venv else q("")
    lines.append(f"HL_VENV={venv_word}")
    return lines


def _generated_functions(
    *,
    run_spec: RunSpec,
    static: Mapping[str, str],
    config: Mapping[str, object],
    cluster: ResolvedCluster,
) -> list[str]:
    q = shlex.quote
    knobs, user_env = _user_env(run_spec, static, cluster)
    setup = [
        f"_hl_setup_line {i} {q(_check_no_nul(line, f'runtime.setup[{i}]'))} || return 1"
        for i, line in enumerate(run_spec.runtime.setup, start=1)
    ]
    argv = [bash_word(_check_no_nul(a, "argv element"), static) for a in run_spec.argv]
    fallback_outputs = [
        "_hl_out_entry {name} {kind} {req} {path}".format(
            name=q(o.name),
            kind=q(o.kind),
            req="true" if o.required else "false",
            path=bash_word(_output_template(o, static, static["code_dir"]), static),
        )
        for o in run_spec.outputs
    ]
    # One line starting with "{" (json escapes newlines), so it can never equal the terminator.
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return [
        _function("_hl_export_knobs", knobs),
        _function("_hl_export_user_env", user_env),
        _function("_hl_run_setup", [*setup, "return 0"]),
        _function("_hl_build_argv", ["HL_ARGV=( " + " ".join(argv) + " )"]),
        _function("_hl_outputs_fallback_entries", fallback_outputs),
        f"_hl_emit_config() {{\n  cat <<'{_HEREDOC_CONFIG}'\n{config_json}\n{_HEREDOC_CONFIG}\n}}",
    ]


def render_job_script(
    *,
    run_spec: RunSpec,
    plan: ExecutionPlan,
    cluster: ResolvedCluster,
    run_id: str,
    attempt_id: str,
    token: str,
    job_name: str,
    layout: AttemptLayout,
) -> str:
    """Render ``job.sbatch`` (deterministic for identical inputs).

    Raises ``InvalidArgument`` for values that cannot be embedded safely (paths outside the
    safe charset, unsafe sbatch values, bad or wrapper-reserved env names, NUL bytes,
    conflicting flags), for a plan that belongs to another run or another version of the run
    spec (``plan.run_spec_hash`` must equal ``run_spec.spec_hash()``: it is exported as
    ``HPCLAB_RUN_SPEC_HASH`` and keys checkpoint compatibility), and ``Unsupported`` for
    ``runtime.container`` (not handled by wrapper v1: put ``apptainer exec ...`` in the
    entrypoint argv instead).
    """
    if plan.run_id != run_id:
        raise InvalidArgument(f"execution plan belongs to run {plan.run_id}, not {run_id}")
    spec_hash = run_spec.spec_hash()
    if plan.run_spec_hash != spec_hash:
        raise InvalidArgument(
            "execution plan run_spec_hash does not match the run spec being rendered",
            details={"plan_run_spec_hash": plan.run_spec_hash, "run_spec_hash": spec_hash},
        )
    _check_id(run_id, "run_id")
    _check_id(attempt_id, "attempt_id")
    _check_token(token)
    if run_spec.runtime.container:
        raise Unsupported(
            "runtime.container is not supported by hpclab-wrapper/1",
            details={"hint": "put 'apptainer exec --nv <image> ...' at the start of entrypoint.argv"},
        )
    if not run_spec.argv:
        raise InvalidArgument("run_spec.argv is empty")
    static = _static_placeholders(run_id, attempt_id, layout)
    outputs = _outputs_config(run_spec, static, layout.work)
    config = _config(
        run_spec=run_spec,
        plan=plan,
        cluster=cluster,
        run_id=run_id,
        attempt_id=attempt_id,
        layout=layout,
        outputs=outputs,
    )
    directives = sbatch_directives(
        resources=plan.resources, plan=plan, cluster=cluster, job_name=job_name, token=token, layout=layout
    )
    code = run_spec.code
    header = [
        "#!/bin/bash",
        *(f"#SBATCH {flag}" for flag in directives),
        "#",
        f"# Generated by hpclab ({WRAPPER_VERSION}). Part of the attempt record: do not edit.",
        _comment_safe(f"# run {run_id} | attempt {attempt_id} | run spec {plan.run_spec_hash}"),
        _comment_safe(
            f"# code {code.digest} at {plan.snapshot_path}"
            + (f" | git {code.git_commit}{' (dirty)' if code.git_dirty else ''}" if code.git_commit else "")
        ),
        _comment_safe(f"# cluster {plan.cluster_id} | profile {plan.profile_revision}"),
        "",
        "# ---- values (generated) ----",
    ]
    values = _values_block(
        run_spec=run_spec,
        plan=plan,
        run_id=run_id,
        attempt_id=attempt_id,
        token=token,
        layout=layout,
        cluster=cluster,
        static=static,
    )
    functions = _generated_functions(run_spec=run_spec, static=static, config=config, cluster=cluster)
    helper = f"_hl_emit_helper() {{\n  cat <<'{_HEREDOC_HELPER}'\n{HELPER_SOURCE}{_HEREDOC_HELPER}\n}}"
    parts = [
        "\n".join(header),
        "\n".join(values),
        "\n\n".join(functions),
        WRAPPER_RUNTIME.strip("\n"),
        helper,
        "_hl_main\nexit $?\n",
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Static bash runtime (bash 3.2 compatible; `set -u`, never `set -e`)
# ---------------------------------------------------------------------------

WRAPPER_RUNTIME = r"""
# ---- runtime (static for this wrapper version) ----
set -u
HL_PY=
HL_HELPER="$HL_INTERNAL_DIR/helper.py"
HL_CONFIG="$HL_INTERNAL_DIR/config.json"
HL_CHILD=
HL_FORWARDED=
HL_SIDECAR_PID=
HL_TERM_SIG=
HL_STOP_SIGNAL=
HL_SIGNALS=
HL_WAIT_INTERRUPTED=0
HL_DONE=0
HL_PHASE=init
HL_RC=
HL_START_TIME=
HL_HOST=unknown
HL_RESUME_SOURCE=
HL_TIMEOUT_BIN=
HL_SETUP_FAILED_INDEX=
HL_SETUP_FAILED_RC=
HL_OUT_FIRST=1
HL_OUT_MISSING=
HL_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
HL_REQUEUE_SET_ASIDE=
HPCLAB_RESUME_FROM=$HL_PLAN_RESUME_FROM

_hl_log() { printf '[hpclab] %s\n' "$*" >&2; }
_hl_now() { date +%s; }

_hl_jstr() {
  if [ -z "$1" ]; then printf 'null'; return 0; fi
  printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr -d '\000-\037')"
}
_hl_jnum() { case "$1" in ''|*[!0-9]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }
_hl_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }
_hl_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
_hl_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" 2>/dev/null | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
  fi
}

# _hl_timeout SECONDS CMD... : coreutils timeout when present, else a watchdog.
# Returns the command's own status: a trapped signal interrupts `wait` (status > 128) while
# the command keeps running, so the watchdog path waits again until it has really ended.
_hl_timeout() {
  local secs=$1 p w rc rc2
  shift
  if [ -n "$HL_TIMEOUT_BIN" ]; then "$HL_TIMEOUT_BIN" -k 5 "$secs" "$@"; return $?; fi
  "$@" &
  p=$!
  ( sleep "$secs"; kill -9 "$p" 2>/dev/null ) </dev/null >/dev/null 2>&1 &
  w=$!
  while :; do
    wait "$p"
    rc=$?
    [ "$rc" -gt 128 ] || break
    if kill -0 "$p" 2>/dev/null; then continue; fi
    wait "$p" 2>/dev/null
    rc2=$?
    [ "$rc2" -eq 127 ] || rc=$rc2
    break
  done
  kill "$w" 2>/dev/null
  return $rc
}

# ---- signals ----
# _hl_forward SIG: TERM/INT go to the program's main process; USR1/USR2 (the checkpoint
# signal) to the topmost processes of its tree that handle it, so the workers behind a
# launcher that does not (torchrun, accelerate, `bash train.sh`) get it instead of the
# launcher dying of it. If nothing handles it (or no helper), the main process gets it.
_hl_forward() {
  local sig=$1 targets=$HL_CHILD out= pid sent= IFS=' '
  case "$sig" in USR1|USR2)
    [ -z "$HL_PY" ] || out=$(_hl_timeout 10 "$HL_PY" -I "$HL_HELPER" sigtargets "$HL_CHILD" "$sig" 2>/dev/null) ;;
  esac
  case "$out" in ''|*[!0-9\ ]*) ;; *) targets=$out ;; esac
  for pid in $targets; do kill -s "$sig" "$pid" 2>/dev/null && sent="${sent:+$sent }$pid"; done
  [ -n "$sent" ] || return 1
  HL_FORWARDED="the program (pid $sent)"
  [ "$sent" = "$HL_CHILD" ] || HL_FORWARDED="the program's processes that handle it (pid $sent, not the main process $HL_CHILD)"
}
_hl_on_signal() {
  local sig=$1
  HL_WAIT_INTERRUPTED=1
  HL_SIGNALS="${HL_SIGNALS:+$HL_SIGNALS,}$sig"
  case "$sig" in TERM|INT) [ -n "$HL_TERM_SIG" ] || HL_TERM_SIG=$sig ;; esac
  if [ "$sig" = TERM ] || [ "$sig" = INT ] || [ "$sig" = "$HL_RESILIENCE_SIGNAL" ]; then
    [ -n "$HL_STOP_SIGNAL" ] || HL_STOP_SIGNAL=$sig
    # $HPCLAB_STOP_FILE names the first stop signal (hpclab_io exits 128+<that signal>).
    if [ ! -s "$HL_STOP_FILE" ]; then printf '%s\n' "$HL_STOP_SIGNAL" > "$HL_STOP_FILE" 2>/dev/null; fi
  fi
  if [ -n "$HL_CHILD" ] && _hl_forward "$sig"; then
    _hl_log "received SIG$sig; forwarded to $HL_FORWARDED"
  else
    # No program yet: TERM/INT make the pre-run checks (_hl_bail_if_signalled) stop the
    # attempt. A checkpoint signal is not replayed later (the program may not have installed
    # its handler yet, and USR1's default action kills it); $HPCLAB_STOP_FILE is touched.
    _hl_log "received SIG$sig (phase $HL_PHASE)"
  fi
}
_hl_on_exit() {
  local rc=$?
  if [ "$HL_DONE" != 1 ]; then
    HL_DONE=1
    if [ -n "$HL_CHILD" ]; then kill -s TERM "$HL_CHILD" 2>/dev/null; fi
    _hl_log "wrapper exited unexpectedly during phase $HL_PHASE (exit $rc)"
    _hl_write_status exited "$rc" "" "$HL_PHASE" "wrapper exited unexpectedly during $HL_PHASE (exit $rc)"
  fi
  _hl_stop_sidecar
}
_hl_install_traps() {
  trap '_hl_on_signal TERM' TERM
  trap '_hl_on_signal INT' INT
  trap '_hl_on_signal USR1' USR1
  trap '_hl_on_signal USR2' USR2
  trap '_hl_on_exit' EXIT
}

# ---- helper (python3 >= 3.6, stdlib only) ----
_hl_find_python() {
  local cand path
  HL_PY=
  if [ -n "${HPCLAB_HELPER_PYTHON:-}" ]; then
    [ "$HPCLAB_HELPER_PYTHON" = none ] && return 1
    set -- "$HPCLAB_HELPER_PYTHON"
  else
    set -- python3 "$HL_RUNTIME_PYTHON"
  fi
  for cand in "$@"; do
    path=$(command -v "$cand" 2>/dev/null) || continue
    [ -n "$path" ] || continue
    if "$path" -I -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' >/dev/null 2>&1; then
      HL_PY=$path
      return 0
    fi
  done
  return 1
}
_hl_setup_helper() {
  _hl_find_python || return 1
  if _hl_emit_helper > "$HL_HELPER.tmp.$$" && mv -f "$HL_HELPER.tmp.$$" "$HL_HELPER" \
    && _hl_emit_config > "$HL_CONFIG.tmp.$$" && mv -f "$HL_CONFIG.tmp.$$" "$HL_CONFIG"; then
    return 0
  fi
  rm -f "$HL_HELPER.tmp.$$" "$HL_CONFIG.tmp.$$"
  HL_PY=
  return 1
}
_hl_helper() { "$HL_PY" -I "$HL_HELPER" "$@"; }

# ---- status.json ----
# _hl_write_status STATE [EXIT_CODE] [SIGNAL] [FAILED_PHASE] [ERROR] [GUARD_FAILED]
_hl_write_status() {
  local state=$1 rc=${2:-} sig=${3:-} phase=${4:-} err=${5:-} guard=${6:-} end= stop=0
  [ "$state" = started ] || end=$(_hl_now)
  [ -z "$HL_STOP_SIGNAL" ] || stop=1
  if [ -n "$HL_PY" ] && _hl_helper status "$HL_STATUS_JSON" \
    "state=$state" "attempt_id=$HL_ATTEMPT_ID" "run_id=$HL_RUN_ID" "token=$HL_TOKEN" \
    "job_id=${SLURM_JOB_ID:-}" "host=$HL_HOST" "start_time=$HL_START_TIME" "end_time=$end" \
    "exit_code=$rc" "signal=$sig" "wrapper_version=$HL_WRAPPER_VERSION" \
    "restart_count=$HL_RESTART_COUNT" "failed_phase=$phase" "error=$err" "guard_failed=$guard" \
    "signals_received=$HL_SIGNALS" "resume_from=$HPCLAB_RESUME_FROM" "resume_source=$HL_RESUME_SOURCE" \
    "stop_requested=$stop" "stop_signal=$HL_STOP_SIGNAL" "pid=$$" "resumable=$HL_RESUMABLE" \
    "requeue_set_aside=$HL_REQUEUE_SET_ASIDE"; then
    return 0
  fi
  _hl_write_status_fallback "$state" "$rc" "$sig" "$phase" "$err" "$guard" "$end" "$stop"
}
_hl_write_status_fallback() {
  local tmp="$HL_STATUS_JSON.tmp.$$" sigs= s stop=false resumable=false IFS=,
  for s in $HL_SIGNALS; do sigs="${sigs:+$sigs,}\"$s\""; done
  IFS=' '
  [ "$8" = 1 ] && stop=true
  [ "$HL_RESUMABLE" = 1 ] && resumable=true
  {
    printf '{"schema":"hpclab.status/v1","state":%s,"attempt_id":%s,"run_id":%s,"token":%s,' \
      "$(_hl_jstr "$1")" "$(_hl_jstr "$HL_ATTEMPT_ID")" "$(_hl_jstr "$HL_RUN_ID")" "$(_hl_jstr "$HL_TOKEN")"
    printf '"job_id":%s,"host":%s,"start_time":%s,"end_time":%s,"exit_code":%s,"signal":%s,' \
      "$(_hl_jstr "${SLURM_JOB_ID:-}")" "$(_hl_jstr "$HL_HOST")" "$(_hl_jnum "$HL_START_TIME")" \
      "$(_hl_jnum "$7")" "$(_hl_jnum "$2")" "$(_hl_jstr "$3")"
    printf '"wrapper_version":%s,"restart_count":%s,"failed_phase":%s,"error":%s,"guard_failed":%s,' \
      "$(_hl_jstr "$HL_WRAPPER_VERSION")" "$(_hl_jnum "$HL_RESTART_COUNT")" "$(_hl_jstr "$4")" \
      "$(_hl_jstr "$5")" "$(_hl_jstr "$6")"
    printf '"signals_received":[%s],"resume_from":%s,"resume_source":%s,"stop_requested":%s,' \
      "$sigs" "$(_hl_jstr "$HPCLAB_RESUME_FROM")" "$(_hl_jstr "$HL_RESUME_SOURCE")" "$stop"
    printf '"stop_signal":%s,"pid":%s,"resumable":%s,"requeue_set_aside":%s,"degraded":true,"written_at":%s}\n' \
      "$(_hl_jstr "$HL_STOP_SIGNAL")" "$$" "$resumable" "$(_hl_jstr "$HL_REQUEUE_SET_ASIDE")" "$(_hl_now)"
  } > "$tmp" 2>/dev/null && mv -f "$tmp" "$HL_STATUS_JSON" 2>/dev/null
}

# ---- outputs.json ----
_hl_write_outputs() {
  if [ -n "$HL_PY" ] && _hl_helper outputs "$HL_OUTPUTS_JSON" "$HL_CONFIG" "$1"; then return 0; fi
  [ "$1" = quick ] || return 0
  _hl_write_outputs_fallback
}
_hl_out_entry() {
  local name=$1 kind=$2 req=$3 p=$4 exists=false size=null mtime=null sha=null skipped=null h
  if [ -e "$p" ]; then
    exists=true
    mtime=$(_hl_jnum "$(_hl_mtime "$p")")
    if [ "$kind" = file ] && [ -f "$p" ]; then
      size=$(_hl_jnum "$(_hl_size "$p")")
      if [ "$size" = null ]; then
        skipped='"unreadable"'
      elif [ "$size" -gt "$HL_MAX_HASH_BYTES" ]; then
        skipped='"too_large"'
      else
        h=$(_hl_sha256 "$p")
        h=${h#\\}
        if [ "${#h}" -eq 64 ]; then sha="\"sha256:$h\""; else skipped='"no_hash_tool"'; fi
      fi
    elif [ "$kind" = dir ]; then
      skipped='"no_python"'
    fi
  elif [ "$req" = true ]; then
    HL_OUT_MISSING="${HL_OUT_MISSING:+$HL_OUT_MISSING,}$(_hl_jstr "$name")"
  fi
  [ "$HL_OUT_FIRST" = 1 ] || printf ','
  HL_OUT_FIRST=0
  printf '{"name":%s,"path":%s,"kind":"%s","required":%s,"exists":%s,"size":%s,"mtime":%s,"sha256":%s,"members":null,"skipped":%s}' \
    "$(_hl_jstr "$name")" "$(_hl_jstr "$p")" "$kind" "$req" "$exists" "$size" "$mtime" "$sha" "$skipped"
}
_hl_write_outputs_fallback() {
  local tmp="$HL_OUTPUTS_JSON.tmp.$$"
  HL_OUT_FIRST=1
  HL_OUT_MISSING=
  {
    printf '{"schema":"hpclab.outputs/v1","attempt_id":%s,"generated_at":%s,"hash_state":"complete","degraded":true,"files":[' \
      "$(_hl_jstr "$HL_ATTEMPT_ID")" "$(_hl_now)"
    _hl_outputs_fallback_entries
    printf '],"missing_required":[%s]}\n' "$HL_OUT_MISSING"
  } > "$tmp" 2>/dev/null && mv -f "$tmp" "$HL_OUTPUTS_JSON" 2>/dev/null
}

# ---- env.json (degraded path; the helper writes the full version) ----
_hl_write_env_fallback() {
  local tmp="$HL_ENV_JSON.tmp.$$" enabled=false
  [ "$HL_GUARD" = 1 ] && enabled=true
  {
    printf '{"schema":"hpclab.env/v1","degraded":true,"host":%s,"slurm":{"job_id":%s,"job_name":%s,' \
      "$(_hl_jstr "$HL_HOST")" "$(_hl_jstr "${SLURM_JOB_ID:-}")" "$(_hl_jstr "${SLURM_JOB_NAME:-}")"
    printf '"partition":%s,"nodelist":%s,"cpus":%s,"gpus":%s,"cuda_visible_devices":%s,"restart_count":%s},' \
      "$(_hl_jstr "${SLURM_JOB_PARTITION:-}")" "$(_hl_jstr "${SLURM_JOB_NODELIST:-}")" \
      "$(_hl_jnum "${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}")" "$(_hl_jnum "${SLURM_GPUS_ON_NODE:-}")" \
      "$(_hl_jstr "${CUDA_VISIBLE_DEVICES:-}")" "$(_hl_jnum "$HL_RESTART_COUNT")"
    printf '"python":null,"packages_digest":null,"packages_tool":null,"torch":null,"nvidia_smi":null,'
    printf '"lockfiles":{},"inputs":[],"guard":{"enabled":%s,"status":%s,"failed":%s},' \
      "$enabled" "$(_hl_jstr "$1")" "$(_hl_jstr "$2")"
    printf '"warnings":[],"errors":["python3 >= 3.6 is not available on the compute node: environment probes were skipped"],'
    printf '"captured_at":%s}\n' "$(_hl_now)"
  } > "$tmp" 2>/dev/null && mv -f "$tmp" "$HL_ENV_JSON" 2>/dev/null
}

# ---- telemetry sidecar (heartbeat + nvidia-smi samples; advisory, never blocks the job) ----
_hl_sidecar_loop() {
  local interval=$1 n=0 size stride smi=0 s=
  trap 'if [ -n "$s" ]; then kill "$s" 2>/dev/null; fi; exit 0' TERM
  trap '' INT USR1 USR2 HUP
  # A CPU-only attempt owns no GPU: nvidia-smi would report other jobs' devices as its own.
  if [ "$HL_GPUS_PER_NODE" != 0 ] && command -v nvidia-smi >/dev/null 2>&1; then smi=1; fi
  while :; do
    _hl_now >> "$HL_HEARTBEAT" 2>/dev/null
    if [ "$smi" = 1 ]; then
      size=$(_hl_size "$HL_GPU_CSV")
      size=${size:-0}
      stride=1
      if [ "$size" -ge $((2 * HL_TELEMETRY_CAP)) ]; then
        stride=0
      elif [ "$size" -ge "$HL_TELEMETRY_CAP" ]; then
        stride=10
      fi
      if [ "$stride" -gt 0 ] && [ $((n % stride)) -eq 0 ]; then
        _hl_timeout 15 nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
          --format=csv,noheader,nounits >> "$HL_GPU_CSV" 2>/dev/null
      fi
    fi
    n=$((n + 1))
    sleep "$interval" &
    s=$!
    wait "$s"
    s=
  done
}
_hl_start_sidecar() {
  local interval=${HPCLAB_TELEMETRY_INTERVAL_S:-60}
  case "$interval" in ''|*[!0-9]*|0) interval=60 ;; esac
  ( _hl_sidecar_loop "$interval" ) </dev/null >/dev/null 2>&1 &
  HL_SIDECAR_PID=$!
}
_hl_stop_sidecar() {
  if [ -n "$HL_SIDECAR_PID" ]; then
    kill "$HL_SIDECAR_PID" 2>/dev/null
    HL_SIDECAR_PID=
  fi
}

# ---- lifecycle ----
_hl_finish() {
  local state=$1 rc=$2 sig=${3:-} phase=${4:-} err=${5:-} guard=${6:-} ran=${7:-} num
  HL_DONE=1
  # A stop signal that arrived before the program ran explains any failure it caused
  # (e.g. Slurm's TERM killing cp or a probe): record "killed", never a bogus guard/prepare error.
  if [ -z "$ran" ] && [ -n "$HL_TERM_SIG" ] && [ "$state" = exited ]; then
    num=$(kill -l "$HL_TERM_SIG" 2>/dev/null) || num=15
    state=killed
    sig=$HL_TERM_SIG
    rc=$((128 + num))
    err="received SIG$HL_TERM_SIG during $phase${err:+ ($err)}"
    guard=
  fi
  HL_PHASE=finalize
  _hl_write_status "$state" "$rc" "$sig" "$phase" "$err" "$guard"
  _hl_log "finished: state=$state exit_code=$rc${sig:+ signal=$sig}${phase:+ failed_phase=$phase}${err:+ ($err)}"
  _hl_write_outputs quick
  if [ -n "$ran" ]; then _hl_write_outputs full; fi
  _hl_stop_sidecar
  exit "$rc"
}
_hl_bail_if_signalled() {
  [ -n "$HL_TERM_SIG" ] || return 0
  _hl_finish exited 1 "" "$HL_PHASE" "the program was not started"
}
_hl_guard_fail() {
  printf 'HPCLAB_GUARD_FAILED:%s\n' "$1" >&2
  _hl_log "guard check '$1' failed: $2"
  _hl_finish exited "$HL_EXIT_GUARD" "" guard "$2" "$1"
}
_hl_banner() {
  local line
  line="===== hpclab attempt $HL_ATTEMPT_ID | run $HL_RUN_ID | job ${SLURM_JOB_ID:-?} | restart $HL_RESTART_COUNT | host $HL_HOST | $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) ====="
  printf '%s\n' "$line"
  printf '%s\n' "$line" >&2
}

# An attempt dir belongs to the first Slurm job that runs it (requeues keep the job id). A job
# with another id is a duplicate submission (possible after an uncertain sbatch; hpclab never
# cancels it automatically): it leaves BEFORE writing anything, so the two jobs cannot clobber
# one attempt's work dir, outputs, checkpoints and status.json. The claim is created
# atomically (hardlink of a complete temp file); any failure to record or read it fails open.
_hl_claim_attempt() {
  local claim="$HL_INTERNAL_DIR/owner_job_id" tmp owner
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  tmp="$claim.tmp.$$"
  printf '%s\n' "$SLURM_JOB_ID" > "$tmp" 2>/dev/null || { rm -f "$tmp"; return 0; }
  if ln "$tmp" "$claim" 2>/dev/null; then rm -f "$tmp"; return 0; fi
  rm -f "$tmp"
  owner=$(sed -n 1p "$claim" 2>/dev/null)
  if [ -z "$owner" ] || [ "$owner" = "$SLURM_JOB_ID" ]; then return 0; fi
  HL_DONE=1  # the EXIT trap must not write this attempt's status.json
  printf '%s%s\n' "$HL_DUPLICATE_MARKER" "$owner" >&2
  _hl_log "attempt $HL_ATTEMPT_ID already belongs to Slurm job $owner: this job ($SLURM_JOB_ID) is a duplicate submission and exits without touching the attempt"
  exit "$HL_EXIT_DUPLICATE"
}

# Remove a (partial) work tree. Only directories are chmod-ed: files may be hardlinks into the
# read-only snapshot and must never change mode.
_hl_rm_work_tree() {
  [ -e "$1" ] || return 0
  find "$1" -type d -exec chmod u+w {} + 2>/dev/null
  rm -rf "$1"
}
_hl_prepare_workdir() {
  local tmp="$HL_WORK_DIR.tmp.$$" d
  if [ -d "$HL_WORK_DIR" ]; then
    _hl_log "reusing work dir $HL_WORK_DIR (restart $HL_RESTART_COUNT)"
    return 0
  fi
  for d in "$HL_WORK_DIR".tmp.*; do
    if [ -d "$d" ]; then _hl_rm_work_tree "$d"; fi
  done
  if cp -al "$HL_SNAPSHOT" "$tmp" 2>/dev/null; then
    :
  else
    _hl_rm_work_tree "$tmp"
    if [ -n "$HL_PY" ] && _hl_helper linkfarm "$HL_SNAPSHOT" "$tmp"; then
      :
    else
      _hl_rm_work_tree "$tmp"
      cp -R "$HL_SNAPSHOT" "$tmp" || { _hl_rm_work_tree "$tmp"; return 1; }
    fi
  fi
  find "$tmp" -type d -exec chmod u+w {} + 2>/dev/null
  rm -f "$tmp/.hpclab-complete" "$tmp/.hpclab-manifest.json"
  mv "$tmp" "$HL_WORK_DIR" || { _hl_rm_work_tree "$tmp"; return 1; }
  return 0
}
# A requeue of a program that is not resilience.resumable reruns it from scratch: the previous
# segment's outputs/, checkpoints/ and work dir (declared outputs may be relative to it) move to
# requeue-<restart>/ so the rerun cannot mix with them; the work dir is rebuilt from the snapshot.
_hl_set_aside_segment() {
  local dest="$HL_ATTEMPT_DIR/requeue-$HL_RESTART_COUNT" d
  [ "$HL_RESTART_COUNT" != 0 ] && [ "$HL_RESUMABLE" != 1 ] || return 0
  [ ! -e "$dest" ] || dest="$dest.$$"
  for d in "$HL_OUTPUT_DIR" "$HL_CHECKPOINT_DIR" "$HL_WORK_DIR"; do
    [ -n "$(ls -A "$d" 2>/dev/null)" ] || continue
    mkdir -p "$dest" && mv "$d" "$dest/" && { [ "$d" = "$HL_WORK_DIR" ] || mkdir -p "$d"; } \
      || _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "cannot set the previous segment's ${d##*/} aside in $dest"
    HL_REQUEUE_SET_ASIDE=$dest
  done
  [ -z "$HL_REQUEUE_SET_ASIDE" ] || _hl_log "not resumable: the previous segment's files were set aside in $dest; starting from scratch"
}
_hl_resolve_resume() {
  local found
  HPCLAB_RESUME_FROM=$HL_PLAN_RESUME_FROM
  HL_RESUME_SOURCE=
  [ -z "$HPCLAB_RESUME_FROM" ] || HL_RESUME_SOURCE=plan
  [ "$HL_RESUMABLE" = 1 ] || return 0  # never hand a non-resumable program its own checkpoints
  [ -n "$HL_PY" ] || return 0
  found=$(_hl_timeout 60 "$HL_PY" -I "$HL_HELPER" resume "$HL_CHECKPOINT_DIR" 2>/dev/null) || return 0
  if [ -n "$found" ]; then
    HPCLAB_RESUME_FROM=$found
    HL_RESUME_SOURCE=attempt_checkpoint
    _hl_log "resuming from this attempt's latest complete checkpoint: $found"
  fi
}
_hl_export_core_env() {
  export HPCLAB_RUN_ID="$HL_RUN_ID" HPCLAB_ATTEMPT_ID="$HL_ATTEMPT_ID" HPCLAB_TOKEN="$HL_TOKEN"
  export HPCLAB_OUTPUT_DIR="$HL_OUTPUT_DIR" HPCLAB_CHECKPOINT_DIR="$HL_CHECKPOINT_DIR"
  export HPCLAB_RESUME_FROM HPCLAB_SEED="$HL_SEED" HPCLAB_CODE_DIR="$HL_WORK_DIR"
  export HPCLAB_ATTEMPT_DIR="$HL_ATTEMPT_DIR" HPCLAB_METRICS_PATH="$HL_OUTPUT_DIR/metrics.jsonl"
  export HPCLAB_PARAMS_JSON="$HL_PARAMS_JSON" HPCLAB_RUN_SPEC_HASH="$HL_RUN_SPEC_HASH"
  export HPCLAB_RESTART_COUNT="$HL_RESTART_COUNT" HPCLAB_STOP_FILE="$HL_STOP_FILE"
  export HPCLAB_REQUEUE_SET_ASIDE="$HL_REQUEUE_SET_ASIDE"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  if [ -z "${MASTER_PORT:-}" ]; then
    case "${SLURM_JOB_ID:-}" in
      ''|*[!0-9]*) ;;
      *) export MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000)) ;;
    esac
  fi
}
_hl_activate_venv() {
  local rc
  [ -n "$HL_VENV" ] || return 0
  if [ ! -f "$HL_VENV/bin/activate" ]; then
    _hl_guard_fail venv "virtualenv $HL_VENV not found on $HL_HOST (no bin/activate)"
  fi
  set +u
  . "$HL_VENV/bin/activate"
  rc=$?
  set -u
  if [ "$rc" -ne 0 ]; then _hl_guard_fail venv "activating $HL_VENV failed (exit $rc)"; fi
  _hl_log "activated virtualenv $HL_VENV"
}
_hl_setup_line() {
  local _hl_idx=$1 _hl_line=$2 _hl_rc
  _hl_log "setup[$_hl_idx]: $_hl_line"
  set --
  set +u
  eval "$_hl_line"
  _hl_rc=$?
  set -u
  if [ "$_hl_rc" -ne 0 ]; then
    HL_SETUP_FAILED_INDEX=$_hl_idx
    HL_SETUP_FAILED_RC=$_hl_rc
    _hl_log "setup[$_hl_idx] failed with exit code $_hl_rc"
    return 1
  fi
  return 0
}
_hl_capture_env() {
  local out rc check msg
  if [ -z "$HL_PY" ] && [ "${HPCLAB_HELPER_PYTHON:-}" != none ]; then
    _hl_setup_helper || true
  fi
  if [ -n "$HL_PY" ]; then
    out=$(_hl_helper env "$HL_ENV_JSON" "$HL_CONFIG")
    rc=$?
    if [ "$rc" -eq 3 ] && [ -n "$out" ]; then
      check=$(printf '%s\n' "$out" | sed -n 1p)
      msg=$(printf '%s\n' "$out" | sed -n 2p)
      _hl_guard_fail "$check" "$msg"
    elif [ "$rc" -ne 0 ]; then
      _hl_log "environment capture failed (exit $rc); continuing without env.json"
    fi
    return 0
  fi
  if [ "$HL_GUARD" = 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! _hl_timeout 60 nvidia-smi -L >/dev/null 2>&1; then
      _hl_write_env_fallback fail nvidia_smi
      _hl_guard_fail nvidia_smi "nvidia-smi is missing or failing on $HL_HOST although GPUs were requested"
    fi
    _hl_write_env_fallback unchecked ""
  else
    _hl_write_env_fallback skipped ""
  fi
}
_hl_run_program() {
  local rc rc2
  _hl_build_argv
  _hl_log "exec in $PWD:$(printf ' %q' "${HL_ARGV[@]}")"
  # Last check right before the fork: a TERM that lands after it can only be forwarded, and a
  # signal forwarded before the child has reset bash's inherited trap is silently dropped.
  _hl_bail_if_signalled
  "${HL_ARGV[@]}" &
  HL_CHILD=$!
  while :; do
    HL_WAIT_INTERRUPTED=0
    wait "$HL_CHILD"
    rc=$?
    if [ "$HL_WAIT_INTERRUPTED" = 1 ]; then
      if kill -0 "$HL_CHILD" 2>/dev/null; then continue; fi
      wait "$HL_CHILD" 2>/dev/null
      rc2=$?
      [ "$rc2" -eq 127 ] || rc=$rc2
    fi
    break
  done
  HL_CHILD=
  HL_RC=$rc
}
_hl_after_program() {
  local state=exited sig= name
  if [ -n "$HL_TERM_SIG" ]; then
    state=killed
    sig=$HL_TERM_SIG
  elif [ "$HL_RC" -gt 128 ] && [ "$HL_RC" -lt 193 ]; then
    name=$(kill -l "$((HL_RC - 128))" 2>/dev/null)
    if [ -n "$name" ]; then
      state=killed
      sig=${name#SIG}
    fi
  fi
  _hl_finish "$state" "$HL_RC" "$sig" "" "" "" 1
}

_hl_main() {
  HL_START_TIME=$(_hl_now)
  HL_HOST=$(hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown)
  HL_TIMEOUT_BIN=$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)
  _hl_install_traps
  _hl_export_knobs
  HL_PHASE=prepare
  if ! mkdir -p "$HL_ATTEMPT_DIR" "$HL_LOGS_DIR" "$HL_OUTPUT_DIR" "$HL_CHECKPOINT_DIR" \
    "$HL_TELEMETRY_DIR" "$HL_INTERNAL_DIR"; then
    _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "cannot create the attempt directories under $HL_ATTEMPT_DIR"
  fi
  _hl_claim_attempt
  rm -f "$HL_STOP_FILE"
  # A requeued segment rewrites the outputs: the previous segment's manifest must not look current.
  if [ -f "$HL_OUTPUTS_JSON" ]; then
    mv -f "$HL_OUTPUTS_JSON" "$HL_INTERNAL_DIR/outputs.before-restart-$HL_RESTART_COUNT.json" 2>/dev/null \
      || rm -f "$HL_OUTPUTS_JSON"
  fi
  _hl_set_aside_segment
  _hl_setup_helper || _hl_log "python3 >= 3.6 not found: using the degraded bash-only paths"
  _hl_banner
  _hl_write_status started
  _hl_start_sidecar
  if [ ! -f "$HL_SNAPSHOT/.hpclab-complete" ]; then
    _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "code snapshot $HL_SNAPSHOT is missing or incomplete (no .hpclab-complete marker)"
  fi
  _hl_prepare_workdir || _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "could not build the work dir $HL_WORK_DIR"
  _hl_resolve_resume
  # "started" was written before the checkpoint scan; never leave it advertising a stale path
  if [ "$HL_RESUME_SOURCE" = attempt_checkpoint ]; then _hl_write_status started; fi
  _hl_export_core_env
  _hl_export_user_env
  _hl_bail_if_signalled
  HL_PHASE=setup
  cd "$HL_WORK_DIR" || _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "cannot enter $HL_WORK_DIR"
  _hl_activate_venv
  if ! _hl_run_setup; then
    set +e
    set -u
    _hl_finish exited "$HL_EXIT_SETUP" "" setup "setup line $HL_SETUP_FAILED_INDEX failed with exit code $HL_SETUP_FAILED_RC"
  fi
  set +e
  set -u
  _hl_install_traps
  _hl_export_core_env
  _hl_bail_if_signalled
  HL_PHASE=env
  cd "$HL_WORK_DIR/$HL_ENTRY_WORKDIR" \
    || _hl_finish exited "$HL_EXIT_PREPARE" "" prepare "entrypoint workdir '$HL_ENTRY_WORKDIR' does not exist in the code snapshot"
  _hl_capture_env
  _hl_bail_if_signalled
  HL_PHASE=run
  _hl_run_program
  _hl_after_program
}
"""


# ---------------------------------------------------------------------------
# Compute-node helper (python3 >= 3.6, stdlib only). Embedded verbatim in job.sbatch.
# ---------------------------------------------------------------------------

HELPER_SOURCE = r'''# hpclab job-wrapper helper (hpclab-wrapper/1). Generated by hpclab: do not edit.
# Standard library only; runs under any python3 >= 3.6 on the compute node. Every command is
# best effort: problems go to stderr and into the JSON it writes, and the bash wrapper never
# lets a helper failure change the exit code of the user's program.
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

STATUS_SCHEMA = "hpclab.status/v1"
ENV_SCHEMA = "hpclab.env/v1"
OUTPUTS_SCHEMA = "hpclab.outputs/v1"
CKPT_SCHEMA = "hpclab.checkpoint/v1"
CKPT_MANIFEST = "hpclab-ckpt.json"
MAX_HASH_BYTES = 50 * 1024 ** 3
MAX_MEMBERS = 10000
MAX_WALK = 200000
MAX_CKPT_CANDIDATES = 10000
MAX_LOCKFILE_BYTES = 64 * 1024 * 1024
MAX_INPUT_VERIFY_BYTES = 4 * 1024 ** 3
MAX_ENV_VALUE = 2048
MAX_DIR_ENTRIES = 100000
AUTO_LOCKFILES = ("uv.lock", "poetry.lock", "pixi.lock", "Pipfile.lock", "requirements.txt",
                  "environment.yml", "conda-lock.yml")
ENV_PREFIXES = ("CUDA_", "NCCL_", "OMP_", "MKL_", "TORCH_", "PYTORCH_", "HF_", "TRANSFORMERS_",
                "XLA_", "JAX_", "TF_", "WANDB_", "CUBLAS_", "CUDNN_", "NVIDIA_", "TRITON_", "UV_",
                "SLURM_", "MASTER_", "CONDA_")
ENV_NAMES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHASHSEED", "VIRTUAL_ENV", "WORLD_SIZE",
             "LOCAL_RANK", "RANK", "XDG_CACHE_HOME")
SECRETISH = re.compile(r"(TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|KEY|PRIVATE|COOKIE|SESSION)", re.I)
URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
# user[:password]@ inside URL-valued variables (UV_INDEX_URL=https://user:token@host/...)
URL_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")

PY_INFO = ("import json, platform, sys; print(json.dumps({'executable': sys.executable, "
           "'version': platform.python_version(), 'implementation': platform.python_implementation(), "
           "'prefix': sys.prefix}))")

PKG_LIST = r"""
import re, sys
try:
    from importlib import metadata
except ImportError:
    sys.exit(3)
seen = {}
for dist in metadata.distributions():
    try:
        name = dist.metadata["Name"]
    except Exception:
        name = None
    if not name:
        continue
    key = re.sub(r"[-_.]+", "-", name).lower()
    if key not in seen:
        seen[key] = dist.version
for key in sorted(seen):
    print("%s==%s" % (key, seen[key]))
"""

TORCH_PROBE = r"""
import json, sys
try:
    import torch
except ImportError:
    print(json.dumps({"installed": False}))
    sys.exit(0)
info = {"installed": True, "version": str(getattr(torch, "__version__", "")) or None}
def attempt(key, fn):
    try:
        info[key] = fn()
    except Exception as exc:
        info[key] = None
        info.setdefault("probe_errors", []).append("%s: %s" % (key, str(exc)[:300]))
attempt("cuda", lambda: torch.version.cuda)
attempt("cudnn", lambda: torch.backends.cudnn.version())
attempt("cuda_available", lambda: bool(torch.cuda.is_available()))
attempt("arch_list", lambda: list(torch.cuda.get_arch_list()))
names, caps = [], []
if info.get("cuda_available"):
    attempt("device_count", lambda: int(torch.cuda.device_count()))
    for i in range(min(info.get("device_count") or 0, 16)):
        try:
            names.append(str(torch.cuda.get_device_name(i)))
            caps.append([int(x) for x in torch.cuda.get_device_capability(i)])
        except Exception as exc:
            info.setdefault("probe_errors", []).append("device %d: %s" % (i, str(exc)[:300]))
info["device_names"] = names
info["device_capabilities"] = caps
print(json.dumps(info))
"""


def warn(msg):
    sys.stderr.write("[hpclab] WARNING: %s\n" % msg)
    sys.stderr.flush()


def load_json(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def write_json_atomic(path, doc):
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, ".%s.%d.tmp" % (os.path.basename(path), os.getpid()))
    data = json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=True) + "\n"
    with open(tmp, "w") as fh:
        fh.write(data)
    os.replace(tmp, path)


def env_float(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return float(default)


def as_int(value):
    if value is None:
        return None
    text = str(value).strip()
    return int(text) if re.match(r"^-?\d+$", text) else (text or None)


def sha256_file(path, deadline=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if deadline is not None and time.time() > deadline:
                return None
    return "sha256:" + h.hexdigest()


def call_with_timeout(fn, timeout):
    """Run fn in a daemon thread: a hung network filesystem must not hang the job."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa
            box["error"] = exc

    worker = threading.Thread(target=target)
    worker.daemon = True
    worker.start()
    worker.join(max(0.0, timeout))
    if worker.is_alive():
        raise TimeoutError("timed out after %.0fs (hung filesystem?)" % timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------------------------------------------------------------- status
STATUS_INT = ("exit_code", "restart_count", "pid", "start_time", "end_time")


def cmd_status(args):
    path = args[0]
    doc = {"schema": STATUS_SCHEMA, "signals_received": [], "stop_requested": False}
    for item in args[1:]:
        key, _, value = item.partition("=")
        if key in STATUS_INT:
            doc[key] = int(value) if re.match(r"^-?\d+$", value) else None
        elif key in ("stop_requested", "resumable"):
            doc[key] = value in ("1", "true")
        elif key == "signals_received":
            doc[key] = [v for v in value.split(",") if v]
        else:
            doc[key] = value if value != "" else None
    doc["written_at"] = time.time()
    write_json_atomic(path, doc)
    return 0


# ---------------------------------------------------------------- probes
class ProbeResult(object):
    def __init__(self, rc=None, out="", err="", error=None, missing=False, timed_out=False):
        self.rc = rc
        self.out = out
        self.err = err
        self.error = error
        self.missing = missing
        self.timed_out = timed_out

    @property
    def ok(self):
        return self.rc == 0


class Prober(object):
    def __init__(self, timeout, budget):
        self.timeout = timeout
        self.deadline = time.time() + budget
        self.errors = []

    def remaining(self):
        return max(0.0, self.deadline - time.time())

    def run(self, argv, what, env=None):
        limit = min(self.timeout, self.remaining())
        if limit <= 0:
            self.errors.append("%s: skipped (probe time budget exhausted)" % what)
            return ProbeResult(error="budget exhausted", timed_out=True)
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True, env=env)
        except OSError as exc:
            return ProbeResult(error=str(exc), missing=exc.errno in (errno.ENOENT, errno.EACCES))
        try:
            out, err = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa
                pass
            self.errors.append("%s: timed out after %.0fs" % (what, limit))
            return ProbeResult(error="timed out", timed_out=True)
        return ProbeResult(rc=proc.returncode, out=out.decode("utf-8", "replace"),
                           err=err.decode("utf-8", "replace"))


def last_json_line(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def resolve_python(name):
    if not name:
        return None
    if os.sep in name:
        return name if os.access(name, os.X_OK) else None
    found = shutil.which(name)
    if found is None and name == "python":
        found = shutil.which("python3")
    return found


def probe_python(prober, py):
    """(info, responsive): a runtime python that hangs here would hang every later probe."""
    if py is None:
        return None, False
    res = prober.run([py, "-c", PY_INFO], "python")
    if res.timed_out:
        prober.errors.append("runtime python %s is unresponsive: package and torch probes skipped" % py)
        return None, False
    if not res.ok:
        prober.errors.append("python probe failed: %s" % (res.error or res.err.strip()[-300:]))
        return None, True
    return last_json_line(res.out), True


def normalize_packages(text):
    lines = sorted(set(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")))
    return "\n".join(lines) + ("\n" if lines else "")


def probe_packages(prober, py):
    """(normalized listing, tool) - importlib.metadata first: pip-free, tool-independent."""
    if py is None:
        return None, None
    res = prober.run([py, "-c", PKG_LIST], "packages (importlib.metadata)")
    if res.ok:
        return normalize_packages(res.out), "importlib.metadata"
    if res.timed_out:
        return None, None
    if shutil.which("uv"):
        res = prober.run(["uv", "pip", "freeze", "--python", py], "packages (uv pip freeze)")
        if res.ok:
            return normalize_packages(res.out), "uv pip freeze"
    res = prober.run([py, "-m", "pip", "freeze", "--all"], "packages (pip freeze)")
    if res.ok:
        return normalize_packages(res.out), "pip freeze"
    prober.errors.append("could not list installed packages: %s" % (res.error or res.err.strip()[-300:]))
    return None, None


def probe_torch(prober, py):
    """(info, state) with state in ok|absent|timeout|failed."""
    if py is None:
        return None, "failed"
    res = prober.run([py, "-c", TORCH_PROBE], "torch")
    if res.timed_out:
        return None, "timeout"
    info = last_json_line(res.out) if res.ok else None
    if info is None:
        prober.errors.append("torch probe failed: %s" % (res.error or res.err.strip()[-500:]))
        return None, "failed"
    if not info.get("installed"):
        return None, "absent"
    if not info.get("cuda_available") and res.err.strip():
        info["cuda_error"] = res.err.strip()[-500:]
    return info, "ok"


def probe_nvidia(prober):
    """(info, state) with state in ok|missing|timeout|failed."""
    query = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,uuid",
             "--format=csv,noheader,nounits"]
    res = prober.run(query, "nvidia-smi")
    if res.missing:
        return None, "missing"
    if res.timed_out:
        return None, "timeout"
    if not res.ok:
        prober.errors.append("nvidia-smi exited %s: %s" % (res.rc, res.err.strip()[-300:]))
        return None, "failed"
    gpus, driver = [], None
    for line in res.out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        try:
            mem = float(cols[2])
        except ValueError:
            mem = None
        gpus.append({"index": as_int(cols[0]), "name": cols[1], "memory_total_mib": mem, "uuid": cols[4]})
        driver = driver or cols[3]
    cuda_version = None
    head = prober.run(["nvidia-smi"], "nvidia-smi (header)")
    if head.ok:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", head.out)
        cuda_version = m.group(1) if m else None
    return {"driver_version": driver, "cuda_version": cuda_version, "gpus": gpus}, "ok"


def version_tuple(text):
    try:
        return tuple(int(p) for p in str(text).split(".")[:2])
    except ValueError:
        return None


def arch_compatible(cap, arch_list):
    """True if torch ships kernels usable on compute capability ``cap`` (major, minor).

    sm_XY cubins run on the same major with minor >= Y; 'a'/'f' variants only on the exact
    arch; compute_XY PTX can be JIT-compiled for any capability >= XY.
    """
    major, minor = int(cap[0]), int(cap[1])
    for arch in arch_list or []:
        m = re.match(r"^(sm|compute)_(\d+)(\d)([a-z]?)$", str(arch))
        if not m:
            continue
        kind, amaj, amin, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if kind == "sm":
            if suffix:
                if (amaj, amin) == (major, minor):
                    return True
            elif amaj == major and amin <= minor:
                return True
        elif not suffix and (amaj, amin) <= (major, minor):
            return True
    return False


def cuda_unavailable_message(torch_info, nvidia):
    version, built = torch_info.get("version"), torch_info.get("cuda")
    if not built:
        return "torch %s is a CPU-only build; install a CUDA build of torch" % version
    msg = "torch %s (built for CUDA %s) cannot use the allocated GPU(s)" % (version, built)
    driver_cuda = (nvidia or {}).get("cuda_version")
    if driver_cuda:
        msg += "; the driver supports CUDA %s" % driver_cuda
        dv, bv = version_tuple(driver_cuda), version_tuple(built)
        if dv and bv and bv > dv:
            msg += ("; install a torch wheel built for CUDA <= %s, e.g. pip install torch "
                    "--index-url https://download.pytorch.org/whl/cu%d%d" % (driver_cuda, dv[0], dv[1]))
    if torch_info.get("cuda_error"):
        msg += " [%s]" % torch_info["cuda_error"].splitlines()[-1][:300]
    return msg


def evaluate_guard(cfg, nvidia, nvidia_state, torch_info, torch_state):
    gpus = int(cfg.get("gpus_per_node") or 0)
    enabled = bool(cfg.get("guard")) and gpus > 0
    guard = {"enabled": enabled, "status": "skipped", "checks": [], "failed": None, "message": None}
    if gpus <= 0:
        return guard
    checks = guard["checks"]

    def add(name, status, message):
        checks.append({"check": name, "status": status, "message": message})

    if nvidia_state == "ok" and nvidia and nvidia.get("gpus"):
        add("nvidia_smi", "pass", "%d GPU(s) visible, driver %s" % (len(nvidia["gpus"]), nvidia.get("driver_version")))
    elif nvidia_state == "timeout":
        add("nvidia_smi", "unchecked", "nvidia-smi timed out")
    else:
        add("nvidia_smi", "fail", "nvidia-smi is %s on this node although %d GPU(s) were requested"
            % ("missing" if nvidia_state == "missing" else "failing or shows no GPUs", gpus))
    if torch_state == "absent":
        add("torch_cuda_available", "unchecked", "torch is not installed in the runtime environment")
    elif torch_state != "ok":
        add("torch_cuda_available", "unchecked", "torch probe %s" % ("timed out" if torch_state == "timeout" else "failed"))
    elif not torch_info.get("cuda_available"):
        add("torch_cuda_available", "fail", cuda_unavailable_message(torch_info, nvidia))
    else:
        add("torch_cuda_available", "pass", "torch %s sees %s GPU(s)" % (torch_info.get("version"), torch_info.get("device_count")))
        caps = torch_info.get("device_capabilities") or []
        bad = [(n, c) for n, c in zip(torch_info.get("device_names") or [], caps) if not arch_compatible(c, torch_info.get("arch_list"))]
        if not caps:
            add("torch_gpu_arch", "unchecked", "no device capabilities reported")
        elif bad:
            name, cap = bad[0]
            add("torch_gpu_arch", "fail", "GPU %s (sm_%d%d) is not supported by torch %s (CUDA %s), built for %s; "
                "install a torch build that includes sm_%d%d (e.g. a cu128 wheel for Blackwell GPUs)"
                % (name, cap[0], cap[1], torch_info.get("version"), torch_info.get("cuda"),
                   " ".join(torch_info.get("arch_list") or []), cap[0], cap[1]))
        else:
            add("torch_gpu_arch", "pass", "torch kernels match the GPU architecture")
    failed = [c for c in checks if c["status"] == "fail"]
    if failed:
        guard["status"] = "fail"
        guard["failed"], guard["message"] = failed[0]["check"], failed[0]["message"]
    elif all(c["status"] == "pass" for c in checks):
        guard["status"] = "pass"
    else:
        guard["status"] = "unchecked"
    return guard


def slurm_info():
    e = os.environ.get
    return {
        "job_id": e("SLURM_JOB_ID"),
        "job_name": e("SLURM_JOB_NAME"),
        "partition": e("SLURM_JOB_PARTITION"),
        "nodelist": e("SLURM_JOB_NODELIST"),
        "num_nodes": as_int(e("SLURM_JOB_NUM_NODES")),
        "cpus": as_int(e("SLURM_CPUS_PER_TASK") or e("SLURM_CPUS_ON_NODE")),
        "gpus": as_int(e("SLURM_GPUS_ON_NODE") or e("SLURM_GPUS_PER_NODE") or e("SLURM_GPUS")),
        "cuda_visible_devices": e("CUDA_VISIBLE_DEVICES"),
        "account": e("SLURM_JOB_ACCOUNT"),
        "qos": e("SLURM_JOB_QOS"),
        "cluster_name": e("SLURM_CLUSTER_NAME"),
        "mem_per_node_mb": as_int(e("SLURM_MEM_PER_NODE")),
        "restart_count": as_int(e("SLURM_RESTART_COUNT") or "0"),
    }


def system_info():
    info = {"os": platform.platform(), "kernel": platform.release(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "cpu_affinity": None, "mem_total_bytes": None, "load_avg": None}
    try:
        info["cpu_affinity"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["mem_total_bytes"] = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:
        info["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        pass
    return info


def interesting_env():
    """Reproducibility-relevant variables; secret-named keys are dropped and URL credentials
    masked, because env.json is copied into the local ledger and the vault."""
    out = {}
    for key in sorted(os.environ):
        if not (key.startswith(ENV_PREFIXES) or key in ENV_NAMES) or SECRETISH.search(key):
            continue
        value = URL_USERINFO_RE.sub(r"\1***@", os.environ[key])
        out[key] = value if len(value) <= MAX_ENV_VALUE else value[:MAX_ENV_VALUE] + "...(truncated)"
    return out


def hash_lockfiles(cfg, errors):
    code_dir = cfg.get("code_dir") or "."
    declared = list(cfg.get("lockfiles") or [])
    names = declared + [n for n in AUTO_LOCKFILES if n not in declared and os.path.isfile(os.path.join(code_dir, n))]
    result = {}
    for rel in names:
        path = os.path.join(code_dir, rel)
        try:
            if os.path.getsize(path) > MAX_LOCKFILE_BYTES:
                result[rel] = None
                errors.append("lockfile %s: larger than %d bytes, not hashed" % (rel, MAX_LOCKFILE_BYTES))
                continue
            result[rel] = sha256_file(path)
        except OSError as exc:
            result[rel] = None
            errors.append("lockfile %s: %s" % (rel, exc.strerror or exc))
    return result


def fingerprint_inputs(cfg, timeout, deadline, errors, warnings):
    out = []
    code_dir = cfg.get("code_dir") or "."
    for spec in cfg.get("inputs") or []:
        uri = spec.get("uri") or ""
        item = {"name": spec.get("name"), "uri": uri, "kind": None, "exists": None, "size": None, "mtime": None}
        out.append(item)
        if URL_RE.match(uri) and not uri.startswith("file://"):
            item["kind"] = "url"
            continue
        path = os.path.expanduser(os.path.expandvars(uri[7:] if uri.startswith("file://") else uri))
        if not os.path.isabs(path):
            path = os.path.join(code_dir, path)
        item["path"] = path
        try:
            st = call_with_timeout(lambda: os.stat(path), min(timeout, max(1.0, deadline - time.time())))
        except FileNotFoundError:
            item["exists"] = False
            warnings.append("input %s: %s does not exist on this node" % (spec.get("name"), path))
            continue
        except Exception as exc:  # noqa
            errors.append("input %s: stat failed: %s" % (spec.get("name"), exc))
            continue
        item["exists"] = True
        item["mtime"] = st.st_mtime
        if stat.S_ISDIR(st.st_mode):
            item["kind"] = "dir"
            try:
                item["entries"] = call_with_timeout(lambda: count_entries(path), min(timeout, max(1.0, deadline - time.time())))
            except Exception as exc:  # noqa
                errors.append("input %s: listing failed: %s" % (spec.get("name"), exc))
            continue
        item["kind"] = "file"
        item["size"] = st.st_size
        declared = spec.get("digest")
        if declared and str(declared).startswith("sha256:") and st.st_size <= MAX_INPUT_VERIFY_BYTES:
            try:
                found = call_with_timeout(lambda: sha256_file(path, deadline), max(1.0, deadline - time.time()))
            except Exception as exc:  # noqa
                found = None
                errors.append("input %s: hashing failed: %s" % (spec.get("name"), exc))
            if found is not None:
                item["sha256"] = found
                item["digest_match"] = found == declared
                if found != declared:
                    warnings.append("input %s: sha256 %s differs from the declared digest %s" % (spec.get("name"), found, declared))
    return out


def count_entries(path):
    n = 0
    for _ in os.scandir(path):
        n += 1
        if n >= MAX_DIR_ENTRIES:
            break
    return n


def cmd_env(args):
    out_path, cfg = args[0], load_json(args[1])
    started = time.time()
    timeout = env_float("HPCLAB_PROBE_TIMEOUT_S", cfg.get("probe_timeout_s", 60))
    prober = Prober(timeout, budget=4 * timeout)
    warnings = []
    requested = cfg.get("python") or "python"
    py = resolve_python(requested)
    if py is None:
        prober.errors.append("runtime python %r not found on PATH" % requested)
    python, responsive = probe_python(prober, py)
    if python is not None:
        python["requested"] = requested
    if not responsive:
        py = None
    listing, tool = probe_packages(prober, py)
    packages_digest = None
    if listing is not None:
        packages_digest = "sha256:" + hashlib.sha256(listing.encode("utf-8")).hexdigest()
        target = cfg.get("packages_file")
        if target:
            try:
                tmp = "%s.%d.tmp" % (target, os.getpid())
                with open(tmp, "wb") as fh:
                    fh.write(listing.encode("utf-8"))
                os.replace(tmp, target)
            except OSError as exc:
                prober.errors.append("could not write %s: %s" % (target, exc))
    torch_info, torch_state = probe_torch(prober, py)
    nvidia, nvidia_state = probe_nvidia(prober)
    guard = evaluate_guard(cfg, nvidia, nvidia_state, torch_info, torch_state)
    for check in guard["checks"]:
        if check["status"] == "fail":
            warnings.append("%s: %s" % (check["check"], check["message"]))
    doc = {
        "schema": ENV_SCHEMA,
        "host": socket.gethostname(),
        "slurm": slurm_info(),
        "python": python,
        "packages_digest": packages_digest,
        "packages_tool": tool,
        "packages_count": len(listing.splitlines()) if listing is not None else None,
        "packages_file": cfg.get("packages_file") if listing is not None else None,
        "torch": torch_info,
        "nvidia_smi": nvidia,
        "lockfiles": hash_lockfiles(cfg, prober.errors),
        "inputs": fingerprint_inputs(cfg, timeout, prober.deadline + timeout, prober.errors, warnings),
        "system": system_info(),
        "env_vars": interesting_env(),
        "guard": guard,
        "hpclab": {
            "wrapper_version": cfg.get("wrapper_version"),
            "run_id": cfg.get("run_id"),
            "attempt_id": cfg.get("attempt_id"),
            "code_digest": cfg.get("code_digest"),
            "snapshot_path": cfg.get("snapshot_path"),
            "code_dir": cfg.get("code_dir"),
            "resume_from": os.environ.get("HPCLAB_RESUME_FROM") or None,
            "restart_count": as_int(os.environ.get("HPCLAB_RESTART_COUNT") or "0"),
            "resumable": cfg.get("resumable") is True,
            "requeue_set_aside": os.environ.get("HPCLAB_REQUEUE_SET_ASIDE") or None,
        },
        "warnings": warnings,
        "errors": prober.errors,
        "captured_at": started,
        "capture_duration_s": round(time.time() - started, 3),
    }
    write_json_atomic(out_path, doc)
    for message in warnings:
        warn(message)
    if guard["enabled"] and guard["failed"]:
        sys.stdout.write("%s\n%s\n" % (guard["failed"], (guard["message"] or "").replace("\n", " ")))
        return 3
    return 0


# ---------------------------------------------------------------- outputs
def hash_bounded(path, size, deadline):
    if size > MAX_HASH_BYTES:
        return None, "too_large"
    if time.time() > deadline:
        return None, "time_budget"
    try:
        found = sha256_file(path, deadline)
    except OSError:
        return None, "unreadable"
    return (found, None) if found is not None else (None, "time_budget")


def output_path(template, cfg):
    path = template.replace("{job_id}", os.environ.get("SLURM_JOB_ID", ""))
    path = path.replace("{resume_from}", os.environ.get("HPCLAB_RESUME_FROM", ""))
    if not os.path.isabs(path):
        path = os.path.join(cfg.get("code_dir") or ".", path)
    return os.path.normpath(path)


def walk_members(root):
    """Sorted (relpath, stat) of regular files below root; symlinked dirs are not followed."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            count += 1
            if count > MAX_WALK:
                return
            yield os.path.relpath(full, root).replace(os.sep, "/"), full, st


def describe_output(decl, cfg, full, deadline, state):
    path = output_path(decl["path"], cfg)
    entry = {"name": decl["name"], "path": path, "kind": decl.get("kind", "file"), "role": decl.get("role"),
             "required": bool(decl.get("required", True)), "exists": False, "size": None, "mtime": None,
             "sha256": None, "members": None}
    try:
        st = os.stat(path)
    except OSError:
        return entry
    entry["exists"] = True
    entry["mtime"] = st.st_mtime
    if entry["kind"] == "file":
        if not stat.S_ISREG(st.st_mode):
            entry["error"] = "declared as a file but is not a regular file"
            return entry
        entry["size"] = st.st_size
        if full:
            entry["sha256"], skipped = hash_bounded(path, st.st_size, deadline)
            if skipped:
                entry["skipped"] = skipped
                state["partial"] = True
        return entry
    if not stat.S_ISDIR(st.st_mode):
        entry["error"] = "declared as a directory but is not one"
        return entry
    members, total, count = [], 0, 0
    for rel, member_path, mst in walk_members(path):
        count += 1
        total += mst.st_size
        if len(members) >= MAX_MEMBERS:
            continue
        member = {"relpath": rel, "size": mst.st_size, "sha256": None}
        if full:
            member["sha256"], skipped = hash_bounded(member_path, mst.st_size, deadline)
            if skipped:
                member["skipped"] = skipped
                state["partial"] = True
        members.append(member)
    entry.update(size=total, members=members, member_count=count, truncated=count > len(members))
    if full and entry["truncated"]:
        state["partial"] = True
    if full and not entry["truncated"] and all(m["sha256"] for m in members):
        tree = json.dumps([[m["relpath"], m["sha256"]] for m in members], separators=(",", ":"))
        entry["sha256"] = "sha256:" + hashlib.sha256(tree.encode("utf-8")).hexdigest()
    return entry


def cmd_outputs(args):
    out_path, cfg, mode = args[0], load_json(args[1]), args[2]
    full = mode == "full"
    budget = env_float("HPCLAB_OUTPUT_HASH_BUDGET_S", cfg.get("output_hash_budget_s", 1800))
    deadline = time.time() + budget
    state = {"partial": False}
    files = [describe_output(decl, cfg, full, deadline, state) for decl in cfg.get("outputs") or []]
    doc = {
        "schema": OUTPUTS_SCHEMA,
        "attempt_id": cfg.get("attempt_id"),
        "generated_at": time.time(),
        "hash_state": ("partial" if state["partial"] else "complete") if full else "pending",
        "files": files,
        "missing_required": [f["name"] for f in files if f["required"] and not f["exists"]],
    }
    write_json_atomic(out_path, doc)
    return 0


# ---------------------------------------------------------------- resume
def read_checkpoint(directory):
    """The manifest of a complete, resumable hpclab.checkpoint/v1 dir (sizes checked) or None."""
    path = os.path.join(directory, CKPT_MANIFEST)
    try:
        if os.path.getsize(path) > 4 * 1024 * 1024:
            return None
        manifest = load_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != CKPT_SCHEMA:
        return None
    if manifest.get("complete") is not True or manifest.get("resumable") is False:
        return None
    for item in manifest.get("files") or []:
        rel = str(item.get("path") or "")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return None
        try:
            size = os.path.getsize(os.path.join(directory, rel))
        except OSError:
            return None
        if item.get("size") is not None and size != item.get("size"):
            return None
    return manifest


def cmd_resume(args):
    root = args[0]
    try:
        names = sorted(os.listdir(root))[:MAX_CKPT_CANDIDATES]
    except OSError:
        return 0
    best = None
    for candidate in [root] + [os.path.join(root, n) for n in names]:
        if not os.path.isdir(candidate):
            continue
        manifest = read_checkpoint(candidate)
        if manifest is None:
            continue
        step = manifest.get("step") if isinstance(manifest.get("step"), int) else -1
        created = manifest.get("created_at") if isinstance(manifest.get("created_at"), (int, float)) else 0
        key = (step, created, os.path.getmtime(os.path.join(candidate, CKPT_MANIFEST)))
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is not None:
        sys.stdout.write(best[1] + "\n")
    return 0


# ---------------------------------------------------------------- work dir
def cmd_linkfarm(args):
    """Hardlink farm of src at dst (per-file copy fallback) when `cp -al` is unavailable."""
    src, dst = args[0], args[1]
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        os.chmod(target, 0o755)
        for name in filenames:
            if rel == "." and name in (".hpclab-complete", ".hpclab-manifest.json"):
                continue
            source = os.path.join(dirpath, name)
            try:
                os.link(source, os.path.join(target, name))
            except OSError:
                shutil.copy2(source, os.path.join(target, name))
    return 0


# ---------------------------------------------------------------- signal recipients
PLATFORM = sys.platform


def process_children(proc="/proc"):
    """{ppid: [pid, ...]} of every process (Linux /proc, else ps); {} if unknown."""
    pairs = []
    if PLATFORM.startswith("linux") and os.path.isdir(proc):
        for name in os.listdir(proc):
            try:  # comm (2nd field) may contain spaces and ")"
                with open(os.path.join(proc, name, "stat"), "rb") as fh:
                    pairs.append((int(name), int(fh.read().rsplit(b")", 1)[1].split()[1])))
            except (OSError, ValueError, IndexError):
                pass  # not a pid, or gone
    if not pairs:
        try:
            out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, timeout=10).stdout.split()
            pairs = [(int(a), int(b)) for a, b in zip(out[::2], out[1::2])]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    children = {}
    for pid, ppid in sorted(pairs):
        if pid != ppid:
            children.setdefault(ppid, []).append(pid)
    return children


def caught_mask(pid, proc="/proc"):
    """Mask (bit n-1 = signal n) of the signals pid has a handler for; None if unknown."""
    try:
        if PLATFORM.startswith("linux"):
            with open(os.path.join(proc, str(pid), "status")) as fh:
                for line in fh:
                    if line.startswith("SigCgt:"):
                        return int(line.split()[1], 16)
        elif PLATFORM == "darwin":  # LP64 struct kinfo_proc (648 bytes): p_pid @40, p_sigcatch @236
            import ctypes
            buf, size = ctypes.create_string_buffer(648), ctypes.c_size_t(648)
            mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
            ok = ctypes.sizeof(ctypes.c_void_p) == 8 and not ctypes.CDLL(None).sysctl(
                mib, 4, buf, ctypes.byref(size), None, ctypes.c_size_t(0))
            raw = buf.raw
            if ok and size.value == 648 and int.from_bytes(raw[40:44], sys.byteorder) == pid:
                return int.from_bytes(raw[236:240], sys.byteorder)
    except Exception:  # noqa - unreadable/unknown layout: the caller looks through it
        pass
    return None


def signal_targets(root, signum, children, caught):
    """The topmost processes of root's tree with a handler for signum ([] if none).

    One that handles it owns its subtree (data-loader workers forked after the handler was
    installed inherit it); one that does not (a launcher, a shell running a script) is looked
    through.
    """
    bit = 1 << (signum - 1)
    targets, stack, seen = [], [root], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if (caught(pid) or 0) & bit:
            targets.append(pid)
        else:
            stack.extend(reversed(children.get(pid, ())))
    return targets


def cmd_sigtargets(args):
    """`sigtargets ROOT SIG`: print the pids to signal (space-separated; nothing if none)."""
    name = args[1].upper()
    signum = int(name) if name.isdigit() else int(getattr(signal, name if name[:3] == "SIG" else "SIG" + name))
    targets = signal_targets(int(args[0]), signum, process_children(), caught_mask)
    if targets:
        print(" ".join(map(str, targets)))
    return 0


COMMANDS = {"status": cmd_status, "env": cmd_env, "outputs": cmd_outputs, "resume": cmd_resume,
            "linkfarm": cmd_linkfarm, "sigtargets": cmd_sigtargets}


def main(argv):
    if len(argv) < 2 or argv[1] not in COMMANDS:
        sys.stderr.write("usage: helper.py {%s} ...\n" % ",".join(sorted(COMMANDS)))
        return 2
    try:
        return COMMANDS[argv[1]](argv[2:])
    except Exception as exc:  # noqa
        sys.stderr.write("[hpclab] helper %s failed: %s: %s\n" % (argv[1], type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

if _HEREDOC_HELPER in HELPER_SOURCE:  # pragma: no cover - guards future edits of the helper
    raise RuntimeError("helper source contains its heredoc terminator")
