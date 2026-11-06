# Stanford HAI Compute Cluster (HAIC)

The first real site hpclab supports. The public site profile ships with hpclab
(`profiles/hai.yaml`, cluster id `hai`); your identity goes into a private overlay. This page
walks through setup and the site rules hpclab enforces or reports.

Placeholders: `<sunet-id>` is your SUNet ID, `<slurm-account>` the Slurm account your PI gave
you. Nothing personal belongs in a profile, an ExperimentSpec or a commit.

> **Site facts age.** Node inventory, the driver version, quota behaviour and bad nodes change.
> The profile records each of them as an *observation* with a source and an expiry date (below);
> hpclab treats an expired fact as unknown, and live probes (`sinfo`, `scontrol`, `sacctmgr`)
> always win over the file. Facts on this page were measured in 2026-08.

## 1. Before you start

* A HAIC account with a SUNet ID and Duo, and a Slurm account from your PI.
* Off campus: connect the **Stanford VPN in full-tunnel mode** first.
* hpclab installed on your laptop or workstation (see the [quickstart](../quickstart.md)). The
  controller runs *there*, not on the cluster: the HAIC login node is for bounded metadata
  commands only, and hpclab never runs anything heavy on it.

Recommended `~/.ssh/config` entry:

```sshconfig
Host haic
    HostName haic.stanford.edu
    User <sunet-id>
```

## 2. Register the cluster

```console
$ hpclab cluster profiles
3 site profile(s)
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ name          ┃ cluster_id    ┃ display_name                        ┃ source  ┃ error ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ generic-slurm │ generic-slurm │ Generic Slurm site (template)       │ bundled │       │
│ hai           │ hai           │ Stanford HAI Compute Cluster (HAIC) │ bundled │       │
│ sim           │ sim           │ Local Slurm simulator               │ bundled │       │
└───────────────┴───────────────┴─────────────────────────────────────┴─────────┴───────┘

$ hpclab cluster add hai --profile hai --ssh-host haic --user <sunet-id> --account <slurm-account>
cluster hai registered
cluster hai (Stanford HAI Compute Cluster (HAIC)) · transport ssh · user <sunet-id> · account <slurm-account>
remote root: /hai/scratch/<sunet-id>/hpclab
next: authenticate once per session: hpclab auth hai
next: check the cluster: hpclab cluster inspect hai
```

This writes your **overlay**, `~/.hpclab/clusters/hai.yaml`, with mode 0600:

```yaml
# hpclab private cluster overlay. Holds YOUR identity for this cluster (remote user,
# Slurm account, QoS, SSH alias). Never commit or share this file; it is written 0600.
cluster_id: hai
ssh_host: haic
user: <sunet-id>
account: <slurm-account>
```

A fully commented template is `profiles/overlays/hai.example.yaml` (it also shows `qos`,
`remote_root`, `default_partition`, `ssh_options`, `lab_users` and `notify` channels).

* **Account.** Every HAIC job needs `--account`; the profile sets `require_account: true` and
  `experiment_prepare` blocks a plan without one (`account_present`).
* **QoS.** Operator notes call `--qos` mandatory, but batch jobs with only `-p hai` have been
  accepted and the site's QoS names are not recorded. Leave it unset. If a submission is rejected
  for QoS, look yours up on the login node and add it:
  `sacctmgr show user $USER withassoc format=user,account%20,qos%30`, then
  `hpclab cluster add hai --qos <qos>`.
* **Remote root on scratch.** hpclab keeps code snapshots and attempt directories under
  `/hai/scratch/<sunet-id>/hpclab` (the profile's `{scratch}/hpclab`). The home directory
  `/hai/users/<sunet-id>` has a **50 GB** quota; a remote root there is a *blocking* finding
  (`storage_home_avoidance`). Override with `--remote-root /hai/scratch/<sunet-id>/<dir>` only.

## 3. Authenticate: Duo stays in your terminal

hpclab never sees your password or Duo codes. It runs `ssh` with `BatchMode=yes` over an OpenSSH
**ControlMaster** connection that you open once:

```console
$ hpclab auth hai
running: ssh -o ControlMaster=auto -o ControlPath=<control dir>/ssh-%C -o ControlPersist=30m -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o LogLevel=INFO -l <sunet-id> -- haic true
(Duo prompt here)
connected: hpclab will reuse this connection for hai
resumed 0 waiting step(s)
```

`hpclab auth hai --print` only prints the command (to run it inside `tmux`, for example).

When the connection expires, nothing fails silently: submissions, cancels and polls that need
the cluster **block** with `auth_required` and the remediation `hpclab auth hai` (the MCP tools
tell the agent to ask *you* to run it; the agent cannot answer Duo). Running `hpclab auth hai`
wakes the blocked work immediately. A submission that was in flight when the login expired is
reconciled, never re-sent.

`hpclab cluster inspect hai` distinguishes `auth_required`, `vpn_unreachable` (connect the VPN),
`host_key_changed` (confirm the new host key with the cluster admins before accepting it) and
`unknown`.

## 4. Check the cluster, then prove it with one job

```bash
hpclab cluster inspect hai     # auth, identity, scheduler, storage, runtime, queue, capacity, limits, nodes, profile facts
hpclab onboard hai             # the same checks, then a 1-GPU, 10-minute smoke job (≤ 0.17 GPU-h) end to end
```

Every check reports `ok | warn | fail | unavailable | unsupported | skipped` with its source and
time; what Slurm does not let you see is `unavailable`, never zero. `onboard` asks before spending
GPU time and prints the host, `CUDA_VISIBLE_DEVICES` and `nvidia-smi -L` from inside the job.

For a first real experiment, the example project has a HAI spec. It is CPU-only (HAIC runs CPU
jobs on `hai` without `--gres`; if your account is refused, set `gpus: 1` and `gpu_type: h100`):

```console
$ hpclab examples copy tiny-classifier tc
$ cd tc && git init -q && git add . && git commit -qm "tiny-classifier" && cd ..
$ hpclab prepare tc/hpclab.hai.yaml --no-remote-checks --all-findings
plan pln_01M3KPDY2KHYH6RDP329P4JS0C ready (matrix) · experiment exp_01M3KPDY2KCC33WXZ6S44RWN90
validation: warn — 23 pass, 2 warn, 4 unchecked, 0 fail
│ warn      │ info     │ cache_locations             │ ['HF_HOME', 'UV_CACHE_DIR', 'TORCH_HOME',             │
│           │          │                             │ 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR'] are unset and   │
│           │          │                             │ default to $HOME/.cache on the home role (small       │
│           │          │                             │ quota) if the program downloads models/datasets or    │
│           │          │                             │ compiles kernels                                      │
│           │          │                             │ → set them in runtime.env to scratch paths (see the   │
│           │          │                             │ profile's runtime recipes)                            │
│ pass      │ info     │ remote_root                 │ remote root /hai/scratch/<sunet-id>/hpclab            │
│ pass      │ info     │ account_present             │ account <slurm-account>                               │
│ pass      │ info     │ storage_home_avoidance      │ no attempt data on storage roles that avoid it        │
...
```

That run was made from a laptop without a HAIC login, so `--no-remote-checks` left the three
remote checks (`env_guard_static`, `snapshot_bypass`, `scheduler_test`) `unchecked`. With a live
connection, drop the flag and they run. The `cache_locations` warning is worth fixing for any real
project (next section).

## 5. Python environments and the CUDA 12.8 driver

HAIC has no conda and no root: Python comes from **uv**, and virtualenvs, caches, models and
datasets live on scratch. The profile ships two runtime recipes:

* **`uv-scratch`**: move every cache off the 50 GB home. Put these in your spec's `runtime.env`
  (`{scratch}` expands to `/hai/scratch/<sunet-id>`):

  ```yaml
  runtime:
    venv: "{scratch}/venvs/myproject"         # create once on the login node: uv venv /hai/scratch/<sunet-id>/venvs/myproject --python 3.12
    env:
      UV_CACHE_DIR: "{scratch}/.cache/uv"
      PIP_CACHE_DIR: "{scratch}/.cache/pip"
      HF_HOME: "{scratch}/.cache/huggingface"
      TORCH_HOME: "{scratch}/.cache/torch"
      TRITON_CACHE_DIR: "{scratch}/.cache/triton"
      XDG_CACHE_HOME: "{scratch}/.cache"
  ```

  Keep the venv outside the project tree so snapshots stay small. Downloads work on the login
  node; compute nodes may have no internet, so fetch models and datasets beforehand.

* **`torch-cu128`**: the HAIC driver is a 570-series driver whose newest supported CUDA runtime
  is **12.8** (profile fact `driver_cuda`). Default PyPI torch wheels now target CUDA 13.0 and
  fail with `driver too old (found version 12080)` (`torch.cuda.is_available()` is False); cu124
  wheels lack `sm_100` kernels and break on B200. Install:

  ```bash
  uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # sm_90 (H100/H200) and sm_100 (B200)
  ```

  Verify only inside a job, never on the login node.

### The prepare-time driver check

When a spec declares `runtime.venv` and requests GPUs, `experiment_prepare` reads the venv's
`pyvenv.cfg` and `torch/version.py` over SSH (a few small files, bounded) and compares the torch
CUDA build with the profile fact `driver_cuda` and the GPU architectures in `gpu_arch`
(`h100`/`h200` → `sm_90`, `b200` → `sm_100`). A mismatch **blocks the plan before any GPU-hour is
spent**. This is the real finding, reproduced on the simulator with the same fact and a venv
whose torch was built for CUDA 13.0:

```console
$ hpclab prepare tc/hpclab.yaml
plan pln_01M3KPEKMN3ZSMNA27AK3ZGE8T invalid (matrix) · experiment exp_01M3KPEKMN2VDA3202CW1Z2ERC
validation: fail — 25 pass, 1 warn, 2 unchecked, 1 fail · blocking: env_guard_static
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ status    ┃ severity ┃ check             ┃ finding                                                         ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ fail      │ block    │ env_guard_static  │ torch is built for CUDA 13.0 but the driver supports CUDA 12.8: │
│           │          │                   │ torch.cuda.is_available() will be False                         │
│           │          │                   │ → install a torch wheel matching the driver, e.g. uv pip        │
│           │          │                   │ install 'torch>=2.7' --index-url                                │
│           │          │                   │ https://download.pytorch.org/whl/cu128                          │
...
next:
  fix the blocking finding(s) above and run `hpclab prepare` again
  or, only if you accept the risk (recorded): hpclab prepare ... --override env_guard_static='<reason>'
```

The same check blocks an **editable install** of your project in the venv (`snapshot_bypass`:
the job would import your live working tree instead of the frozen snapshot) and warns when cache
variables resolve to the home directory.

Two more layers catch what a static check cannot:

* **Compute-side guard** (`runtime.guard`, on by default for GPU jobs): before your program
  starts, the wrapper checks that `nvidia-smi` works, torch sees the allocated GPUs and their
  architecture is in torch's build. Otherwise the job stops within seconds with exit code 3 and
  `HPCLAB_GUARD_FAILED:<check>` on stderr, instead of silently training on the CPU.
* **Circuit breaker**: a workflow pauses dispatch after 2 cells fail with the same environment
  error (import error, CUDA driver mismatch, missing file, guard failure). Infrastructure failures
  (node failure, preemption, timeout) never trip it. Resume with `hpclab workflow resume <wf>`
  after the fix.

## 6. Partitions

| Partition | Max time | GPUs (profile) | Notes |
|---|---|---|---|
| `hai` (default) | 7 days | h100, b200 | Main batch partition on HAI-owned nodes. CPU-only jobs: try without `--gres` first. |
| `hai-interactive` | 1 day | 1 GPU per job, 1 per user | For `srun --pty` sessions only. The profile marks it `interactive_only`, so a batch plan on it is **blocked** (`interactive_partition_batch`). |
| `hai-lo` | 14 days | h100, h200, b200 | **Preemptible** scavenger partition that borrows every node, including nodes owned by other groups (the H200s are reachable only here; H100 venvs run unchanged on H200, both `sm_90`). Often idle: the best fallback when `hai` queues are long. |

**Preemption rule.** A plan on `hai-lo` is **blocked** (`preemptible_gate`) unless the program
declares that it can resume, because a preemption would restart it from scratch. Declare:

```yaml
resources: {partition: hai-lo, gpus: 1, gpu_type: h200, time: "2-00:00:00"}
resilience:
  resumable: true                     # the program resumes from $HPCLAB_RESUME_FROM (or its own checkpoint dir on requeue)
  stop_behavior: checkpoint_and_exit  # adds --signal=B:USR1@lead_s: checkpoint and exit before the time limit
  signal: USR1
  lead_s: 900                         # >= checkpoint time + 60 s (Slurm may deliver the signal late)
```

With `resumable: true` on a preemptible partition hpclab submits with `--requeue` (set
`requeue_on_preemption: false` to opt out). A program that is not resumable always gets an
explicit `--no-requeue`, on `hai` too: after a node failure (e.g. an NFS black hole) Slurm's
default `JobRequeue=1` would otherwise restart it from scratch over the attempt's own outputs. A
requeue is recorded as a new segment of the same attempt. After a preemption or timeout,
`run_recover` (`hpclab recover <attempt>`) builds a `resume` plan from the newest *validated*
checkpoint, first creating a small probe run that loads it on a compute node if its loadability
has not been proven. See
[experiment-spec.md](../experiment-spec.md#resilience) and `examples/tiny-classifier/hpclab_io.py`
for a checkpoint format that supports this.

**Limits.** HAIC documents 8 running + 8 queued GPUs per user and 16 per account, but observed
behaviour has been looser (profile fact `gpu_limits`). hpclab does not hard-code per-user caps:
submit, and if Slurm holds a job, `run_inspect` decodes the pending reason (for example
`QOSMaxGRESPerUser` is a self-limit that clears when your own jobs finish; `Resources`/`Priority`
means the cluster is busy).

**Time limits.** Request estimate × 1.15 plus evaluation/checkpoint overhead; jobs sized "just
enough" have been killed at 98 % of training. After three comparable completed runs,
`experiment_prepare` suggests a limit from your own history (`resource_advice`), and
`run_inspect` shows progress/ETA against the remaining walltime from your metrics log.

## 7. Known issues are expiring observations

The profile does not hard-code bad nodes or driver versions as policy. Each is a
`known_issues` / `facts` entry with `observed_at`, `expires_at` and a source. Current entries:

| Key | Kind | Observed | Expires | What it says |
|---|---|---|---|---|
| `node.haic-hgx-2.nfs_client_blackhole` | known issue (warning) | 2026-08-23 | 2026-10-23 | New jobs end in `NODE_FAIL` after ~10 min without even a `slurm-*.out`; running jobs stay `R` while their log mtime freezes (process stuck in D state, often during a large checkpoint write); cancelled jobs hang in `CG`. The file server is healthy, only that node's NFS client is broken. |
| `driver_cuda` | fact | 2026-08 | 2026-11-30 | `12.8` (570-series driver). Drives the prepare-time torch check. |
| `node_inventory` | fact | 2026-08 | 2026-11-30 | `haic-hgx-[1-5]` 8×H100 and `haic-hgx-8` 8×B200 in `hai` and `hai-lo`; `haic-hgx-[6,9]` 8×H200 and `haic-hgx-7` 8×B200 owned by another group, reachable only through `hai-lo`. |
| `gpu_limits` | fact | 2026-08 | 2026-11-30 | Documented 8+8 per user, 16 per account; a single user ran 11 GPUs at once without rejection. |

How hpclab uses them:

* `hpclab cluster inspect hai` lists active known issues in the `nodes` check, together with
  drained/down nodes from `sinfo` and nodes your own attempts found suspect.
* Known issues are **reported, not auto-applied**. While the `haic-hgx-2` entry is active, add
  `exclude_nodes: [haic-hgx-2]` to `resources` for long jobs. Independently, `run_recover`'s
  `retry` excludes nodes where two or more of your attempts failed within the last 72 hours (a
  single failure is reported as suspect, never excluded silently).
* An **expired** fact is treated as unknown: after 2026-11-30 the driver check becomes
  `unchecked` ("no current driver_cuda fact in the profile") instead of trusting an old number.
  To renew a fact, verify it (e.g. `nvidia-smi` inside a job), then update `expires_at`,
  `observed_at` and `source` in a local copy (`~/.hpclab/profiles/hai.yaml` overrides the bundled
  profile) or send a pull request.
* Recovering a zombie run on a black-hole node: cancel it (hpclab verifies the job identity
  first), let `run_recover` build a retry/resume plan with the node excluded (each attempt already
  writes to its own directory, so the stuck process cannot touch the new attempt's outputs), and
  report the `NODE_FAIL` line, the missing `slurm-*.out` and the mtime evidence to hai-support.

## 8. Usage reports and `PrivateData`

```bash
hpclab usage --cluster hai                                  # your jobs, last 7 days, by partition
hpclab usage --cluster hai --account <slurm-account> --by user,state --since 30d
```

GPU-hours are *allocated* hours (GPUs × wall time), not utilization. Whether you can see other
users' jobs is Slurm's decision: `PrivateData` in `slurm.conf` (reported by
`cluster inspect`'s scheduler check) and in `slurmdbd.conf` (which cannot be read from a login
node). When accounting returns only your own jobs for an account report, hpclab labels it
`visibility: self_only`, prints **no lab total**, and says so:

```text
visibility: only your jobs are visible (Slurm PrivateData hides other users' jobs) — no lab totals can be
computed from here; ask an account coordinator for a lab-wide report
```

This is deliberate: a partial sum presented as "the lab's usage" would be wrong. A PI or account
coordinator who can see the account's jobs gets `visibility: full`. See [for-pis.md](../for-pis.md).

## 9. House rules hpclab encodes

* **No compute on the login node** (`login_node_compute: forbidden`): hpclab runs only `squeue`,
  `sacct`, `sinfo`, `sacctmgr`, `scontrol show`, `stat` and small reads there, polls `squeue` at
  most every 30 s and `sacct` every 60 s per cluster (shared by all your agent sessions), and hashes
  files over 256 MiB inside a Slurm job.
* **No double writes**: one active execution per run, and each attempt has its own output
  directory. Two live jobs writing the same run directory cannot happen through hpclab.
* **Never cancel by remembered job id**: `run_cancel` re-verifies job id, owner, job-name token
  and submit time right before `scancel` and refuses on any mismatch. For a manual check:
  `sacct -j <id> -X -n --format=SubmitLine%600`.
* **Judge hangs by evidence**: `run_inspect` shows log and metrics age, heartbeat and (with
  `probe="gpu"`) `nvidia-smi` inside the job's own allocation. Evaluation heartbeats can pause
  ~3 minutes and checkpoint writes ~1 minute; hpclab's health flags are advisory and never cancel
  anything.
* **No transfer node**: keep rsync concurrency low. hpclab uploads a snapshot once per content
  digest; unchanged code is not re-uploaded.

## Sources

The profile lists its sources: the HAI Compute Cluster public page (retrieved 2026-09-23; its node
counts and time limits disagree with the operator notes, so neither is hard-coded as current),
hands-on operator notes of HAIC users (measurements from 2026-08, personal identifiers removed)
and the Slurm `sbatch` documentation. Support contact: hai-support.
