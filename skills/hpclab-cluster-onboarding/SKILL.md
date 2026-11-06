---
name: hpclab-cluster-onboarding
description: Set up HPC Lab for a user on a Slurm cluster (or the local simulator) — register the cluster from a site profile plus a private overlay, open the MFA-safe SSH connection, run cluster_inspect, fix storage/environment findings, and prove the setup with a smoke job. Use for first-time setup, a new student joining a lab, "hpclab doesn't connect", or moving to a new cluster.
---

# Onboarding a cluster

Setup steps are CLI commands the **user** runs in their terminal (registration writes a private file;
authentication needs their MFA). From the agent side you check the result with `cluster_inspect`.

## 0. Try it without a cluster first (2 minutes)

```bash
hpclab init --sim          # registers a local Slurm simulator as cluster "sim"
hpclab demo                # prepare → smoke → sweep → failure → recover → table, all local
```

## 1. Register the cluster

* A **site profile** (public rules: partitions, storage roles, poll limits, known issues with expiry) is
  bundled for supported sites (`hpclab cluster profiles`), e.g. `hai`. For other sites start from
  `generic-slurm` and fill in what the site documents.
* A **private overlay** holds the user's SSH alias, username, Slurm account, QoS and remote root. It
  never goes into a shared repo.

```bash
hpclab cluster add hai --profile hai --ssh-host haic --user <sunet> --account <slurm-account> \
  --remote-root /hai/scratch/<sunet>/hpclab
```

Put outputs, venvs, caches and checkpoints on scratch, never on a small home quota.

## 2. Authenticate once per session (MFA/Duo sites)

`hpclab auth hai` opens an SSH master connection *in the user's terminal* (they approve Duo);
the controller reuses it and never prompts on its own. When a tool returns `auth_required`, ask the
user to run exactly the `remediation` command it contains — don't retry in a loop.

## 3. Inspect and fix

`cluster_inspect(cluster_id="hai")` → one line per check with status and findings:

| check | typical finding | fix |
|---|---|---|
| auth / identity | remote user ≠ overlay user | fix the overlay |
| scheduler | `sacct` unavailable | results still work; accounting-based budget settlement becomes `unknown` |
| storage | remote root under home / quota nearly full | move to scratch |
| limits | per-user GPU caps, QoS | informs `max_parallel` |
| capacity | free GPUs per partition/type | informs partition choice |
| nodes | drained nodes, locally observed bad nodes (with expiry) | nothing to do; recovery uses them |

`cluster_inspect(cluster_id="hai", checks=["capacity"], candidate_shapes=[{"partition": "hai", "gpu_type": "h100", "gpus": 1, "time": "04:00:00"}])`
adds low-confidence start estimates for up to three placements.

Environment pitfalls worth checking before the first real job (the prepare step checks these
statically when `runtime.venv` is set): torch CUDA build vs driver (on HAI the driver is CUDA 12.8 →
use `cu128` wheels; B200 needs them), editable installs that bypass the code snapshot, HF/uv/torch
caches on home.

## 4. Prove it with a smoke job

`hpclab onboard <cluster>` (CLI) inspects the cluster and runs a tiny 1-GPU smoke job end to end. Or
prepare the user's experiment with a `smoke` block and dispatch it: the smoke run goes first, and a
passing smoke = the environment, paths and outputs work on a compute node. Keep compute off the login
node entirely — HPC Lab never runs training, tests or data prep there, and neither should you.
