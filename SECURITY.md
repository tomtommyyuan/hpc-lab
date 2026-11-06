# Security policy

## Supported versions

HPC Lab is alpha software. Security fixes go into the latest release (currently 0.1.x) and
the main branch.

## Trust model

hpclab runs on your own machine, as your own Unix user, and acts on a shared cluster with your
own SSH login. What it trusts and what it does not:

* **You and your ExperimentSpec are trusted.** Your entrypoint and your `runtime.setup` lines run
  verbatim, as you, on the compute node. Read-only snapshots and content addressing protect
  against accidents (editing code while a job queues), not against code running as your own
  user. hpclab is not a sandbox.
* **Your coding agent is a client, not an authority.** MCP tool annotations are UI hints. Every
  mutation still passes the runtime's checks (idempotency, the one-execution-per-run claim,
  live job identity before `scancel`, the budget); no parameter skips them.
* **Everything that comes back from the cluster is data**: logs, `squeue`/`sacct` output, file
  contents, manuscript text. It is parsed and bounded, never executed or treated as an
  instruction.
* **Authentication stays with you.** hpclab never sees passwords or Duo codes. It runs `ssh` with
  `BatchMode=yes` over an OpenSSH ControlMaster connection you open yourself
  (`hpclab auth <cluster>`). Its own ssh calls disable agent and X11 forwarding (`-a -x`) to
  the shared login node.
* **Local state is private to your user.** The hpclab home and its `run/` directory are 0700,
  the controller socket 0600 in a 0700 directory, cluster overlays (your username, account, SSH
  alias) 0600. The ledger is per user and per machine.
* **Secrets stay out of what hpclab uploads and records.** Code snapshots exclude likely secrets
  (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, `credentials*`, `.aws/`, `.ssh/`, files
  containing a private key block, ...) even when they are tracked by git. Notifications carry
  ids, states and labels only; webhook URLs never enter the ledger, logs or errors. Public site
  profiles are linted for personal identifiers.
* **Values that reach a shell are quoted.** Everything hpclab embeds in a job script (paths,
  ids, argv, environment values) is shell-quoted; `#SBATCH` values are checked against a safe
  character set; `extra_sbatch` accepts only allow-listed flags.
* **Reads are confined.** `artifact_read` only reads inside the attempt's own directory
  (lexical check, then `realpath`); a symlink that escapes it is refused.

More detail: [docs/architecture.md#trust-boundaries](docs/architecture.md#trust-boundaries).

## What is a vulnerability here

For example: hpclab cancelling or signalling a job that is not the caller's own; a second
`sbatch` for one attempt; command injection through spec values, profile or overlay fields,
cluster output or file names; the socket, ledger, overlays or vault being readable by other
users; a secret ending up in a snapshot, the ledger, a log or a notification; `artifact_read`
escaping the attempt directory; a public profile or doc that leaks personal data.

Out of scope: code you run yourself through your own spec (`entrypoint`, `runtime.setup`), a
compromised login node or cluster, and Slurm's own access control (`PrivateData`, QoS limits).

## Reporting

Please report vulnerabilities privately: use the private vulnerability reporting of the
repository host (on GitHub: *Security → Report a vulnerability*), or contact a maintainer
privately and ask for a secure channel before sending details. Do not open a public issue.

Include the hpclab version (`hpclab --version`), what you did, what happened and what you
expected. Do not send credentials, cluster usernames or private data; a reproduction on the
simulator (`hpclab init --sim`) is ideal. The maintainers will acknowledge the report, keep you
informed, and agree on a disclosure date with you.
