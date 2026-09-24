# HPC Lab

MCP infrastructure for agent-driven research on Slurm GPU clusters.

HPC Lab is being designed to connect experiment planning, cluster execution, recovery, and research results through a shared MCP and CLI runtime.

> **Status:** Early development. This repository currently contains the project overview; the capabilities below are planned.

## Planned capabilities

- **Reliable execution:** Persistent job tracking, submission reconciliation, and recovery across interrupted sessions.
- **Reproducible experiments:** Immutable code snapshots, versioned configurations, and isolated execution attempts.
- **Research workflows:** Resource-bounded experiment matrices, checkpoint-aware recovery, and results comparison across seeds.
- **Result provenance:** Traceable links from paper tables to evaluations, artifacts, configurations, and code snapshots.
- **Extensible integrations:** Cluster profiles, starting with Stanford HAI, plus research and paper-writing skills.

## Design

Coding agents and CLI clients connect to a shared controller that manages experiment state, communicates with Slurm over SSH, and exposes structured tools and research artifacts through MCP.

The initial scope is a single-user runtime for Slurm-based GPU research, with 15 planned MCP tools.
