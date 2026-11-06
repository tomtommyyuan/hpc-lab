"""Local Slurm simulator: fake ``sbatch/squeue/sacct/scancel/sinfo/sacctmgr/sstat/srun/scontrol``
that reproduce real Slurm output formats, run jobs as local processes and inject failures.

Entry points: ``cluster.SimCluster`` (create/inspect/fault/shutdown), ``transport.SimTransport``
(what ``transport: sim`` clusters use), ``testing.sim_cluster`` (pytest helper) and
``python -m hpclab.sim.cli`` (the fake commands and the ``_admin`` control plane).

Kept import-free on purpose: every fake command is a fresh interpreter.
"""
