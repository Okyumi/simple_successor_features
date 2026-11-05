#!/usr/bin/env bash
# Source after the site's module initialization (as in sgcrl/DRAFT_jubail.sh).
module purge
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate "${SF_CONDA_ENV:-simple_sfs}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/.cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
# Modern dm-control uses mujoco, not the old mujoco_py installation.
