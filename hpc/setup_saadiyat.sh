#!/usr/bin/env bash
# Run once on a login node with package-download access; never inside each job.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
SF_CONDA_ENV="${SF_CONDA_ENV:-simple_sfs}"
conda create -y -n "$SF_CONDA_ENV" python=3.11 pip
conda activate "$SF_CONDA_ENV"
export PYTHONNOUSERSITE=1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$REPO_DIR/requirements-experiment.txt"
cd "$REPO_DIR"
python -m pytest -q
