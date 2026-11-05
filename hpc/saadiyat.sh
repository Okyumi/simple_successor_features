#!/usr/bin/env bash
#SBATCH --job-name=sf_nrc2
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=sf_nrc2_%A_%a.out
#SBATCH --error=sf_nrc2_%A_%a.err
# Submit via hpc/submit_saadiyat.sh to set array bounds and log paths.
set -eo pipefail
: "${SF_REPO_DIR:?Use hpc/submit_saadiyat.sh}"
: "${SF_MANIFEST:?Missing frozen experiment manifest}"
: "${SF_OUTPUT_ROOT:?Missing output directory}"
: "${SLURM_ARRAY_TASK_ID:?Must be a Slurm array job}"
cd "$SF_REPO_DIR"
source hpc/environment.sh
set -u
# Refuse to run changed code after submission (untracked logs are permitted).
EXPECTED_COMMIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_commit"])' "$SF_MANIFEST")
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo 'Code changed since submission; resubmit.' >&2; exit 1; }
git diff --quiet HEAD || { echo 'Tracked code is dirty; resubmit after committing.' >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip freeze > "${SF_OUTPUT_ROOT}/environment_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
srun python experiment_configs.py --manifest "$SF_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --output-root "$SF_OUTPUT_ROOT" --run
