#!/usr/bin/env bash
# Usage: bash hpc/submit_saadiyat.sh --stage smoke [grid flags]
set -euo pipefail
export SF_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SF_REPO_DIR"
git diff --quiet HEAD || { echo 'Commit tracked changes before submitting.' >&2; exit 1; }
# Setup runs in a child shell, so activate again before generating the manifest.
# Site module and Conda initialization may reference unset shell variables.
set +u
source "$SF_REPO_DIR/hpc/environment.sh"
set -u
python -c 'import sys; assert sys.version_info >= (3, 11), "Activate the Python 3.11 experiment environment"'
SF_RUNS_DIR="${SF_RUNS_DIR:-/scratch/$USER/simple_sf_runs}"
mkdir -p "$SF_RUNS_DIR"
# Unique directory and immutable config snapshot per submission.
BATCH_DIR=$(mktemp -d "$SF_RUNS_DIR/batch_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export SF_MANIFEST="$BATCH_DIR/experiments.json"
export SF_OUTPUT_ROOT="$BATCH_DIR/runs"
mkdir -p "$BATCH_DIR/logs" "$SF_OUTPUT_ROOT"
TOTAL=$(python experiment_configs.py "$@" --write "$SF_MANIFEST" --total)
[[ "$TOTAL" =~ ^[1-9][0-9]*$ ]] || { echo 'Empty or invalid experiment grid' >&2; exit 1; }
chmod a-w "$SF_MANIFEST"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_PARALLEL must be positive' >&2; exit 1; }
echo "Submitting $TOTAL experiments; files: $BATCH_DIR"
sbatch --export=ALL --array="0-$((TOTAL-1))%$MAX_PARALLEL" \
  --output="$BATCH_DIR/logs/%A_%a.out" --error="$BATCH_DIR/logs/%A_%a.err" hpc/saadiyat.sh
