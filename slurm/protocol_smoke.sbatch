#!/bin/bash
#SBATCH --job-name=lego-protocol-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$PWD/scratch/protocol_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
python tests/smoke.py --configs feedback lego
