#!/bin/bash
#SBATCH --job-name=lego
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.log
#
# One array task = one (arm, shard) job of an experiment:
#   python -m lego.launch $EXPERIMENT --job $((OFFSET + SLURM_ARRAY_TASK_ID))
# Submit with slurm/submit.sh, which sets --array from the experiment's job count.
# The work is model-API latency plus pytest, so CPU nodes are sufficient.
#
# Environment (exported by submit.sh or your shell):
#   EXPERIMENT     path to experiments/<id>/experiment.yaml   (required)
#   OFFSET         added to the array index (for experiments larger than MaxArraySize)
#   LEGO_VENV      virtualenv with this package installed      (default .venv)
#   LEGO_ENV_FILE  file exporting API keys / endpoints         (optional)
#   LEGO_SCRATCH_BASE  where per-job scratch dirs (venvs, clones) are created
#                      (default $TMPDIR; each job gets its own subdirectory)
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
: "${EXPERIMENT:?set EXPERIMENT}"
IDX=$(( ${OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
source "${LEGO_VENV:-.venv}/bin/activate"
[ -n "${LEGO_ENV_FILE:-}" ] && source "$LEGO_ENV_FILE"
export LEGO_SCRATCH="${LEGO_SCRATCH_BASE:-${TMPDIR:-/tmp}}/lego_${SLURM_JOB_ID:-$$}_${IDX}"
mkdir -p "$LEGO_SCRATCH"
echo "[lego] $EXPERIMENT job $IDX on $(hostname) $(date -u)"
python -m "${LEGO_ENTRYPOINT:-lego.launch}" "$EXPERIMENT" --job "$IDX"
rc=$?
rm -rf "$LEGO_SCRATCH"
echo "[lego] job $IDX done rc=$rc $(date -u)"
exit $rc
