#!/bin/bash
# Submit every job of an experiment as a SLURM array.
#
#   bash slurm/submit.sh experiments/rq3_paradigms/experiment.yaml [sbatch args...]
# Set LEGO_ENTRYPOINT=lego.special for special experiment jobs.
#
# THROTTLE (default 64) caps concurrently running tasks; CHUNK (default 1000)
# splits experiments with more jobs than the cluster's MaxArraySize into
# several arrays. Jobs are resumable, so resubmitting after a failure or a key
# rotation only runs what has no record yet.
set -eu
EXP="$1"; shift || true
ENTRYPOINT="${LEGO_ENTRYPOINT:-lego.launch}"
N=$(python -m "$ENTRYPOINT" "$EXP" --count)
THROTTLE="${THROTTLE:-64}"
CHUNK="${CHUNK:-1000}"
mkdir -p runs/slurm
off=0
while [ "$off" -lt "$N" ]; do
  n=$(( N - off < CHUNK ? N - off : CHUNK ))
  sbatch --array=0-$((n - 1))%"$THROTTLE" \
         --export=ALL,EXPERIMENT="$EXP",OFFSET="$off",LEGO_ENTRYPOINT="$ENTRYPOINT" "$@" slurm/array.sbatch
  off=$(( off + n ))
done
echo "submitted $N jobs for $EXP"
