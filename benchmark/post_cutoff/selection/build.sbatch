#!/bin/bash
#SBATCH --job-name=lego-postcutoff
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.log
#
# Build the post-cutoff tasks with the LEGO-REPO procedure (lego.bench_build). One array task = one
# shard of the cards. Resume-safe: a card with a record in any runs/post_cutoff_build_s*.jsonl is
# skipped, so every card is attempted exactly once. Harness test tooling is held below pytest 9.2
# (9.2.0, 2026-10-27, turns a deprecation into a collection error; LEGO-REPO predates it).
#   sbatch --array=0-31%16 benchmark/post_cutoff/build_cluster.sbatch
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
export NSHARDS="${NSHARDS:-32}" I="${SLURM_ARRAY_TASK_ID:-0}"
export PATH="$HOME/.local/bin:$PATH"
export PIP_CONSTRAINT="$PWD/benchmark/post_cutoff/constraints.txt" UV_CONSTRAINT="$PWD/benchmark/post_cutoff/constraints.txt"
source .venv/bin/activate
export LEGO_SCRATCH="${TMPDIR:-/tmp}/lego_pc_${SLURM_JOB_ID:-$$}_${I}"
mkdir -p "$LEGO_SCRATCH" runs/slurm
export TODO="$LEGO_SCRATCH/cards_todo.jsonl" CARDS="${CARDS:-benchmark/post_cutoff/cards.jsonl}" LOGP="${LOGP:-runs/post_cutoff_build}" OUTD="${OUTD:-benchmark/post_cutoff/tasks}"
python - <<'PY'
import glob, json, os
i, n = int(os.environ['I']), int(os.environ['NSHARDS'])
cards = [l for l in open(os.environ['CARDS']) if l.strip()][i::n]
done = set()
for f in glob.glob(os.environ['LOGP'] + '_s*.jsonl'):
    done |= {json.loads(l)['name'] for l in open(f) if l.strip()}
todo = [l for l in cards if json.loads(l)['name'] not in done]
open(os.environ['TODO'], 'w').writelines(todo)
print(f'shard {i}: {len(cards)} cards, {len(todo)} still to build', flush=True)
PY
python -m lego.bench_build --cards "$TODO" --pins benchmark/pins.json \
  --out "$OUTD" --log "${LOGP}_s${I}.jsonl" --shard 0 --nshards 1
rc=$?
rm -rf "$LEGO_SCRATCH"
exit $rc
