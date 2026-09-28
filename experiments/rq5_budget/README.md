# RQ5: Attempt-budget ladder

b=1..5 are read from the per-round trajectories of the b=5 runs (a b=k run
is a prefix of the b=5 run). b=8 is a separate run.

| arm | config |
|---|---|
| Feedback | `feedback` |
| Feedback, b=8 | `feedback` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/rq5_budget/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq5_budget/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq5_budget/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq5_budget/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq5_budget/experiment.yaml --out results
```

writes `results/rq5_budget/*.md|csv|tex`.
