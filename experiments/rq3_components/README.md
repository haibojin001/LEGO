# RQ3: Component removals

Leave-one-out removals of LEGO's mechanisms.

| arm | config |
|---|---|
| LEGO | `lego` |
| - diagnosis | `retrieval_adapt` |
| - adaptation | `retrieval_diag` |
| - library | `feedback_diag` |
| - collaboration (mu) | `lego_no_collab` |
| - carried tests | `lego_no_V` |

## Run

```bash
python -m lego.launch experiments/rq3_components/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_components/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_components/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_components/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_components/experiment.yaml --out results
```

writes `results/rq3_components/*.md|csv|tex`.
