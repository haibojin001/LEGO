# RQ1: Single pass vs. execution feedback

Single-shot is the first-round prefix of the matching Feedback trajectory.

| arm | config |
|---|---|
| Single-shot | first-round prefix of `feedback` |
| Feedback | `feedback` |

## Run

```bash
python -m lego.launch experiments/rq1_single_pass/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq1_single_pass/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq1_single_pass/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq1_single_pass/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq1_single_pass/experiment.yaml --out results
```

writes `results/rq1_single_pass/*.md|csv|tex`.
