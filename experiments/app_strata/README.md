# Appendix: Contamination strata (star count)

Same arms as `rq3_paradigms`; only the table differs. A post-cutoff subset
can be added as another split file.

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/app_strata/experiment.yaml --list     # jobs
python -m lego.launch experiments/app_strata/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/app_strata/experiment.yaml             # SLURM array
python -m lego.launch experiments/app_strata/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/app_strata/experiment.yaml --out results
```

writes `results/app_strata/*.md|csv|tex`.
