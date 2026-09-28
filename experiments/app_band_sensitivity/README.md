# Appendix: Difficulty-band sensitivity and scoring checks

Band definition (Eq. 6) versus two size-only binnings, span weighting,
zero-floor tasks and raw pass rate:
`python -m lego.analysis.robustness experiments/app_band_sensitivity/experiment.yaml --a LEGO --b Feedback`

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |
| Feedback + Diagnosis | `feedback_diag` |

## Run

```bash
python -m lego.launch experiments/app_band_sensitivity/experiment.yaml --list     # jobs
python -m lego.launch experiments/app_band_sensitivity/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/app_band_sensitivity/experiment.yaml             # SLURM array
python -m lego.launch experiments/app_band_sensitivity/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/app_band_sensitivity/experiment.yaml --out results
```

writes `results/app_band_sensitivity/*.md|csv|tex`.
