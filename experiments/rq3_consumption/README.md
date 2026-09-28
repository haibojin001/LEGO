# RQ3: How reused components are consumed

All library rows share LEGO's retrieval, relevance assessment, diagnosis
and budget. Import + Call vendors the primitive byte-identical under
`_lego_vendor/` outside the target prefix; Context-only shows it read-only and
never delivers a file from it.

| arm | config |
|---|---|
| Feedback + Diagnosis | `feedback_diag` |
| Context-only | `context_only` |
| Import + Call | `import_call` |
| Structured-schema adaptation | `structured_adapt` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/rq3_consumption/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_consumption/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_consumption/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_consumption/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_consumption/experiment.yaml --out results
```

writes `results/rq3_consumption/*.md|csv|tex`.
