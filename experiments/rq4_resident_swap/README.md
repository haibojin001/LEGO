# RQ4: Adaptation-model substitution

Backbone and diagnosis fixed to the default; the resident model that
performs ASSESS/ADAPT is swept. The *no primitive* reference row is
`retrieval_diag` in `rq3_paradigms`.

| arm | config |
|---|---|
| LEGO | `lego` |

**Sweep** (applied to every arm): `models.resident` over gpt-5.6-terra, claude-sonnet-5, gpt-5.4, gpt-5.6-luna, deepseek-v4, gemma-4-31b, glm-5, qwen3-coder-480b, gpt-oss-20b

## Run

```bash
python -m lego.launch experiments/rq4_resident_swap/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq4_resident_swap/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq4_resident_swap/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq4_resident_swap/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq4_resident_swap/experiment.yaml --out results
```

writes `results/rq4_resident_swap/*.md|csv|tex`.
