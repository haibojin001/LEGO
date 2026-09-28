# RQ4: Diagnosis-model substitution

Backbone and adaptation fixed to the default; the diagnosis model is swept.
The *no diagnosis* reference row is `retrieval_adapt` in `rq3_paradigms`.

| arm | config |
|---|---|
| LEGO | `lego` |

**Sweep** (applied to every arm): `models.diagnosis` over gpt-5.6-terra, claude-sonnet-5, gpt-5.4, gpt-5.6-luna, deepseek-v4, gemma-4-31b, glm-5, qwen3-coder-480b, gpt-oss-20b

## Run

```bash
python -m lego.launch experiments/rq4_diagnosis_swap/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq4_diagnosis_swap/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq4_diagnosis_swap/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq4_diagnosis_swap/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq4_diagnosis_swap/experiment.yaml --out results
```

writes `results/rq4_diagnosis_swap/*.md|csv|tex`.
