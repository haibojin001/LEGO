# RQ1: Construction from scratch across 13 backbones

Every backbone under the same execution-feedback configuration (no
library, b=5). Band columns, All, Dead@0 and Ceil@1 come from `main`; the
aggregate rows (proprietary / open-weight / field mean) are means of the
per-backbone rows. The single-pass score of each backbone is the b=1 prefix of
its trajectory (see `rq5_budget` for the ladder, or `trajectory[0]` of each
record).

| arm | config |
|---|---|
| Feedback | `feedback` |

**Sweep** (applied to every arm): `models.backbone` over gpt-5.6-terra, claude-sonnet-5, gpt-5.4, claude-opus-5, gpt-5.6-luna, grok-4.3, claude-haiku-4.5, gemma-4-31b, kimi-k2.5, qwen3-coder-480b, glm-5, deepseek-v3.2, nemotron-super-3-120b

## Run

```bash
python -m lego.launch experiments/rq1_scratch/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq1_scratch/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq1_scratch/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq1_scratch/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq1_scratch/experiment.yaml --out results
```

writes `results/rq1_scratch/*.md|csv|tex`.
