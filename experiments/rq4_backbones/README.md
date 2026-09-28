# RQ4: Backbone substitution

Adaptation and diagnosis stay on GPT-5.6-terra; only the backbone moves.
The Feedback arms are identical to `rq1_scratch` (identical configs share
records), so the two experiments together cost one set of Feedback runs.

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |

**Sweep** (applied to every arm): `models.backbone` over gpt-5.6-terra, claude-sonnet-5, gpt-5.4, claude-opus-5, gpt-5.6-luna, grok-4.3, claude-haiku-4.5, gemma-4-31b, kimi-k2.5, qwen3-coder-480b, glm-5, deepseek-v3.2, nemotron-super-3-120b

## Run

```bash
python -m lego.launch experiments/rq4_backbones/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq4_backbones/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq4_backbones/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq4_backbones/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq4_backbones/experiment.yaml --out results
```

writes `results/rq4_backbones/*.md|csv|tex`.
