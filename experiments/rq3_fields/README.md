# RQ3: Primitive-field decomposition

Adds the primitive's fields one at a time to scoped generic editing, and
withholds carried tests from full adaptation, at matched budget.

| arm | config |
|---|---|
| RAG + Edit | `rag_edit` |
| + V_i | `rag_edit_V` |
| + I_i, X_i | `rag_edit_VIX` |
| + D_i | `rag_edit_VIXD` |
| LEGO, V_i withheld | `lego_no_V` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/rq3_fields/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_fields/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_fields/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_fields/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_fields/experiment.yaml --out results
```

writes `results/rq3_fields/*.md|csv|tex`.
