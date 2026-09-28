# Appendix: Replicate runs

Three runs per configuration. Replicates differ only in sampling
(`replicate` is part of the arm id, so each replicate keeps its own records).
Set `temperature` in the arms if your endpoints are deterministic by default.

| arm | config |
|---|---|
| Single-shot | first-round prefix of `feedback` at the same replicate |
| Feedback | `feedback` |
| Feedback + Diagnosis | `feedback_diag` |
| RAG + Edit | `rag_edit` |
| LEGO, donor-disjoint | `lego_no_near_dup` |
| LEGO | `lego` |

**Sweep** (applied to every arm): `replicate` over 0, 1, 2

## Run

```bash
python -m lego.launch experiments/app_replicates/experiment.yaml --list     # jobs
python -m lego.launch experiments/app_replicates/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/app_replicates/experiment.yaml             # SLURM array
python -m lego.launch experiments/app_replicates/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/app_replicates/experiment.yaml --out results
```

writes `results/app_replicates/*.md|csv|tex`.
