# RQ3: Construction-stage attribution

Each row adds one mechanism. File RAG needs its index first:
`python -m lego.library.file_rag build --library codeface --out file_rag`
(same donor repositories, same embedder and depth as CodeFace). *step* is the
difference to the previous row; *% tot.* is step / (LEGO - Feedback).

| arm | config |
|---|---|
| Feedback | `feedback` |
| + diagnosis | `feedback_diag` |
| + File RAG, editing | `file_rag_edit` |
| + constructed primitives | `rag_edit` |
| + primitive adaptation | `lego` |

## Run

```bash
python -m lego.launch experiments/rq3_ablation/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_ablation/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_ablation/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_ablation/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_ablation/experiment.yaml --out results
```

writes `results/rq3_ablation/*.md|csv|tex`.
