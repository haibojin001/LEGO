# RQ3: Construction paradigms

R/A/E/V/D = retrieval, primitive adaptation, generic editing, execution
feedback, diagnosis (b=5). Stage increments are differences between rows of
the `paired` tables.

| arm | config |
|---|---|
| Single-shot | first-round prefix of `feedback` |
| Retrieval, one attempt | first-round prefix of `retrieval_feedback` |
| Feedback | `feedback` |
| Retrieval + Feedback | `retrieval_feedback` |
| Feedback + Diagnosis | `feedback_diag` |
| Retrieval + Diagnosis | `retrieval_diag` |
| Retrieval + Adapt | `retrieval_adapt` |
| RAG + Edit | `rag_edit` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_paradigms/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_paradigms/experiment.yaml --out results
```

writes `results/rq3_paradigms/*.md|csv|tex`.
