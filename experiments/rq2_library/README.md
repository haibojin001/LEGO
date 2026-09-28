# RQ2: CodeFace reach and the runtime adaptation funnel

Two parts.

1. **Offline coverage** (no construction):
   `python -m lego.analysis.coverage --split lego_repo_522 --config lego --assess --model gpt-5.6-terra`
   writes per-module coverage and per-band / per-domain summaries to
   `results/coverage/`. It uses the same backbone decomposition, retrieval
   query, and resident assessment as construction. Each completed task is
   flushed to the JSONL file with a progress marker so an interrupted audit can
   resume. Run it once per library view you report (e.g.
   `--set 'view.exclude=["same_repo","same_org","near_dup"]'`).
2. **Runtime funnel**: the `funnel` table aggregates `query / cand / retain /
   adapt_ok` from the LEGO run's records. *share* = adapt_ok / modules;
   *used* = adapt_ok / covered modules (covered from part 1).

Library composition by provenance kind:
`python -c "from lego.library.codeface import CodeFace; print(CodeFace('codeface').counts())"`.

| arm | config |
|---|---|
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/rq2_library/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq2_library/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq2_library/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq2_library/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq2_library/experiment.yaml --out results
```

writes `results/rq2_library/*.md|csv|tex`.
