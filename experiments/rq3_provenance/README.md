# RQ3: Donor provenance and contamination control

The default `lego` configuration uses the full CodeFace view, as in the
paper's headline comparison. `lego_no_same_repo` removes entries from the
target repository; the remaining controls remove progressively more donors.
Use the disjoint control for the strongest provenance comparison.

The re-mined library uses a pinned corpus with evaluated repositories,
organizations, and recorded forks removed. Per-task views additionally drop
MinHash>=0.6 near-duplicates:

```bash
python -m lego.mining.disjoint_plan --corpus corpora/codeface_sources.txt \
  --out corpora/disjoint.txt --split lego_repo_522
python -m lego.mining.mine --corpus corpora/disjoint.txt \
  --out codeface_disjoint --describe --synth-tests --model <model-alias>
```

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |
| same-repository excluded | `lego_no_same_repo` |
| same-organization excluded | `lego_no_same_org` |
| + fork / near-duplicate excluded | `lego_no_near_dup` |
| re-mined disjoint library | `lego_remined` |

## Run

```bash
python -m lego.launch experiments/rq3_provenance/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_provenance/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq3_provenance/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq3_provenance/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq3_provenance/experiment.yaml --out results
```

writes `results/rq3_provenance/*.md|csv|tex`.
