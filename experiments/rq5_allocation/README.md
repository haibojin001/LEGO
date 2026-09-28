# RQ5: Quality-cost trade-off under model allocation

c/a/d = backbone / adaptation / diagnosis model. Manuscript cost accounting
uses recorded backbone and diagnosis tokens, successful adaptation responses
only, and a fixed $0.011 per task for library retrieval. The rates in
`configs/models.yaml` are the manuscript's dated hosted-rate snapshot
(USD per million tokens). Infrastructure cost is excluded. The reporter
leaves costs blank for old records without successful-adaptation attribution
or a split that has not finished.

| arm | config |
|---|---|
| gpt-5.6-terra / gpt-5.6-terra / gpt-5.6-terra | `lego` |
| gpt-5.6-terra / claude-sonnet-5 / claude-sonnet-5 | `lego` |
| gpt-5.6-terra / deepseek-v4 / deepseek-v4 | `lego` |
| gpt-5.6-terra / gpt-oss-20b / gpt-oss-20b | `lego` |
| claude-sonnet-5 / claude-sonnet-5 / claude-sonnet-5 | `lego` |
| claude-sonnet-5 / gpt-oss-20b / gpt-oss-20b | `lego` |
| deepseek-v3.2 / gpt-5.6-terra / gpt-5.6-terra | `lego` |
| deepseek-v3.2 / deepseek-v4 / deepseek-v4 | `lego` |
| gpt-oss-20b / gpt-oss-20b / gpt-oss-20b | `lego` |
| Feedback + Diagnosis, b=5 | `feedback_diag` |
| Feedback, b=8 | `feedback` |

## Run

```bash
python -m lego.launch experiments/rq5_allocation/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq5_allocation/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq5_allocation/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq5_allocation/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq5_allocation/experiment.yaml --out results
```

writes `results/rq5_allocation/*.md|csv|tex`.
