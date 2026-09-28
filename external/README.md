# RQ6 external benchmark transfer

The adapters in this directory run LEGO and the matched feedback arm on
external benchmarks. Each benchmark keeps its official grader and native
metric. Results are stored under `runs/external/<benchmark>/`.

`benchmarks.yaml` contains templates for the three benchmarks named in the
paper. Before a run, set each `release_path` and `release_commit`, map the
release's task fields with `fields` or a custom `loader`, configure its
official `grader_cmd` and `metric_parse`, and state the published protocol.
The template placeholders are rejected by the special-job validator.

Filter CodeFace against every target repository before running LEGO:

```bash
python -m external.filter_library --benchmark repogenesis \
  --library codeface --out codeface_repogenesis
python -m external.coverage --benchmark repogenesis \
  --library codeface_repogenesis
python -m lego.special experiments/rq6_transfer/experiment.yaml --validate
python -m lego.special experiments/rq6_transfer/experiment.yaml --job 0
python -m external.report experiments/rq6_transfer/experiment.yaml
```

The library path in `benchmarks.yaml` must point to the filtered output.
`external/run.py` checks the filter report at run time. The result report
uses observed native scores and includes task and record counts; it does
not put distinct benchmark metrics on a common scale.
