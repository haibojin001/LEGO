# RQ6: Transfer to external benchmarks

Runs under each external benchmark's own tasks, metric and grader via
`external/`; see `external/README.md`. The arms listed here only fix the two
configurations compared (`feedback` = matched execution-feedback baseline).
CodeFace is filtered per benchmark before any task runs
(`python -m external.filter_library`), and offline coverage on each benchmark
is reported by `python -m external.coverage`.

Configure the pinned release, task loader, official grader, metric parser,
and filtered library for each benchmark in `external/benchmarks.yaml`.

```bash
python -m lego.special experiments/rq6_transfer/experiment.yaml --validate
python -m lego.special experiments/rq6_transfer/experiment.yaml --count
# For a local check after configuring the releases:
python -m lego.special experiments/rq6_transfer/experiment.yaml --job 0
python -m external.report experiments/rq6_transfer/experiment.yaml
```

The report keeps each benchmark's native metric and shows how many tasks
were actually graded. On SLURM, use `cluster/submit_special.sh`.

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |
