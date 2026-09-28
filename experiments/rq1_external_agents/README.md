# RQ1: Independently developed repository agents

External systems (OpenHands, Claude Code, SWE-agent, Agentless) run under
their own released harness via the task adapter in `baselines/`; see
`baselines/README.md`. Their records are written to `runs/arms/ext_*` in the
same format. Set the release pins and commands in `baselines/agents.yaml`
before running them.

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |

## Tables

```bash
python -m lego.special experiments/rq1_external_agents/experiment.yaml --validate
python -m lego.special experiments/rq1_external_agents/experiment.yaml --count
# For a local check after configuring the external agents:
python -m lego.special experiments/rq1_external_agents/experiment.yaml --job 0
python -m lego.analysis.report experiments/rq1_external_agents/experiment.yaml --out results
python -m baselines.report experiments/rq1_external_agents/experiment.yaml
```

The first reporter summarizes the two in-house arms. `baselines.report`
writes a separate comparison with the four external systems over the same
522-task denominator. On SLURM, use `cluster/submit_special.sh`.
