# RQ1 external repository agents

`baselines/run.py` prepares each LEGO-REPO task for an external agent, runs
the pinned release, and grades the delivered tree with the same frozen key and
provenance check used by LEGO. `baselines/agents.yaml` describes the four
systems and their workspace, command, version, model, and patch output.

The checked-in commands are templates. For each agent, pin an upstream
release, verify its command and environment variables against that release,
and replace every angle-bracket placeholder in `agents.yaml`. The runner
records the configured pin, observed version, agent output and token usage.
`python -m lego.special
experiments/rq1_external_agents/experiment.yaml --validate` checks the
configuration before jobs are submitted.

```bash
python -m baselines.run --system openhands --model gpt-5.6-terra \
  --split lego_repo_522 --shard 0 --nshards 32
python -m baselines.report experiments/rq1_external_agents/experiment.yaml
```

The `instruct` workspace has interface stubs and an instruction file; the
`issue` workspace has the target package removed and an issue file. Agents
that emit patches use the configured `patch_glob`; agents that edit a tree
deliver that tree. Every record is stored in `runs/arms/ext_*` and resumed
by task name. The comparison report includes missing and inadmissible tasks
as zero under the fixed split.
