# LEGO: Large-scale Repository Engineering via Agent-Native Reusable Code Primitives

Code for the paper *Large-scale Repository Engineering via Agent-Native Reusable
Code Primitives*: the **LEGO** construction pipeline, the **CodeFace** library of
Code Primitives (and the pipeline that mines it), the **LEGO-REPO** benchmark
harness, and one runnable definition per experiment.

```
lego/            the package
  harness/       clone, per-task environment, native-suite execution, provenance,
                 interface stubs, scoring (Eq. 7-8)
  library/       Code Primitive format, CodeFace views and retrieval, resident
                 model (ASSESS / ADAPT + carried validation), File RAG control
  pipeline/      DECOMPOSE, retrieval + activation, adaptation + collaboration,
                 integration, execution feedback, diagnosis (Sec. 3.4)
  mining/        CodeFace construction: mined, harvested and web-sourced primitives
  analysis/      tables, paired statistics, coverage audit, special-casing audit
  prompts/       the three system prompts (activation, primitive, diagnosis)
  config.py      every configuration of the paper as a named RunConfig
  run.py         run one configuration over a task list
  launch.py      expand an experiment file into jobs
experiments/     one directory per RQ / appendix experiment (yaml + README)
benchmark/       task index builder, splits, repository metadata, commit pins
data/            archived flat primitive snippets and their original manifest
baselines/       external repository agents on LEGO-REPO (task adapter + grading)
external/        transfer to external benchmarks (adapters, library filtering)
configs/         model aliases and endpoints
slurm/           generic SLURM array launcher
tests/           smoke test (all construction modes, mock models) and unit tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add [embed] for sentence-transformers retrieval
pip install --user uv              # recommended: per-task interpreters/venvs
python tests/smoke.py              # every construction mode end to end, mock models
```

The harness builds a fresh virtualenv per task, with the interpreter the
repository declares (`requires-python` / classifiers; 3.12 by default), using
`uv` when available. Tasks need `git` and network access to GitHub and PyPI.

## 1. Benchmark

A task is a frozen directory `benchmark/tasks/<name>/` (task statement, native
tests, pinned commit and environment, and the grading key as node-id lists) plus
one line of `benchmark/tasks.jsonl`.
The fixed [522-task split](benchmark/splits/README.md) is checked in. Frozen
task directories are supplied separately. The checked-in inventory records frozen
task metadata without source code or grading node IDs.
The [522-row inventory](benchmark/lego_repo_inventory.csv) lists the frozen
task metadata without source code or grading node IDs.

```bash
# place the frozen task directories at benchmark/tasks/, then
python benchmark/build_index.py           # tasks.jsonl + available, dev_25, strata; checks fixed 522 IDs
python benchmark/export_inventory.py       # rebuild the 522-row CSV
python tools/audit_tasks_security.py      # held-repository and content audit
```

To rebuild task directories from scratch (clone, environment, identity ceiling,
empty-package floor; no model calls): `python -m lego.bench_build --cards
benchmark/metadata.jsonl --out benchmark/tasks`.

## 2. CodeFace

```bash
python -m lego.mining.source_plan \
  --manifest data/primitives_library/manifest.json \
  --out corpora/codeface_sources.txt
python -m lego.mining.mine --corpus corpora/codeface_sources.txt \
  --out codeface --describe --synth-tests --model <model-alias>
python -m lego.analysis.coverage --split lego_repo_522 --config lego \
  --assess --out results/coverage
python -m lego.mining.web_plan \
  --coverage results/coverage/<arm>_assess.jsonl \
  --out corpora/web_requests.txt
python -m lego.mining.web --requests corpora/web_requests.txt \
  --out codeface --describe --synth-tests --model <model-alias>
python -m lego.mining.harvest --arm runs/arms/<separate-construction-pass> \
  --out codeface --describe --synth-tests --model <model-alias>
python -m lego.library.file_rag build --library codeface --out file_rag   # File RAG control
python -m lego.library.audit_snapshot --library codeface --paper
```

Primitive records contain the source text, provenance kind, donor repository,
commit, paths, license metadata, optional descriptions, and carried validation
tests when available. Harvested primitives must come from a construction pass
that is disjoint from the evaluation runs; the library is read-only during
evaluation. The source planner resolves donor repositories from the old flat
manifest and writes unresolved hints to a separate report.
The separate donor pass is defined in
[experiments/codeface_harvest_seed/](experiments/codeface_harvest_seed/).
It does not turn snippets into primitives. A new run may produce a different
entry count and source mix; the paper audit must pass before its results can be
compared as a replication.

The older flat snippet directory is checked in at `data/primitives_library/`.
Convert it for an exploratory run:

```bash
python -m lego.library.convert_legacy --src data/primitives_library --out codeface_legacy
LEGO_LIBRARY=codeface_legacy LEGO_ALLOW_UNVALIDATED=1 \
  python -m lego.run --config lego --split dev_25
```

The archived 1,424 snippets preserve the earlier source text and manifest.
The converted proxy records per-entry syntax status; the archive does not
include carried validation tests. See
[data/primitives_library/README.md](data/primitives_library/README.md).

## 3. Models

Edit `configs/models.yaml`: every alias used by an experiment maps to a provider
(`openai_chat`, `openai_responses`, `anthropic`, or a plugin class), a model id,
the environment variables holding endpoint and key, and prices for the cost
tables. Open-weight models can be served with any OpenAI-compatible server.

## 4. Run an experiment

```bash
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --local    # run here
bash slurm/submit.sh experiments/rq3_paradigms/experiment.yaml             # or SLURM
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --status
python -m lego.analysis.report experiments/rq3_paradigms/experiment.yaml   # tables
```

A single configuration can also be run directly:

```bash
python -m lego.run --config lego --split dev_25 --backbone gpt-5.6-terra
python -m lego.run --config feedback --split lego_repo_522 --shard 3 --nshards 32
python -m lego.run --config lego --print-arm        # resolved config and arm id
```

## License

Benchmark tasks and CodeFace entries are derived from third-party repositories
and remain under their upstream licenses; provenance (repository, commit,
paths, license) is recorded for every task and primitive. The authors should
set the release license before publication.
