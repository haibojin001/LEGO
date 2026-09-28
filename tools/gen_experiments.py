"""Regenerate experiments/*/experiment.yaml and README.md.

The experiment files are the source of truth for what each RQ runs; this
script only exists so that shared model lists stay consistent across them.

    python tools/gen_experiments.py
"""

import os

import yaml

B13 = ["gpt-5.6-terra", "claude-sonnet-5", "gpt-5.4", "claude-opus-5",
       "gpt-5.6-luna", "grok-4.3", "claude-haiku-4.5", "gemma-4-31b",
       "kimi-k2.5", "qwen3-coder-480b", "glm-5", "deepseek-v3.2",
       "nemotron-super-3-120b"]
SWAP9 = ["gpt-5.6-terra", "claude-sonnet-5", "gpt-5.4", "gpt-5.6-luna",
         "deepseek-v4", "gemma-4-31b", "glm-5", "qwen3-coder-480b", "gpt-oss-20b"]
SPLIT = "lego_repo_522"
E = {}


def exp(key, title, rq, arms, tables, readme, sweep=None, shards=32):
    d = {"name": key, "title": title, "rq": rq, "split": SPLIT,
         "shards": shards, "arms": arms}
    if sweep:
        d["sweep"] = sweep
    d["tables"] = tables
    E[key] = (d, readme)


def A(config, display, **kw):
    return {"config": config, "display": display, **kw}


exp("rq1_scratch", "Construction from scratch across 13 backbones", "RQ1",
    [A("feedback", "Feedback")],
    [{"type": "main"}, {"type": "domains"}, {"type": "cost"}],
    """Every backbone under the same execution-feedback configuration (no
library, b=5). Band columns, All, Dead@0 and Ceil@1 come from `main`; the
aggregate rows (proprietary / open-weight / field mean) are means of the
per-backbone rows. The single-pass score of each backbone is the b=1 prefix of
its trajectory (see `rq5_budget` for the ladder, or `trajectory[0]` of each
record).""",
    sweep={"models.backbone": B13})

exp("rq1_single_pass", "Single pass vs. execution feedback", "RQ1",
    [A("single_shot", "Single-shot"), A("feedback", "Feedback")],
    [{"type": "main"}, {"type": "paired", "ref": "Single-shot"}],
    """Single-shot is an independent b=1 run with the default backbone. The
configuration table also reads b=1 as the first-round prefix of the b=5
trajectory; `rq5_budget` reports that reading.""")

exp("rq1_external_agents", "Independently developed repository agents", "RQ1",
    [A("feedback", "Feedback"), A("lego", "LEGO")],
    [{"type": "main"}],
    """External systems (OpenHands, Claude Code, SWE-agent, Agentless) run under
their own released harness via the task adapter in `baselines/`; see
`baselines/README.md`. Their records are written to `runs/arms/ext_*` in the
same format, so after running them add them to the report with
`python -m lego.analysis.report experiments/rq1_external_agents/experiment.yaml`
plus `python -m baselines.report` (lists every `ext_*` arm next to these two).""")

exp("rq2_library", "CodeFace reach and the runtime adaptation funnel", "RQ2",
    [A("lego", "LEGO")],
    [{"type": "funnel", "arm": "LEGO"}, {"type": "main"}],
    """Two parts.

1. **Offline coverage** (no construction):
   `python -m lego.analysis.coverage --split lego_repo_522 --config lego --assess --model gpt-5.6-terra`
   writes per-module coverage and per-band / per-domain summaries to
   `results/coverage/`. Run it once per library view you report (e.g.
   `--set 'view.exclude=["same_repo","same_org","near_dup"]'`).
2. **Runtime funnel**: the `funnel` table aggregates `query / cand / retain /
   adapt_ok` from the LEGO run's records. *share* = adapt_ok / modules;
   *used* = adapt_ok / covered modules (covered from part 1).

Library composition by provenance kind:
`python -c "from lego.library.codeface import CodeFace; print(CodeFace('codeface').counts())"`.""")

exp("rq2_library_mix", "Library source, size and retrieval depth", "RQ2",
    [A("lego", "LEGO"), A("lego_mined_only", "mined only"),
     A("lego_mined_web", "mined+web"), A("lego_mined_harvested", "mined+harvested"),
     A("lego_web_only", "web only"), A("lego_harvested_only", "harvested only"),
     A("lego_half_s0", "half s0"), A("lego_half_s1", "half s1"),
     A("lego_half_s2", "half s2"), A("lego_m1", "top-1"), A("lego_m4", "top-4"),
     A("lego_m8", "top-8")],
    [{"type": "main"}, {"type": "paired", "ref": "LEGO"}],
    """Restricts CodeFace by provenance kind, keeps a seeded half of it (three
seeds), or changes retrieval depth m. Everything else is LEGO.""")

exp("rq3_paradigms", "Construction paradigms", "RQ3",
    [A("single_shot", "Single-shot"), A("retrieval_one", "Retrieval, one attempt"),
     A("feedback", "Feedback"), A("retrieval_feedback", "Retrieval + Feedback"),
     A("feedback_diag", "Feedback + Diagnosis"),
     A("retrieval_diag", "Retrieval + Diagnosis"),
     A("retrieval_adapt", "Retrieval + Adapt"), A("rag_edit", "RAG + Edit"),
     A("lego", "LEGO")],
    [{"type": "main"},
     {"type": "paired", "ref": "Feedback", "by_band": True},
     {"type": "paired", "ref": "Feedback + Diagnosis", "by_band": True,
      "name": "paired_vs_fbdiag"},
     {"type": "domains"}, {"type": "cost"}],
    """R/A/E/V/D = retrieval, primitive adaptation, generic editing, execution
feedback, diagnosis (b=5). Stage increments are differences between rows of
the `paired` tables.""")

exp("rq3_ablation", "Construction-stage attribution", "RQ3",
    [A("feedback", "Feedback"), A("feedback_diag", "+ diagnosis"),
     A("file_rag_edit", "+ File RAG, editing"),
     A("rag_edit", "+ constructed primitives"),
     A("lego", "+ primitive adaptation")],
    [{"type": "main"}, {"type": "paired", "ref": "Feedback"}],
    """Each row adds one mechanism. File RAG needs its index first:
`python -m lego.library.file_rag build --library codeface --out file_rag`
(same donor repositories, same embedder and depth as CodeFace). *step* is the
difference to the previous row; *% tot.* is step / (LEGO - Feedback).""")

exp("rq3_consumption", "How reused components are consumed", "RQ3",
    [A("feedback_diag", "Feedback + Diagnosis"), A("context_only", "Context-only"),
     A("import_call", "Import + Call"),
     A("structured_adapt", "Structured-schema adaptation"), A("lego", "LEGO")],
    [{"type": "main"},
     {"type": "paired", "ref": "Feedback + Diagnosis", "by_band": True}],
    """All library rows share LEGO's retrieval, relevance assessment, diagnosis
and budget. Import + Call vendors the primitive byte-identical under
`_lego_vendor/` outside the target prefix; Context-only shows it read-only and
never delivers a file from it.""")

exp("rq3_fields", "Primitive-field decomposition", "RQ3",
    [A("rag_edit", "RAG + Edit"), A("rag_edit_V", "+ V_i"),
     A("rag_edit_VIX", "+ I_i, X_i"), A("rag_edit_VIXD", "+ D_i"),
     A("lego_no_V", "LEGO, V_i withheld"), A("lego", "LEGO")],
    [{"type": "main"}, {"type": "paired", "ref": "RAG + Edit"}],
    """Adds the primitive's fields one at a time to scoped generic editing, and
withholds carried tests from full adaptation, at matched budget.""")

exp("rq3_provenance", "Donor provenance and contamination control", "RQ3",
    [A("feedback", "Feedback"), A("lego", "LEGO (same-repo donors excluded)"),
     A("lego_with_same_repo", "LEGO, same-repo donors allowed"),
     A("lego_no_same_org", "same-organization excluded"),
     A("lego_no_near_dup", "+ fork / near-duplicate excluded"),
     A("lego_remined", "re-mined disjoint library")],
    [{"type": "main"}, {"type": "paired", "ref": "Feedback", "by_band": True}],
    """**Default differs from the draft paper:** the released `lego` config
excludes primitives mined from the target's own repository, because such a
primitive can contain the answer. `lego_with_same_repo` reproduces the
same-repository-allowed setting, so the provenance effect is measured
explicitly rather than folded into the headline configuration.

The re-mined library must be built from a corpus disjoint from LEGO-REPO (no
benchmark repository, organization, fork, or MinHash>=0.6 near-duplicate):
`python -m lego.mining.mine --corpus corpora/disjoint.txt --out codeface_disjoint`.""")

exp("rq3_components", "Component removals", "RQ3",
    [A("lego", "LEGO"), A("retrieval_adapt", "- diagnosis"),
     A("retrieval_diag", "- adaptation"), A("feedback_diag", "- library"),
     A("lego_no_collab", "- collaboration (mu)"),
     A("lego_no_V", "- carried tests")],
    [{"type": "main"}, {"type": "paired", "ref": "LEGO"}],
    """Leave-one-out removals of LEGO's mechanisms.""")

exp("rq4_backbones", "Backbone substitution", "RQ4",
    [A("feedback", "Feedback"),
     A("lego", "LEGO", models={"resident": "gpt-5.6-terra",
                               "diagnosis": "gpt-5.6-terra"})],
    [{"type": "pairs", "a": "LEGO", "b": "Feedback"}, {"type": "main"}],
    """Adaptation and diagnosis stay on GPT-5.6-terra; only the backbone moves.
The Feedback arms are identical to `rq1_scratch` (identical configs share
records), so the two experiments together cost one set of Feedback runs.""",
    sweep={"models.backbone": B13})

exp("rq4_resident_swap", "Adaptation-model substitution", "RQ4",
    [A("lego", "LEGO")], [{"type": "main"}, {"type": "cost"}],
    """Backbone and diagnosis fixed to the default; the resident model that
performs ASSESS/ADAPT is swept. The *no primitive* reference row is
`retrieval_diag` in `rq3_paradigms`.""",
    sweep={"models.resident": SWAP9})

exp("rq4_diagnosis_swap", "Diagnosis-model substitution", "RQ4",
    [A("lego", "LEGO")], [{"type": "main"}, {"type": "cost"}],
    """Backbone and adaptation fixed to the default; the diagnosis model is swept.
The *no diagnosis* reference row is `retrieval_adapt` in `rq3_paradigms`.""",
    sweep={"models.diagnosis": SWAP9})

GRID = [("gpt-5.6-terra", "gpt-5.6-terra", "gpt-5.6-terra"),
        ("gpt-5.6-terra", "claude-sonnet-5", "claude-sonnet-5"),
        ("gpt-5.6-terra", "deepseek-v4", "deepseek-v4"),
        ("gpt-5.6-terra", "gpt-oss-20b", "gpt-oss-20b"),
        ("claude-sonnet-5", "claude-sonnet-5", "claude-sonnet-5"),
        ("claude-sonnet-5", "gpt-oss-20b", "gpt-oss-20b"),
        ("deepseek-v3.2", "gpt-5.6-terra", "gpt-5.6-terra"),
        ("deepseek-v3.2", "deepseek-v4", "deepseek-v4"),
        ("gpt-oss-20b", "gpt-oss-20b", "gpt-oss-20b")]
exp("rq5_allocation", "Quality-cost trade-off under model allocation", "RQ5",
    [A("lego", f"{c} / {a} / {d}",
       models={"backbone": c, "resident": a, "diagnosis": d}) for c, a, d in GRID]
    + [A("feedback_diag", "Feedback + Diagnosis, b=5"),
       A("feedback", "Feedback, b=8", budget=8)],
    [{"type": "cost"}, {"type": "main"},
     {"type": "paired", "ref": "gpt-5.6-terra / gpt-5.6-terra / gpt-5.6-terra"}],
    """c/a/d = backbone / adaptation / diagnosis model. Costs are recorded-call
costs: per-role token usage stored in every record x the prices in
`configs/models.yaml` (fill in `price_in` / `price_out` per 1M tokens before
running the report). Infrastructure cost is excluded.""")

exp("rq5_budget", "Attempt-budget ladder", "RQ5",
    [A("feedback", "Feedback"), A("feedback", "Feedback, b=8", budget=8),
     A("lego", "LEGO")],
    [{"type": "ladder", "arm": "Feedback", "budgets": [1, 2, 3, 4, 5]},
     {"type": "ladder", "arm": "Feedback, b=8", "budgets": list(range(1, 9)),
      "name": "ladder_b8"},
     {"type": "ladder", "arm": "LEGO", "budgets": [1, 2, 3, 4, 5],
      "name": "ladder_lego"},
     {"type": "cost"}],
    """b=1..5 are read from the per-round trajectories of the b=5 runs (a b=k run
is a prefix of the b=5 run). b=8 is a separate run.""")

exp("rq6_transfer", "Transfer to external benchmarks", "RQ6",
    [A("feedback", "Feedback"), A("lego", "LEGO")], [],
    """Runs under each external benchmark's own tasks, metric and grader via
`external/`; see `external/README.md`. The arms listed here only fix the two
configurations compared (`feedback` = matched execution-feedback baseline).
CodeFace is filtered per benchmark before any task runs
(`python -m external.filter_library`), and offline coverage on each benchmark
is reported by `python -m external.coverage`.""")

exp("app_replicates", "Replicate runs", "Appendix",
    [A(c, d) for c, d in [("single_shot", "Single-shot"), ("feedback", "Feedback"),
                          ("feedback_diag", "Feedback + Diagnosis"),
                          ("rag_edit", "RAG + Edit"),
                          ("lego_no_near_dup", "LEGO, donor-disjoint"),
                          ("lego", "LEGO")]],
    [{"type": "main"}],
    """Three runs per configuration. Replicates differ only in sampling
(`replicate` is part of the arm id, so each replicate keeps its own records).
Set `temperature` in the arms if your endpoints are deterministic by default.""",
    sweep={"replicate": [0, 1, 2]})

exp("app_strata", "Contamination strata (star count)", "Appendix",
    [A("feedback", "Feedback"), A("lego", "LEGO")],
    [{"type": "paired", "ref": "Feedback",
      "splits": ["stars_ge10k", "stars_3k_10k", "stars_1k_3k"]}],
    """Same arms as `rq3_paradigms`; only the table differs. A post-cutoff subset
can be added as another split file.""")

exp("app_band_sensitivity", "Difficulty-band sensitivity and scoring checks",
    "Appendix",
    [A("feedback", "Feedback"), A("lego", "LEGO"),
     A("feedback_diag", "Feedback + Diagnosis")],
    [{"type": "paired", "ref": "Feedback", "by_band": True}],
    """Band definition (Eq. 6) versus two size-only binnings, span weighting,
zero-floor tasks and raw pass rate:
`python -m lego.analysis.robustness experiments/app_band_sensitivity/experiment.yaml --a LEGO --b Feedback`""")

exp("app_special_casing", "Test-targeted special-casing audit", "Appendix",
    [A("feedback", "Feedback"), A("lego", "LEGO")], [],
    """Band-stratified sample of 120 delivered trees per configuration, two
independent annotators, Cohen's kappa:

```bash
python -m lego.analysis.audit sample --arm-dir runs/arms/<lego arm id> --n 120 --out audit/lego
python -m lego.analysis.audit sample --arm-dir runs/arms/<feedback arm id> --n 120 --out audit/feedback
# two annotators fill copies of annotations_template.csv independently, then
python -m lego.analysis.audit score audit/lego/annotator_a.csv audit/lego/annotator_b.csv
```

Arm ids: `python -m lego.run --config lego --print-arm`.""")


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "experiments")
    index = ["# Experiments\n",
             "Each directory holds `experiment.yaml` (arms, sweep, tables) and a "
             "README with the exact commands. Identical arm configurations are "
             "shared across experiments (records are keyed by arm id), so an arm "
             "listed in several experiments runs once.\n",
             "| experiment | RQ | what | arms x sweep |", "|---|---|---|---|"]
    for key, (d, readme) in E.items():
        os.makedirs(os.path.join(root, key), exist_ok=True)
        with open(os.path.join(root, key, "experiment.yaml"), "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False, width=100)
        rows = "\n".join(f"| {a.get('display', a['config'])} | `{a['config']}` |"
                         for a in d["arms"])
        sweep = ""
        n_pts = 1
        if d.get("sweep"):
            for v in d["sweep"].values():
                n_pts *= len(v)
            sweep = ("\n**Sweep** (applied to every arm): " + "; ".join(
                f"`{k}` over {', '.join(map(str, v))}"
                for k, v in d["sweep"].items()) + "\n")
        runs = "" if key in ("rq6_transfer", "rq1_external_agents") else f"""
## Run

```bash
python -m lego.launch experiments/{key}/experiment.yaml --list     # jobs
python -m lego.launch experiments/{key}/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/{key}/experiment.yaml             # SLURM array
python -m lego.launch experiments/{key}/experiment.yaml --status   # progress
```
"""
        tables = "" if not d["tables"] else f"""
## Tables

```bash
python -m lego.analysis.report experiments/{key}/experiment.yaml --out results
```

writes `results/{key}/*.md|csv|tex`.
"""
        with open(os.path.join(root, key, "README.md"), "w") as fh:
            fh.write(f"# {d['rq']}: {d['title']}\n\n{readme}\n\n| arm | config |\n"
                     f"|---|---|\n{rows}\n{sweep}{runs}{tables}")
        index.append(f"| [`{key}`]({key}/) | {d['rq']} | {d['title']} | "
                     f"{len(d['arms'])} x {n_pts} |")
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write("\n".join(index) + "\n")
    print(f"{len(E)} experiments -> {root}")


if __name__ == "__main__":
    main()
