# Post-cutoff tasks

Tasks built from public repositories created after the knowledge cutoff of GPT-5.6-terra, the backbone of the
contamination analysis in the appendix of the paper. `inventory.csv` lists the 106 tasks with the same columns as
`../lego_repo_inventory.csv`, plus the creation date, the first-commit date and the license of each repository.
As for the 522 LEGO-REPO tasks, the frozen task directories are supplied separately; each row pins the commit,
package, source prefix, test path, floor and ceiling needed to rebuild the task with `python -m lego.bench_build`.

## Selection

The rules below were fixed before any system was run on these tasks.

1. **Discovery** (GitHub search on 2026-11-13, `selection/discover.py`): public repositories whose primary language
   is Python, with at least 300 stars, not archived, created on or after 2026-02-17, the day after the
   2026-02-16 knowledge cutoff listed for GPT-5.6-terra on OpenAI's model page. 2,992 repositories.
2. **Screen** (read-only, `selection/screen.py`): not a fork or template; a license that GitHub identifies; not
   held by the LEGO-REPO security screen (`HELD_REPOS` and the pattern of `tools/audit_tasks_security.py`); not a
   LEGO-REPO target or a CodeFace donor; a packaging file, an importable package and native test files; a first
   commit on or after 2026-02-17 and no PyPI release of the same distribution name before that date, so that the
   code itself is new. 788 repositories (`cards.jsonl`).
3. **Build** (`python -m lego.bench_build`, the LEGO-REPO procedure): clone, per-task environment, native suite,
   identity ceiling and empty-package floor. A task is admitted when the identity run's provenance is `ok`,
   c - f >= 5 and the package has at most 300 modules. The harness's own test tooling is held below pytest 9.2
   (`selection/constraints.txt`): pytest 9.2.0 (2026-10-27) turns a deprecation into a collection error and
   postdates LEGO-REPO. Each repository is attempted once, in a fixed order (sorted by name and interleaved over 32
   job shards) that does not depend on any result. The set was frozen when 216 of the 788 screened repositories
   had been processed: 109 passed the build, and the post-build run of `tools/audit_tasks_security.py`'s pattern
   held 3 of them, leaving 106 tasks.
4. **Provenance**: no CodeFace entry (mined, harvested or web-sourced) comes from any of the 106 repositories,
   checked against `corpora/codeface_sources.txt` and the CodeFace manifest.

## Files

- `inventory.csv`: the 106 tasks.
- `splits/post_cutoff.txt`: their names.
- `cards.jsonl`: the 788 screened repositories.
- `funnel.csv`: every discovered repository, with the stage where it stopped and why (screen reason, build status,
  post-build hold, or outside the released subset).
- `selection/`: the scripts and the job file used for steps 1-3.

## Shape

Difficulty bands D1-D5 (Eq. 6): 17 / 24 / 38 / 20 / 7 tasks. Median module count 38. Every repository has at
least 300 stars and was created, and received its first commit, on 2026-02-18 or later.
