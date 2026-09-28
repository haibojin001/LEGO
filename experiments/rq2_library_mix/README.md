# RQ2: Library source, size and retrieval depth

Restricts CodeFace by provenance kind, keeps a seeded half of it (three
seeds), or changes retrieval depth m. Everything else is LEGO.

| arm | config |
|---|---|
| LEGO | `lego` |
| mined only | `lego_mined_only` |
| mined+web | `lego_mined_web` |
| mined+harvested | `lego_mined_harvested` |
| web only | `lego_web_only` |
| harvested only | `lego_harvested_only` |
| half s0 | `lego_half_s0` |
| half s1 | `lego_half_s1` |
| half s2 | `lego_half_s2` |
| top-1 | `lego_m1` |
| top-4 | `lego_m4` |
| top-8 | `lego_m8` |

## Run

```bash
python -m lego.launch experiments/rq2_library_mix/experiment.yaml --list     # jobs
python -m lego.launch experiments/rq2_library_mix/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/rq2_library_mix/experiment.yaml             # SLURM array
python -m lego.launch experiments/rq2_library_mix/experiment.yaml --status   # progress
```

## Tables

```bash
python -m lego.analysis.report experiments/rq2_library_mix/experiment.yaml --out results
```

writes `results/rq2_library_mix/*.md|csv|tex`.
