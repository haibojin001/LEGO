# Appendix: Test-targeted special-casing audit

Band-stratified sample of 120 delivered trees per configuration, two
independent annotators, Cohen's kappa:

```bash
python -m lego.analysis.audit sample --arm-dir runs/arms/<lego arm id> --n 120 --out audit/lego
python -m lego.analysis.audit sample --arm-dir runs/arms/<feedback arm id> --n 120 --out audit/feedback
# two annotators fill copies of annotations_template.csv independently, then
python -m lego.analysis.audit score audit/lego/annotator_a.csv audit/lego/annotator_b.csv
```

Arm ids: `python -m lego.run --config lego --print-arm`.

| arm | config |
|---|---|
| Feedback | `feedback` |
| LEGO | `lego` |

## Run

```bash
python -m lego.launch experiments/app_special_casing/experiment.yaml --list     # jobs
python -m lego.launch experiments/app_special_casing/experiment.yaml --local    # all jobs here
bash slurm/submit.sh experiments/app_special_casing/experiment.yaml             # SLURM array
python -m lego.launch experiments/app_special_casing/experiment.yaml --status   # progress
```
