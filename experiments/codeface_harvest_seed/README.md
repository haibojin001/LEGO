# CodeFace harvest donor pass

This is a separate, no-library construction pass used only to generate
candidate implementations for the harvested part of CodeFace. Its `replicate`
value gives the pass its own arm and records; these records are not paper
evaluation observations. The selection step still requires a completed task
and a passing isolated carried test for each admitted component.

On the cluster:

```bash
bash cluster/submit.sh experiments/codeface_harvest_seed/experiment.yaml
python -m lego.launch experiments/codeface_harvest_seed/experiment.yaml --status
sbatch cluster/harvest.sbatch
python -m lego.library.audit_snapshot --library codeface --paper
```

Submit `harvest.sbatch` after the donor construction pass completes. It is
resumable by task through the harvest log. Run the paper evaluation only after
the mined, harvested, and web-sourced library has been frozen.
