# Running on SLURM

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
export LEGO_ENV_FILE=$PWD/keys.sh          # exports the API keys named in configs/models.yaml
bash slurm/submit.sh experiments/rq3_paradigms/experiment.yaml --partition=<cpu partition>
python -m lego.launch experiments/rq3_paradigms/experiment.yaml --status
```

* One array task runs one shard of one arm; tasks inside a shard run serially,
  each in its own per-task virtualenv under `$LEGO_SCRATCH`.
* Per-task environments are built with `uv` when it is on `PATH`
  (`pip install --user uv`), which also provisions interpreters the node does
  not have. Without `uv` the node's own Python is used.
* Point `LEGO_SCRATCH_BASE` at a disk with a few GB free per concurrent task
  (machine-learning repositories install large wheels); if `/tmp` is a small
  shared tmpfs, use a shared filesystem path instead.
* A dead API credential makes the task exit with code 3 *before* writing a
  record, so resubmitting with a fresh key resumes cleanly.
