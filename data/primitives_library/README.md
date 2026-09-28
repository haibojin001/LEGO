# Archived primitive snippets

This directory stores the earlier flat primitive library: 1,424 Python source
files and their original `manifest.json`. The source files and manifest are
preserved as data. Each manifest row points to one source file and records its
historical source hints and labels.

The archive is separate from the reconstructed CodeFace library. To convert it
to the layout accepted by the current loader:

```bash
python -m lego.library.convert_legacy \
  --src data/primitives_library --out codeface_legacy
```

`codeface_legacy/` is generated locally and is excluded from Git. The original
manifest labels do not imply that this repository has rerun validation.
The manifest has no license inventory. Individual source files are third-party
code and may use the network; check source terms before public distribution and
use an isolated execution environment for experiments.
