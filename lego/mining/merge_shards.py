"""Combine independently mined SLURM shards into one CodeFace directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def merge(shards: str, out: str) -> dict:
    source = Path(shards)
    dest = Path(out)
    parts = sorted(p for p in source.glob("shard_*") if p.is_dir())
    if not parts:
        raise ValueError(f"no mining shards under {source}")
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    identical = 0
    rows = []
    for shard in parts:
        for primitive in sorted(shard.iterdir()):
            if not primitive.is_dir() or not (
                    primitive / "primitive.json").is_file():
                continue
            target = dest / primitive.name
            if target.exists():
                if _digest(target) != _digest(primitive):
                    raise ValueError(f"primitive collision: {primitive.name}")
                identical += 1
            else:
                shutil.copytree(primitive, target)
                copied += 1
        log = shard / "_mining_log.jsonl"
        if log.exists():
            rows.extend(log.read_text().splitlines())
    (dest / "_mining_log.jsonl").write_text(
        "\n".join(rows) + ("\n" if rows else ""))
    return {"shards": len(parts), "copied": copied, "identical": identical,
            "log_rows": len(rows), "total": len(list(dest.glob(
                "*/primitive.json")))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(merge(a.shards, a.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
