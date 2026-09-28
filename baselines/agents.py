"""External repository agents, each driven through its own released entry point.

One runner per system (``openhands``, ``claude_code``, ``swe_agent``,
``agentless``). A runner is a command template from ``baselines/agents.yaml``;
it (1) fills in the workspace, the instruction and the backbone, (2) runs the
command in the workspace under a wall-clock limit, killing the whole process
group on expiry, (3) for systems that deliver a patch instead of editing the
tree, applies the newest patch they wrote, and (4) records the agent version,
exit code, wall time and whatever token/cost figures the tool reports. The
system's scaffolding, tool set and stopping rules are not touched.

Usage parsing is best effort: JSON on stdout and in ``usage_files`` is scanned
for the keys the known tools use (cumulative values; the largest is kept), and
the raw logs stay in ``DIR/agent/`` next to ``agent.json``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import signal
import subprocess
import tempfile
import time

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "agents.yaml")
SYSTEMS = ("openhands", "claude_code", "swe_agent", "agentless")

_COST = ("total_cost_usd", "accumulated_cost", "instance_cost", "total_cost",
         "cost_usd")
_IN = ("input_tokens", "prompt_tokens", "tokens_sent")
_OUT = ("output_tokens", "completion_tokens", "tokens_received")
_CACHE = ("cache_read_input_tokens", "cache_creation_input_tokens")
_CALLS = ("api_calls", "num_turns")
_COST_TEXT = re.compile(r"(?i)(?:total|accumulated)[ _]cost[^0-9$\n]{0,20}"
                        r"\$?\s*([0-9]+(?:\.[0-9]+)?)")


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh) or {}


def system_spec(system: str, path: str | None = None) -> dict:
    cfg = load_config(path)
    systems = cfg.get("systems") or {}
    if system not in systems:
        raise SystemExit(f"unknown system {system!r}; known: {sorted(systems)}")
    spec = dict(cfg.get("defaults") or {})
    spec.update(systems[system])
    spec["name"] = system
    return spec


def model_id(spec: dict, alias: str) -> str:
    m = (spec.get("models") or {}).get(alias)
    if m:
        return str(m)
    from lego.llm.client import model_spec
    return str(model_spec(alias).get("model_id") or alias)


class _Fields(dict):
    def __missing__(self, k):
        raise KeyError(f"unknown template field {{{k}}}")


def render(template, fields: dict):
    if isinstance(template, list):
        return [render(x, fields) for x in template]
    return str(template).format_map(_Fields(fields))


_VERSIONS: dict = {}


def version(spec: dict) -> str | None:
    """First line printed by ``version_cmd`` (cached per process)."""
    cmd = spec.get("version_cmd")
    if not cmd:
        return None
    key = json.dumps(cmd)
    if key not in _VERSIONS:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            txt = (r.stdout or r.stderr).strip().splitlines()
            _VERSIONS[key] = txt[0][:200] if txt and r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            _VERSIONS[key] = None
    return _VERSIONS[key]


def _kill(p: subprocess.Popen):
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ---------------------------------------------------------------- usage
def _json_objects(text: str):
    text = (text or "").strip()
    if not text:
        return
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _scan(obj, found: dict, depth: int = 0):
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                found.setdefault(k, []).append(v)
            else:
                _scan(v, found, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:5000]:
            _scan(v, found, depth + 1)


def parse_usage(stdout: str, log_dir: str, patterns=()) -> dict:
    """Token and cost figures the tool reported, or {} if none parse."""
    found: dict = {}
    for o in _json_objects(stdout):
        _scan(o, found)
    for pat in patterns or ():
        for f in glob.glob(os.path.join(log_dir, pat), recursive=True):
            if not os.path.isfile(f) or os.path.getsize(f) > 64 * 1024 * 1024:
                continue
            try:
                text = open(f, errors="ignore").read()
            except OSError:
                continue
            for o in _json_objects(text):
                _scan(o, found)

    def top(keys):
        vals = [v for k in keys for v in found.get(k, [])]
        return max(vals) if vals else None
    out = {"cost_usd": top(_COST), "tokens_in": top(_IN),
           "tokens_out": top(_OUT), "calls": top(_CALLS)}
    for k in _CACHE:
        if found.get(k):
            out[k] = max(found[k])
    if out["cost_usd"] is None:
        m = _COST_TEXT.findall(stdout or "")
        if m:
            out["cost_usd"] = float(m[-1])
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------- patches
def _patch_text(path: str) -> str:
    if path.endswith((".json", ".jsonl")):
        found = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "model_patch" and isinstance(v, str) and v.strip():
                        found.append(v)
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        for o in _json_objects(open(path, errors="ignore").read()):
            walk(o)
        return found[-1] if found else ""
    return open(path, errors="ignore").read()


def apply_patch(tree: str, patterns: list[str]) -> dict:
    """Apply the newest non-empty patch matching ``patterns`` to ``tree``."""
    cands = sorted({f for p in patterns for f in glob.glob(p, recursive=True)
                    if os.path.isfile(f)}, key=os.path.getmtime, reverse=True)
    for f in cands:
        text = _patch_text(f)
        if not text.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".diff",
                                         delete=False) as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
            tmp = fh.name
        try:
            for cmd in (["git", "-C", tree, "apply", "--whitespace=nowarn", tmp],
                        ["patch", "-p1", "-d", tree, "-i", tmp, "--batch"]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    return {"applied": True, "source": f, "bytes": len(text),
                            "tool": cmd[0]}
            return {"applied": False, "source": f, "bytes": len(text),
                    "error": (r.stderr or r.stdout)[-500:]}
        finally:
            os.remove(tmp)
    return {"applied": False, "source": None, "error": "no patch found"}


# ---------------------------------------------------------------- run
def run(system: str, meta: dict, model: str, timeout: int | None = None,
        config: str | None = None) -> dict:
    """Run one system in the workspace described by ``meta`` (the dict
    ``baselines.workspace.build`` returns). Writes ``DIR/agent/agent.json``."""
    spec = system_spec(system, config)
    timeout = int(timeout or spec.get("timeout") or 3600)
    out = meta["out"]
    log_dir = os.path.join(out, "agent")
    os.makedirs(log_dir, exist_ok=True)
    instruction = open(meta["instruction"]).read()
    fields = {"repo": meta["tree"], "instruction_file": meta["instruction"],
              "instruction": instruction, "model": model,
              "model_id": model_id(spec, model), "log_dir": log_dir,
              "timeout": timeout, "task": meta["task"], "out": out}
    cmd = render(spec["cmd"], fields)
    env = dict(os.environ)
    for k, v in (spec.get("env") or {}).items():
        env[k] = os.path.expandvars(render(v, fields))
    agent_bin = (meta.get("env") or {}).get("agent_bin")
    if spec.get("runs_on", "host") == "host" and agent_bin:
        env["PATH"] = agent_bin + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = os.path.dirname(agent_bin)
    stdin = instruction if spec.get("stdin") == "instruction" else None
    info = {"system": system, "version_pin": spec.get("version_pin"),
            "version": version(spec), "model": model,
            "model_id": fields["model_id"], "mode": spec.get("mode"),
            "cmd": [c if c != instruction else "<instruction>" for c in cmd],
            "timeout": timeout, "delivery": spec.get("delivery", "tree")}
    so_path = os.path.join(log_dir, "stdout.log")
    se_path = os.path.join(log_dir, "stderr.log")
    t0 = time.time()
    rc, timed_out, err = None, False, None
    with open(so_path, "w") as so, open(se_path, "w") as se:
        try:
            p = subprocess.Popen(cmd, cwd=render(spec.get("cwd", "{repo}"),
                                                 fields),
                                 env=env, stdout=so, stderr=se, text=True,
                                 stdin=subprocess.PIPE if stdin is not None
                                 else subprocess.DEVNULL,
                                 start_new_session=True)
        except OSError as e:
            p, err = None, f"{type(e).__name__}: {e}"
        if p is not None:
            try:
                p.communicate(input=stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(p)
            rc = p.returncode
    info.update(exit_code=rc, timed_out=timed_out,
                wall_seconds=round(time.time() - t0, 1))
    if err:
        info["error"] = err
    stdout = open(so_path, errors="ignore").read()
    info["usage"] = parse_usage(stdout, log_dir, spec.get("usage_files") or [])
    if info["delivery"] == "patch":
        info["patch"] = apply_patch(meta["tree"],
                                    render(spec.get("patch_glob") or [], fields))
    info["logs"] = {"stdout": so_path, "stderr": se_path}
    with open(os.path.join(log_dir, "agent.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info
