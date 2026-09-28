"""Model client with per-role usage accounting.

Every model call in LEGO goes through ``LLM``. A model is named by an *alias*
declared in ``configs/models.yaml``::

    gpt-5.6-terra:
      provider: openai_responses      # openai_chat | openai_responses | anthropic | <plugin>
      model_id: gpt-5.6-terra
      base_url_env: OPENAI_BASE_URL   # optional; env var holding the endpoint
      api_key_env: OPENAI_API_KEY
      price_in: 0.0                   # USD per 1M input tokens (fill in)
      price_out: 0.0                  # USD per 1M output tokens (fill in)

An alias that is not declared is used verbatim as a model id on the default
provider (``LEGO_DEFAULT_PROVIDER``, default ``openai_chat``).

Each ``LLM`` carries a *role* (``backbone``, ``resident``, ``diagnosis``,
``retrieval``, ``decompose``) so that RQ5's cost table can attribute tokens to the
position that spent them. Usage is accumulated in ``USAGE`` and written into every
task record by the runner.
"""

from __future__ import annotations

import collections
import importlib
import json
import os
import re
import sys
import threading
import time

import yaml

MODELS_FILE = os.environ.get("LEGO_MODELS", "configs/models.yaml")

_AUTH_DEAD = (
    "invalid_api_key", "Signature expired", "InvalidSignatureException",
    "ExpiredToken", "UnrecognizedClientException", "AccessDeniedException",
    "Invalid Authentication", "authentication_error",
)

_lock = threading.Lock()


class Usage:
    """Thread-safe token/call counter keyed by (role, model alias)."""

    def __init__(self):
        self.reset()

    def reset(self):
        with _lock:
            self.calls = collections.Counter()
            self.tok_in = collections.Counter()
            self.tok_out = collections.Counter()
            self.failed = collections.Counter()

    def add(self, role, alias, tin, tout, ok=True):
        with _lock:
            k = f"{role}|{alias}"
            self.calls[k] += 1
            self.tok_in[k] += int(tin or 0)
            self.tok_out[k] += int(tout or 0)
            if not ok:
                self.failed[k] += 1

    def snapshot(self) -> dict:
        with _lock:
            keys = sorted(set(self.calls) | set(self.tok_in))
            return {k: {"calls": self.calls[k], "in": self.tok_in[k],
                        "out": self.tok_out[k], "failed": self.failed[k]}
                    for k in keys}


USAGE = Usage()


def _load_models() -> dict:
    if not os.path.exists(MODELS_FILE):
        return {}
    with open(MODELS_FILE) as fh:
        return yaml.safe_load(fh) or {}


MODELS = _load_models()


def model_spec(alias: str) -> dict:
    spec = dict(MODELS.get(alias) or {})
    spec.setdefault("provider",
                    os.environ.get("LEGO_DEFAULT_PROVIDER", "openai_chat"))
    spec.setdefault("model_id", alias)
    return spec


def price(alias: str) -> tuple[float, float]:
    s = model_spec(alias)
    return float(s.get("price_in") or 0.0), float(s.get("price_out") or 0.0)


# ---------------------------------------------------------------- providers
class _OpenAIChat:
    def __init__(self, spec):
        from openai import OpenAI
        kw = {}
        if spec.get("base_url_env") and os.environ.get(spec["base_url_env"]):
            kw["base_url"] = os.environ[spec["base_url_env"]]
        elif spec.get("base_url"):
            kw["base_url"] = spec["base_url"]
        key = os.environ.get(spec.get("api_key_env") or "OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=key or "EMPTY", timeout=600, max_retries=2,
                             **kw)
        self.model = spec["model_id"]
        self.extra = spec.get("extra_body") or {}

    def __call__(self, system, prompt, max_tokens, temperature):
        kw = {"temperature": temperature} if temperature is not None else {}
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max(max_tokens, 16),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            extra_body=self.extra or None, **kw)
        u = getattr(r, "usage", None)
        return (r.choices[0].message.content or "",
                getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


class _OpenAIResponses(_OpenAIChat):
    def __call__(self, system, prompt, max_tokens, temperature):
        kw = {"temperature": temperature} if temperature is not None else {}
        r = self.client.responses.create(
            model=self.model, instructions=system, input=prompt,
            max_output_tokens=max(max_tokens, 16), **kw)
        u = getattr(r, "usage", None)
        return (r.output_text or "", getattr(u, "input_tokens", 0),
                getattr(u, "output_tokens", 0))


class _Anthropic:
    def __init__(self, spec):
        import anthropic
        kw = {}
        if spec.get("base_url_env") and os.environ.get(spec["base_url_env"]):
            kw["base_url"] = os.environ[spec["base_url_env"]]
        self.client = anthropic.Anthropic(
            api_key=os.environ.get(spec.get("api_key_env") or
                                   "ANTHROPIC_API_KEY"), **kw)
        self.model = spec["model_id"]

    def __call__(self, system, prompt, max_tokens, temperature):
        kw = {"temperature": temperature} if temperature is not None else {}
        m = self.client.messages.create(
            model=self.model, system=system, max_tokens=max(max_tokens, 16),
            messages=[{"role": "user", "content": prompt}], **kw)
        text = "\n".join(b.text for b in m.content
                         if getattr(b, "type", "") == "text")
        return text, m.usage.input_tokens, m.usage.output_tokens


PROVIDERS = {"openai_chat": _OpenAIChat, "openai_responses": _OpenAIResponses,
             "anthropic": _Anthropic}


def _provider(spec):
    name = spec["provider"]
    if name in PROVIDERS:
        return PROVIDERS[name](spec)
    # plugin: "package.module:Class"
    mod, _, cls = name.partition(":")
    return getattr(importlib.import_module(mod), cls or "Provider")(spec)


def _is_credential_dead(err: BaseException) -> bool:
    text = f"{type(err).__name__}: {err}"
    return any(k in text for k in _AUTH_DEAD)


# ---------------------------------------------------------------- front end
class LLM:
    """One model in one pipeline position."""

    def __init__(self, alias: str, role: str = "backbone",
                 temperature: float | None = None):
        self.alias = alias
        self.role = role
        self.temperature = temperature
        self.spec = model_spec(alias)
        self._impl = None
        # A resident model instance belongs to one assessment or adaptation.
        # Keep its usage separately so successful adaptations can be priced
        # without including assessments or rejected adaptations.
        self.usage = {"calls": 0, "in": 0, "out": 0}

    @property
    def model(self) -> str:
        return self.alias

    def _call(self, system: str, prompt: str, max_tokens: int) -> str:
        if self._impl is None:
            self._impl = _provider(self.spec)
        for attempt in range(3):
            try:
                text, tin, tout = self._impl(system, prompt, max_tokens,
                                             self.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"[llm:{self.role}:{self.alias}] attempt {attempt + 1}/3 "
                      f"failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                if _is_credential_dead(e):
                    # A dead credential would otherwise be recorded as a model
                    # that writes empty modules. Exit before any record is
                    # written so the shard can be resumed with a fresh key.
                    print("[llm] FATAL: credential rejected; exiting without "
                          "writing a record", flush=True)
                    sys.stdout.flush()
                    os._exit(3)
                USAGE.add(self.role, self.alias, 0, 0, ok=False)
                time.sleep(4 * (attempt + 1))
                continue
            USAGE.add(self.role, self.alias, tin, tout, ok=bool(text))
            self.usage["calls"] += 1
            self.usage["in"] += int(tin or 0)
            self.usage["out"] += int(tout or 0)
            if text and text.strip():
                return text.strip()
            time.sleep(2)
        return ""

    def text(self, prompt: str, system: str = "You are a careful software "
             "engineer.", max_tokens: int = 4000) -> str:
        return self._call(system, prompt, max_tokens)

    def code(self, prompt: str, max_tokens: int = 8000) -> str:
        text = self._call("Output ONLY valid Python code. No markdown fences.",
                          prompt, max_tokens)
        return strip_fences(text)

    def json(self, prompt: str, max_tokens: int = 2000):
        text = self._call("Respond with ONLY a JSON object. No markdown.",
                          prompt, max_tokens)
        return parse_json(text)


def strip_fences(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def parse_json(text: str):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None


def preflight(aliases) -> None:
    """Fail fast (exit 1) if any model cannot generate. Run before a sweep."""
    for a in dict.fromkeys(x for x in aliases if x):
        out = LLM(a, role="preflight").code(
            "Write a function add(a, b) that returns a + b.", max_tokens=256)
        if "def add" not in out:
            print(f"[preflight] FAILED: {a} returned {out[:80]!r}", flush=True)
            sys.exit(1)
        print(f"[preflight] ok: {a}", flush=True)
