"""Repositories excluded from LEGO-REPO and from CodeFace.

The same set gates two doors: the benchmark task list, and the primitive library
(an entry harvested or mined from a held repository is never served).
"""

from __future__ import annotations

import re

# Offensive-security / anti-detection / provenance-defeating repos. Over-exclude
# on purpose: missing one costs far more than dropping one library from 565.
HELD_REPOS = frozenset("""
bbot sqlmap flaresolverr undetected-chromedriver qiling
pentestgpt agent-cloakbrowser faraday bunkerweb checkov
agent-decepticon rsactftool
jumpserver agent-integuru agent-ai-infra-guard snoop
agent-pentestagent agent-agentic_security one-lin3r bypass-url-parser
ufonet crypto-attacks featherduster xortool crypton
agent-hcaptcha-challenger sherlock agent-hexstrike-ai cai objection
maigret pydoll oletools remove-ai-watermarks
""".split())

# Every shape a repo name takes in a manifest entry. Exact-token extraction, not
# substring matching: `cai` is a held repo and a substring of "cairo", "cached"
# and "certificate", so a substring test would quarantine a large slice of a
# legitimate library and teach everyone to switch the check off.
_FORMS = (
    re.compile(r"^\(verified:repo_(?P<n>[^)]+)\)$"),   # from_repo, harvested
    re.compile(r"^\(verified:(?P<n>[^)]+)\)$"),        # from_repo, synthesized
    re.compile(r"^repo_(?P<n>.+)$"),                   # used_in_case
    re.compile(r"^[^/]+/(?P<n>[^/]+)$"),               # from_repo "owner/name"
    re.compile(r"^(?P<n>[^/()]+)$"),                   # bare name
)
_FILE = re.compile(r"^verified__repo_(?P<n>.+?)__")    # file basename


def provenance_repos(entry: dict) -> set[str]:
    """Every repo name a manifest entry claims to come from."""
    out = set()
    for field in ("from_repo", "used_in_case"):
        val = (entry.get(field) or "").strip()
        if not val:
            continue
        for pat in _FORMS:
            m = pat.match(val)
            if m:
                out.add(m.group("n"))
                break
    m = _FILE.match((entry.get("file") or "").split("/")[-1])
    if m:
        out.add(m.group("n"))
    # from_path is the path INSIDE the source repo ("objection/commands/ios/
    # jailbreak.py"), so its first component is the package, which for most
    # repos equals the repo name. Included because it is the only provenance a
    # hand-added entry may carry.
    head = (entry.get("from_path") or "").split("/")[0]
    if head and not head.endswith(".py"):
        out.add(head)
    return out


def is_held_primitive(entry: dict, held=HELD_REPOS) -> str | None:
    """The held repo this entry came from, or None. Truthy = quarantine it."""
    for name in provenance_repos(entry):
        if name in held:
            return name
    return None
