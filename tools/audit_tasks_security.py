#!/usr/bin/env python3
"""Audit the BUILT tasks for offensive-security content, not the repo list.

The first audit matched only `name`/`note` in `repos.py` and ran once.
`agent-decepticon` ("Autonomous Hacking Agent for Red Team") and `rsactftool`
(an RSA attack tool for CTFs) both got as far as a materialized task directory
before a wider pattern caught them — so the invariant this enforces is about
`benchmark/tasks/`, which is what a run actually reads, and it is meant to be
re-run after every sweep rather than trusted from last time.

Two independent checks, because either alone has already failed:

  1. HOLD_REPOS ∩ built task dirs must be empty. Cheap and exact: a held repo
     with a task dir means the hold did not take effect (a stale directory from
     before the hold, or a sweep run with LEGO_HOLD_REPOS overridden).
  2. Every built task's own text — name, task.json, the description, and the
     upstream URL — is pattern-matched. This is the net that catches a repo
     nobody thought to add to HOLD_REPOS.

Check 2 has false positives by design (it is tuned to over-match, since missing
one costs more than reviewing one); they are printed separately from check 1's
hard failures. Exit code is non-zero only when check 1 fails or an unreviewed
pattern hit appears.

  python3 tools/audit_tasks_security.py
  python3 tools/audit_tasks_security.py --tasks benchmark/tasks --quiet
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lego.harness.core import HOLD_REPOS  # noqa: E402
from lego.benchmark_meta import is_held_primitive  # noqa: E402

# Word-boundaried. The first version of this had bare `rce` and `ids`, which
# matched inside "open-sou(rce)" and "prov(ids)" and produced 103 hits of which
# 87 were noise — a check whose output nobody reads is not a check.
PATTERN = re.compile(r"\b("
    r"pentest\w*|penetration.test\w*|privesc|priv.escalation|exploit\w*|"
    r"payloads?|malware|ransomware|botnet|c2|command.and.control|red.?team\w*|"
    r"offensive.security|vulnerabilit\w+|vuln.scan\w*|cve|0.?day|backdoor\w*|"
    r"rootkit|keylogg\w+|sql.?injection|xss|rce|ssrf|lfi|"
    r"port.scan\w*|nmap|reconnaissance|osint|subdomain.enum\w*|"
    r"brute.?forc\w+|password.crack\w*|hash.?crack\w*|hashcat|"
    r"deauth\w*|packet.sniff\w*|mitm|man.in.the.middle|"
    r"anti.?detect\w*|evasion|captcha.solv\w*|fingerprint.spoof\w*|"
    r"ctf|capture.the.flag|reverse.engineer\w*|disassembl\w+|deobfuscat\w+|"
    r"firewall|waf|honeypot|forensics?|phishing|spoofing|ddos|denial.of.service|"
    r"cracking|hacking|hacker\w*|attack.tool|cryptanalysis|"
    r"security.(scanner|testing|toolkit)|"
    r"bastion|privileged.access.management|"
    # mobile runtime manipulation: the class `objection` belongs to. Added
    # after it reached the gradeable set — its card, package name and URL are
    # all the neutral word "objection", so only these betray it.
    r"jailbreak\w*|frida|ssl.?pinning|pinning.bypass|root.?detection|"
    # OSINT people-search (`maigret`, `sherlock`). These tools describe
    # themselves in ordinary words — "collect a dossier on a person by
    # username" contains no security term at all — so the words to match are
    # the euphemisms, not the threat.
    r"dossier|people.search|username.search|username.enum\w*|"
    # anti-automation-detection (`pydoll`, `undetected-chromedriver`). A library
    # whose selling point is that it does not look automated.
    r"undetected|stealth.(browser|mode|automation)|anti.?bot|bot.detection|"
    # "without a WebDriver" is `pydoll`'s actual pitch, and the phrase only has
    # a selling point if not being detectable is the point.
    r"without.a.?web.?driver|web.?driver.?less|"
    # malicious-document analysis (`oletools`). Held as forensics-adjacent; its
    # own card names only the file format, so match the tool names.
    r"olevba|oleid|vba.deobfusc\w*|macro.malware|"
    # content-provenance defeat (`remove-ai-watermarks`). Not security at all,
    # which is why nothing above would ever fire on it.
    r"watermark.remov\w*|remove.?\w*.?watermark|watermark.strip\w*|synthid|c2pa"
    r")\b", re.I)

# Hits already looked at and judged not to be offensive-security tooling. Keyed
# by task name so a repo cannot be quietly forgiven by editing the regex.
# `presidio` is PII redaction, `textattack` is academic adversarial-NLP
# robustness, `tf-encrypted` is privacy-preserving ML; the crypto libraries
# (`pyotp`, `python-jose`, `bcrypt-py`, `pyca-nacl`, `authlib-jose`,
# `itsdangerous*`, `hashids-py`) implement standards, they do not break them.
REVIEWED = {
    "presidio", "textattack", "tf-encrypted", "pyotp", "pyotp-full",
    "python-jose", "authlib-jose", "bcrypt-py", "pyca-nacl", "hashids-py",
    "itsdangerous", "itsdangerous-full", "django-cors-headers",
    "python-magic-mime", "jsonschema-validator",
    # `brute_force` here is imagededup's own nearest-neighbour search mode.
    "imagededup",
    # `river.datasets.Phishing` is a stock binary-classification dataset shipped
    # with an online-learning library, and `engineio.payload.Payload` is the
    # Engine.IO wire format's packet container. Both matched on a single word in
    # the generated API surface, which is the price of tuning this to over-match.
    "river-online", "python-engineio",
    # `optuna.samplers.BruteForceSampler` is exhaustive search over a finite
    # space, a hyperparameter-tuning strategy, not credential guessing.
    "optuna",
}


def task_text(d):
    """Everything about a task that is cheap to read and describes its purpose.

    Deliberately not the sources or the test bodies: a 900-module repo's code
    mentions "attack" for a hundred innocent reasons, and a check that cries
    wolf on every ML repo would get switched off.
    """
    parts = [os.path.basename(d)]
    for rel in ("task.json", "task.md"):
        p = os.path.join(d, rel)
        if os.path.exists(p):
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if rel == "task.json":
                try:                     # only the descriptive fields
                    j = json.loads(txt)
                    txt = " ".join(str(j.get(k, "")) for k in
                                   ("name", "clone", "package", "note"))
                except json.JSONDecodeError:
                    pass
            else:
                # Whole file, not the first 4000 chars. The header is the
                # least informative part: `sherlock`, `cai` and `objection`
                # were each caught by a symbol in the generated API surface,
                # which starts after the header and can run past 4000 chars
                # on a large package. Truncating here is truncating the
                # evidence.
                pass
            parts.append(txt)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="benchmark/tasks")
    ap.add_argument("--library", default="data/primitives_library/manifest.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    built = sorted(d for d in os.listdir(args.tasks)
                   if os.path.isfile(os.path.join(args.tasks, d, "task.json")))

    # ---- check 1: nothing held may have a task dir ----
    leaked = [n for n in built if n in HOLD_REPOS]

    # ---- check 2: pattern the built tasks themselves ----
    hits = []
    for n in built:
        m = PATTERN.search(task_text(os.path.join(args.tasks, n)))
        if m:
            hits.append((n, m.group(0)))
    unreviewed = [(n, w) for n, w in hits if n not in REVIEWED]

    # ---- check 3: the primitive library's provenance ----
    # Checks 1 and 2 both look at benchmark/tasks/, which is what a RUN reads.
    # That missed an entire second door: the library is harvested from repos the
    # loop ran, persists, and is served back to every later run, so a repo held
    # after its harvest stayed in the system with no task dir at all. Measured:
    # 44 primitives from `objection` (including jailbreak.py) were still being
    # retrieved by arm A weeks after the hold. Hard failure, not a warning.
    lib_hits = []
    if os.path.exists(args.library):
        try:
            entries = json.load(open(args.library))
        except (ValueError, OSError):
            entries = []
        for e in entries:
            hit = is_held_primitive(e, HOLD_REPOS)
            if hit:
                lib_hits.append((e.get("file") or "?", hit))

    print("built tasks: %d    held: %d" % (len(built), len(HOLD_REPOS)))
    if leaked:
        print("\nFAIL — held repos with a built task dir (delete these):")
        for n in leaked:
            print("  %s/%s" % (args.tasks, n))
    else:
        print("held ∩ built = 0  ✓")

    if unreviewed:
        print("\nFAIL — pattern hits not in REVIEWED (judge each, then either "
              "add to HOLD_REPOS and delete the task dir, or add to REVIEWED):")
        for n, w in unreviewed:
            print("  %-34s [%s]" % (n, w))
    elif not args.quiet:
        print("pattern hits: %d, all previously reviewed  ✓" % len(hits))
        for n, w in hits:
            print("    %-32s [%s] reviewed-benign" % (n, w))

    if lib_hits:
        by = collections.Counter(h for _f, h in lib_hits)
        print("\nFAIL — primitives in the library came from a held repo "
              "(quarantine them; the library filters on load and save, so a "
              "hit here means the filter was bypassed or the file predates it):")
        for repo, n in by.most_common():
            print("  %-34s %d primitives" % (repo, n))
        for f, h in lib_hits[:10]:
            print("      %s  <- %s" % (f, h))
    elif not args.quiet:
        print("library provenance: 0 primitives from a held repo  ✓ "
              "(%s)" % args.library)

    return 1 if (leaked or unreviewed or lib_hits) else 0


if __name__ == "__main__":
    sys.exit(main())
