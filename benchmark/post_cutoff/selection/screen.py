"""Stage 2: pre-specified repository screening, read-only (GitHub API + PyPI JSON), no code execution.
Mirrors the LEGO-REPO screen (security/domain hold, importable package, native tests) and adds the
post-cutoff checks. Every candidate gets one recorded outcome in funnel.jsonl."""
import concurrent.futures as cf, datetime as dt, json, os, re, subprocess, sys, time, tomllib, urllib.request
W = 'benchmark/post_cutoff/selection'
REPO = '.'
CUTOFF = '2026-02-17T00:00:00Z'          # first day after the 2026-02-16 knowledge cutoff of GPT-5.6-terra
sys.path.insert(0, REPO)
from lego.benchmark_meta import HELD_REPOS
import importlib.util
spec = importlib.util.spec_from_file_location('audit', os.path.join(REPO, 'tools/audit_tasks_security.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
PATTERN = audit.PATTERN
TEST = re.compile(r"^(test_.*\.py|.*_tests?\.py|tests?\.py)$", re.I)
PACKAGING = ('pyproject.toml', 'setup.py', 'setup.cfg')
SKIPDIR = re.compile(r'(^|/)(tests?|docs?|examples?|benchmarks?|scripts?|\.github)(/|$)', re.I)
inv = {l.split(',')[1].lower().removesuffix('.git') for l in open('benchmark/lego_repo_inventory.csv').read().splitlines()[1:]}
cf_src = open('corpora/codeface_sources.txt').read().lower()

def gh(path, headers=False):
    for attempt in range(8):
        cmd = ['gh', 'api'] + (['-i'] if headers else []) + [path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        if 'rate limit' in (p.stdout + p.stderr).lower():
            time.sleep(60); continue
        if '404' in p.stdout + p.stderr or '409' in p.stdout + p.stderr:
            return None
        time.sleep(5 * (attempt + 1))
    return None

def pypi_first(name):
    try:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json', timeout=20) as r:
            d = json.load(r)
    except Exception:
        return None
    times = [f['upload_time_iso_8601'] for fs in d.get('releases', {}).values() for f in fs]
    return min(times) if times else None

def dist_name(full, path):
    raw = gh(f'repos/{full}/contents/{path}')
    if not raw:
        return None
    import base64
    try:
        text = base64.b64decode(json.loads(raw)['content']).decode('utf-8', 'ignore')
    except Exception:
        return None
    if path.endswith('pyproject.toml'):
        try:
            t = tomllib.loads(text)
            return (t.get('project') or {}).get('name') or ((t.get('tool') or {}).get('poetry') or {}).get('name')
        except Exception:
            return None
    m = re.search(r'''name\s*[=:]\s*['"]?([A-Za-z0-9_.\-]+)''', text)
    return m.group(1) if m else None

def screen(c):
    full = c['full_name']; r = {'full_name': full, 'stars': c['stargazers_count'], 'created_at': c['created_at'],
                                'license': c['license']}
    url = c['html_url'].lower()
    if c['fork'] or c['archived'] or c['is_template']:
        return r | {'stage': 'metadata', 'reason': 'fork/archived/template'}
    if not c['license'] or c['license'] == 'NOASSERTION':
        return r | {'stage': 'metadata', 'reason': 'no redistributable license'}
    if url in inv or url.split('github.com/')[1] in cf_src:
        return r | {'stage': 'metadata', 'reason': 'in LEGO-REPO or a CodeFace donor'}
    name = full.split('/')[1].lower()
    text = ' '.join([full, c['description'] or '', ' '.join(c['topics'] or [])])
    if name in HELD_REPOS or PATTERN.search(text):
        return r | {'stage': 'metadata', 'reason': 'security/domain hold'}
    raw = gh(f'repos/{full}/git/trees/{c["default_branch"]}?recursive=1')
    if not raw:
        return r | {'stage': 'tree', 'reason': 'tree unavailable'}
    tree = json.loads(raw); paths = [t['path'] for t in tree.get('tree', []) if t['type'] == 'blob']
    pack = sorted((p for p in paths if p.split('/')[-1] in PACKAGING and p.count('/') <= 2), key=lambda p: (p.count('/'), p))
    tests = [p for p in paths if p.endswith('.py') and TEST.match(p.split('/')[-1])]
    pkgs = [p for p in paths if p.endswith('__init__.py') and not SKIPDIR.search(p)]
    r |= {'n_py': sum(p.endswith('.py') for p in paths), 'n_test_files': len(tests), 'truncated': tree.get('truncated', False)}
    if not pack:
        return r | {'stage': 'tree', 'reason': 'no packaging file'}
    if not pkgs:
        return r | {'stage': 'tree', 'reason': 'no importable package'}
    if not tests:
        return r | {'stage': 'tree', 'reason': 'no native tests'}
    hdr = gh(f'repos/{full}/commits?per_page=1&sha={c["default_branch"]}', headers=True) or ''
    m = re.search(r'[?&]page=(\d+)>; rel="last"', hdr)
    last = int(m.group(1)) if m else 1
    body = gh(f'repos/{full}/commits?per_page=1&page={last}&sha={c["default_branch"]}')
    try:
        root = json.loads(body)[0]['commit']
        r['root_commit_date'] = min(root['author']['date'], root['committer']['date'])
    except Exception:
        return r | {'stage': 'history', 'reason': 'history unavailable'}
    if r['root_commit_date'] < CUTOFF:
        return r | {'stage': 'history', 'reason': 'root commit before cutoff'}
    dn = dist_name(full, pack[0]) if pack[0].endswith(('pyproject.toml', 'setup.cfg', 'setup.py')) else None
    r['dist'] = dn
    if dn:
        first = pypi_first(dn); r['pypi_first'] = first
        if first and first < CUTOFF:
            return r | {'stage': 'history', 'reason': 'PyPI release before cutoff'}
    return r | {'stage': 'build', 'reason': 'passed screen', 'clone': c['clone_url'], 'name': name}

cands = [json.loads(l) for l in open(f'{W}/candidates_raw.jsonl')]
out = []
with cf.ThreadPoolExecutor(6) as ex:
    for i, r in enumerate(ex.map(screen, cands)):
        out.append(r)
        if i % 200 == 0: print(i, len(cands), flush=True)
with open(f'{W}/funnel.jsonl', 'w') as fh:
    for r in out: fh.write(json.dumps(r) + '\n')
import collections
print(collections.Counter((r['stage'], r['reason']) for r in out))
