"""Stage 1: list every public Python repository created after the GPT-5.6-terra knowledge cutoff
(2026-02-16, developers.openai.com model page) with >= 300 stars. Read-only GitHub search API."""
import datetime as dt, json, subprocess, sys, time
OUT = 'benchmark/post_cutoff/selection/candidates_raw.jsonl'
START, END = dt.date(2026, 2, 17), dt.date(2026, 11, 13)
BASE = 'language:Python stars:>=300 archived:false'

def search(q, page):
    for attempt in range(6):
        p = subprocess.run(['gh', 'api', '-X', 'GET', 'search/repositories', '-f', f'q={q}',
                            '-f', 'per_page=100', '-f', f'page={page}', '-f', 'sort=stars'],
                           capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)
        time.sleep(20 * (attempt + 1))
    raise SystemExit(p.stderr)

def window(a, b, out):
    q = f'{BASE} created:{a.isoformat()}..{b.isoformat()}'
    r = search(q, 1); n = r['total_count']; time.sleep(2.2)
    if n > 1000 and a < b:
        mid = a + (b - a) // 2
        window(a, mid, out); window(mid + dt.timedelta(days=1), b, out); return
    items = r['items']
    for page in range(2, (min(n, 1000) + 99) // 100 + 1):
        items += search(q, page)['items']; time.sleep(2.2)
    for it in items:
        out[it['full_name']] = {k: it.get(k) for k in (
            'full_name', 'html_url', 'clone_url', 'created_at', 'pushed_at', 'stargazers_count', 'fork',
            'archived', 'is_template', 'size', 'default_branch', 'description', 'topics', 'language')}
        out[it['full_name']]['license'] = (it.get('license') or {}).get('spdx_id')
    print(a, b, n, len(items), file=sys.stderr, flush=True)

out = {}
window(START, END, out)
with open(OUT, 'w') as fh:
    for v in sorted(out.values(), key=lambda v: v['created_at']):
        fh.write(json.dumps(v) + '\n')
print('repos', len(out), 'query date', dt.date.today().isoformat())
