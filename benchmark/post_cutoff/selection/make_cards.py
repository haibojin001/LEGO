"""Stage 3 input: one card per repository that passed the screen (name collisions get the owner prefix)."""
import collections, json
W = 'benchmark/post_cutoff/selection'
ok = [r for r in map(json.loads, open(f'{W}/funnel.jsonl')) if r['stage'] == 'build']
cnt = collections.Counter(r['name'] for r in ok)
with open(f'{W}/cards.jsonl', 'w') as fh:
    for r in sorted(ok, key=lambda r: r['full_name'].lower()):
        name = r['name'] if cnt[r['name']] == 1 else r['full_name'].replace('/', '__').lower()
        fh.write(json.dumps({'name': name, 'clone': r['clone'], 'full_name': r['full_name'], 'stars': r['stars'],
                             'created_at': r['created_at'], 'root_commit_date': r['root_commit_date'],
                             'license': r['license']}) + '\n')
print('cards', len(ok))
