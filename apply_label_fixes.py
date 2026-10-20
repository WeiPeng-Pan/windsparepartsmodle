"""
应用高置信标签修正到 train/val (test 不动,保持指标可比)
- 修正来源: data/relabel_suggestions.json 中 flash==model≠old 的维度
- 修正固化到 data/label_fixes.json,可重复应用(重跑 data_pipeline.py 后需重放)
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
DIMS = ['type', 'physical', 'scene']

with open(os.path.join(BASE, 'data', 'relabel_suggestions.json')) as f:
    sugg = json.load(f)

# ---- 1. 生成修正清单 ----
fixes = []
for s in sugg:
    for i, d in enumerate(DIMS):
        if s['flash'][i] == s['model'][i] != s['old'][i]:
            fixes.append({'split': s['split'], 'name': s['name'], 'dim': d,
                          'old': s['old'][i], 'new': s['flash'][i]})

fix_path = os.path.join(BASE, 'data', 'label_fixes.json')
with open(fix_path, 'w') as f:
    json.dump(fixes, f, ensure_ascii=False, indent=1)
print(f'高置信修正 {len(fixes)} 条 → {fix_path}')

# ---- 2. 应用到 train/val ----
applied = skipped = 0
for split in ['train', 'val']:
    path = os.path.join(BASE, 'data', f'{split}.json')
    with open(path) as f:
        rows = json.load(f)
    by_name = {r['name']: r for r in rows}
    for fx in fixes:
        if fx['split'] != split:
            continue
        r = by_name.get(fx['name'])
        if r is None:
            skipped += 1; continue
        if r[fx['dim']] == fx['new']:      # 已应用过
            continue
        if r[fx['dim']] != fx['old']:      # 原值对不上,保守跳过
            skipped += 1; continue
        r[fx['dim']] = fx['new']
        applied += 1
    with open(path, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

print(f'已应用 {applied} 条, 跳过 {skipped} 条')
print('注意: test 未修正(保持与历史报告可比); 修正明细见 data/label_fixes.json')
