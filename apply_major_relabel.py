"""
应用权威大部件口径: 只改 type 维度
- flash 判定 + 用户明确裁定覆盖(减速机/液压站/变压器等总成→备件)
- train_v3 → train_v4; val/test 原地更新(备份为 *.preMajorRelabel.json)
- 审计日志 data/major_relabel_changes.json
"""
import json, os, re, shutil
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
FORCE_SPARE = re.compile(r'减速机|减速器|减速箱|液压站|变压器|齿条|变桨电机')

def load(p):
    with open(os.path.join(BASE, 'data', f'{p}.json')) as f:
        return json.load(f)

with open(os.path.join(BASE, 'data', 'major_relabel_progress.json')) as f:
    flash_verdict = {r['name']: r['type'] for r in json.load(f)}

changes = []
def apply_to(rows, split):
    n_changed = 0
    for it in rows:
        new = flash_verdict.get(it['name'])
        if not new or new not in ('大部件', '备件', '耗材', '工器具'):
            continue
        if it['type'] == '大部件' and FORCE_SPARE.search(it['name']):
            new = '备件'  # 用户明确裁定优先
        if new != it['type']:
            changes.append({'split': split, 'name': it['name'], 'old': it['type'], 'new': new})
            it['type'] = new
            n_changed += 1
    return n_changed

# train_v3 → train_v4
train = load('train_v3')
n1 = apply_to(train, 'train')
with open(os.path.join(BASE, 'data', 'train_v4.json'), 'w') as f:
    json.dump(train, f, ensure_ascii=False, indent=1)

# val/test 原地更新 + 备份
for sp in ['val', 'test']:
    src = os.path.join(BASE, 'data', f'{sp}.json')
    bak = os.path.join(BASE, 'data', f'{sp}.preMajorRelabel.json')
    if not os.path.exists(bak):
        shutil.copy(src, bak)
    rows = load(sp)
    n = apply_to(rows, sp)
    with open(src, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f'{sp}: 修正 {n} 条 (备份 {os.path.basename(bak)})')
print(f'train: 修正 {n1} 条 → data/train_v4.json')

with open(os.path.join(BASE, 'data', 'major_relabel_changes.json'), 'w') as f:
    json.dump(changes, f, ensure_ascii=False, indent=1)
print(f'审计日志: {len(changes)} 条变更 → data/major_relabel_changes.json')
print('变更方向:', Counter((c['old'], c['new']) for c in changes).most_common())

# 口径核验: 新大部件构成
for sp, rows in [('train_v4', train), ('test', load('test'))]:
    majors = [it for it in rows if it['type'] == '大部件']
    kws = ['叶片', '发电机', '齿轮箱', '主轴', '主轴承', '变桨轴承', '偏航轴承',
           '塔筒', '机舱', '导流罩', '轮毂', '主机架', '变流器']
    outside = [it['name'] for it in majors if not any(k in it['name'] for k in kws)]
    print(f'[{sp}] 大部件 {len(majors)} 条, 不含定义关键词的残留 {len(outside)} 条: {outside[:6]}')
