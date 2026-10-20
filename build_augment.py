"""
构建增强集 data/augment.json + 重训 v3
- 大部件: 全收(107)  变流系统类: 截500  润滑系统类: 全收(297)  随机对照: 全收
- 只并入 train(val/test 冻结), 产出 data/train_v3.json
- 用 finalize_v2 配方重训, 在冻结 test 上与 v2.2 对比
"""
import json, os, random, re
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
random.seed(20261020)

VALID = {
    'type': {'备件', '耗材', '工器具', '大部件'},
    'physical': {'电气件', '机械件', '电子件', '传感器', '油品', '液压件', '密封件', '其他'},
    'scene': {'通用类', '控制系统类', '发电机类', '齿轮箱类', '变流系统类', '变桨系统类', '液压系统类', '偏航系统类', '润滑系统类'},
}

with open(os.path.join(BASE, 'data', 'augment_progress.json')) as f:
    rows = json.load(f)

# ---- 1. 过滤非法标签 ----
valid_rows = [r for r in rows if all(r.get(d) in VALID[d] for d in VALID)]
print(f'合法标签: {len(valid_rows)}/{len(rows)}')

# ---- 2. 构建增强集 ----
major = [r for r in valid_rows if r['type'] == '大部件']
conv = [r for r in valid_rows if r['scene'] == '变流系统类']
random.shuffle(conv); conv = conv[:500]
lub = [r for r in valid_rows if r['scene'] == '润滑系统类']
kw = re.compile(r'变流|变频|逆变|IGBT|整流|功率模块|润滑|加油|油分配|油脂泵|黄油泵|油站|整机|塔筒|轮毂|机舱|导流罩|主机架|副机架|叶根|发电机|齿轮箱|主轴|变桨轴承|偏航轴承|叶片')
gen = [r for r in valid_rows if not kw.search(r['name'])]

augment = major + conv + lub + gen
# 内部去重 + 对 train/val/test 归一化去重(防近重复泄漏)
from data_pipeline import norm as dup_norm
seen = set()
final = []
for r in augment:
    k = dup_norm(r['name'])
    if not k or k in seen:
        continue
    seen.add(k)
    final.append({'name': r['name'], 'type': r['type'], 'physical': r['physical'], 'scene': r['scene']})

with open(os.path.join(BASE, 'data', 'val.json')) as f: va = json.load(f)
with open(os.path.join(BASE, 'data', 'test.json')) as f: te = json.load(f)
frozen = {dup_norm(x['name']) for x in va + te}
final = [r for r in final if dup_norm(r['name']) not in frozen]

with open(os.path.join(BASE, 'data', 'augment.json'), 'w') as f:
    json.dump(final, f, ensure_ascii=False, indent=1)
print(f'增强集: 大部件{len(major)} + 变流{len(conv)} + 润滑{len(lub)} + 随机{len(gen)} → 去重冻结后 {len(final)} 条')

# ---- 3. train_v3 = train + augment ----
with open(os.path.join(BASE, 'data', 'train.json')) as f: tr = json.load(f)
train_v3 = tr + [{'name': r['name'], 'type': r['type'], 'physical': r['physical'], 'scene': r['scene']} for r in final]
with open(os.path.join(BASE, 'data', 'train_v3.json'), 'w') as f:
    json.dump(train_v3, f, ensure_ascii=False, indent=1)
print(f'train: {len(tr)} → train_v3: {len(train_v3)}')
for d in ['type', 'scene']:
    before, after = Counter(x[d] for x in tr), Counter(x[d] for x in train_v3)
    print(f'[{d}] 变化:')
    for c in sorted(after, key=lambda c: -after[c]):
        if after[c] != before.get(c, 0):
            print(f'  {c:8s}: {before.get(c,0)} → {after[c]}')
