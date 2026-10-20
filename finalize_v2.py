"""
最终定稿: 用 V2 子词配方重训三维度模型 → test 最终确认 → 替换生产模型
配方(val选优结果):
  type/physical: lr=0.05 epoch=60 wordNgrams=3 dim=32 bucket=200000 minn=2 maxn=4
  scene:         lr=0.1  epoch=50 wordNgrams=2 dim=16 bucket=100000 minn=2 maxn=4
训练数据: data/train_v5.json (v4 + 同义词/能级规则提升45条, 见 synonym_promotions.json)
"""
import json, os, re, tempfile
from collections import Counter
import fasttext
from classifier import WindClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DIMS = ['type', 'physical', 'scene']
RECIPES = {
    'type':     dict(lr=0.05, epoch=60, wordNgrams=3, dim=32, bucket=200000, minn=2, maxn=4),
    'physical': dict(lr=0.05, epoch=60, wordNgrams=3, dim=32, bucket=200000, minn=2, maxn=4),
    'scene':    dict(lr=0.1, epoch=50, wordNgrams=2, dim=16, bucket=100000, minn=2, maxn=4),
}

def load(name):
    with open(os.path.join(BASE, 'data', f'{name}.json')) as f:
        return json.load(f)

train, val, test = load('train_v5'), load('val'), load('test')

def write_ft(data, path, dim):
    lines = [f"__label__{re.sub(r'\s+', '_', d[dim].strip())} {d['name']}" for d in data]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

# ---- 1. 训练并保存 v2 模型 ----
models = {}
for dim in DIMS:
    tp, vp = tempfile.mktemp(), tempfile.mktemp()
    write_ft(train, tp, dim); write_ft(val, vp, dim)
    m = fasttext.train_supervised(input=tp, loss='softmax', minCount=1, thread=4, verbose=0, **RECIPES[dim])
    p = m.test(vp)[1]
    models[dim] = m
    out = os.path.join(BASE, 'models', f'fasttext_v2_{dim}.bin')
    m.save_model(out)
    print(f'{dim}: val P={p:.1%} → {out} ({os.path.getsize(out)//1024}KB)')
    os.unlink(tp); os.unlink(vp)

# ---- 2. 混合分类器 test 最终确认 ----
clf = WindClassifier(model_dir='/nonexistent')
for dim in DIMS:
    setattr(clf, f'model_{dim}', models[dim])

def macro_f1(data, dim):
    tp, fp, fn = Counter(), Counter(), Counter()
    for it in data:
        r = clf.classify(it['name'])
        if r[dim] == it[dim]: tp[it[dim]] += 1
        else: fp[r[dim]] += 1; fn[it[dim]] += 1
    labels = set(tp) | set(fp) | set(fn)
    f1s = []
    for l in labels:
        prec = tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l] else 0
        rec = tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l] else 0
        f1s.append(2*prec*rec/(prec+rec) if prec+rec else 0)
    return sum(f1s)/len(f1s)

n = len(test)
ok = {d: 0 for d in DIMS}; all3 = 0
cls_tot, cls_ok = {d: Counter() for d in DIMS}, {d: Counter() for d in DIMS}
m_stat, m_ok = {d: Counter() for d in DIMS}, {d: Counter() for d in DIMS}
for it in test:
    r = clf.classify(it['name'])
    oks = []
    for d in DIMS:
        m_stat[d][r['method'][d]] += 1
        cls_tot[d][it[d]] += 1
        good = r[d] == it[d]
        oks.append(good)
        if good: ok[d] += 1; m_ok[d][r['method'][d]] += 1; cls_ok[d][it[d]] += 1
    all3 += all(oks)

lines = []
lines.append('最终版(规则v2 + 子词fastText) test 集报告 (n=%d)' % n)
lines.append('=' * 52)
for d in DIMS:
    lines.append(f'{d:9s}: acc {ok[d]/n*100:.1f}%   macro-F1 {macro_f1(test, d)*100:.1f}%')
    parts = [f"{m}={m_stat[d][m]}({m_ok[d][m]/m_stat[d][m]*100:.0f}%准)" for m in ['rule','model','default'] if m_stat[d][m]]
    lines.append(f'          归因: {" ".join(parts)}')
lines.append(f'三维全对 : {all3/n*100:.1f}%')
lines.append('')
for d in DIMS:
    lines.append(f'[{d} 分类别recall]')
    for cls in sorted(cls_tot[d], key=lambda c: -cls_tot[d][c]):
        t, k = cls_tot[d][cls], cls_ok[d][cls]
        lines.append(f'  {cls:8s}: {k:>4d}/{t:>4d} = {k/t*100:3.0f}%')
report = '\n'.join(lines)
print(report)
with open(os.path.join(BASE, 'eval_report_final.txt'), 'w') as f:
    f.write(report + '\n')

# ---- 3. 冒烟测试 ----
print('\n冒烟测试:')
for name_ in ['偏航刹车片 SK8010', '油位传感器', '注油接头', '齿轮箱润滑油', '发电机进水管', 'PLC通讯模块', '碳刷 20*30*50']:
    r = clf.classify(name_)
    print(f'  {name_:16s} → {r["type"]}/{r["physical"]}/{r["scene"]}')
