"""
干净重训 + 对比报告
- 模型只用 data/train.json 训练, data/val.json 选参, data/test.json 只出报告
- 对比: ①交付版(泄漏参考) ②旧规则+新模型 ③新规则+新模型(候选交付)
- 指标: 各维准确率 / macro-F1 / 分类recall / 三维全对 / 方法归因
"""
import json, os, re, tempfile, importlib.util
from collections import Counter, defaultdict
import fasttext
from classifier import WindClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("轻量", dict(lr=0.1, epoch=50, wordNgrams=2, dim=16, bucket=50000)),
    ("中量", dict(lr=0.05, epoch=60, wordNgrams=3, dim=16, bucket=100000)),
    ("重量", dict(lr=0.1, epoch=60, wordNgrams=3, dim=32, bucket=100000)),
]
DIMS = ['type', 'physical', 'scene']

def load(name):
    with open(os.path.join(BASE, 'data', f'{name}.json')) as f:
        return json.load(f)

train, val, test = load('train'), load('val'), load('test')
print(f'train {len(train)} / val {len(val)} / test {len(test)}')

# ---------- 1. 训练(仅train, val选参) ----------
def write_ft(data, path, dim):
    lines = [f"__label__{re.sub(r'\s+', '_', d[dim].strip())} {d['name']}" for d in data]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

models = {}
for dim in DIMS:
    tp, vp = tempfile.mktemp(), tempfile.mktemp()
    write_ft(train, tp, dim); write_ft(val, vp, dim)
    best = (0, None, '')
    for name, cfg in CONFIGS:
        try:
            m = fasttext.train_supervised(input=tp, loss='softmax', minCount=1, thread=4, verbose=0, **cfg)
            p = m.test(vp)[1]
            if p > best[0]:
                best = (p, m, name)
        except RuntimeError:
            pass
    models[dim] = best[1]
    print(f'{dim}: 最优={best[2]} 纯模型val P={best[0]:.1%}')
    os.unlink(tp); os.unlink(vp)
    out = os.path.join(BASE, 'models', f'fasttext_clean_{dim}.bin')
    best[1].save_model(out)

# ---------- 2. 三套分类器 ----------
# ①交付版(参考,数字偏高: 训练含全部数据)
clf_shipped = WindClassifier()

# ②旧规则 + 新模型
old_src = os.popen('git show HEAD:classifier.py').read()
old_path = os.path.join(BASE, '_old_classifier_tmp.py')
with open(old_path, 'w') as f:
    f.write(old_src)
spec = importlib.util.spec_from_file_location('old_classifier', old_path)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)
clf_oldrule = old_mod.WindClassifier(model_dir='/nonexistent')
for dim in DIMS:
    setattr(clf_oldrule, f'model_{dim}', models[dim])

# ③新规则 + 新模型(候选)
clf_new = WindClassifier(model_dir='/nonexistent')
for dim in DIMS:
    setattr(clf_new, f'model_{dim}', models[dim])

# ---------- 3. 评估 ----------
def macro_f1(items, pred_fn, dim):
    tp, fp, fn = Counter(), Counter(), Counter()
    for it in items:
        r = pred_fn(it)
        p, t = r[dim], it[dim]
        if p == t: tp[t] += 1
        else: fp[p] += 1; fn[t] += 1
    labels = set(tp) | set(fp) | set(fn)
    f1s = []
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0
        rec = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return sum(f1s) / len(f1s)

def evaluate(clf, data, tag, file):
    n = len(data)
    ok = {d: 0 for d in DIMS}
    all3 = 0
    m_stat = {d: Counter() for d in DIMS}
    m_ok = {d: Counter() for d in DIMS}
    cls_tot, cls_ok = {d: Counter() for d in DIMS}, {d: Counter() for d in DIMS}
    for it in data:
        r = clf.classify(it['name'])
        oks = []
        for d in DIMS:
            m = r['method'][d] if isinstance(r.get('method'), dict) else 'model'
            m_stat[d][m] += 1
            cls_tot[d][it[d]] += 1
            good = r[d] == it[d]
            oks.append(good)
            if good: ok[d] += 1; m_ok[d][m] += 1; cls_ok[d][it[d]] += 1
        all3 += all(oks)
    print(f'\n===== {tag} (test n={n}) =====', file=file)
    print(f"{'维度':10s} accuracy  macro-F1", file=file)
    for d in DIMS:
        f1 = macro_f1(data, lambda it: clf.classify(it['name']), d)
        print(f"{d:10s} {ok[d]/n*100:6.1f}%   {f1*100:6.1f}%", file=file)
    print(f"三维全对    {all3/n*100:6.1f}%", file=file)
    for d in DIMS:
        parts = [f"{m}={m_stat[d][m]}({m_ok[d][m]/m_stat[d][m]*100:.0f}%准)" for m in ['rule', 'model', 'default'] if m_stat[d][m]]
        print(f"  {d} 归因: {' '.join(parts)}", file=file)
    return {'acc': {d: ok[d]/n for d in DIMS}, 'all3': all3/n, 'cls_tot': cls_tot, 'cls_ok': cls_ok}

report_lines = []
report = lambda s='': report_lines.append(s)
report('分类器修复效果报告')
report('================')
report(f'数据: train {len(train)} / val {len(val)} / test {len(test)} (近重复同折, 13组冲突已裁定)')
report(f'模型: 仅用train训练, val选参; test从未参与任何环节\n')

import sys
with open(os.path.join(BASE, 'eval_report.txt'), 'w') as rf:
    evaluate(clf_shipped, test, '① 交付版模型+交付版规则 (参考: 训练含全部数据, 数字偏高)', rf)
    evaluate(clf_oldrule, test, '② 旧规则 + 新训练模型 (隔离规则修复效果)', rf)
    res_new = evaluate(clf_new, test, '③ 新规则 + 新训练模型 (候选交付)', rf)
    rf.write('\n===== ③ 分类别 recall =====\n')
    for d in DIMS:
        rf.write(f'[{d}]\n')
        for cls in sorted(res_new['cls_tot'][d], key=lambda c: -res_new['cls_tot'][d][c]):
            t, k = res_new['cls_tot'][d][cls], res_new['cls_ok'][d][cls]
            rf.write(f'  {cls:8s}: {k:>4d}/{t:>4d} = {k/t*100:3.0f}%\n')
    rf.write('\n===== ③ 错误样例(前15) =====\n')
    n_err = 0
    for it in test:
        r = clf_new.classify(it['name'])
        if any(r[d] != it[d] for d in DIMS):
            rf.write(f"  {it['name'][:30]:32s} | 真 {it['type']}/{it['physical']}/{it['scene']} | 预 {r['type']}/{r['physical']}/{r['scene']}\n")
            n_err += 1
            if n_err >= 15: break
os.unlink(old_path)
print(open(os.path.join(BASE, 'eval_report.txt')).read())
