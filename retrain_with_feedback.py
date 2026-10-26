"""
反馈闭环重训: 纠错记录 → 数据 → 重训 → 冻结test报告, 一条命令
数据源: data/corrections.json (来源: apply_audit / 复核队列 / 手工)
  - 纠错只作用于 train(val/test 冻结; test 的纠错单独另存 test_corrected.json 用于诚实报告)
  - 输出 data/train_next.json + models/fasttext_v{N}_*.bin (版本号自动递增)
"""
import json, os, re, sys, tempfile
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
DIMS = ['type', 'physical', 'scene']
VALID = {
    'type': {'备件', '耗材', '工器具', '大部件'},
    'physical': {'电气件', '机械件', '电子件', '传感器', '油品', '液压件', '密封件', '其他'},
    'scene': {'通用类', '控制系统类', '发电机类', '齿轮箱类', '变流系统类', '变桨系统类',
              '液压系统类', '偏航系统类', '润滑系统类'},
}


def load(p):
    with open(os.path.join(BASE, 'data', f'{p}.json')) as f:
        return json.load(f)


def next_version():
    vers = [int(m.group(1)) for m in
            (re.search(r'fasttext_v(\d+)_', f) for f in os.listdir(os.path.join(BASE, 'models')))
            if m]
    return max(vers) + 1 if vers else 1


def main():
    import fasttext
    with open(os.path.join(BASE, 'data', 'corrections.json')) as f:
        corrections = json.load(f)
    if not corrections:
        print('data/corrections.json 为空, 无需重训')
        return

    # 1. 应用纠错: train 直接改; val/test 的改错单独记录(test 另存 corrected 版本)
    train = load('train_v5')
    by_name = {r['name']: r for r in train}
    n_train = n_test = skipped = 0
    test_corr = []
    for c in corrections:
        if c['dim'] not in DIMS or c.get('new') not in VALID[c['dim']]:
            skipped += 1
            continue
        r = by_name.get(c['name'])
        if r is not None:
            if r[c['dim']] == c['new']:
                continue  # 已应用
            r[c['dim']] = c['new']
            n_train += 1
        else:
            test_corr.append(c)
    with open(os.path.join(BASE, 'data', 'train_next.json'), 'w') as f:
        json.dump(train, f, ensure_ascii=False, indent=1)
    if test_corr:
        test = load('test')
        tb = {r['name']: r for r in test}
        n_test = 0
        for c in test_corr:
            r = tb.get(c['name'])
            if r is not None and r[c['dim']] != c['new']:
                r[c['dim']] = c['new']
                n_test += 1
        with open(os.path.join(BASE, 'data', 'test_corrected.json'), 'w') as f:
            json.dump(test, f, ensure_ascii=False, indent=1)
    print(f'纠错: train 应用 {n_train} / test 另存 {n_test} / 跳过 {skipped}')

    # 2. 重训(finalize 配方; 与 finalize_v2.py 保持一致, 独立副本避免 import 副作用)
    RECIPES = {
        'type':     dict(lr=0.05, epoch=60, wordNgrams=3, dim=32, bucket=200000, minn=2, maxn=4),
        'physical': dict(lr=0.05, epoch=60, wordNgrams=3, dim=32, bucket=200000, minn=2, maxn=4),
        'scene':    dict(lr=0.1, epoch=50, wordNgrams=2, dim=16, bucket=100000, minn=2, maxn=4),
    }
    val, test_raw = load('val'), load('test')
    test_fixed = load('test_corrected') if test_corr and os.path.exists(
        os.path.join(BASE, 'data', 'test_corrected.json')) else None

    def write_ft(data, path, dim):
        with open(path, 'w') as f:
            f.write('\n'.join(f"__label__{re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]+', '_', d[dim].strip())} {d['name']}"
                              for d in data))

    v = next_version()
    print(f'重训 fasttext_v{v} (train {len(train)} 条)...')
    for dim in DIMS:
        tp, vp = tempfile.mktemp(), tempfile.mktemp()
        write_ft(train, tp, dim)
        write_ft(val, vp, dim)
        m = fasttext.train_supervised(input=tp, loss='softmax', minCount=1, thread=4, verbose=0,
                                      **RECIPES[dim])
        p = m.test(vp)[1]
        m.save_model(os.path.join(BASE, 'models', f'fasttext_v{v}_{dim}.bin'))
        print(f'  {dim}: val P={p:.1%}')
        os.unlink(tp)
        os.unlink(vp)

    # 3. 新旧模型在 test 上对比(原标签 + 纠错后标签各报一版)
    from classifier import WindClassifier
    clf_old = WindClassifier()
    clf_new = WindClassifier(model_dir='/nonexistent')
    for dim in DIMS:
        setattr(clf_new, f'model_{dim}',
                fasttext.load_model(os.path.join(BASE, 'models', f'fasttext_v{v}_{dim}.bin')))

    def acc(clf, data):
        ok = {d: 0 for d in DIMS}
        all3 = 0
        for it in data:
            r = clf.classify(it['name'])
            for d in DIMS:
                ok[d] += r[d] == it[d]
            all3 += all(r[d] == it[d] for d in DIMS)
        n = len(data)
        return ok, all3, n

    for tag, data in [('原test', test_raw)] + ([('纠错后test', test_fixed)] if test_fixed else []):
        (o_old, a_old, n) = acc(clf_old, data)
        (o_new, a_new, _) = acc(clf_new, data)
        print(f'\n[{tag} n={n}]')
        for d in DIMS:
            print(f'  {d:9s}: v6 {o_old[d]/n*100:.1f}% → v{v} {o_new[d]/n*100:.1f}%')
        print(f'  三维全对 : {a_old/n*100:.1f}% → {a_new/n*100:.1f}%')

    print(f'\n新模型: models/fasttext_v{v}_*.bin (classifier.py 尚未切换, 确认后手动改路径或 git mv)')


if __name__ == '__main__':
    main()
