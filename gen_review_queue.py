"""
复核队列生成器: 置信度分层的落地入口
对任意物料清单(xlsx/csv, 含名称列)跑 v6 分类器, 输出:
  1. 原文件 + 分类列(prob/tier)  → <输入名>_已分类.xlsx
  2. 复核队列(tier=review 的行)  → <输入名>_复核队列.xlsx
     每维给出 top3 模型建议供人工点选, 复核结果可直接喂给 retrain_with_feedback.py

用法:
  python3 gen_review_queue.py <input.xlsx> [名称列名]
  名称列默认依次尝试: 清洗后名称 / 物料名称 / name
"""
import json, os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import WindClassifier

NAME_COLS = ['清洗后名称', '物料名称', 'name', '名称']
DIMS = ['type', 'physical', 'scene']
DIM_CN = {'type': '类型', 'physical': '物理属性', 'scene': '场景'}
VALID = {
    'type': ['备件', '耗材', '工器具', '大部件'],
    'physical': ['电气件', '机械件', '电子件', '传感器', '油品', '液压件', '密封件', '其他'],
    'scene': ['通用类', '控制系统类', '发电机类', '齿轮箱类', '变流系统类', '变桨系统类',
              '液压系统类', '偏航系统类', '润滑系统类'],
}


def top3(clf, dim, name):
    labels, probs = getattr(clf, f'model_{dim}').predict(name, k=3)
    return ' / '.join(f"{l.replace('__label__', '')}({p:.0%})" for l, p in zip(labels, probs))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    src = sys.argv[1]
    df = pd.read_excel(src) if src.endswith(('.xlsx', '.xls')) else pd.read_csv(src)
    name_col = None
    if len(sys.argv) > 2:
        name_col = sys.argv[2]
    else:
        for c in NAME_COLS:
            if c in df.columns:
                name_col = c
                break
    if name_col is None:
        print(f'未找到名称列, 现有列: {list(df.columns)}')
        return
    print(f'输入: {len(df)} 行, 名称列 [{name_col}]')

    clf = WindClassifier()
    t0 = time.time()
    names = [str(n) if pd.notna(n) else '' for n in df[name_col]]
    uniq = sorted(set(n for n in names if len(n) >= 2))
    results = {n: clf.classify(n) for n in uniq}
    print(f'分类 {len(uniq)} 个唯一名, {time.time()-t0:.1f}s')

    def get(n, key, dim):
        r = results.get(n)
        return r[key][dim] if r and key in r else (0.0 if key == 'prob' else '')

    for dim in DIMS:
        df[f'分类_{DIM_CN[dim]}'] = [results.get(n, {}).get(dim, '') or
                                     {'type': '备件', 'physical': '其他', 'scene': '通用类'}[dim]
                                     for n in names]
        df[f'概率_{DIM_CN[dim]}'] = [round(get(n, 'prob', dim), 3) for n in names]
        df[f'置信_{DIM_CN[dim]}'] = [get(n, 'tier', dim) or 'review' for n in names]

    # 复核队列: 任一维 review
    review_mask = df[[f'置信_{DIM_CN[d]}' for d in DIMS]].eq('review').any(axis=1)
    n_review = int(review_mask.sum())
    n_auto = len(df) - n_review
    tier_cnt = {t: int((df[[f'置信_{DIM_CN[d]}' for d in DIMS]] == t).any(axis=1).sum())
                for t in ['auto', 'low', 'review']}
    print(f'分层: 任一维auto {tier_cnt["auto"]} | 含low {tier_cnt["low"]} | 含review {n_review}')

    out_classified = src.rsplit('.', 1)[0] + '_已分类.xlsx'
    df.to_excel(out_classified, index=False)
    print(f'已保存: {out_classified}')

    if n_review:
        rv = df[review_mask].copy()
        for dim in DIMS:
            rv[f'建议_{DIM_CN[dim]}_top3'] = [top3(clf, dim, n) if len(str(n)) >= 2 else '' for n in rv[name_col]]
            rv[f'人工_{DIM_CN[dim]}'] = ''
        rv['_维度需复核'] = rv.apply(
            lambda r: ','.join(d for d in DIMS if r[f'置信_{DIM_CN[d]}'] == 'review'), axis=1)
        out_review = src.rsplit('.', 1)[0] + '_复核队列.xlsx'
        rv.to_excel(out_review, index=False)
        print(f'复核队列: {n_review} 行 → {out_review}')
        print('人工列填法: 在 人工_类型/人工_物理属性/人工_场景 填正确标签(参考 建议_top3),')
        print('填完的文件用 retrain_with_feedback.py 直接并入训练。')


if __name__ == '__main__':
    main()
