"""物料分类.xlsx 批量分类(2026-10-30)
口径与 classify_full_data.py 一致: 物料名称+规格+型号 拼接后过 v6 混合分类器,
输出格式与 gen_review_queue.py 一致: <名>_已分类.xlsx + <名>_复核队列.xlsx
"""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import WindClassifier
from gen_review_queue import top3, DIMS, DIM_CN

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '物料分类.xlsx')


def main():
    df = pd.read_excel(SRC)
    parts = [df[c].fillna('').astype(str).str.replace(r'[\r\n]+', ' ', regex=True)
             for c in ['物料名称', '规格', '型号']]
    df['_cls_input'] = parts[0].str.cat(parts[1:], sep=' ').str.strip()
    print(f'输入: {len(df)} 行')

    clf = WindClassifier()
    t0 = time.time()
    names = df['_cls_input'].tolist()
    uniq = sorted(set(n for n in names if len(n) >= 2))
    results = {n: clf.classify(n) for n in uniq}
    print(f'分类 {len(uniq)} 个唯一名, {time.time()-t0:.1f}s')

    default = {'type': '备件', 'physical': '其他', 'scene': '通用类'}
    for dim in DIMS:
        df[f'分类_{DIM_CN[dim]}'] = [results.get(n, {}).get(dim) or default[dim] for n in names]
        df[f'概率_{DIM_CN[dim]}'] = [round(results.get(n, {}).get('prob', {}).get(dim, 0.0), 3)
                                     for n in names]
        df[f'置信_{DIM_CN[dim]}'] = [results.get(n, {}).get('tier', {}).get(dim) or 'review'
                                     for n in names]

    review_mask = df[[f'置信_{DIM_CN[d]}' for d in DIMS]].eq('review').any(axis=1)
    n_review = int(review_mask.sum())
    tier_cnt = {t: int((df[[f'置信_{DIM_CN[d]}' for d in DIMS]] == t).any(axis=1).sum())
                for t in ['auto', 'low', 'review']}
    print(f'分层: 任一维auto {tier_cnt["auto"]} | 含low {tier_cnt["low"]} | 含review {n_review}')

    out = SRC.rsplit('.', 1)[0] + '_已分类.xlsx'
    df.drop(columns=['_cls_input']).to_excel(out, index=False)
    print(f'已保存: {out}')

    if n_review:
        rv = df[review_mask].drop(columns=['_cls_input']).copy()
        inputs = df.loc[review_mask, '_cls_input']
        for dim in DIMS:
            rv[f'建议_{DIM_CN[dim]}_top3'] = [top3(clf, dim, n) if len(str(n)) >= 2 else ''
                                             for n in inputs]
            rv[f'人工_{DIM_CN[dim]}'] = ''
        rv['_维度需复核'] = df.loc[review_mask].apply(
            lambda r: ','.join(d for d in DIMS if r[f'置信_{DIM_CN[d]}'] == 'review'), axis=1).values
        out_rv = SRC.rsplit('.', 1)[0] + '_复核队列.xlsx'
        rv.to_excel(out_rv, index=False)
        print(f'复核队列: {n_review} 行 → {out_rv}')

    for dim in DIMS:
        print(f'\n===== {DIM_CN[dim]} 分布 =====')
        print(df[f'分类_{DIM_CN[dim]}'].value_counts().to_string())


if __name__ == '__main__':
    main()
