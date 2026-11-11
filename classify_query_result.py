"""批量查询结果 xlsx 批量分类
口径与 classify_wuliao.py 一致: 物资名称+规格/型号 拼接后过 v6 混合分类器,
输出: 原列 + 分类_类型/分类_物理属性/分类_场景 三列 → <名>_已分类.xlsx

用法: python3 classify_query_result.py <input.xlsx>
"""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import WindClassifier
from gen_review_queue import DIMS, DIM_CN

NAME_COL = '物资名称'
SPEC_COLS = ['规格/型号']


def main():
    src = sys.argv[1]
    df = pd.read_excel(src)
    parts = [df[NAME_COL].fillna('').astype(str).str.replace(r'[\r\n]+', ' ', regex=True)]
    for c in SPEC_COLS:
        if c in df.columns:
            parts.append(df[c].fillna('').astype(str).str.replace(r'[\r\n]+', ' ', regex=True))
    df['_cls_input'] = parts[0].str.cat(parts[1:], sep=' ').str.strip()
    print(f'输入: {len(df)} 行, 分类输入 = {" + ".join([NAME_COL] + [c for c in SPEC_COLS if c in df.columns])}')

    clf = WindClassifier()
    t0 = time.time()
    names = df['_cls_input'].tolist()
    uniq = sorted(set(n for n in names if len(n) >= 2))
    results = {n: clf.classify(n) for n in uniq}
    print(f'分类 {len(uniq)} 个唯一名, {time.time()-t0:.1f}s')

    default = {'type': '备件', 'physical': '其他', 'scene': '通用类'}
    for dim in DIMS:
        df[f'分类_{DIM_CN[dim]}'] = [results.get(n, {}).get(dim) or default[dim] for n in names]

    out = src.rsplit('.', 1)[0] + '_已分类.xlsx'
    df.drop(columns=['_cls_input']).to_excel(out, index=False)
    print(f'已保存: {out}')

    for dim in DIMS:
        print(f'\n===== {DIM_CN[dim]} 分布 =====')
        print(df[f'分类_{DIM_CN[dim]}'].value_counts().to_string())


if __name__ == '__main__':
    main()
