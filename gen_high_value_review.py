"""
高价值物料复核清单(单价辅助识别大部件)
思路: 单价是识别大部件的强信号——风机整机级物料极少低于30万
  1. 按单价倒序, 取单价 >= 阈值(默认30万)的物料
  2. 当前判为非大部件的 → 高优先级复核(可能是漏识别的大部件)
  3. 一并列出所有高单价物料供参考(不仅大部件)
输出: <输入名>_高价值复核.xlsx
用法: python3 gen_high_value_review.py <已分类.xlsx> [阈值万] [单价列名]
"""
import os, sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    src = sys.argv[1]
    threshold = float(sys.argv[2]) * 10000 if len(sys.argv) > 2 else 300000
    price_col = sys.argv[3] if len(sys.argv) > 3 else '单价'

    df = pd.read_excel(src)
    if price_col not in df.columns:
        print(f'缺列 {price_col}; 现有: {list(df.columns)[:20]}')
        return

    name_col = next((c for c in ['清洗后名称', '物料名称', 'name'] if c in df.columns), None)
    type_col = next((c for c in ['分类_类型', '分类_type'] if c in df.columns), None)
    if not name_col or not type_col:
        print('缺名称列或分类列'); return

    print(f'{len(df)} 行, 单价>={threshold/10000:.0f}万: {(df[price_col] > threshold).sum()} 行')

    # 按唯一物料名聚合(取最高单价)
    agg = df.groupby(name_col, as_index=False).agg(
        最高单价=(price_col, 'max'), 出库行数=(price_col, 'size'))
    agg = agg.merge(df[[name_col, type_col]].drop_duplicates(name_col), on=name_col, how='left')
    hi = agg[agg['最高单价'] >= threshold].sort_values('最高单价', ascending=False)

    suspects = hi[hi[type_col] != '大部件']
    print(f'\n高单价唯一物料 {len(hi)} 个:')
    print(f'  已判大部件 {len(hi) - len(suspects)} | 非大部件(待确认) {len(suspects)}')

    out = src.replace('.xlsx', '_高价值复核.xlsx')
    with pd.ExcelWriter(out) as w:
        suspects.to_excel(w, sheet_name='待确认类型', index=False)
        hi.to_excel(w, sheet_name='全部高价值', index=False)
    print(f'\n→ {out}')
    print(f'  待确认类型 {len(suspects)} 个(单价≥{threshold/10000:.0f}万但未判大部件)')
    print('  确认是整机大部件的, 把名称加到 classifier.py 的 MAJOR_TERMS 或告知即可')
    for _, r in suspects.head(15).iterrows():
        print(f'    {r["最高单价"]/10000:>8.1f}万 | {str(r[name_col])[:40]:42s} | {r[type_col]}')


if __name__ == '__main__':
    main()
