"""
纠错入库(通用): 把人工填好的复核队列/抽检清单 并入 data/corrections.json
兼容两种表(都靠 人工_* 列识别填写, 物料名称列定位):
  - gen_review_queue.py 产出的 *_复核队列.xlsx (现标签在 分类_类型/物理属性/场景 列)
  - apply_audit/make_audit_sample 产出的 audit_sample.xlsx   (现标签在 现标签_type/... 列)
填法: 与现标签一致留空; 不同意填正确值。
用法: python3 apply_review.py <填好的.xlsx>
"""
import json, os, sys
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
DIM_COLS = {  # 人工列名 -> (维度, 现标签列的候选名)
    '人工_type': ('type', ['现标签_type', '分类_类型']),
    '人工_physical': ('physical', ['现标签_physical', '分类_物理属性']),
    '人工_scene': ('scene', ['现标签_scene', '分类_场景']),
}


def main(path):
    import pandas as pd
    df = pd.read_excel(path)
    if '物料名称' not in df.columns:
        print('缺 物料名称 列'); return

    corr_path = os.path.join(BASE, 'data', 'corrections.json')
    corr = json.load(open(corr_path)) if os.path.exists(corr_path) else []
    known = {(c['name'], c['dim']) for c in corr}

    added = unchanged = 0
    for _, r in df.iterrows():
        name = r['物料名称']
        for col, (dim, old_candidates) in DIM_COLS.items():
            if col not in df.columns:
                continue
            v = r[col]
            if pd.isna(v) or not str(v).strip():
                unchanged += 0  # 留空=同意现标签
                continue
            new = str(v).strip()
            old = next((str(r[c]) for c in old_candidates if c in df.columns and pd.notna(r[c])), '')
            if new == old:
                continue
            if (name, dim) in known:
                continue
            corr.append({'name': str(name), 'dim': dim, 'old': old, 'new': new,
                         'source': 'review', 'date': str(date.today())})
            added += 1

    with open(corr_path, 'w') as f:
        json.dump(corr, f, ensure_ascii=False, indent=1)
    print(f'新增纠错 {added} 条 → data/corrections.json (累计 {len(corr)})')
    print('下一步: python3 retrain_with_feedback.py')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(sys.argv[1])
