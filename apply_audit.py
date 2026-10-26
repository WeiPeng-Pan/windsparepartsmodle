"""
消费人工抽检结果: 统计标签噪声率 + 生成纠错记录
输入: data/audit_sample.xlsx (人工已填 人工_* 列)
输出:
  1. 噪声率报告(各维度: 人工改判率 = 标签噪声率的上界估计)
  2. 纠错追加到 data/corrections.json (统一格式, 供 retrain_with_feedback.py 使用)
"""
import json, os, sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DIMS = ['type', 'physical', 'scene']


def load_corrections():
    path = os.path.join(BASE, 'data', 'corrections.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_corrections(corr):
    with open(os.path.join(BASE, 'data', 'corrections.json'), 'w') as f:
        json.dump(corr, f, ensure_ascii=False, indent=1)


def main(path=None):
    import pandas as pd
    path = path or os.path.join(BASE, 'data', 'audit_sample.xlsx')
    df = pd.read_excel(path)
    filled = df[df['人工_type'].notna() | df['人工_physical'].notna() | df['人工_scene'].notna()]
    n_audit = len(df)
    n_filled = len(filled)
    print(f'抽检 {n_audit} 条, 人工填了 {n_filled} 条')

    corr = load_corrections()
    known = {(c['name'], c['dim']) for c in corr}
    stats = {d: [0, 0] for d in DIMS}  # [改判数, 有效填写数]
    new_corr = 0
    for _, r in filled.iterrows():
        for d in DIMS:
            human = r[f'人工_{d}']
            if pd.isna(human) or not str(human).strip():
                continue
            human = str(human).strip()
            stats[d][1] += 1
            if human != str(r[f'现标签_{d}']):
                stats[d][0] += 1
                if (r['物料名称'], d) not in known:
                    corr.append({'name': r['物料名称'], 'dim': d, 'old': str(r[f'现标签_{d}']),
                                 'new': human, 'source': 'audit', 'date': '2026-10-20'})
                    new_corr += 1
    save_corrections(corr)

    print('\n标签噪声率(人工改判率, 噪声上界):')
    for d in DIMS:
        c, n = stats[d]
        rate = f'{c/n*100:.1f}%' if n else '无数据'
        print(f'  {d:9s}: {c}/{n} = {rate}')
    print(f'\n新增纠错 {new_corr} 条 → data/corrections.json (累计 {len(corr)})')
    print('下一步: FLASH_API_KEY 可选, python3 retrain_with_feedback.py')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
