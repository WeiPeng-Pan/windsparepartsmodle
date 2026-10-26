"""
test 集人工抽检工具(评估协议第2步: 量化标签噪声天花板)
1. 从 test 分层抽样: 优先抽 模型与标签不一致 / 低置信 / 少数类, 外加随机对照
2. 每条附 flash 预填建议(可选, 需 FLASH_API_KEY)——人工只需 确认/改判, 不用从零标
3. 输出 data/audit_sample.xlsx: 人工在 人工_type/人工_physical/人工_scene 列填
   (同意现标签就留空, 不同意填正确值), 完成后跑:
   python3 apply_audit.py data/audit_sample.xlsx  → 统计噪声率 + 生成纠错记录

用法:
  python3 make_audit_sample.py                # 默认 150 条(不含flash预填)
  FLASH_API_KEY=xxx python3 make_audit_sample.py   # 附flash预填建议
"""
import json, os, sys, random
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

N_TOTAL = 150


def main():
    from classifier import WindClassifier
    random.seed(20261020)
    with open(os.path.join(BASE, 'data', 'test.json')) as f:
        test = json.load(f)
    clf = WindClassifier()

    rows = []
    for it in test:
        r = clf.classify(it['name'])
        dims = ['type', 'physical', 'scene']
        mismatch = [d for d in dims if r[d] != it[d]]
        low_conf = [d for d in dims if r['tier'][d] in ('review', 'low')]
        minority = it['type'] in ('大部件', '工器具') or it['scene'] in ('变流系统类', '润滑系统类', '液压系统类')
        rows.append({**it, 'pred': r, 'mismatch': mismatch, 'low_conf': low_conf, 'minority': minority})

    # 分层: 不一致优先 → 低置信 → 少数类 → 随机
    pool_p1 = [x for x in rows if x['mismatch']]
    pool_p2 = [x for x in rows if not x['mismatch'] and x['low_conf']]
    pool_p3 = [x for x in rows if not x['mismatch'] and not x['low_conf'] and x['minority']]
    pool_p4 = [x for x in rows if x not in pool_p1 + pool_p2 + pool_p3]
    random.shuffle(pool_p4)
    sample = (pool_p1[:60] + pool_p2[:40] + pool_p3[:30] + pool_p4[:20])[:N_TOTAL]
    print(f'抽样: 不一致 {min(60, len(pool_p1))} + 低置信 {min(40, len(pool_p2))} + '
          f'少数类 {min(30, len(pool_p3))} + 随机 {max(0, N_TOTAL - len(sample) + 20) - (N_TOTAL - len(sample))} '
          f'= {len(sample)} 条')

    # flash 预填建议(可选)
    key = os.environ.get('FLASH_API_KEY')
    if key:
        from relabel_weak import call_flash
        BATCH = 50
        for i in range(0, len(sample), BATCH):
            chunk = sample[i:i + BATCH]
            res = call_flash([x['name'] for x in chunk], key, 'https://api.deepseek.com', 'deepseek-chat')
            if res and len(res) == len(chunk):
                for x, fr in zip(chunk, res):
                    x['flash'] = (fr.get('type'), fr.get('physical'), fr.get('scene'))
                print(f'  flash 预填 {i // BATCH + 1} 批')
            else:
                print(f'  flash 批 {i // BATCH + 1} 失败(跳过预填)')

    # 输出
    try:
        import pandas as pd
        out = []
        for x in sample:
            out.append({
                '物料名称': x['name'],
                '现标签_type': x['type'], '模型_type': x['pred']['type'],
                '现标签_physical': x['physical'], '模型_physical': x['pred']['physical'],
                '现标签_scene': x['scene'], '模型_scene': x['pred']['scene'],
                'flash_type': x.get('flash', ('', '', ''))[0],
                'flash_physical': x.get('flash', ('', '', ''))[1],
                'flash_scene': x.get('flash', ('', '', ''))[2],
                '人工_type': '', '人工_physical': '', '人工_scene': '',
                '备注': '',
                '_抽样原因': ('不一致' if x['mismatch'] else '') +
                            ('+低置信' if x['low_conf'] else '') +
                            ('+少数类' if x['minority'] and not x['mismatch'] and not x['low_conf'] else ''),
            })
        path = os.path.join(BASE, 'data', 'audit_sample.xlsx')
        pd.DataFrame(out).to_excel(path, index=False)
        print(f'\n抽检清单 → {path}')
        print('填法: 与现标签一致留空; 不同意在 人工_* 列填正确值。完成后运行:')
        print('  python3 apply_audit.py data/audit_sample.xlsx')
    except ImportError:
        print('需要 pandas+openpyxl')


if __name__ == '__main__':
    main()
