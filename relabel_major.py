"""
按权威大部件口径重打标 type 维度
- 候选: train_v3/val/test 中 名称含大部件关键词 或 现标为大部件 的全部物料
- flash 按新口径打标, 只采用 type 维度变更; 用户明确裁定项作为最终覆盖
- 产出: data/major_relabel_changes.json (审计日志)
"""
import json, os, sys, re
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from relabel_weak import call_flash

# 权威口径: 大部件本体关键词(名称含即候选)
MAJOR_KW = ['叶片', '发电机', '齿轮箱', '主轴', '主轴承', '变桨轴承', '偏航轴承',
            '塔筒', '机舱', '导流罩', '轮毂', '主机架', '副机架', '变流器', '整机', '主机']
# 用户明确裁定: 这些总成即使像整机也是备件
FORCE_SPARE = re.compile(r'减速机|减速器|减速箱|液压站|变压器|齿条|变桨电机')
# 用户明确裁定: 这些本体是大部件
FORCE_MAJOR = re.compile(r'机舱罩|机舱壳|导流罩|轮毂|主机架|塔筒|变流器整机')

def load(p):
    with open(os.path.join(BASE, 'data', f'{p}.json')) as f:
        return json.load(f)

def main():
    key = os.environ.get('FLASH_API_KEY')
    if not key:
        print('缺 FLASH_API_KEY'); return

    splits = {'train_v3': load('train_v3'), 'val': load('val'), 'test': load('test')}
    cand = []  # (split, name, old_type)
    seen = set()
    for sp, rows in splits.items():
        for it in rows:
            hit = it['type'] == '大部件' or any(k in it['name'] for k in MAJOR_KW)
            if hit and (sp, it['name']) not in seen:
                seen.add((sp, it['name']))
                cand.append({'split': sp, 'name': it['name'], 'old': it['type']})
    print(f'候选(含关键词或现标大部件): {len(cand)} 条')
    print('按split:', Counter(c['split'] for c in cand))

    prog_path = os.path.join(BASE, 'data', 'major_relabel_progress.json')
    results = {}
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            for r in json.load(f):
                results[r['name']] = r
    todo = [c for c in cand if c['name'] not in results]
    print(f'已完成 {len(results)}, 待打标 {len(todo)}')

    BATCH = 50
    def label_chunk(chunk):
        """返回本批中成功打标的 name→type 字典; 失败批自动切小块重试"""
        res = call_flash([c['name'] for c in chunk], key, 'https://api.deepseek.com', 'deepseek-chat')
        got = {}
        if res:
            for r in res:
                if isinstance(r, dict) and r.get('name') and r.get('type'):
                    got[r['name']] = r['type']
        missing = [c for c in chunk if c['name'] not in got]
        if missing and len(chunk) > 10:
            for j in range(0, len(missing), 10):
                sub = missing[j:j+10]
                r2 = call_flash([c['name'] for c in sub], key, 'https://api.deepseek.com', 'deepseek-chat')
                if r2:
                    for r in r2:
                        if isinstance(r, dict) and r.get('name') and r.get('type'):
                            got[r['name']] = r['type']
            missing = [c for c in chunk if c['name'] not in got]
        return got, missing

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        got, missing = label_chunk(chunk)
        for c in chunk:
            if c['name'] in got:
                results[c['name']] = {'name': c['name'], 'type': got[c['name']]}
        with open(prog_path, 'w') as f:
            json.dump(list(results.values()), f, ensure_ascii=False)
        print(f'  批 {i//BATCH+1}/{(len(todo)+BATCH-1)//BATCH}: +{len(got)} (累计{len(results)}' +
              (f', 未回 {len(missing)}' if missing else '') + ')')

    # 预览变更方向
    changes = Counter()
    for c in cand:
        r = results.get(c['name'])
        if r and r['type'] in ('大部件', '备件', '耗材', '工器具') and r['type'] != c['old']:
            changes[(c['old'], r['type'])] += 1
    print('\nflash 建议的 type 变更方向:')
    for (o, n), v in changes.most_common(10):
        print(f'  {o}→{n}: {v}')

if __name__ == '__main__':
    main()
