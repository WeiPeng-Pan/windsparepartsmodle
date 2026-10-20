"""
定向补样: 大部件/变流系统类/润滑系统类 三个弱类
1. 从原始业务清单(询价+呆滞+定制, 不读22MB出库表以节省时间)抽取候选名称
2. flash 打标(断点续跑, 进度存 data/augment_progress.json)
3. 构建增强集 data/augment.json (弱类命中 + 随机对照样本)

用法: FLASH_API_KEY=xxx python3 augment_data.py
"""
import json, os, re, sys, random
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from clean_desc import clean_name
from relabel_weak import call_flash
from data_pipeline import norm as dup_norm  # 同一归一化,防近重复泄漏

random.seed(20261020)

# ============ 1. 候选池 ============
def build_pool():
    cache = os.path.join(BASE, 'data', 'pool_names.json')
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    names = set()
    for fn in ['运达历史询价清单.xlsx', '运达呆滞件清单.xlsx', '运达定制件清单.xlsx']:
        path = os.path.join(BASE, fn)
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path)
        cols = [c for c in df.columns if c in ('物资描述', '规格型号', '部件名称', '型号', '规格')]
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in cols if pd.notna(row[c])]
            nm = clean_name(' '.join(parts))
            if len(nm) >= 2:
                names.add(nm)
        print(f'{fn}: 累计 {len(names)}')
    names = sorted(names)
    with open(cache, 'w') as f:
        json.dump(names, f, ensure_ascii=False)
    return names

# 弱类定向关键词
TIER1_MAJOR = re.compile(r'整机|塔筒|轮毂|机舱|导流罩|主机架|副机架|叶根|叶片[^螺电]')  # 强大部件信号
TIER2_MAJOR = re.compile(r'发电机|齿轮箱|主轴|主轴承|变桨轴承|偏航轴承|叶片')
CONVERTER = re.compile(r'变流|变频|逆变|IGBT|整流|功率模块|变流器')
LUBRIC = re.compile(r'润滑|加油|油分配|油脂泵|黄油泵|油站|集中润滑')

def select_candidates(pool, existing_norms):
    seen = set(existing_norms)  # 已在 train/val/test 的(归一化后)排除
    cand = []
    for nm in pool:
        k = dup_norm(nm)
        if k in seen or not k or k in cand:
            continue
        cand.append((nm, k))
    tier1 = [nm for nm, k in cand if TIER1_MAJOR.search(nm)][:2500]
    tier2 = [nm for nm, k in cand if TIER2_MAJOR.search(nm) and not TIER1_MAJOR.search(nm)][:1500]
    conv = [nm for nm, k in cand if CONVERTER.search(nm)][:900]
    lub = [nm for nm, k in cand if LUBRIC.search(nm)][:900]
    used = set(tier1) | set(tier2) | set(conv) | set(lub)
    general = [nm for nm, k in cand if nm not in used]
    general = random.sample(general, min(500, len(general)))
    all_sel = tier1 + tier2 + conv + lub + general
    print(f'候选: 大部件T1 {len(tier1)} + T2 {len(tier2)} + 变流 {len(conv)} + 润滑 {len(lub)} + 随机 {len(general)} = {len(all_sel)}')
    return all_sel

# ============ 2. flash 打标(断点续跑) ============
def label_all(names, key):
    prog_path = os.path.join(BASE, 'data', 'augment_progress.json')
    results = []
    done_names = set()
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            results = json.load(f)
        done_names = {r['name'] for r in results}
    todo = [n for n in names if n not in done_names]
    print(f'已完成 {len(done_names)}, 待打标 {len(todo)}')
    BATCH = 50
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        res = call_flash(chunk, key, 'https://api.deepseek.com', 'deepseek-chat')
        if res and len(res) == len(chunk):
            for nm, r in zip(chunk, res):
                results.append({'name': nm, 'type': r.get('type'), 'physical': r.get('physical'), 'scene': r.get('scene')})
            with open(prog_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
            print(f'  批 {i//BATCH+1}/{(len(todo)+BATCH-1)//BATCH}: +{len(chunk)} (累计{len(results)})')
        else:
            print(f'  批 {i//BATCH+1} 失败,下轮重试')
            with open(prog_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
            break
    return results

def main():
    key = os.environ.get('FLASH_API_KEY')
    if not key:
        print('缺 FLASH_API_KEY'); return

    pool = build_pool()
    with open(os.path.join(BASE, 'data', 'train.json')) as f: tr = json.load(f)
    with open(os.path.join(BASE, 'data', 'val.json')) as f: va = json.load(f)
    with open(os.path.join(BASE, 'data', 'test.json')) as f: te = json.load(f)
    existing = {dup_norm(x['name']) for x in tr + va + te}

    cand = select_candidates(pool, existing)
    results = label_all(cand, key)
    print(f'打标累计: {len(results)}')

    from collections import Counter
    print('\ntype 分布:', Counter(r['type'] for r in results).most_common())
    print('scene 弱类命中:', Counter(r['scene'] for r in results if r['scene'] in ('变流系统类', '润滑系统类')).most_common())

if __name__ == '__main__':
    main()
