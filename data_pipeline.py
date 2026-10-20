"""
数据清洗 + 固定三切分
1. 清洗: 变流器类→变流系统类, 丢弃 type=其他, 归一化去重合并(13组冲突人工裁定)
2. 切分: 按归一化名称分组 80/10/10, 近重复绝不跨折
3. 落盘: data/train.json, data/val.json, data/test.json —— 固定不变, 所有实验共用
"""
import json, os, re, random
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261020  # 固定切分种子,永不改动

# ============ 13 组近重复标签冲突的人工裁定 (归一化名称 → 三维标签) ============
ADJUDICATED = {
    '变桨电机碳刷':            ('耗材', '电气件', '变桨系统类'),   # 碳刷属耗材,电刷为电气件
    '一字螺丝刀':              ('工器具', '机械件', '通用类'),
    '光纤跳线接头2芯多模MILPCFCONNECTORANDENDPROTECTION2芯光缆米': ('备件', '电子件', '控制系统类'),  # 光纤通讯属控制网络
    'UPS电源延时继电器CT-ERD或': ('备件', '电气件', '控制系统类'),   # 继电器→电气件(与规则库口径一致)
    '轴刹车模块不含元件':        ('备件', '机械件', '偏航系统类'),   # EM-ENBR550MN28 为偏航刹车模块
    '偏航开关':                ('备件', '传感器', '偏航系统类'),   # 偏航限位/扭缆开关属位置传感
    'Emergencystop按钮头':      ('备件', '电气件', '控制系统类'),   # 急停回路属控制系统
    '轴流风机FDA_克菲尔':       ('备件', '电气件', '通用类'),      # 风机含电机,数据口径电气件略多
    '变频器风扇小':            ('备件', '机械件', '变流系统类'),
    '油管米直度':              ('备件', '液压件', '液压系统类'),   # 带长度/角度的油管为液压软管
    '针连接器公头母头一端公头一端母头双层屏蔽电缆长': ('备件', '电子件', '通用类'),  # 3条中2条为此标签
    '光端机小':                ('备件', '电子件', '控制系统类'),   # 光端机用于SCADA通讯
    '齿轮箱油泵电机':           ('备件', '电气件', '齿轮箱类'),    # 电机类→电气件
}

def norm(n):
    """与切分一致的归一化: 去规格数字/单位/空白"""
    n = re.sub(r'[0-9０-９.*×xX/\\\-—()+()mM]{2,}', ' ', n)
    return re.sub(r'\s+', '', n)

def main():
    with open(os.path.join(BASE, 'training_samples_v10k_merged.json')) as f:
        samples = json.load(f)
    print(f'原始样本: {len(samples)}')

    # ---- 1. 标签清洗 ----
    cleaned = []
    for s in samples:
        if s.get('scene') == '变流器类':
            s = dict(s, scene='变流系统类')
        if s.get('type') in ('其他', '', None):
            continue
        if not (s.get('type') and s.get('physical') and s.get('scene') and s.get('name')):
            continue
        if len(s['name']) < 2:
            continue
        cleaned.append(s)
    print(f'标签清洗后: {len(cleaned)}')

    # ---- 2. 归一化分组去重合并 ----
    groups = {}
    for s in cleaned:
        groups.setdefault(norm(s['name']), []).append(s)

    merged, conflicts_resolved, dropped_conflicts = [], 0, 0
    for nn, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        labels = {(g['type'], g['physical'], g['scene']) for g in group}
        # 代表名称取 source_count 最大(信息最全)的
        rep = max(group, key=lambda g: g.get('source_count') or 0)
        if len(labels) == 1:
            item = dict(rep)
        elif nn in ADJUDICATED:
            item = dict(rep)
            item['type'], item['physical'], item['scene'] = ADJUDICATED[nn]
            conflicts_resolved += 1
        else:
            dropped_conflicts += 1
            continue  # 未裁定的冲突组直接丢弃(保守)
        merged.append(item)
    print(f'合并去重后: {len(merged)} (裁定冲突 {conflicts_resolved} 组, 丢弃未裁定 {dropped_conflicts} 组)')

    # ---- 3. 按组切分 80/10/10 ----
    random.seed(SEED)
    keys = list(groups.keys())
    random.shuffle(keys)
    n = len(keys)
    n_test = round(n * 0.10)
    n_val = round(n * 0.10)
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])

    splits = {'train': [], 'val': [], 'test': []}
    for item in merged:
        k = norm(item['name'])
        splits['val' if k in val_keys else 'test' if k in test_keys else 'train'].append({
            'name': item['name'], 'type': item['type'],
            'physical': item['physical'], 'scene': item['scene'],
        })

    os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(BASE, 'data', f'{name}.json')
        with open(path, 'w') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f'{name}: {len(rows)} 条 → {path}')

    # ---- 4. 分布核对 ----
    for split, rows in splits.items():
        print(f'\n[{split}] n={len(rows)}')
        for dim in ['type', 'physical', 'scene']:
            dist = Counter(r[dim] for r in rows)
            top = ', '.join(f'{k}:{v/len(rows)*100:.0f}%' for k, v in dist.most_common(4))
            print(f'  {dim:9s} {top}')

if __name__ == '__main__':
    main()
