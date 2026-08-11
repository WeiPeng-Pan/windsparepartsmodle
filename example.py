"""WindSparePartsModel 使用示例"""
import os
from classifier import WindClassifier

# 加载分类器（默认读取本文件同级的 models/ 目录）
clf = WindClassifier()

print("=" * 60)
print("风电物料三维分类示例")
print("=" * 60)

test_cases = [
    "偏航刹车片 SK8010",
    "齿轮箱滤芯密封圈",
    "发电机集电环",
    "液压扳手",
    "PLC通讯模块",
    "清洗剂 环保型",
    "密封圈 175*3.55mm",
    "变桨电机",
    "塔筒灯",
    "温控阀总成",
    "润滑油 壳牌 XMP 320",
    "碳刷 20*30*50",
    "偏航减速器",
    "变流器IGBT模块",
]

print(f"\n{'物料名称':<24s} | {'type':6s} | {'physical':6s} | {'scene':8s}")
print("-" * 60)
for name in test_cases:
    r = clf.classify(name)
    m = r.pop('method')
    print(f"{name:<24s} | {r['type']:<6s} | {r['physical']:<6s} | {r['scene']:<8s} | t={m['type']} p={m['physical']} s={m['scene']}")

# 批量分类演示
print("\n" + "=" * 60)
print("批量分类 (classify_batch)")
print("=" * 60)
batch = ["发电机碳刷", "偏航轴承", "齿轮箱油泵", "变桨控制柜", "压力传感器"]
results = clf.classify_batch(batch)
for name, r in zip(batch, results):
    print(f"  {name:<10s} → {r['type']}/{r['physical']}/{r['scene']}")
