"""
风电物料三维分类器 - 混合方案
规则 + fastText 互补

规则优先（高置信度关键词 → 直接出结果）
模型兜底（规则覆盖不了的用 fastText）

v2.1: 配套子词 fastText 模型(fasttext_v2_*,minn/maxn=2/4)
- 规则库按「规则 vs 模型在各自触发集上的准确率」裁剪,只保留规则不劣于模型的高置信组
- 油品判定用专用谓词: 以"油"结尾 或 含润滑脂/黄油等成品油词
"""

import fasttext
import os
import re

# ============ 规则库 ============

# scene 规则（关键词 → 场景）
# v2.1: 子词模型变强后按 val 集逐组对比裁剪——只保留规则≥模型的高置信组
SCENE_RULES = [
    (["偏航"], "偏航系统类"),
    (["变桨"], "变桨系统类"),
    (["齿轮箱"], "齿轮箱类"),
    (["发电机"], "发电机类"),
    (["PLC"], "控制系统类"),
]

# type 规则（仅用于高置信度明确关键词）
# v2.1: 删"碳刷"(精度31%)和"滤芯"(模型100% vs 规则90%)
TYPE_RULES = [
    # 耗材 - 明确消耗品
    (["滤棉", "滤网", "O型圈", "油封", "扎带", "擦机布", "擦拭纸"], "耗材"),
    # 工器具 - 明确工具
    (["扳手", "螺丝刀", "万用表", "电钻", "角磨机", "钳子", "吊带", "千斤顶", "手锤", "撬棍", "测试仪", "示波器", "兆欧表", "液压扳手", "扭力扳手", "吊环"], "工器具"),
]

# physical 规则（v2.1: 子词模型在原各规则组触发集上达 85%+,全面超过规则,
# 仅保留与模型打平且确定性高的「传感器」组;油品走专用谓词）
PHYSICAL_RULES = [
    (["传感器", "接近开关", "压力开关"], "传感器"),
]

# 油品判定：以"油"结尾（齿轮油/液压油/润滑油/变压器油等成品油）或含成品油脂词
# v2: 裸"油"子串误伤653条非油品物料（油管/油泵/注油接头等），已废弃
OIL_PRODUCTS = ["润滑脂", "黄油", "锂基脂", "废油"]


def _is_oil_product(name):
    return name.rstrip().endswith("油") or any(kw in name for kw in OIL_PRODUCTS)


class WindClassifier:
    """风电物料三维分类器"""

    def __init__(self, model_dir=None):
        """加载 fastText 模型（如果存在）

        model_dir: 模型目录，默认使用本文件同级的 models/ 目录
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.model_dir = model_dir
        self.model_type = None
        self.model_physical = None
        self.model_scene = None

        type_path = os.path.join(model_dir, 'fasttext_v2_type.bin')
        physical_path = os.path.join(model_dir, 'fasttext_v2_physical.bin')
        scene_path = os.path.join(model_dir, 'fasttext_v2_scene.bin')

        if os.path.exists(type_path):
            self.model_type = fasttext.load_model(type_path)
        if os.path.exists(physical_path):
            self.model_physical = fasttext.load_model(physical_path)
        if os.path.exists(scene_path):
            self.model_scene = fasttext.load_model(scene_path)

    def _rule_match(self, name, rules):
        """关键词匹配，返回第一个命中的标签；条目可含第3个元素(正则列表)"""
        if not name:
            return None
        for entry in rules:
            keywords, label = entry[0], entry[1]
            for kw in keywords:
                if kw in name:
                    return label
            for rx in (entry[2] if len(entry) > 2 else []):
                if rx.search(name):
                    return label
        return None

    def classify(self, name):
        """三维分类：规则优先 + 模型兜底"""
        if not name or len(name) < 2:
            return {'type': '备件', 'physical': '其他', 'scene': '通用类', 'method': 'default'}

        result = {}
        methods = {}

        # ---- type ----
        type_rule = self._rule_match(name, TYPE_RULES)
        if type_rule:
            result['type'] = type_rule
            methods['type'] = 'rule'
        elif self.model_type:
            pred = self.model_type.predict(name)
            result['type'] = pred[0][0].replace('__label__', '')
            methods['type'] = 'model'
        else:
            result['type'] = '备件'
            methods['type'] = 'default'

        # ---- physical ----
        if _is_oil_product(name):
            physical_rule = '油品'
        else:
            physical_rule = self._rule_match(name, PHYSICAL_RULES)
        if physical_rule:
            result['physical'] = physical_rule
            methods['physical'] = 'rule'
        elif self.model_physical:
            pred = self.model_physical.predict(name)
            result['physical'] = pred[0][0].replace('__label__', '')
            methods['physical'] = 'model'
        else:
            result['physical'] = '其他'
            methods['physical'] = 'default'

        # ---- scene ----
        scene_rule = self._rule_match(name, SCENE_RULES)
        if scene_rule:
            result['scene'] = scene_rule
            methods['scene'] = 'rule'
        elif self.model_scene:
            pred = self.model_scene.predict(name)
            result['scene'] = pred[0][0].replace('__label__', '')
            methods['scene'] = 'model'
        else:
            result['scene'] = '通用类'
            methods['scene'] = 'default'

        result['method'] = methods
        return result

    def classify_batch(self, names):
        return [self.classify(n) for n in names]


if __name__ == '__main__':
    # 简单测试
    clf = WindClassifier()

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
    ]

    print(f"{'物料名称':35s} | {'type':6s} | {'physical':6s} | {'scene':6s} | 方法")
    print("-" * 85)
    for name in test_cases:
        r = clf.classify(name)
        m = r.pop('method')
        print(f"{name:35s} | {r['type']:6s} | {r['physical']:6s} | {r['scene']:6s} | type={m['type']} phy={m['physical']} scn={m['scene']}")
