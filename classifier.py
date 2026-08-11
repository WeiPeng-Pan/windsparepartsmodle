"""
风电物料三维分类器 - 混合方案
规则 + fastText 互补

规则优先（高置信度关键词 → 直接出结果）
模型兜底（规则覆盖不了的用 fastText）
"""

import fasttext
import os
import re

# ============ 规则库 ============

# scene 规则（关键词 → 场景）
# 优先级从高到低
SCENE_RULES = [
    (["偏航"], "偏航系统类"),
    (["变桨"], "变桨系统类"),
    (["齿轮箱", "齿轮箱类", "齿轮箱系统"], "齿轮箱类"),
    (["发电机", "发电机类", "发电机系统"], "发电机类"),
    (["液压", "液压系统", "液压站"], "液压系统类"),
    (["润滑", "润滑油", "润滑脂", "润滑系统"], "润滑系统类"),
    (["控制", "控制器", "控制柜", "控制模块", "PLC", "控制系统"], "控制系统类"),
    (["变流", "变流器", "变频器"], "变流系统类"),
]

# type 规则（仅用于高置信度明确关键词）
# 注意：不含"发电机""齿轮箱"等易误伤的大部件关键词
# 数据表明含这些词的物料 78%+ 是备件/耗材而非大部件
TYPE_RULES = [
    # 耗材 - 明确消耗品
    (["滤芯", "滤棉", "滤网", "O型圈", "油封", "碳刷", "扎带", "擦机布", "擦拭纸"], "耗材"),
    # 工器具 - 明确工具
    (["扳手", "螺丝刀", "万用表", "电钻", "角磨机", "钳子", "吊带", "千斤顶", "手锤", "撬棍", "测试仪", "示波器", "兆欧表", "液压扳手", "扭力扳手", "吊环"], "工器具"),
    # 大部件规则已移除——含"塔筒""发电机""齿轮箱"的物料 90%+是备件/耗材
    # (如"塔筒灯""发电机集电环""齿轮箱滤芯")，规则匹配误伤率太高
    # 大部件分类完全交给 fastText 模型处理
]

# physical 规则（可选增强）
PHYSICAL_RULES = [
    (["油", "润滑脂", "润滑油", "液压油", "齿轮油"], "油品"),
    (["密封圈", "O型圈", "油封", "密封垫", "密封条", "密封件"], "密封件"),
    (["传感器", "探头", "变送器", "编码器", "接近开关", "压力开关"], "传感器"),
    (["PLC", "CPU", "模块", "芯片", "电路板", "PCB", "电源模块", "通讯模块"], "电子件"),
    (["液压", "液压阀", "液压泵", "液压缸", "油缸", "液压站", "液压件"], "液压件"),
    (["电机", "断路器", "接触器", "继电器", "变压器", "变频器", "开关", "熔断器", "电缆", "电线"], "电气件"),
    (["轴承", "螺栓", "螺母", "垫圈", "弹簧", "销", "轴", "齿轮", "链条", "联轴器", "法兰", "管接头", "支架", "壳体", "罩", "盖板", "法兰"], "机械件"),
]


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

        type_path = os.path.join(model_dir, 'fasttext_10k_type.bin')
        physical_path = os.path.join(model_dir, 'fasttext_10k_physical.bin')
        scene_path = os.path.join(model_dir, 'fasttext_10k_scene.bin')

        if os.path.exists(type_path):
            self.model_type = fasttext.load_model(type_path)
        if os.path.exists(physical_path):
            self.model_physical = fasttext.load_model(physical_path)
        if os.path.exists(scene_path):
            self.model_scene = fasttext.load_model(scene_path)

    def _rule_match(self, name, rules):
        """关键词匹配，返回第一个命中的标签"""
        if not name:
            return None
        for keywords, label in rules:
            for kw in keywords:
                if kw in name:
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
