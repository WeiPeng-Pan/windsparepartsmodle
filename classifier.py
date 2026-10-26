"""
风电物料三维分类器 - 混合方案
规则 + fastText 互补

规则优先（高置信度关键词 → 直接出结果）
模型兜底（规则覆盖不了的用 fastText）

v2.1: 配套子词 fastText 模型(fasttext_v6_*,minn/maxn=2/4)
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
    # 大部件规则已移除——含"塔筒""发电机""齿轮箱"的物料 90%+是备件/耗材
    # 大部件分类完全交给 fastText 模型处理
]

# ============ 大部件同义词与能级规则(业务确认口径) ============
# 同义词: 叶片回转轴承=变桨轴承; 大齿圈=回转轴承; 桨叶=叶片;
#         高原配/及高原/超高原/速比/型号等只是整机规格限定词,不改变本体
# 能级规则: 名称含整机词时,若同时含配件词(螺栓/密封/泵/传感器等)则按配件归类
# 口径确认(2026-10-20): 轴承连接螺栓套件(含垫片螺母)一律按备件,不算大部件
BEARING_TERMS = re.compile(r"变桨轴承|偏航轴承|回转轴承|大齿圈")
BEARING_ACCESSORY = re.compile(
    r"润滑|加脂|油脂|密封|传感|编码|变送|监测|监视|视频|注油|废油|费油|油嘴|分配|"
    r"收集|集油|袋|垫片|垫圈|螺杆|螺栓|螺柱|螺母|螺钉|裂纹|螺栓孔|孔裂|模块|吊")
QUALIFIER_RE = re.compile(r"不含\S*|含\S*")
MAJOR_TERMS = re.compile(
    r"机舱罩壳|机舱前盖板|主轴总成|主轴轴承|主轴承|主轴组件|主轴轴系|风轮叶片|"
    r"变桨轴承|偏航轴承|回转轴承|大齿圈|叶片|桨叶|发电机|齿轮箱|主轴|塔筒|机舱罩|"
    r"机舱|导流罩|轮毂|主机架|副机架|变流器整机|双馈变流器|全功率变流器|风冷一体机|整机|主机")
# 配件降级词: 在「剥掉整机词后的剩余部分」里匹配(避免"发电机"被"电机"子串误伤)
# 注意: "润滑"单独出现常是规格限定词(如"高原配 润滑 速比"),仅部件形态(润滑泵/系统等)降级
MAJOR_ACCESSORY = re.compile(
    r"螺栓|螺柱|螺母|螺杆|螺钉|垫片|垫圈|垫|挡圈|密封|油封|轴封|碳刷|电刷|刷|刷架|"
    r"滑环|集电环|轴承座|轴承|滤|传感|感应|编码器|变送|发讯|监测|监视|测量|视频|声纹|"
    r"电缆|光纤|接头|线|绳|防水布|照明|灯|灭火|封边|爬梯|助爬|吊|架|支架|支撑|支承|"
    r"座|泵|加脂|风扇|通风机|轴流风机|风机$|转子风机|扇叶|通风|风口|风冷机|水冷机|"
    r"冷机|冷媒|换热|散热|扇热|冷却|(?<!风)(?<!发)(?<!电动)电机|电动机|高速轴|中速轴|低速轴|中心轴|高速|"
    r"低速|半环|半月环|月环|胀紧|涨套|轴套|套筒|转换|定心|联轴|齿条|减速机|减速器|"
    r"减速箱|变压器|液压站|总成|油|脂|胶|锤|钳|闸钳|阀|制动|刹车|集中|系统|开关|装置|控制|"
    r"模块|模件|模组|板|PCBA|继电器|接触器|断路器|短路|面板|UPS|ups|保险|熔断|熔芯|"
    r"组件|铜|销|导向|栓|记录|卡|表|游标|指示|温度|液位|管道|管路|管|软管|罩|壳|"
    r"端盖|锻件|透气|帽|天窗|标签|柜|机箱|箱|盒|手操|手柄|挡板|绝缘|防雷器|防雷箱|"
    r"转子防雷|浪涌|主控|整流|加热器|呼吸|干燥|储能|罐|蓄能|分配器|CMS|测速|驱动|"
    r"离心盘|收缩盘|盘|电池|充电|交换机|门锁|锁|减震|地脚|按钮|空开|安装|标识|标志|"
    r"扶手|拉手|滑轨|雨眉|顶盖|盖|梯|除湿|雷达|插头|拉伸|胀套|警告|测风|存储|"
    r"运输|费|部装|电源|不间断|电容|电阻|电抗|互感|二极管|母排|预充|单元|通讯|第|"
    r"适配|程序|Crowbar|crowbar|备品|清洗|弹力|内窥|限制|调整|锁紧|逃生包|"
    r"圈|上置|下置|空冷|接地|防尘|桶|PLC|逆变|监控|水冷|免爬|子站|从站|变频|软件|内存|备件|工控|电脑|"
    r"润滑系统|润滑站|润滑泵|润滑器|润滑分配|辅料|工装|服务")


def _major_component_match(name):
    """整机词+同义词+能级判定: 返回 '大部件' / 'demoted'(配件,排除大部件) / None(交给模型)

    能级实现: 回转轴承类本体优先;其他整机词命中后,把整机词从名称中剥离,
    若剩余部分含配件词则降级——降级时模型仍可判备件/耗材/工器具,但不得判大部件。
    """
    if BEARING_TERMS.search(name):
        if not BEARING_ACCESSORY.search(name):
            return '大部件'  # 回转轴承类本体(螺栓套件等配件除外)
        return 'demoted'
    if MAJOR_TERMS.search(name):
        # 先剥"含X/不含X"限定词(再做整机词剥离,避免"含叶片监测"被拆断漏出裸词)
        residue = QUALIFIER_RE.sub(' ', name)
        residue = MAJOR_TERMS.sub(' ', residue)
        if not MAJOR_ACCESSORY.search(residue):
            return '大部件'
        return 'demoted'
    return None

# physical 规则（v2.1: 子词模型在原各规则组触发集上达 85%+,全面超过规则,
# 仅保留与模型打平且确定性高的「传感器」组;油品走专用谓词）
PHYSICAL_RULES = [
    (["传感器", "接近开关", "压力开关"], "传感器"),
    (["碳刷", "电刷"], "电气件"),  # 业务裁定2026-10-20: 碳刷/电刷及其配件属电气件
]

# ============ 油品牌号 → scene (牌号用途经 websearch 确认, 2026-10-20) ============
# 仅对油品语境的名称生效; 型号可辨识系统者按系统归类(业务裁定)
# 依据: Omala=壳牌工业/风电齿轮油; OPTIGEAR/UNISYN=福斯合成齿轮油; EMGARD=BASF齿轮油;
#       Klüberplex BEM(PLEX)/STABYL=风电主轴/发电机轴承脂; RHODINA BBZ=壳牌叶片轴承脂;
#       GADUS=壳牌通用脂; GLEITMO=福斯变桨/偏航脂; CEPLATTYN=福斯开式齿轮脂;
#       DYNAGEAR=美孚得耐格偏航/变桨开式齿轮脂; MOLYKOTE=多功能脂
OIL_BRAND_RULES = [
    (["OMALA", "OPTIGEAR", "UNISYN", "EMGARD"], "齿轮箱类"),
    (["RHODINA", "BBZ", "GADUS", "STABYL", "GLEITMO", "MOLYKOTE",
      "CEPLATTYN", "DYNAGEAR", "PLEX", "BEM"], "润滑系统类"),
]
OILISH_RE = re.compile(r'润滑|油脂|黄油|锂基脂|GREASE|GEAR OIL', re.IGNORECASE)


def _oil_brand_scene(name):
    """油品牌号识别: 返回 scene 或 None; SHC 细分(XMP/GEAR=齿轮油, GREASE/脂=润滑脂)"""
    n = name.upper()
    if 'SHC' in n:
        if 'XMP' in n or 'GEAR' in n:
            return '齿轮箱类'
        if 'GREASE' in n or '脂' in name or '油' in name:
            return '润滑系统类'
        return None  # 裸SHC语境不明, 交模型
    for kws, scene in OIL_BRAND_RULES:
        if any(re.search(rf'\b{kw}\b', n) for kw in kws):
            return scene
    return None


# 油品判定：以"油"结尾（齿轮油/液压油/润滑油/变压器油等成品油）或含成品油脂词
# v2: 裸"油"子串误伤653条非油品物料（油管/油泵/注油接头等），已废弃
OIL_PRODUCTS = ["润滑脂", "黄油", "锂基脂", "废油"]

# ============ 同义词归一(全局预处理, 业务确认口径) ============
# 仅作用于规则层匹配(scene/physical/type规则); 模型输入保持原始名称,
# 避免与训练数据产生分布偏移。新同义词直接往表里追加。
SYNONYMS = [
    (r"叶片回转轴承", "变桨轴承"),   # 叶片回转轴承=变桨轴承
    (r"大齿圈", "回转轴承"),         # 大齿圈=回转轴承
    (r"回转轴承", "变桨轴承"),       # 回转轴承默认归变桨(偏航大齿圈等含系统词的不受影响)
    (r"桨叶", "叶片"),              # 桨叶=叶片
]
_SYN_RE = re.compile("|".join(p for p, _ in SYNONYMS))
_SYN_MAP = dict(SYNONYMS)


def _canon(name):
    """同义词归一: 返回规则层用标准名(原始名不动)"""
    return _SYN_RE.sub(lambda m: _SYN_MAP[m.group(0)], name)


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

        type_path = os.path.join(model_dir, 'fasttext_v6_type.bin')
        physical_path = os.path.join(model_dir, 'fasttext_v6_physical.bin')
        scene_path = os.path.join(model_dir, 'fasttext_v6_scene.bin')

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
        major = _major_component_match(name)
        type_rule = ('大部件' if major == '大部件' else None) or self._rule_match(_canon(name), TYPE_RULES)
        if type_rule:
            result['type'] = type_rule
            methods['type'] = 'rule'
        elif self.model_type:
            pred = self.model_type.predict(name)
            label = pred[0][0].replace('__label__', '')
            if label == '大部件' and major == 'demoted':
                # 能级降级: 配件不得判大部件,取剩余标签中概率最高者
                labels, probs = self.model_type.predict(name, k=-1)
                alt = max(((l, p) for l, p in zip(labels, probs)
                           if l != '__label__大部件'), key=lambda x: x[1])
                label = alt[0].replace('__label__', '')
                result['type'] = label
                methods['type'] = 'model'
                result.setdefault('prob', {})['type'] = alt[1]
            else:
                result['type'] = label
                methods['type'] = 'model'
                result.setdefault('prob', {})['type'] = pred[1][0]
        else:
            result['type'] = '备件'
            methods['type'] = 'default'
            result.setdefault('prob', {})['type'] = 0.0

        # ---- physical ----
        if _is_oil_product(name):
            physical_rule = '油品'
        else:
            physical_rule = self._rule_match(_canon(name), PHYSICAL_RULES)
        if physical_rule:
            result['physical'] = physical_rule
            methods['physical'] = 'rule'
        elif self.model_physical:
            pred = self.model_physical.predict(name)
            result['physical'] = pred[0][0].replace('__label__', '')
            methods['physical'] = 'model'
            result.setdefault('prob', {})['physical'] = pred[1][0]
        else:
            result['physical'] = '其他'
            methods['physical'] = 'default'
            result.setdefault('prob', {})['physical'] = 0.0

        # ---- scene ----
        scene_rule = self._rule_match(_canon(name), SCENE_RULES)
        if not scene_rule:
            scene_rule = _oil_brand_scene(name)  # 油品牌号识别(型号定系统)
        if not scene_rule and _is_oil_product(name) and re.search(r'主轴|主轴承', name):
            scene_rule = '润滑系统类'  # 业务裁定2026-10-20: 主轴润滑油脂属润滑系统(如主轴轴承润滑油)
        if scene_rule:
            result['scene'] = scene_rule
            methods['scene'] = 'rule'
        elif self.model_scene:
            pred = self.model_scene.predict(name)
            result['scene'] = pred[0][0].replace('__label__', '')
            methods['scene'] = 'model'
            result.setdefault('prob', {})['scene'] = pred[1][0]
        else:
            result['scene'] = '通用类'
            methods['scene'] = 'default'
            result.setdefault('prob', {})['scene'] = 0.0

        # ---- 置信度分层(val校准): auto≥0.9 / low 0.7~0.9 / review<0.7 ----
        # rule/default 判定: rule 置信等同 auto; default(名称过短)进 review
        tiers = {}
        for d in ('type', 'physical', 'scene'):
            if methods[d] == 'rule':
                p = 1.0
            elif methods[d] == 'default':
                p = 0.0
            else:
                p = result.get('prob', {}).get(d, 0.0)
            result.setdefault('prob', {})[d] = p
            tiers[d] = 'auto' if p >= 0.9 else ('review' if p < 0.7 else 'low')
        result['tier'] = tiers
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
