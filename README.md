# WindSparePartsModel 风电物料三维分类模型

风电备件物料智能分类模型（fastText 子词 + 规则混合方案），对物料名称做三维自动标注，毫秒级推理、零 API 成本。

## 三维分类体系

| 维度 | 标签 |
|:-----|:-----|
| **type 类型** | 备件 / 耗材 / 工器具 / 大部件 |
| **physical 物理属性** | 电气件 / 机械件 / 电子件 / 传感器 / 油品 / 液压件 / 密封件 / 其他 |
| **scene 使用场景** | 通用类 / 控制系统类 / 发电机类 / 齿轮箱类 / 变流系统类 / 变桨系统类 / 液压系统类 / 偏航系统类 / 润滑系统类 |

> **大部件权威口径**：仅限整机本体——叶片（桨叶/风轮叶片）、发电机、齿轮箱、主轴（主轴总成/轴系）、
> 主轴承（主轴轴承）、变桨轴承（叶片回转轴承）、偏航轴承、塔筒、机舱罩/机舱壳/机舱前盖板、导流罩、
> 轮毂、主机架、变流器整机；变桨减速机、液压站总成、变压器等总成级部件属备件；上述整机的配件
> （螺栓/碳刷/密封圈等）按备件/耗材归类。
> **同义词与能级**（classifier.py 内置规则）：大齿圈=回转轴承；"高原配/速比/型号"等是整机规格限定词
> 不改变本体；名称含多个部件词时按能级判定——整机词+配件词并存则按配件归类；
> **轴承连接螺栓套件（含垫片螺母）一律按备件，不算大部件**。
> **耗材** = 定检消耗品（滤芯滤网、密封圈/O型圈/油封、润滑油/脂、清洗剂、胶、扎带）；
> **备件** = 其余替换件；**工器具** = 维修检测工具。

## 模型效果

基于约 **1 万条 LLM 辅助标注样本**训练（固定三切分 train/val/test = 7967/996/996，
近重复合并且同折、13 组标签冲突人工裁定，test 集从未参与训练与调参；
train 另并入 1280 条 flash 定向补样——大部件/变流/润滑弱类，见 augment_data.py）：

| 维度 | 纯模型(val) | 混合方案(test) | macro-F1 |
|:-----|:-----------:|:------------:|:--------:|
| type | 94.1% | **93.7%** | 85.6 |
| physical | 83.2% | **81.5%** | 81.4 |
| scene | 82.3% | **81.7%** | 80.7 |

三维全对率 **63.6%**。推理速度约 **8 万条/秒**（Apple M 系列 CPU）。

> 标签质量说明：训练/验证集经 flash 复核修正 88 处标签、按大部件权威口径重打标 206 处、
> 按同义词/能级规则提升 45 处（审计日志见 data/ 下 label_fixes / major_relabel_changes /
> synonym_promotions 三个 json），test 标签随口径同步更新。
> type 大部件 recall 80%（test 15 条）。

## 安装

```bash
pip install fasttext
```

> 注意：fasttext 依赖 C++ 编译，macOS/Linux 上通常直接可用；Windows 用户建议通过 WSL 或 Conda 安装。

## 快速开始

```python
from classifier import WindClassifier

clf = WindClassifier()

result = clf.classify("偏航刹车片 SK8010")
print(result)
# {'type': '备件', 'physical': '机械件', 'scene': '偏航系统类', 'method': {...}}
```

`result['method']` 表示每个维度的判定来源（`rule` 规则 / `model` 模型 / `default` 兜底）。

### 批量分类

```python
names = ["齿轮箱滤芯密封圈", "PLC通讯模块", "液压扳手", "塔筒灯"]
for name in names:
    r = clf.classify(name)
    print(name, "→", r['type'], r['physical'], r['scene'])
```

## 文件结构

```
├── classifier.py              # 分类器（规则 + fastText 混合）
├── models/
│   ├── fasttext_v2_type.bin       # type 分类模型
│   ├── fasttext_v2_physical.bin   # physical 分类模型
│   └── fasttext_v2_scene.bin      # scene 分类模型
├── data_pipeline.py           # 数据清洗 + 固定三切分（近重复同折）
├── apply_label_fixes.py       # 高置信标签修正重放（flash+模型双确认）
├── augment_data.py            # 弱类定向补样（业务清单抽候选 + flash 打标）
├── relabel_weak.py            # 弱类可疑样本 flash 重打标
├── retrain_clean.py           # 干净重训 + test 报告
├── finalize_v2.py             # 子词模型最终配方（生产模型由它产出）
├── example.py                 # 使用示例
└── data/                      # 固定切分与增强集（业务数据,不入库）
```

## 技术说明

- **模型层（主力）**：fastText 开启子词特征（minn=2, maxn=4），中文物料名按字符 n-gram 泛化——
  相比无子词版本纯模型平均准确率提升约 19 个百分点。
- **规则层（高置信兜底）**：仅保留在验证集上不劣于模型的关键词组（偏航/变桨/齿轮箱/发电机/PLC
  等场景词、工器具词、传感器词）；油品用专用谓词（以"油"结尾或含润滑脂等），规避"油管/油泵"类误伤。
- **数据**：训练样本来自风力发电行业备件物料，涵盖控制系统、发电机、齿轮箱、变流器、变桨、偏航、液压、润滑等主要子系统。


## License

MIT
