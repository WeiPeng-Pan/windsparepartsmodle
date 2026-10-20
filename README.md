# WindSparePartsModel 风电物料三维分类模型

风电备件物料智能分类模型（fastText 子词 + 规则混合方案），对物料名称做三维自动标注，毫秒级推理、零 API 成本。

## 三维分类体系

| 维度 | 标签 |
|:-----|:-----|
| **type 类型** | 备件 / 耗材 / 工器具 / 大部件 |
| **physical 物理属性** | 电气件 / 机械件 / 电子件 / 传感器 / 油品 / 液压件 / 密封件 / 其他 |
| **scene 使用场景** | 通用类 / 控制系统类 / 发电机类 / 齿轮箱类 / 变流系统类 / 变桨系统类 / 液压系统类 / 偏航系统类 / 润滑系统类 |

## 模型效果

基于约 **1 万条 LLM 辅助标注样本**训练（固定三切分 train/val/test = 7967/996/996，
近重复合并且同折、13 组标签冲突人工裁定，test 集从未参与训练与调参）：

| 维度 | 纯模型(val) | 混合方案(test) | macro-F1 |
|:-----|:-----------:|:------------:|:--------:|
| type | 92.2% | **92.0%** | 76.5 |
| physical | 80.6% | **81.0%** | 81.0 |
| scene | 80.9% | **81.7%** | 79.9 |

三维全对率 **62.3%**。推理速度约 **8 万条/秒**（Apple M 系列 CPU）。

> 弱项（样本少/标签歧义）：type 大部件 recall 33%（test 仅 21 条）、scene 变流系统类 55%、润滑系统类 68%。

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
├── retrain_clean.py           # 干净重训 + test 报告
├── finalize_v2.py             # 子词模型最终配方（生产模型由它产出）
├── example.py                 # 使用示例
└── data/                      # train/val/test 固定切分（业务数据,不入库）
```

## 技术说明

- **模型层（主力）**：fastText 开启子词特征（minn=2, maxn=4），中文物料名按字符 n-gram 泛化——
  相比无子词版本纯模型平均准确率提升约 19 个百分点。
- **规则层（高置信兜底）**：仅保留在验证集上不劣于模型的关键词组（偏航/变桨/齿轮箱/发电机/PLC
  等场景词、工器具词、传感器词）；油品用专用谓词（以"油"结尾或含润滑脂等），规避"油管/油泵"类误伤。
- **数据**：训练样本来自风力发电行业备件物料，涵盖控制系统、发电机、齿轮箱、变流器、变桨、偏航、液压、润滑等主要子系统。


## License

MIT
