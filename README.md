# WindSparePartsModel 风电物料三维分类模型

风电备件物料智能分类模型（fastText + 规则混合方案），对物料名称做三维自动标注，毫秒级推理、零 API 成本。

## 三维分类体系

| 维度 | 标签 |
|:-----|:-----|
| **type 类型** | 备件 / 耗材 / 工器具 / 大部件 |
| **physical 物理属性** | 电气件 / 机械件 / 电子件 / 传感器 / 油品 / 液压件 / 密封件 / 其他 |
| **scene 使用场景** | 通用类 / 控制系统类 / 发电机类 / 齿轮箱类 / 变流系统类 / 变桨系统类 / 液压系统类 / 偏航系统类 / 润滑系统类 |

## 模型效果

模型基于 **10,000 条人工标注（DeepSeek 辅助打标）样本** 训练，配合规则库形成混合分类器：

| 维度 | 纯模型准确率 | 混合方案准确率 |
|:-----|:-----------:|:------------:|
| type | 86.4% | **98.3%** |
| physical | 57.0% | **79.0%** |
| scene | 59.7% | **96.7%** |

推理速度约 **12 万条/秒**（Apple M 系列 CPU）。

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
│   ├── fasttext_10k_type.bin      # type 分类模型
│   ├── fasttext_10k_physical.bin  # physical 分类模型
│   └── fasttext_10k_scene.bin     # scene 分类模型
└── example.py                 # 使用示例
```

## 技术说明

- **规则层**：优先命中高置信度关键词（如"密封圈→密封件"、"偏航→偏航系统类"），规则未覆盖时走模型。
- **模型层**：fastText 分类模型，基于 n-gram 子词特征，对中文物料名有较好泛化。
- **数据**：训练样本来自风力发电行业备件物料，涵盖控制系统、发电机、齿轮箱、变流器、变桨、偏航、液压、润滑等主要子系统。

## License

MIT
