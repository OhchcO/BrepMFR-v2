# MFRInstSeg 小样本 BIN 转换器设计

## 目标

新增 `tools/mfrinstseg_to_bin.py`，将均衡抽样清单中的 MFRInstSeg STEP 模型转换为当前 BrepMFR 基线可直接读取的 DGL `.bin`。脚本只生成转换能力，本次不实际批量转换数据。

## 输入

- 原始数据集根目录，包含：
  - `steps/<sample_id>.step`
  - `labels/<sample_id>.json`
- 小样本清单目录，包含：
  - `train.txt`
  - `val.txt`
- 旧版几何提取器：
  - `D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py`
- AAG 属性配置：
  - `D:\Code\BrepMFR-g\data_set_line\schema.json`

转换器不读取约 20 GB 的 `aag/graphs.json`。

## 输出

输出根目录结构：

```text
output_root/
├── bin/
│   ├── <sample_id>.bin
│   └── ...
├── train.txt
├── val.txt
└── conversion_failures.json
```

`train.txt` 和 `val.txt` 只包含成功转换或已经存在且验证通过的 BIN 编号，因此输出根目录可直接作为当前 `CADSynth` 数据加载器的 `root_dir`。

## 几何图字段

每个 BIN 使用 DGL `save_graphs` 保存一张面邻接图。

### 面节点字段

- `x`: `[N, 5, 5, 7]`，面 UV 采样点坐标、法向和裁剪掩码。
- `y`: `[N]`，面面积，`float32`。
- `z`: `[N]`，面类型，`int32`。
- `l`: `[N]`，面环数量，`int32`。
- `a`: `[N]`，面邻接度，`int32`。
- `f`: `[N]`，MFRInstSeg 语义标签，`int32`。

### 边字段

- `x`: `[E, 5, 7]`，边采样点坐标、切向和有符号二面角。
- `t`: `[E]`，边曲线类型，`int32`。
- `l`: `[E]`，边长度，`float32`。
- `a`: `[E]`，有符号二面角，`float32`。
- `c`: `[E]`，光滑/凹/凸类别，`int32`。

有符号二面角由边两侧法向计算。凹边为正、凸边为负、光滑边为零，与现有 CADSynth BIN 的编码方向一致。

## 注意力辅助字段

BIN 元数据保存：

- `spatial_pos`: `[N, N]`，图最短路径距离。
- `edges_path`: `[N, N, 6]`，路径上的零基边索引；空位置为 `-1`。
- `d2_distance`: `[N, N, 64]`，面中心欧氏距离的 RBF 编码。
- `angle_distance`: `[N, N, 64]`，面法向夹角的 RBF 编码。

这些字段与 `data/dataset.py` 和当前 Graph Transformer 的输入约定一致。

## 实例标注字段

BIN 元数据额外保存：

- `instance`: `[N, N]`，实例亲和矩阵。
- `bottom`: `[N]`，底面标注。

当前纯基线加载器可以忽略这两个额外字段；后续实例损失、动态实例偏置和实例后处理可以直接读取。

## 转换流程

1. 从指定划分清单读取模型编号并保持首次出现顺序。
2. 动态导入 `AAGExtractor.py`，使用 `scale_body=True` 提取 STEP 几何。
3. 读取官方 `[[sample_id, {"seg", "inst", "bottom"}]]` 标签格式。
4. 严格检查提取面数、语义标签数、实例矩阵尺寸和底面标签数一致。
5. 构造 DGL 图、面字段、边字段和注意力辅助矩阵。
6. 先保存到临时文件，再重新加载并检查必需键与张量形状。
7. 验证成功后原子替换为最终 BIN。
8. 分别生成成功的 `train.txt`、`val.txt` 和失败报告。

## 命令行

主要参数：

- `--dataset-root`
- `--subset-root`
- `--output-root`
- `--extractor-file`
- `--schema-file`
- `--split train|val|all`，默认 `all`
- `--overwrite`
- `--limit`，用于以后手工小规模检查

脚本默认顺序转换，避免 Windows 下 PyOCC 多进程稳定性问题。

## 错误处理

- STEP、标签、提取器或 schema 缺失时，该模型记录失败原因。
- 面数或标注维度不一致时不生成 BIN。
- 拓扑检查、PyOCC、DGL 保存或回读验证失败时删除临时文件并继续下一个模型。
- 已存在的 BIN 在未指定 `--overwrite` 时先验证；验证通过则跳过转换并保留在输出清单中。
- 输出目录不得覆盖原始数据集或小样本清单目录。

## 范围限制

- 本次只新增转换脚本，不修改网络、损失函数或数据加载器。
- 本次不运行批量转换。
- 本次不复制或重构旧版 `AAGExtractor.py`。
