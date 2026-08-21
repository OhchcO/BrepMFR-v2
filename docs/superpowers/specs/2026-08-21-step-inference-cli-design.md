# STEP 端到端识别 Python 包与 CLI 设计

## 背景与目标

当前仓库已经具备 MFRInstSeg 数据集转换、BrepMFR 模型推理、动态亲和度预测和拓扑约束实例聚类能力，但这些能力分别面向有标签的数据准备与整套数据集评估。现有 `MFR_Recognition_Project/Python/recognize2.py` 只适配旧 AAGNet 模型，不能加载当前最佳 BrepMFR checkpoint。

本功能在当前 BrepMFR 仓库中新增一个可独立调用的 Python 推理包和薄 CLI，打通以下流程：

```text
无标签 STEP -> 可复用 BIN -> 单图批处理 -> checkpoint 推理
            -> 语义分类 + 动态亲和度 -> 拓扑约束实例聚类
            -> 标准 result.json
```

第一阶段用于验证端到端识别和建立迁移到未来独立项目所需的输入输出、黄金样例与数值基准。UG/NX 集成不在本阶段范围内。

## 范围

### 包含

- 单个 `.step` 或 `.stp` 文件的无标签转换。
- 根据 checkpoint 超参数自动选择 BIN 几何字段。
- 内容与转换契约驱动的 BIN 缓存。
- 单个 BIN 的推理专用加载与批处理。
- 逐面语义概率、动态亲和度和拓扑约束实例聚类。
- 版本化、可机器读取的 JSON 成功或失败结果。
- CPU、CUDA 和自动设备选择。
- 单元测试、缓存测试和一个真实 MFRInstSeg 黄金样例验证。

### 不包含

- UG/NX 调用、Face Tag 回映射、面高亮或 UI。
- TXT 兼容输出。
- 批量 STEP 任务、常驻推理服务或 HTTP API。
- 同一输出目录的并发写入；一个输出目录同一时刻只允许一个 CLI 进程使用。
- 模型训练、重新训练、阈值搜索或模型导出。
- bottom face 预测；当前 BrepMFR checkpoint 不提供该输出头。
- 将 OCC/AAG 图节点编号解释为 NX Face Tag。

## 对外接口

CLI 位于 `tools/infer_step.py`，业务逻辑位于 `brepmfr/inference/`。命令形式为：

```powershell
python tools/infer_step.py `
  --step D:\path\part.step `
  --checkpoint D:\path\model.ckpt `
  --output-dir D:\path\recognition-output `
  --device auto `
  --affinity-threshold 0.5
```

必需参数：

- `--step`：存在的 `.step` 或 `.stp` 文件。
- `--checkpoint`：存在的 PyTorch Lightning `.ckpt` 文件。
- `--output-dir`：一个 STEP 任务独占的结果与缓存目录。

可选参数：

- `--device {auto,cuda,cpu}`，默认 `auto`。
- `--affinity-threshold FLOAT`，默认 `0.5`，范围为 `[0, 1]`。
- `--force-rebuild`，忽略缓存并重新转换 STEP。
- `--seed INTEGER`，默认优先读取 checkpoint 保存值；checkpoint 无值时使用 `2026`。

成功时进程退出码为 `0`。任何输入、转换、加载或推理失败均返回非零退出码。CLI 只负责参数解析、调用流水线、打印简短摘要和映射退出码，不承载转换或模型逻辑。

## 包结构与职责

```text
brepmfr/
└── inference/
    ├── __init__.py
    ├── cache.py
    ├── conversion.py
    ├── loading.py
    ├── postprocess.py
    ├── result.py
    └── pipeline.py
tools/
└── infer_step.py
```

- `cache.py`：计算文件 SHA-256、转换契约指纹、读取和校验 sidecar 元数据、决定命中或重建。
- `conversion.py`：组合现有 AAG 提取和几何特征函数，生成无标签推理 BIN；不读取标签 JSON。
- `loading.py`：把一个推理 BIN 转为现有编码器需要的单图 batch。
- `postprocess.py`：保存生产用的 checkpoint 几何需求解析、类别表和拓扑约束实例聚类。
- `result.py`：构建和原子写入版本化 JSON；不依赖 CLI。
- `pipeline.py`：编排 checkpoint、缓存、转换、加载、推理和结果序列化，并记录阶段耗时。
- `tools/infer_step.py`：用户入口。

现有评估入口继续保持原有 CLI 和指标语义。可复用的 `geometry_requirements` 与 `cluster_instances` 从 `tools/test_mfrinstseg.py` 迁移到生产模块，评估入口改为导入这些函数，避免推理包依赖测试工具或复制聚类算法。

## 无标签 BIN 契约

训练转换器保持原有有标签行为。底层图构建函数扩展为允许缺少 `LabelData`：

- 所有几何、拓扑和可选参数曲面/柱坐标张量与训练 BIN 使用同一形状和 dtype。
- `graph.ndata["f"]` 使用 `-1` 明确表示没有语义真值，只用于满足现有 collator 的结构字段；推理不得读取它计算预测。
- metadata 不写入 `instance` 真值矩阵。
- 推理验证调用 `BinRequirements(require_instance=False, ...)`，可选字段由 checkpoint 决定。
- JSON 结果不得把 `-1` 占位值暴露为预测或真值。

`BrepSeg.predict_with_affinity()` 只读取几何 batch 与 `padding_mask`，不会使用 `label_feature` 或真值 `instance`。动态亲和 checkpoint 必须返回 affinity logits；不支持动态亲和的 checkpoint 以明确错误结束，因为本阶段输出契约要求实例结果。

## Checkpoint 与转换需求

流水线先加载 checkpoint，再从保存的超参数派生：

- 是否使用 UV 网格；
- 是否需要 face pointcloud；
- 是否使用 `param_surface` 编码器；
- 是否需要 cylindrical position；
- checkpoint seed、类别数和 experiment mode。

当前验收 checkpoint 为：

```text
best-mIoU-187-0.939778.ckpt
```

其已核验配置为 `dynamic_affinity`、25 类、参数曲面、intrinsic 邻域、untrimmed、gated fusion、柱坐标位置和 seed 42。CLI 不硬编码该绝对路径，调用者必须显式传入 checkpoint。

## BIN 缓存

输出目录固定包含：

```text
recognition-output/
├── input.bin
├── input.bin.meta.json
└── result.json
```

`input.bin.meta.json` 的缓存身份包含：

- STEP 文件内容 SHA-256；
- `schema.json` 内容 SHA-256；
- AAG extractor、BIN 转换器和几何特征实现的内容指纹；
- point sampling 配置与 seed；
- checkpoint 派生的 BIN 几何需求；
- 推理 BIN 契约版本。

checkpoint 文件本身不属于 BIN 缓存身份。两个 checkpoint 要求相同的输入字段时可以复用同一 BIN；需求不同则自动重建。

缓存命中必须同时满足：

1. BIN 与 sidecar 均存在；
2. sidecar 身份完全匹配；
3. `validate_bin` 通过当前字段、shape、dtype 和点采样契约校验。

任一条件不满足即重建。`--force-rebuild` 无条件重建。BIN 与 sidecar 均先写临时文件并完成回读校验，再替换正式目标。转换或临时文件校验失败不触碰先前有效缓存；正式替换期间若发生文件系统错误，下一次运行必须因 sidecar 或 BIN 不匹配而重建，不能把混合代际文件当作缓存命中。

## 推理与后处理

1. 通过 `BrepSeg.load_from_checkpoint(..., map_location="cpu")` 加载模型。
2. 根据 `--device` 选择设备并进入 `eval()` 与 `torch.inference_mode()`。
3. 推理专用 loader 加载单个 BIN，并用现有 collator 生成 batch size 1 的 batch。
4. `predict_with_affinity()` 返回每个面的 25 类概率和方形 affinity logits。
5. 每面类别取最大概率索引，置信度取该最大概率。
6. affinity logits 经稳定 sigmoid 转为概率。
7. 实例聚类只连接：亲和度达到阈值、拓扑直接相邻、预测语义相同且不是 stock 类 24 的面。
8. 实例 ID 按最小 face index 的升序稳定编号。
9. 实例置信度定义为实例内各面预测类别置信度的算术平均值。

类别 ID 到名称沿用 MFRInstSeg 的固定 25 类顺序：

```text
0 chamfer
1 through_hole
2 triangular_passage
3 rectangular_passage
4 6sides_passage
5 triangular_through_slot
6 rectangular_through_slot
7 circular_through_slot
8 rectangular_through_step
9 2sides_through_step
10 slanted_through_step
11 Oring
12 blind_hole
13 triangular_pocket
14 rectangular_pocket
15 6sides_pocket
16 circular_end_pocket
17 rectangular_blind_slot
18 v_circular_end_blind_slot
19 h_circular_end_blind_slot
20 triangular_blind_step
21 circular_blind_step
22 rectangular_blind_step
23 round
24 stock
```

stock 面保留在逐面结果中，但不生成实例记录。多个相同类别实例分别占一条实例记录。

## JSON 成功契约

`result.json` 使用 UTF-8、两空格缩进和稳定字段顺序。成功文档结构为：

```json
{
  "schema_version": "1.0",
  "status": "success",
  "input": {
    "step_path": "D:/path/part.step",
    "step_sha256": "...",
    "face_count": 41
  },
  "cache": {
    "bin_path": "D:/path/output/input.bin",
    "metadata_path": "D:/path/output/input.bin.meta.json",
    "hit": false,
    "fingerprint": "..."
  },
  "model": {
    "checkpoint_path": "D:/path/model.ckpt",
    "checkpoint_sha256": "...",
    "checkpoint_epoch": 187,
    "experiment_mode": "dynamic_affinity",
    "device": "cuda",
    "num_classes": 25,
    "affinity_threshold": 0.5
  },
  "faces": [
    {
      "face_index": 0,
      "class_id": 24,
      "feature_name": "stock",
      "confidence": 0.999,
      "instance_id": null
    }
  ],
  "instances": [
    {
      "instance_id": 0,
      "class_id": 1,
      "feature_name": "through_hole",
      "face_indices": [6, 7],
      "confidence": 0.96
    }
  ],
  "timings_seconds": {
    "checkpoint_load": 0.0,
    "conversion": 0.0,
    "inference": 0.0,
    "total": 0.0
  },
  "warnings": []
}
```

路径使用解析后的绝对路径。`face_index` 是 STEP 经 OCC/AAG 提取后的 0-based 图节点编号，不承诺等于 NX 面数组位置或 Face Tag。所有浮点数必须有限；实例中的 face index 唯一、升序且与逐面 `instance_id` 一致。

## 错误处理

在已经能够创建输出目录的前提下，任何失败都原子写入同一 `result.json`：

```json
{
  "schema_version": "1.0",
  "status": "error",
  "stage": "conversion",
  "error": {
    "type": "ValueError",
    "message": "..."
  },
  "input": {
    "step_path": "D:/path/part.step"
  }
}
```

阶段值限定为 `arguments`、`checkpoint`、`cache`、`conversion`、`loading`、`inference`、`postprocess` 或 `serialization`。错误 JSON 不写 traceback；完整 traceback 输出到 stderr，便于独立 CLI 调试。输出目录本身无法创建时只写 stderr 并返回非零退出码。

## 测试与验收

### 单元与契约测试

- 无标签图构建产生 `f == -1` 且不含 `instance` metadata。
- 训练转换路径仍写入原语义标签和实例矩阵。
- STEP 内容不变且转换契约相同时命中缓存。
- STEP 内容、schema、转换实现、seed 或几何需求变化时缓存失效。
- 损坏 BIN 即使 sidecar 匹配也强制重建。
- `--force-rebuild` 跳过有效缓存。
- 单图 loader 接受 `-1` 标签占位且不要求 instance 真值。
- 类别名称、stock 排除、稳定实例编号和置信度计算正确。
- JSON 成功与失败文档满足字段、类型、有限浮点数和内部一致性约束。
- CLI 对无效路径、设备、阈值和非动态 checkpoint 返回非零退出码。

### 现有回归

重点运行转换、数据字段、模型亲和度和实例评估相关测试。开始开发前完整基线为：

```text
402 passed, 25 failed, 9 warnings
```

25 项失败属于当前 HEAD 已有的实验启动器和数据准备旧契约失败；本功能不得增加失败数量或改变失败集合。

### 真实黄金样例

固定样例：

```text
sample_id: 20221121_154647_0
STEP: D:\Code\Datasets\MFRInstSeg\steps\20221121_154647_0.step
label: D:\Code\Datasets\MFRInstSeg\labels\20221121_154647_0.json
prepared BIN: D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42\bin\20221121_154647_0.bin
faces: 41
```

推理阶段不得打开 label 文件。验收分为三步：

1. 用现有 prepared BIN 和现有模型调用建立语义概率、affinity 与实例结果基准。
2. 对同一 STEP 运行新 CLI，首次生成无标签 BIN 和 JSON；逐面预测类别、实例分组必须与基准完全相同，概率与 affinity 的最大绝对误差不超过 `1e-5`。
3. 第二次运行必须报告 `cache.hit == true`，跳过转换，并产生相同的 faces 与 instances。

标签只在推理完成后用于报告该单样例的语义准确率和实例对照，不参与转换、batch 或模型计算。黄金结果记录 checkpoint 内容 SHA-256、运行环境版本和数值摘要，作为未来独立项目重写与权重迁移的行为基准。

## 兼容性与未来迁移

- 实现保持 Python 3.8 语法兼容。
- 当前训练和整数据集评估 CLI 保持兼容。
- 新包不导入 `MFR_Recognition_Project` 旧代码。
- UG 集成以后只依赖版本化 JSON，不进入模型包。
- 新包边界按未来迁移到独立项目设计；迁移时可用本设计、黄金样例和数值基准验证新实现，而不是逐文件复制整个 BrepMFR 仓库。

## 完成标准

- 新 CLI 对黄金 STEP 首次运行成功产生 `input.bin`、sidecar 和 `result.json`。
- 第二次运行确认安全缓存命中。
- 推理过程通过监控或 mock 证明未读取标签 JSON。
- 新旧输入路径在黄金样例上的类别、实例与数值误差满足验收阈值。
- focused tests 全部通过，完整测试失败集合不超过既有 25 项。
- CLI `--help`、Python 编译、JSON 回读和 `git diff --check` 全部通过。
