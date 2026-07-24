# MFRInstSeg 小样本划分与 BrepMFR 数据转换设计

日期：2026-07-24

## 1. 目标

为后续四组实验建立完全一致、可复现且无数据泄漏的数据基础：

1. Baseline；
2. Oracle GT Instance Bias；
3. Dynamic Bias（无实例亲和损失）；
4. Dynamic Bias（有实例亲和损失）。

本阶段只实现两个数据准备脚本，不修改网络：

- `tools/build_mfrinstseg_subset.py`：从 MFRInstSeg 官方划分中生成类别均衡的小样本清单；
- `tools/convert_mfrinstseg_to_brepmfr.py`：将清单涉及的 STEP 与标签转换为 BrepMFR 可读取的 `.bin`。

## 2. 固定实验划分

MFRInstSeg 官方划分保持不变，不允许跨划分抽样：

| 清单 | 来源 | 数量 | 使用时机 |
|---|---|---:|---|
| `train_1pct.txt` | 官方训练集 | 437 | 训练及反向传播 |
| `val_small.txt` | 官方验证集 | 437 | 每个 epoch 验证、早停、选最佳权重 |
| `val_full.txt` | 官方验证集 | 9,375 | 训练结束后完整评估最佳权重 |
| `test.txt` | 官方测试集 | 9,375 | 所有设计和超参数冻结后最终评测 |

“1% 小样本”只表示使用官方训练集的 1%。验证集不参与参数梯度，因此不计入训练数据比例。

所有实验必须复用同一组清单、同一随机种子和同一选模指标。为保证 Baseline 与实例分支模型之间公平，主实验统一按照 `val_small` 上的语义分割 mIoU 选择最佳权重，不能只给 Dynamic Bias 使用实例指标选权重。

## 3. 脚本一：均衡小样本清单

### 3.1 输入

- MFRInstSeg 数据集根目录；
- 官方 `train`、`val`、`test` 划分文件；
- 每个模型对应的标签 JSON；
- 固定随机种子，默认 `seed=2026`；
- 训练比例，默认 `1%`；
- 小验证集数量，默认与实际训练子集数量相同。

训练子集数量按官方训练样本数乘比例后向下取整：

```text
floor(43,745 × 0.01) = 437
```

### 3.2 均衡口径

MFRInstSeg 中一个 CAD 模型通常同时包含多个语义类别，因此“每类样本数完全相等”在模型级抽样中一般不可实现。脚本采用多标签、稀有类优先的确定性贪心抽样，目标是让各类别在选中模型中的出现次数尽可能接近，同时保证每个可用类别均被覆盖。

对每个候选模型统计：

- `class_presence[c]`：模型是否包含类别 `c`；
- `face_count[c]`：模型中类别 `c` 的面数量；
- 模型总面数及实例数量。

抽样过程：

1. 只从当前官方划分内部读取候选项；
2. 先按类别在完整划分中的模型出现频率由低到高排序；
3. 优先选择能覆盖当前最欠缺类别、并同时补充其他欠缺类别的模型；
4. 得分相同时使用固定种子打破平局；
5. 达到目标数量后进行有限次交换优化，降低类别出现次数的变异系数；
6. 输出前检查样本唯一性、类别覆盖以及与其他官方划分是否相交。

优化的主口径是“包含该类别的模型数”，因为一个模型是本实验的数据采样单位。面级类别数量仅用于统计和次级平局处理，不能承诺完全相等。

### 3.3 输出

脚本只生成清单和统计文件，不复制 STEP 或标签文件：

```text
<output_root>/
  manifests/
    train_1pct.txt
    val_small.txt
    val_full.txt
    test.txt
  reports/
    subset_statistics.json
    subset_statistics.csv
```

每个清单一行一个不带扩展名的唯一模型 ID，按稳定顺序保存。

统计文件至少包含：

- 数据集路径与官方划分规模；
- 随机种子、目标比例、实际数量；
- 各类别的模型出现数、面数量及占比；
- 每个子集的模型面数和实例数分布；
- 类别覆盖率、类别计数最大最小差值和变异系数；
- 四份清单的 SHA-256；
- 官方划分间交集检查结果；
- 未满足约束时的明确警告。

### 3.4 命令行草案

```powershell
python tools/build_mfrinstseg_subset.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root D:\Code\Datasets\MFRInstSeg_BrepMFR `
  --train-ratio 0.01 `
  --val-small-size 437 `
  --seed 2026
```

支持 `--dry-run`：完成读取、抽样和统计，但不落盘。

## 4. 脚本二：转换为 BrepMFR `.bin`

### 4.1 转换策略

不加载约 20 GB 的 `aag/graphs.json`。脚本根据清单按需读取 STEP，使用与 MFRInstSeg 官方数据兼容的 AAG 特征模式重新提取：

- 缩放实体；
- 面属性与面 UV 网格；
- 边属性与边采样网格；
- B-Rep 邻接图；
- 最短路径及 BrepMFR 所需距离编码。

转换器只处理清单涉及的模型。初次联调只转换 `train_1pct` 和 `val_small` 共 874 个模型；训练流程稳定后再转换 `val_full` 和 `test`。同一模型被多个清单引用时只进行一次几何特征提取。

### 4.2 `.bin` 数据契约

保持当前 BrepMFR 读取器依赖的字段：

```text
graph.ndata:
  x  面 UV 网格
  y  面积
  z  面类型
  l  面环数量
  a  面邻接相关属性
  f  面语义标签

graph.edata:
  x  边采样网格
  t  边类型
  l  边长度
  a  边夹角
  c  边凹凸性

metadata:
  edges_path
  spatial_pos
  d2_distance
  angle_distance
```

为后续 Oracle Bias 和 Dynamic Bias 训练额外保留：

```text
metadata["instance"]  # [N, N]，同实例亲和矩阵
metadata["bottom"]    # [N]，MFRInstSeg 原始 bottom 标签
```

当前 Baseline 可以忽略这些新增 metadata；后续模型修改直接读取，无需再次转换数据。DGL 图级 metadata 只接受 Tensor，因此样本 ID 不重复写入 metadata，直接使用 `<sample_id>.bin` 的文件名。

### 4.3 输出组织

转换结果采用单份缓存，避免 `val_small` 与 `val_full` 重复保存：

```text
<output_root>/
  bins/
    <sample_id>.bin
  manifests/
    train_1pct.txt
    val_small.txt
    val_full.txt
    test.txt
  reports/
    conversion_report.json
    failed_samples.txt
```

清单仍保存模型 ID。后续数据加载器通过 `bins/<sample_id>.bin` 解析文件；修改数据加载器支持清单属于网络实现阶段，不属于这两个数据准备脚本。

转换采用临时文件写入成功后再原子重命名，避免中断时留下表面存在但内容不完整的 `.bin`。默认跳过已经通过校验的结果；只有显式指定 `--overwrite` 才重建。

### 4.4 一致性与失败检查

每个样本必须检查：

- STEP、标签 JSON 均存在且 ID 唯一；
- 图节点数等于 `seg` 标签长度；
- `instance` 为 `N × N`；
- 实例矩阵对称、主对角线有效；
- `bottom` 长度等于 `N`；
- 语义类别 ID 在允许范围内；
- 图边索引、节点和边特征形状合法；
- 所有浮点特征不含 NaN 或 Inf；
- 写入后可以被 DGL 重新加载；
- 重载后的节点数、边数、标签和实例矩阵与写入前一致。

单个样本失败时记录 ID、异常类型和信息并继续转换。进程最终只要存在失败样本就返回非零退出码，防止训练在不完整数据上静默开始。

### 4.5 命令行草案

初次只转换训练和小验证集：

```powershell
python tools/convert_mfrinstseg_to_brepmfr.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root D:\Code\Datasets\MFRInstSeg_BrepMFR `
  --manifests train_1pct val_small `
  --workers 4
```

流程稳定后补充完整验证和测试：

```powershell
python tools/convert_mfrinstseg_to_brepmfr.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root D:\Code\Datasets\MFRInstSeg_BrepMFR `
  --manifests val_full test `
  --workers 4
```

## 5. 验证与验收

### 5.1 脚本一

- 相同输入和种子生成完全相同的文件及 SHA-256；
- `train_1pct` 恰好包含 437 个唯一模型；
- `val_small` 恰好包含 437 个唯一模型；
- `train_1pct` 只来自官方训练集；
- `val_small` 只来自官方验证集；
- 官方训练、验证、测试划分之间无交集；
- 每个可用类别至少被一个选中模型覆盖；
- 与同种子纯随机抽样相比，类别模型出现数的变异系数不得更差；
- 报告实际类别分布，不用“完全均衡”掩盖多标签抽样的客观限制。

### 5.2 脚本二

- 先选取少量不同拓扑规模的样本执行冒烟转换；
- 将生成的 `.bin` 交给当前 BrepMFR 数据读取器读取；
- 校验语义标签逐面一致；
- 校验实例矩阵在写入和重载后完全一致；
- 对同一输入重复执行时安全跳过，不重复提取；
- 中断后重启可以继续；
- 874 个训练及小验证样本全部成功后，才开始基线训练；
- 完整验证和测试转换失败数必须为零，才能生成最终论文结果。

## 6. 实验运行约束

每轮训练仅运行：

```text
train_1pct（437）→ 参数更新
val_small（437） → mIoU、早停、最佳权重
```

训练结束后：

```text
val_small 最佳权重 → val_full 完整评估一次
```

所有结构、损失权重、阈值和后处理规则确定后：

```text
最终配置 → test 完整评估
```

完整验证集不在每个 epoch 运行。若后续需要调实例聚类阈值，可对 `val_full` 只前向推理一次并缓存语义 logits 和实例亲和矩阵，在缓存结果上搜索阈值，避免重复运行网络。测试集不得用于选模型、选阈值或修改方法。

## 7. 非目标

本阶段不包含：

- 修改 BrepMFR 网络和注意力层；
- 实现 Oracle 或 Dynamic Bias；
- 实现实例亲和损失；
- 实现实例聚类后处理；
- 在测试集上反复调参；
- 复制或重排 MFRInstSeg 原始数据集。

这些内容在两个数据脚本验收后分别进入网络实现与实验阶段。
