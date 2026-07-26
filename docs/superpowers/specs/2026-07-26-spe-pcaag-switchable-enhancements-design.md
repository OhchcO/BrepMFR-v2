# SPE-PCAAG 与实例关系偏置开关化实验设计

## 1. 目标与实施边界

本阶段在现有 BrepMFR 代码上增加可独立启停的几何增强与实例关系增强，用于完成消融实验。当前阶段不重构整个项目；实验结束并确定最终有效组合后，再删除无效路径并重构为统一的自有框架。

本次实现包含：

- 柱坐标全局位置编码；
- 每面点云局部细节编码；
- 固定点数与面积自适应两种点云采样策略；
- 现有 `baseline/oracle/dynamic/dynamic_affinity` 实例关系模式；
- “仅实例亲和损失、不注入注意力偏置”的解耦对照；
- 训练、测试与数据转换脚本中的对应开关；
- BIN 字段验证和缺失字段的明确报错。

按用户要求，本次只生成和修改代码，不运行 BIN 转换、训练、测试或性能基准。

## 2. 总体架构

完整网络由三个互补层次组成：

1. **局部几何层**：每个 B-rep 面的点云经过轻量 PointNet，提取面内局部形状与法向变化。
2. **全局位置层**：每个面的柱坐标位置经过 MLP 投影，为局部子图保留其在完整零件中的全局位置。
3. **面间关系层**：动态实例关系预测器利用增强后的节点特征及 SPE-PCAAG 成对描述，产生实例关系注意力偏置；训练期可由实例亲和标签监督，推理期不需要实例标签。

数据流为：

```text
STEP
  -> B-rep 面邻接图
  -> UV 网格 + 面属性
  -> 每面三维候选点
  -> 固定/面积自适应 FPS
  -> face_points / face_point_mask / cylindrical_pos
  -> BIN
  -> SurfaceEncoder
       + PointCloudFaceEncoder（可选）
       + CylindricalPositionEncoder（可选）
  -> Graph Transformer
       + InstanceRelationBias（可选）
  -> 面语义分类
```

## 3. 点云采样设计

### 3.1 采样模式

转换器提供两种模式：

```text
--point-sampling-mode fixed
--point-sampling-mode area_adaptive
```

默认使用 `area_adaptive`。

固定模式对每个面使用 64 个有效点：

```text
--fixed-point-count 64
```

面积自适应模式根据完整零件的面面积中位数计算目标点数：

\[
K_i = \operatorname{round8}\left(
\operatorname{clip}\left(
64\left(\frac{A_i}{A_{\mathrm{median}}}\right)^{0.5},
32,
96
\right)
\right)
\]

对应参数为：

```text
--min-point-count 32
--max-point-count 96
--area-sampling-exponent 0.5
```

`round8` 表示取整到最接近的 8 的倍数。面积中位数必须在完整零件上计算，不能在随机裁剪后的局部子图上重新计算。

不采用点数与面积线性正比，因为这会使小倒角、小圆角和孔壁等面积较小但语义关键的面获得过少采样点。

### 3.2 三维采样位置

不能直接把 UV 参数域中的均匀点当作三维均匀点。转换流程应：

1. 在面有效 UV 范围内生成密集候选点和法向；
2. 使用面内掩码去除裁剪边界外的候选点；
3. 将候选点转换为三维坐标；
4. 在三维坐标上执行确定性的最远点采样（FPS）；
5. 保存被选点的 `XYZ + Normal`。

候选网格默认从 `20×20` 开始。如果有效候选点少于目标点数，按 `20、32、48、64、80` 的网格分辨率依次重采样；若最高分辨率仍不足，则对已有 FPS 序列进行确定性循环补齐。补齐点仍属于真实曲面，PointNet 的最大池化不会因重复点改变结果。

FPS 的首点取距离候选点质心最远的点，后续始终取距离已选集合最远的点；距离相同时取候选数组中索引最小的点，保证同一 STEP 文件重复转换得到相同结果。

### 3.3 BIN 节点字段

点云和柱坐标保存为 DGL 节点字段，使航空机匣随机裁剪子图时这些字段自动随节点保留：

```text
graph.ndata["face_points"]       float32 [N, 96, 6]
graph.ndata["face_point_mask"]   bool    [N, 96]
graph.ndata["face_point_count"]  int32   [N]
graph.ndata["cylindrical_pos"]   float32 [N, 4]
```

点云张量统一使用最大长度 96。固定模式前 64 个位置有效；面积自适应模式前 `K_i` 个位置有效；其余位置补零并由 `face_point_mask` 屏蔽。

BIN 元数据额外保存转换配置，确保已有文件可以判断是否与当前请求兼容：

```text
metadata["point_sampling_mode"]    int8    [1]  # 0=fixed, 1=area_adaptive
metadata["point_sampling_config"]  float32 [4]  # fixed/min/max/exponent
metadata["point_candidate_grid"]   int32   [1]  # 初始候选网格分辨率
```

这些字段对旧 BIN 是可选字段；对启用点云的新 BIN 是必需字段。现有元数据验证由“键集合完全相等”改为“核心字段必须存在、新字段按功能开关验证”，避免破坏旧数据兼容性。

## 4. 柱坐标位置编码

柱坐标基于完整零件、缩放后的统一坐标计算。每个面的代表点使用 PyOCC 面面积属性计算得到的曲面质心，不依赖固定或面积自适应点云的采样结果，保证两种点云策略下柱坐标完全一致。默认旋转轴为全局 Z 轴，编码为：

\[
[r_{\mathrm{norm}}, \sin\theta, \cos\theta, z_{\mathrm{norm}}]
\]

其中：

- `r = sqrt(x² + y²)`；
- `theta = atan2(y, x)`；
- `r_norm` 使用完整零件所有面心的半径范围归一化；
- `z_norm` 使用完整零件所有面心的 Z 范围归一化；
- 分母小于数值阈值时相应归一化值置零；
- 角度使用 `sin/cos`，避免 `-π/π` 处的不连续。

柱坐标必须在完整机匣上计算后再生成子图，局部训练样本不得重新归一化。MFRInstSeg 转换同样在每个完整 STEP 模型上计算。

模型开关关闭时，可以读取不包含该字段的旧 BIN；开关开启但字段缺失时必须在数据加载阶段报错，禁止静默填零。

## 5. 点云与柱坐标编码器

### 5.1 PointCloudFaceEncoder

输入为 `[total_faces, 96, 6]` 和有效点掩码。

预处理：

- 仅使用有效点计算每个面的三维质心；
- XYZ 减去该面质心；
- 使用有效点到质心的最大半径进行缩放；
- 最大半径小于阈值时使用 1，防止除零；
- Normal 保持单位方向；
- 无效位置在池化前填为负无穷。

轻量 PointNet 结构：

```text
6 -> 64 -> 128 -> 256
masked max pooling
256 -> hidden_dim
```

逐点层使用 Linear、LayerNorm 和 GELU，避免小批量下 BatchNorm 统计不稳定。池化后投影到节点隐藏维度，以残差方式加入原 SurfaceEncoder 节点特征。

### 5.2 CylindricalPositionEncoder

结构为：

```text
4 -> 64 -> hidden_dim
```

使用 Linear、GELU 和 Linear，以残差方式加入节点隐藏特征。

### 5.3 特征融合

保留原 SurfaceEncoder 作为主分支：

```text
node_feature =
    surface_feature
    + pointcloud_scale * pointcloud_embedding
    + cylindrical_scale * cylindrical_embedding
```

两个增强分支各自独立启停，缩放系数默认 1.0。开关全部关闭时不实例化增强分支，保持原基线的参数量、随机数消耗和前向路径不变。

增强后的节点特征同时进入 Graph Transformer 和动态实例关系预测器，使局部几何与全局位置可以共同指导语义注意力和实例关系估计。

## 6. 实验开关

训练入口新增：

```text
--use-face-pointcloud
--use-cylindrical-pos
--pointcloud-scale 1.0
--cylindrical-scale 1.0
```

保留现有：

```text
--experiment_mode baseline
--experiment_mode oracle
--experiment_mode dynamic
--experiment_mode dynamic_affinity
--instance_bias_scale
--affinity_loss_weight
```

“仅实例亲和损失、不注入注意力偏置”使用：

```text
--experiment_mode dynamic_affinity
--instance_bias_scale 0
```

该模式仍训练实例关系预测器并计算亲和损失，但关系矩阵不进入注意力，用于区分辅助任务正则化与实例偏置注入的贡献。

PowerShell 实验脚本应支持以下命名模式：

```text
baseline
cyl
point
spe_pcaag
instance
affinity_only
full
```

模式只是参数组合别名，底层始终由独立开关控制。

## 7. 最小消融矩阵

| 实验 | 柱坐标 | 点云 | 动态偏置 | 亲和损失 |
|---|---:|---:|---:|---:|
| Baseline | 否 | 否 | 否 | 否 |
| Cyl | 是 | 否 | 否 | 否 |
| Point | 否 | 是 | 否 | 否 |
| SPE-PCAAG | 是 | 是 | 否 | 否 |
| Instance | 否 | 否 | 是 | 是 |
| Affinity-only | 否 | 否 | 不注入 | 是 |
| Full | 是 | 是 | 是 | 是 |

固定64点与面积自适应采样的比较属于点云采样策略消融，不改变网络结构。结构消融先使用同一随机种子；最终只对 Baseline 与 Full 使用多个数据抽样种子和训练种子。

## 8. 数据兼容与错误处理

- 所有新功能默认关闭，旧 BIN 仍可运行当前 Baseline 和实例关系实验。
- 启用点云但缺少 `face_points/face_point_mask/face_point_count` 时，数据加载器立即给出包含 BIN 路径和缺失字段的错误。
- 启用柱坐标但缺少 `cylindrical_pos` 时立即报错。
- 字段第一维必须等于面节点数；点云形状必须为 `[N,96,6]`；柱坐标形状必须为 `[N,4]`。
- 点云、柱坐标和法向必须全部为有限值。
- `face_point_count` 必须位于 `[1,96]`，且必须与 mask 中有效点数量相同。
- 固定模式要求每面有效点数为64；面积自适应模式要求有效点数位于32至96且为8的倍数。
- 转换失败沿用 `conversion_failures.json`，单个模型失败不得发布不完整或字段不一致的 BIN。
- 已存在旧 BIN 且转换配置要求新字段时，不得因 `--overwrite` 缺失而误判为有效跳过；验证失败后明确要求用户指定覆盖或更换输出目录。

## 9. 代码组织

当前阶段只做与实验开关直接相关的最小拆分：

```text
models/modules/pointcloud_encoder.py
models/modules/cylindrical_encoder.py
models/modules/instance_bias.py
tools/mfrinstseg_to_bin.py
data/dataset.py
data/collator.py
models/modules/brep_encoder.py
models/brepseg_model.py
segmentation.py
tools/run_mfrinstseg_experiments.ps1
tools/run_mfrinstseg_test.ps1
```

转换器中的采样计算应拆为无状态函数，避免继续扩大 `convert_one`。模型编码器只接收已经整理好的张量，不读取 STEP 或解释采样配置。

## 10. 静态交付与用户执行

本次由代码生成阶段交付：

- 修改后的转换器；
- 新增的点云与柱坐标编码模块；
- 数据加载与批处理支持；
- 训练与测试开关；
- 实验脚本参数组合；
- 字段验证代码；
- 可供用户执行的命令说明；
- 测试代码或检查脚本，但不执行。

本次不执行：

- MFRInstSeg 或 Aero-Casing 的 BIN 重建；
- 训练；
- 验证或测试推理；
- GPU 显存基准；
- 自动覆盖现有数据集。

最终运行结果由用户提供后，再进行指标分析和下一轮修改。

## 11. 实验后重构阶段

完成消融后再启动第二阶段重构：

1. 根据实验删除无效分支和临时兼容代码；
2. 将数据契约、几何编码、关系编码、训练任务和评估指标拆为清晰模块；
3. 统一配置系统和实验命名；
4. 统一公开数据集与 Aero-Casing 的数据接口；
5. 将局部训练、完整机匣推理和实例后处理纳入同一推理管线；
6. 形成论文对应的自有框架，而不是继续保留多套实验性类文件。

该重构不属于本次开关化实现范围，必须等待实验结果后再确定最终保留结构。
