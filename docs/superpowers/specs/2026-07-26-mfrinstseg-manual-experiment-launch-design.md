# MFRInstSeg 手动单组实验启动设计

## 目标

在现有统一 STEP、标签和 BIN 缓存流程上，补齐论文消融实验开关，并提供一份可直接复制的 TXT 启动命令。用户每次只修改一个 `$Mode` 变量，即可训练一组实验；训练完成后可自动定位该组最新的最优权重并运行完整测试集评估。

## 实验接口

继续使用 `tools/run_mfrinstseg_experiments.ps1` 作为唯一训练入口，支持以下 Mode：

- `baseline`：原始语义分割基线。
- `dynamic_no_loss`：动态实例关系作为注意力偏置，不使用实例亲和损失。
- `affinity_only`：使用实例亲和损失，但实例偏置缩放为零。
- `instance`：动态实例偏置与实例亲和损失同时启用。
- `oracle`：使用 GT 实例关系作为注意力偏置，仅作为理论上界。
- `cyl`：仅启用柱坐标编码。
- `point`：仅启用面点云编码。
- `spe_pcaag`：同时启用柱坐标与面点云编码。
- `full`：SPE-PCAAG、动态实例偏置和实例亲和损失全部启用。

新增别名严格映射为：

- `dynamic_no_loss` → `experiment_mode=dynamic`、不启用点云、柱坐标或实例亲和损失、`instance_bias_scale=1.0`。
- `oracle` → `experiment_mode=oracle`、不启用点云或柱坐标、`instance_bias_scale=1.0`。

## TXT 启动文件

直接维护用户指定的 `tools/实验.txt`，包含：

1. 推荐实验顺序。
2. 单组训练通用 PowerShell 模板。
3. 自动查找对应结果目录中最新 `best-*.ckpt` 的代码。
4. 单组测试通用 PowerShell 模板。
5. 已完成 baseline 的现有权重测试示例。

文件为每个 Mode 提供一段相互独立、可直接复制的代码。每次只复制一个实验块；该块先训练，随后自动查找本组最新的最优权重并运行测试。所有组固定使用同一数据路径、`SamplePercent=0.01`、`Seed=2026`、`NumWorkers=0` 和 `MaxEpochs=100`。

## 数据与结果约束

- 所有实验复用 `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG`。
- 不使用 `-RebuildPreparedData`，除非 BIN 格式或几何采样参数发生变化。
- 测试脚本读取对应 Mode 的最新最优权重，不使用 `last.ckpt`。
- Oracle 的测试指标是使用 GT 实例信息得到的上界，不作为可部署方法结果。
- 不自动启动训练或测试；只生成代码并运行配置级测试。

## 验证

- 扩展别名映射单元测试，验证 `dynamic_no_loss` 和 `oracle` 的实际开关。
- 检查 PowerShell 的 Mode 白名单包含新增值。
- 对 TXT 中的 PowerShell 片段做语法解析检查，但不执行训练。
