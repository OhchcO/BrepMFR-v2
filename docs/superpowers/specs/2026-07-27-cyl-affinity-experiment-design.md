# Cyl + Affinity 实验模式设计

## 目标

新增正式实验别名 `cyl_affinity`，用于验证柱坐标全局位置编码与实例亲和辅助监督的组合效果。

## 固定配置

`cyl_affinity` 必须解析为以下开关：

- `experiment_mode="dynamic_affinity"`：训练实例亲和预测头并计算亲和损失。
- `use_cylindrical_pos=True`：向节点特征加入柱坐标位置编码。
- `use_face_pointcloud=False`：不启用面点云分支。
- `instance_bias_scale=0.0`：不把预测实例偏置注入注意力。

因此该模式不会包含 Oracle GT Bias、Dynamic Bias 注入或面点云编码。

## 接口范围

- `segmentation.py` 接受并解析 `--experiment_alias cyl_affinity`。
- `tools/run_mfrinstseg_experiments.ps1` 的 `-Mode` 接受 `cyl_affinity`，并为其生成独立结果目录。
- `tools/实验.txt` 增加一段可复制的训练命令和一段自动选择最佳权重的测试命令。
- 自动准备的数据缓存身份不随实验别名变化，继续复用包含柱坐标字段的现有 BIN 缓存。
- 测试脚本继续从 checkpoint 超参数推断需要的柱坐标字段，无需新增测试参数。

## 验证

- 别名映射测试必须证明四个开关精确符合固定配置。
- PowerShell 参数元数据测试必须证明启动器接受新模式。
- 实验命令文件测试必须证明十个正式模式各有一组训练和测试命令。
- 运行相关单元测试及 PowerShell 语法检查，不运行数据转换或模型训练。

