# 表示消融与实例评测实施计划

## 1. 表示开关

1. 先补配置测试，覆盖新别名、默认 UV 行为和旧别名兼容性。
2. 为 `GraphNodeFeature` 增加 `use_face_uv` 参数。
3. 关闭 UV 时使用零向量占据原 UV 编码维度，保持隐藏维度不变。
4. 将开关依次穿过 `BrepEncoder`、`BrepSeg`、checkpoint 超参数和训练入口。
5. 增加 `topo_aag`、`uv_aag`、`pc_aag`、`uv_pc_aag` 启动模式。

## 2. 实例评测

1. 先为 Pairwise 计数、语义约束拓扑聚类、ARI 和实例数量误差编写小型张量
   单元测试。
2. 为 `BrepSeg` 增加统一推理方法，同时返回语义概率和可选亲和 logits，
   原 `forward` 返回值保持不变。
3. 测试脚本增加实例评测参数，并在动态模式下收集亲和 logits。
4. 只将 GT 实例用于指标计算；动态模型前向不接收 GT。
5. 写入汇总指标和逐样本实例分组 JSON。

## 3. 启动与验证

1. 更新训练 PowerShell 的 Mode 白名单和 `-All` 顺序。
2. 更新测试 PowerShell，默认开启可用的实例评测，并允许关闭。
3. 在 `tools/实验.txt` 增加四个表示消融的可复制训练/测试块。
4. 运行聚焦单元测试和 PowerShell 语法检查；不运行训练、不重建全量 BIN。
