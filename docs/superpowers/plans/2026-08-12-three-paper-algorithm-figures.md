# Three Paper Algorithm Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three compact, publication-ready Chinese SVG figures that separately communicate the overall method, the fused input representation, and the IAAB core algorithm while remaining faithful to the current code and experiment configuration.

**Architecture:** Create three independent `1680×900` SVG assets with one shared visual vocabulary: five full-height columns, dense internal network modules, consistent semantic colors, and orthogonal functional arrows. Implement and verify each figure independently, then run a cross-figure consistency audit for symbols, terminology, layout, XML validity, and rendered readability.

**Tech Stack:** Hand-authored SVG 1.1/XML, PowerShell validation, regular-expression structural checks, Microsoft Edge headless rendering, Codex local-image inspection, Git.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-08-12-three-paper-algorithm-figures-design.md`.
- Every canvas is exactly `1680×900` with `viewBox="0 0 1680 900"`.
- Every figure uses five full-height columns and vertically stacked internal modules.
- Visible prose is Chinese; only B-rep, AAG, IAAB, MLP, FPS, mathematical symbols, tensor names, and formulas may remain non-Chinese.
- No visible text may contain `点云`, `Point Cloud`, `PointCloud`, `Point MLP`, or `Surface Point Samples`.
- Functional arrows use only horizontal, vertical, or 90-degree path segments. Arrow paths may contain `M`, `H`, `V`, and `Z`; they may not contain `L`, `C`, `Q`, `A`, `S`, or `T`.
- Geometry icons may use diagonal non-arrow strokes.
- Use gray for inputs, orange for parametric-surface/position/gating content, blue for encoders/fusion, green for the graph-transformer backbone, and purple for IAAB/affinity supervision.
- Innovation labels are orange for 1A/1B and purple for 2.
- Preserve `docs/img/spe_pcaag_iaab_framework.svg` as the historical single-figure version.
- Do not modify model, data, experiment, or result-report files.
- Do not stage or commit pre-existing user changes such as `docs/experiment_summary.md` unless the user explicitly requests it.

---

## File Structure

- Create `docs/img/overall_algorithm_framework.svg`: high-level five-stage method overview.
- Create `docs/img/input_representation_fusion.svg`: exact parametric-surface, AAG, position, encoding, fusion, and tokenization data flow.
- Create `docs/img/iaab_core_algorithm.svg`: eight-layer split, IAAB prediction, two-path late attention, outputs, and training/inference contract.
- Create temporary ignored previews under `tmp/`:
  - `tmp/overall_algorithm_framework_preview.png`
  - `tmp/input_representation_fusion_preview.png`
  - `tmp/iaab_core_algorithm_preview.png`
- Modify no existing source-code or experiment-result file.

---

### Task 1: Draw and Verify the Overall Algorithm Framework

**Files:**
- Create: `docs/img/overall_algorithm_framework.svg`
- Reference: `docs/superpowers/specs/2026-08-12-three-paper-algorithm-figures-design.md`
- Reference: `docs/img/spe_pcaag_iaab_framework.svg`
- Render: `tmp/overall_algorithm_framework_preview.png`

**Interfaces:**
- Consumes: the design specification's Section 3 and the shared visual constraints.
- Produces: `X⁰=[g⁰;H⁰]` as the conceptual bridge to Figures 2 and 3, and a final `H⁸/g⁸`-to-semantic-prediction storyline.

- [ ] **Step 1: Re-read the relevant specification and map exact visible labels**

Record the required five headers before drawing:

```text
B-rep 与 AAG
多源输入表征
融合编码
图变换器与 IAAB
预测与监督
```

Required semantic labels:

```text
创新点 1A
基于内蕴度量的参数曲面表征
创新点 1B
完整零件柱坐标位置
面级初始嵌入 H⁰
可学习图级令牌 g⁰
X⁰=[g⁰;H⁰]
创新点 2 · IAAB
动态实例亲和预测
后层门控残差注入
面级语义识别
实例亲和辅助监督
推理仅需几何与拓扑
```

- [ ] **Step 2: Create the SVG shell and shared visual definitions**

Use `apply_patch` to create an XML SVG with:

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="900" viewBox="0 0 1680 900">
  <defs>
    <!-- blue/orange/purple arrow markers -->
  </defs>
  <style>
    /* Microsoft YaHei / SimHei font stack, column fills, panels, labels, arrows */
  </style>
  <rect width="1680" height="900" fill="#ffffff"/>
  <!-- five adjacent full-height columns -->
</svg>
```

Use approximate column widths `245 / 300 / 330 / 440 / 315` inside 25-pixel outer margins. Reserve a header band of about 58 pixels and use the remaining height for vertically stacked internal panels.

- [ ] **Step 3: Draw Column 1 — B-rep and AAG**

Include:

```text
完整 B-rep 零件
属性邻接图 AAG
面为节点 · 邻接关系为边
面属性
边属性
面对描述符
```

Use a compact B-rep polygon/surface icon, an AAG node-link icon, and three filled horizontal attribute bars. Connect the B-rep icon to AAG with a vertical arrow.

- [ ] **Step 4: Draw Column 2 — Multi-source Input Representation**

Create three vertically stacked panels:

```text
创新点 1A · 基于内蕴度量的参数曲面表征
AAG 属性与结构
创新点 1B · 完整零件柱坐标位置
```

Keep internals deliberately high level:

```text
自适应参数采样 · 10维属性 · 第一基本形式 · 内蕴邻域
面积 · 类型 · 环数 · 节点度 · 最短路径 · 距离／夹角
[r̄, sinθ, cosθ, z̄]
```

Do not include the full equations reserved for Figure 2.

- [ ] **Step 5: Draw Column 3 — Fusion Encoding and Correct Tokenization**

Show three encoder lanes entering a gated residual fusion block:

```text
参数曲面层次编码器 → fᵢ^surf
AAG 属性编码器 → fᵢ^AAG
柱坐标位置编码器 → pᵢ
```

Then show:

```text
两路门控残差融合 → 面级初始嵌入 H⁰
独立可学习图级令牌 g⁰
拼接 X⁰=[g⁰;H⁰]
```

The arrow to `g⁰` must originate from an independent learned-token block, not from the fusion block.

- [ ] **Step 6: Draw Column 4 — Graph Transformer and IAAB Overview**

Use vertically stacked backbone blocks:

```text
第1～4层 · 前层基础图编码
H_mid
第5～6层 · 基础图编码
IAAB · 动态实例亲和预测
第7～8层 · 后层门控残差注入
```

Use one orthogonal side branch from `H_mid` to IAAB and from IAAB back into the last-two-layer block. Do not show detailed IAAB formulas in Figure 1.

- [ ] **Step 7: Draw Column 5 — Prediction and Supervision**

Show:

```text
最终面特征 H⁸
图级表示 g⁸
局部—全局注意力融合
面级语义分类
实例亲和辅助监督
L=L_sem+λ(t)L_aff
训练：语义标签＋实例关系
推理：仅几何与拓扑
```

Use solid blue/green flow for inference and dashed purple arrows for training supervision.

- [ ] **Step 8: Run structural validation**

Run a PowerShell check equivalent to:

```powershell
$path = 'docs\img\overall_algorithm_framework.svg'
$text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
[xml]$xml = $text
if ($xml.svg.width -ne '1680' -or $xml.svg.height -ne '900') { throw 'wrong canvas' }
$required = @(
  'B-rep 与 AAG','多源输入表征','融合编码','图变换器与 IAAB','预测与监督',
  '面级初始嵌入','可学习图级令牌','X⁰=[g⁰;H⁰]','动态实例亲和预测',
  '后层门控残差注入','面级语义识别','推理仅需几何与拓扑'
)
$missing = @($required | Where-Object { -not $text.Contains($_) })
if ($missing) { throw ('missing: ' + ($missing -join ', ')) }
$forbidden = @('点云','Point Cloud','PointCloud','Point MLP','Surface Point Samples')
$bad = @($forbidden | Where-Object { $text.Contains($_) })
if ($bad) { throw ('forbidden: ' + ($bad -join ', ')) }
$arrowPaths = [regex]::Matches($text, '<path\s+class="(?:flow|flow-thin|dynamic|supervision)"\s+d="([^"]+)"')
$nonOrthogonal = @($arrowPaths | Where-Object { $_.Groups[1].Value -match '[LlCcQqAaSsTt]' })
if ($nonOrthogonal) { throw 'non-orthogonal functional arrow' }
Write-Output 'overall SVG structural validation: PASS'
```

Expected: `overall SVG structural validation: PASS`.

- [ ] **Step 9: Render and visually inspect**

Run:

```powershell
& 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' `
  --headless --disable-gpu --hide-scrollbars `
  --window-size=1680,900 `
  --screenshot='D:\Code\BrepMFR\tmp\overall_algorithm_framework_preview.png' `
  'file:///D:/Code/BrepMFR/docs/img/overall_algorithm_framework.svg'
```

Inspect the PNG at original resolution. Reject and revise the SVG if any label crosses a box boundary, any arrow intersects text, any column has a large dead area, or `g⁰` visually appears to be a fusion output.

- [ ] **Step 10: Commit Figure 1**

```powershell
git add -- docs/img/overall_algorithm_framework.svg
git commit -m "docs: add overall algorithm framework figure"
```

Before committing, run `git diff --cached --name-only` and require that only `docs/img/overall_algorithm_framework.svg` is staged for this commit. Preserve the user's pre-staged `docs/experiment_summary.md` by using an exact-path commit if necessary.

---

### Task 2: Draw and Verify the Fused Input Representation Figure

**Files:**
- Create: `docs/img/input_representation_fusion.svg`
- Reference: `tools/geometry_features.py:18-36`
- Reference: `tools/geometry_features.py:88-217`
- Reference: `tools/geometry_features.py:307-356`
- Reference: `tools/geometry_features.py:406-517`
- Reference: `tools/geometry_features.py:537-655`
- Reference: `models/modules/pointcloud_encoder.py:72-295`
- Reference: `models/modules/pointcloud_encoder.py:298-346`
- Reference: `models/modules/layers/brep_encoder_layer.py:240-371`
- Render: `tmp/input_representation_fusion_preview.png`

**Interfaces:**
- Consumes: exact B-rep surface geometry, AAG attributes/relations, complete-part face centroids, and the current `intrinsic + untrimmed + gated` configuration.
- Produces: `fᵢ^surf`, `fᵢ^AAG`, `pᵢ`, `H⁰`, independent `g⁰`, and `X⁰=[g⁰;H⁰]`.

- [ ] **Step 1: Confirm code constants before drawing**

Read the referenced code and record these exact values:

```text
MAX_FACE_POINTS = 96
SURFACE_NEIGHBOR_COUNT = 16
POINT_FEATURE_DIM = 10
ADAPTIVE_BASE_COUNT = 64
candidate grids = 20, 32, 48, 64, 80
min/max samples = 32/96
area exponent = 0.5
parametric sample input = 10D attributes + 3D metric = 13D
point MLP = 13→64→64
edge input = 64+3+2+1 = 70
edge MLP = 70→128→128
local fusion = 192→128
coarse_count = 24
formal configuration = intrinsic + untrimmed + gated
```

Abort the drawing step and update the specification if current code contradicts any value.

- [ ] **Step 2: Create the shared SVG shell**

Use the exact same font stack, colors, line widths, corner radii, header height, markers, and column spacing established by Figure 1. Use approximate column widths `230 / 310 / 355 / 400 / 335` inside the same outer margins.

- [ ] **Step 3: Draw Column 1 — Geometry and Topology Sources**

Show three source lanes:

```text
精确 B-rep 参数曲面 S(u,v)
属性邻接图 AAG
完整零件面质心 {cᵢ}
```

Use a trimmed parametric-patch icon, an AAG graph icon, and a small complete-part coordinate icon. Do not show an external scan or point-set silhouette.

- [ ] **Step 4: Draw Column 2 — Adaptive Parametric Sampling**

Show the exact sequential process:

```text
OCC曲面面积 Aᵢ
Nᵢ=RoundTo8(Clip(64(Aᵢ/A_median)^0.5,32,96))
候选网格 20²→32²→48²→64²→80²
裁剪有效域＋有限几何／法向／曲率过滤
确定性三维 FPS
选中参数位置 {(uᵢⱼ,vᵢⱼ)}
```

The output of this column is selected surface sample locations, not a separate mesh or point-cloud file.

- [ ] **Step 5: Draw Column 3 — Input Contracts and Intrinsic Neighborhood**

Use an orthogonal split from selected sample locations to:

```text
10维样本属性 qᵢⱼ=[x,y,z,nx,ny,nz,ū,v̄,HL,KL²]
第一基本形式 Mᵢⱼ=[[E,F],[F,G]]/L²
```

Then merge `ū,v̄` and `M` into:

```text
d²ᵢⱼ=Δuᵀ((Mᵢ+Mⱼ)/2)Δu
每个样本16个内蕴近邻
当前未启用截剪跨孔过滤
```

Add parallel AAG and position contracts:

```text
AAG：面积／类型／环数／节点度＋面对描述符
位置：[r̄,sinθ,cosθ,z̄]
```

- [ ] **Step 6: Draw Column 4 — Three Encoders with Network Composition**

Parametric-surface lane:

```text
13维输入
Linear 13→64 · Norm · GELU
Linear 64→64 · Norm · GELU
16近邻边构造 [hⱼ,Δxyz,Δuv,d_intrinsic]
Linear 70→128 · Norm · GELU
Linear 128→128 · Norm · GELU
邻域最大聚合
中心＋邻域 192→128
前24样本粗粒度最大池化
输出 fᵢ^surf
```

AAG lane:

```text
面积非线性编码＋类型／环数／节点度嵌入
拼接输出 fᵢ^AAG
```

Position lane:

```text
柱坐标位置 MLP
输出 pᵢ
```

- [ ] **Step 7: Draw Column 5 — Gated Fusion and Tokenization**

Show separate gates for the two auxiliary branches:

```text
gᵢ^surf=tanh(G_surf([fᵢ^AAG;fᵢ^surf]))
gᵢ^pos=tanh(G_pos([fᵢ^AAG;pᵢ]))
hᵢ⁰=fᵢ^AAG+gᵢ^surf⊙fᵢ^surf+gᵢ^pos⊙pᵢ
H⁰={hᵢ⁰}
```

Draw an independent learned-token block:

```text
可学习图级令牌 g⁰
```

Then concatenate:

```text
X⁰=[g⁰;H⁰]
```

- [ ] **Step 8: Run exact content and arrow validation**

Check XML parsing, `1680×900`, absence of forbidden terms, absence of mojibake, and orthogonal functional paths. Require these visible phrases:

```text
自适应参数采样
32～96／面
20²→32²→48²→64²→80²
确定性三维 FPS
10维样本属性
第一基本形式
16个内蕴近邻
当前未启用截剪跨孔过滤
13维输入
70→128
前24样本粗粒度最大池化
AAG属性编码器
柱坐标位置编码器
门控残差融合
面级初始嵌入 H⁰
可学习图级令牌 g⁰
X⁰=[g⁰;H⁰]
```

Expected: no missing labels, no forbidden labels, and zero non-orthogonal functional arrows.

- [ ] **Step 9: Render and visually inspect**

Render with Edge at `1680×900` to `tmp/input_representation_fusion_preview.png`. Inspect at original resolution and specifically reject:

```text
- a diagram that implies E,F,G are computed before sample selection;
- a diagram that implies intrinsic neighbors are exact global geodesics;
- a diagram that claims trim-aware filtering is enabled;
- a diagram that routes g⁰ out of the fusion block;
- formulas touching box borders;
- network micro-block labels below readable size.
```

- [ ] **Step 10: Commit Figure 2**

```powershell
git add -- docs/img/input_representation_fusion.svg
git commit -m "docs: add fused input representation figure"
```

Use exact-path staging and committing; do not include unrelated staged or untracked files.

---

### Task 3: Draw and Verify the IAAB Core Algorithm Figure

**Files:**
- Create: `docs/img/iaab_core_algorithm.svg`
- Reference: `models/modules/instance_bias.py:15-34`
- Reference: `models/modules/instance_bias.py:37-265`
- Reference: `models/modules/brep_encoder.py:283-421`
- Reference: `models/modules/layers/brep_encoder_layer.py:126-200`
- Reference: `models/brepseg_model.py:106-169`
- Reference: `models/brepseg_model.py:342-359`
- Reference: `models/brepseg_model.py:403-446`
- Render: `tmp/iaab_core_algorithm_preview.png`

**Interfaces:**
- Consumes: `X⁰=[g⁰;H⁰]`, fixed graph bias `B_AAG`, pair descriptors, and the current late-gated positive-only dynamic-affinity configuration.
- Produces: affinity logits, `B_inst`, final `H⁸/g⁸`, semantic predictions, and the training losses.

- [ ] **Step 1: Confirm IAAB code semantics before drawing**

Record these exact contracts from code:

```text
8 transformer layers
prediction boundary = min(layer_count/2, late_layer_start) = 4
late layers = last 2 = layers 7–8
affinity dimension = 64
temperature τ = 0.2
relation mapping = positive_only
confidence power γ = 1.0
one independent residual gate per late layer
gate initialization = 0
dynamic_affinity relation is detached before attention injection
graph-token row/column of B_inst = 0
```

If the active experiment configuration differs, update the visible annotation to the verified value before drawing.

- [ ] **Step 2: Create the shared SVG shell**

Reuse Figure 1 and Figure 2 visual definitions exactly. Use approximate column widths `240 / 300 / 360 / 455 / 275` inside the common margins.

- [ ] **Step 3: Draw Column 1 — Graph Sequence and Fixed Structural Bias**

Show:

```text
X⁰=[g⁰;H⁰]
可学习图级令牌＋N个面令牌
B_AAG
最短路径 · 多跳边 · 面间距离 · 面间夹角 · 边属性
```

Route `X⁰` to the transformer stream and `B_AAG` to both base-attention usages.

- [ ] **Step 4: Draw Column 2 — Front and Base Encoding**

Show the main path:

```text
第1～4层前层编码
Attn(X;B_AAG)
中间表示 [g⁴;H_mid]
第5～6层基础编码
```

Use an orthogonal branch from `H_mid` to Column 3. The main path continues directly from layers 1–4 to layers 5–6.

- [ ] **Step 5: Draw Column 3 — IAAB Affinity Prediction**

Use the exact sequence:

```text
排除图级令牌
LayerNorm＋64维投影
图内均值中心化
L₂归一化 → E
内容 logits：EEᵀ/τ
面对 logits：路径＋距离＋夹角
求和并对称化
c=2σ(logits)-1
r⁺=max(0,c|c|^γ)
B_inst
```

Annotate:

```text
τ=0.2，γ=1，positive_only
B_inst 仅作用于面—面
dynamic_affinity 注入分支停止梯度
```

- [ ] **Step 6: Draw Column 4 — Layers 7–8 Shared-Parameter Dual Attention**

For each late layer, show one shared input feeding two attention blocks:

```text
O_base=Softmax(QKᵀ/√d+B_AAG)V
O_bias=Softmax(QKᵀ/√d+B_AAG+B_inst)V
ΔO=O_bias-O_base
O=O_base+g_lΔO
O=O_base+g_l(O_bias-O_base)
```

Annotate:

```text
两路径共享 Q/K/V 与注意力参数
第7层门控 g₇
第8层门控 g₈
零初始化＋训练进度调度
```

Then show the normal transformer residual and FFN continuation without expanding unrelated implementation details.

- [ ] **Step 7: Draw Column 5 — Outputs, Losses, and Inference Contract**

Show:

```text
最终 [g⁸;H⁸]
每个面融合 hᵢ⁸ 与 g⁸
面级语义分类
亲和 logits
平衡亲和损失：正样本／困难负样本／简单负样本
L=L_sem+λ(t)L_aff
训练：实例矩阵仅用于亲和监督
推理：自行预测 B_inst，不需要实例标签
```

The affinity matrix is an auxiliary learned relation, not the final semantic output.

- [ ] **Step 8: Run exact content and arrow validation**

Require these labels/formulas:

```text
第1～4层
H_mid
第5～6层
LayerNorm
图内均值中心化
L₂归一化
EEᵀ/τ
对称化
positive_only
B_inst 仅作用于面—面
第7～8层
O_base
O_bias
O=O_base+g_lΔO
零初始化
平衡亲和损失
推理不需要实例标签
```

Run the same XML, forbidden-text, mojibake, and orthogonal-arrow checks as Tasks 1 and 2.

- [ ] **Step 9: Render and visually inspect**

Render to `tmp/iaab_core_algorithm_preview.png` at `1680×900`. Reject and revise if:

```text
- H_mid appears after layers 5–6 rather than after layer 4;
- IAAB appears serially between layers 4 and 5 instead of as a side branch;
- the base path omits QKᵀ/√d;
- B_inst is added directly to feature tensors;
- the diagram suggests instance labels are required at inference;
- the two late gates appear to be one shared gate;
- affinity supervision arrows merge with inference arrows without visual distinction.
```

- [ ] **Step 10: Commit Figure 3**

```powershell
git add -- docs/img/iaab_core_algorithm.svg
git commit -m "docs: add IAAB core algorithm figure"
```

Use exact-path staging and committing; leave all unrelated changes untouched.

---

### Task 4: Cross-Figure Consistency and Publication-Readiness Audit

**Files:**
- Verify: `docs/img/overall_algorithm_framework.svg`
- Verify: `docs/img/input_representation_fusion.svg`
- Verify: `docs/img/iaab_core_algorithm.svg`
- Reference: `docs/superpowers/specs/2026-08-12-three-paper-algorithm-figures-design.md`
- Render: all three PNG previews under `tmp/`

**Interfaces:**
- Consumes: the three independently validated SVGs.
- Produces: a coherent publication figure set with consistent symbols and no implementation contradictions.

- [ ] **Step 1: Run a combined machine-readable validation**

For each SVG, verify:

```text
XML parse succeeds
width=1680
height=900
viewBox=0 0 1680 900
five column headers exist
no forbidden point-cloud wording
no mojibake patterns
no non-orthogonal functional arrow paths
```

Require the cross-figure bridges:

```text
Figure 1 contains X⁰=[g⁰;H⁰]
Figure 2 ends with X⁰=[g⁰;H⁰]
Figure 3 begins with X⁰=[g⁰;H⁰]
Figure 1 and Figure 3 contain H⁸ and g⁸
```

- [ ] **Step 2: Compare visual-system constants**

Inspect the SVG CSS and require equal values for:

```text
font family
section-header font size
body font size
column-border stroke
panel corner radius
flow-arrow stroke width and marker size
input/orange/blue/green/purple fills
```

If values differ, patch Figures 2 and 3 to match Figure 1 unless a smaller formula font is explicitly required.

- [ ] **Step 3: Render all three figures again from final files**

Run Edge headless three times with `--window-size=1680,900`, overwriting all temporary previews. Confirm each output exists and has non-zero size.

- [ ] **Step 4: Inspect all three previews at original resolution**

Check:

```text
balanced column widths
no large dead zones
no text overflow or clipped formulas
no arrow/text collisions
clear innovation labels
consistent symbols H⁰, g⁰, X⁰, H_mid, B_AAG, B_inst, H⁸, g⁸
readability after likely thesis-page scaling
```

Revise and re-render any figure that fails one item.

- [ ] **Step 5: Run repository checks**

Run:

```powershell
git diff --check
git status --short
git diff -- docs/img/overall_algorithm_framework.svg docs/img/input_representation_fusion.svg docs/img/iaab_core_algorithm.svg
```

Confirm no model, data, experiment, result, or unrelated documentation file was modified by the drawing work.

- [ ] **Step 6: Final figure-set commit if audit required changes**

If Task 4 changed any SVG after its individual commit:

```powershell
git add -- docs/img/overall_algorithm_framework.svg docs/img/input_representation_fusion.svg docs/img/iaab_core_algorithm.svg
git commit -m "docs: align paper algorithm figure set"
```

If the audit required no changes, do not create an empty commit.

- [ ] **Step 7: Handoff**

Return absolute clickable links and rendered previews:

```text
D:\Code\BrepMFR\docs\img\overall_algorithm_framework.svg
D:\Code\BrepMFR\docs\img\input_representation_fusion.svg
D:\Code\BrepMFR\docs\img\iaab_core_algorithm.svg
```

Report:

```text
XML/size validation results
functional-arrow orthogonality result
forbidden-word result
rendered visual-inspection result
git commit hashes created during implementation
confirmation that unrelated existing changes were preserved
```
