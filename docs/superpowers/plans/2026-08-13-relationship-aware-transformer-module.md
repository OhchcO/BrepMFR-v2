# 图一关系感知图变换器模块改版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将图一第四列改为一个包含“前层基础编码、IAAB 实例关系预测、后两层偏置增强”的关系感知图变换器模块，准确表达实例关系偏置的生成与注入顺序。

**Architecture:** 仅修改 `docs/img/overall_algorithm_framework.svg` 中第四列及其连接第五列的主线。第四列使用一个主要外框和三个内部阶段框，外层主要框总数由九个变为八个；第五列预测与损失结构保持不变。

**Tech Stack:** SVG/XML、PowerShell 静态检查、Microsoft Edge 无头渲染、Microsoft Visio COM 导入。

## Global Constraints

- 只修改 `docs/img/overall_algorithm_framework.svg`；不修改图二、图三和其他用户文件。
- 第四列仅保留一个 `data-role="major-box"` 外框，内部必须有三个 `data-role="reasoning-stage"` 阶段框。
- 三阶段顺序必须是前层基础编码、IAAB 实例关系预测、后两层偏置增强。
- 可见文字必须包含 `H_mid`、`B_inst`、`B_AAG + B_inst`、`H⁸` 和 `g⁸`。
- 不展开亲和矩阵、双路径公式、门控标量、正负样本划分和具体前层数量。
- 所有功能箭头保持水平、垂直或 90° 折线。
- 保持 `1680 × 900`、`<style type="text/css">`、`orient="auto"` 和现有颜色语义。
- 必须用浏览器和本机 Visio 验证。

---

### Task 1: 建立第四列改版失败验收

**Files:**
- Test only: `docs/img/overall_algorithm_framework.svg`

**Interfaces:**
- Consumes: 当前九框简化版图一。
- Produces: 能区分旧第四列和新三阶段模块的结构断言。

- [ ] **Step 1: 运行失败验收**

```powershell
$svg = Get-Content -Raw -Encoding UTF8 'docs/img/overall_algorithm_framework.svg'
$major = ([regex]::Matches($svg, 'data-role="major-box"')).Count
$stages = ([regex]::Matches($svg, 'data-role="reasoning-stage"')).Count
$required = @('关系感知图变换器','前层基础编码','H_mid','IAAB 实例关系预测','B_inst','后两层偏置增强','B_AAG + B_inst','H⁸','g⁸')
$missing = @($required | Where-Object { -not $svg.Contains($_) })
if ($major -ne 8 -or $stages -ne 3 -or $missing.Count -gt 0) {
  throw "expected relationship-aware module; major=$major stages=$stages missing=$($missing -join ',')"
}
```

Expected: FAIL，因为当前图一仍有九个主要框，第四列是两个独立主要框且没有三阶段标记。

### Task 2: 重绘关系感知图变换器模块

**Files:**
- Modify: `docs/img/overall_algorithm_framework.svg`

**Interfaces:**
- Consumes: 第三列输出 `X⁰=[g⁰;H⁰]`。
- Produces: 第四列输出最终 `H⁸`、`g⁸` 并连接第五列面级预测。

- [ ] **Step 1: 删除旧第四列两个主要框及内部图元**

删除“图变换器主干”和“IAAB 动态实例关系推理”两个 `data-role="major-box"`，以及旧的圆形关系示意、门控残差框和两框之间连线。

- [ ] **Step 2: 绘制统一外框**

在第四列绘制一个 `data-role="major-box"`，标题为“关系感知图变换器”。外框覆盖原来两个主要框的垂直区域，并在底部保留“IAAB 与后层注入细节见图三”。

- [ ] **Step 3: 绘制三个阶段框**

依次绘制三个 `data-role="reasoning-stage"`：

```text
前层基础编码
输入 X⁰ 与固定结构偏置 B_AAG
学习语义与拓扑上下文 → H_mid

IAAB 实例关系预测（创新点 2）
由 H_mid 预测面—面亲和关系
生成动态实例偏置 B_inst

后两层偏置增强
B_AAG + B_inst
关系感知注意力计算 → H⁸、g⁸
```

阶段一使用绿色，阶段二使用紫色，阶段三使用橙色边框和偏置注入强调色。

- [ ] **Step 4: 重连主数据流**

将第三列 `X⁰` 的连线接入阶段一；三个阶段使用垂直正交箭头依次连接；阶段三输出从第四列右侧接入第五列“面级加工特征识别”。IAAB 到辅助损失的紫色监督虚线保留，但从阶段二右侧引出。

- [ ] **Step 5: 运行静态验收**

重新运行 Task 1 Step 1，并补充：

```powershell
[xml]$xml = $svg
if ($xml.svg.width -ne '1680' -or $xml.svg.height -ne '900') { throw 'bad canvas' }
if ($svg.Contains('auto-start-reverse') -or -not $svg.Contains('<style type="text/css">')) { throw 'Visio incompatibility' }
$forbidden = @('点云','Point Cloud','亲和矩阵','双路径注意力公式','门控标量','正样本','困难负样本')
$found = @($forbidden | Where-Object { $svg.Contains($_) })
if ($found.Count -gt 0) { throw "too detailed: $($found -join ',')" }
$paths = [regex]::Matches($svg, '<path\s+class="(?:flow|dynamic|supervision)"\s+d="([^"]+)"')
$bad = @($paths | Where-Object { $_.Groups[1].Value -match '[LCQASTlcqast]' })
if ($bad.Count -gt 0) { throw 'non-orthogonal functional arrow' }
```

Expected: PASS。

### Task 3: 渲染、Visio 与变更边界验收

**Files:**
- Verify: `docs/img/overall_algorithm_framework.svg`
- Temporary: `tmp/paper-figures-preview/overall_algorithm_framework_relationship_module.png`

**Interfaces:**
- Consumes: Task 2 完成的 SVG。
- Produces: 浏览器和 Visio 验收证据。

- [ ] **Step 1: 渲染 1680 × 900 预览**

使用 Edge 无头模式把目标 SVG 渲染到临时 PNG，确认文件非空。

- [ ] **Step 2: 目视检查**

确认第四列三个阶段自上而下清晰、无文字越界、无框体重叠；主线明确显示 `H_mid → B_inst → B_AAG + B_inst → H⁸/g⁸`；第五列不被改动。

- [ ] **Step 3: Visio 实际导入与颜色检查**

使用 `Visio.InvisibleApp` 导入源 SVG，递归读取图元颜色，确认绿色、紫色、橙色和蓝色均存在，且不出现导入错误。

- [ ] **Step 4: 检查其他文件哈希**

记录修改前图二、图三、实验总结和其他用户文件的 SHA-256；修改后逐一比较，必须全部不变。

- [ ] **Step 5: 只提交图一**

```powershell
git commit -m "docs: clarify relationship-aware transformer flow" -- docs/img/overall_algorithm_framework.svg
```

确认提交只包含图一；其他已暂存、已修改或未跟踪文件保持原状态。
