# 图一简化重绘 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将总体算法框架图重绘为五列、九个主要内容框的论文故事线总览图，同时保持三项创新清晰、数据流正确并兼容 Visio。

**Architecture:** 直接替换现有 `overall_algorithm_framework.svg` 的图元结构，不改图二、图三。图一只保留“几何拓扑输入 → 多源表征 → 融合编码 → 关系感知推理 → 任务输出”主线；实现细节通过“详见图二/图三”下放。验证分为静态结构检查、浏览器渲染检查和本机 Visio 实际导入检查。

**Tech Stack:** SVG 1.1 兼容图元、XML、PowerShell、Microsoft Edge 无头渲染、Microsoft Visio COM 导入。

## Global Constraints

- 只修改 `docs/img/overall_algorithm_framework.svg`，不修改图二、图三和其他用户文件。
- 画布保持 `1680 × 900`，采用五列横向布局和九个主要内容框。
- 创新点 1A、1B 使用橙色标签；创新点 2 使用紫色标签。
- 功能箭头只使用水平、垂直或 90° 折线路径。
- 不出现“点云”措辞，不重复图二、图三的实现级细节。
- 使用 `<style type="text/css">` 和 `orient="auto"`；不使用 `auto-start-reverse`、`foreignObject`、滤镜或蒙版。
- 重绘后必须由浏览器和本机 Visio 分别验证。

---

### Task 1: 建立图一简化验收测试

**Files:**
- Test only: `docs/img/overall_algorithm_framework.svg`

**Interfaces:**
- Consumes: 已确认设计 `docs/superpowers/specs/2026-08-13-simplified-overall-algorithm-framework-design.md`。
- Produces: 一组能区分旧版密集图与新版九框图的结构断言。

- [ ] **Step 1: 运行当前图的失败验收检查**

用 PowerShell 读取 SVG，并断言新版独有约束：

```powershell
$path = 'docs/img/overall_algorithm_framework.svg'
$svg = Get-Content -Raw -Encoding UTF8 $path
$majorCount = ([regex]::Matches($svg, 'data-role="major-box"')).Count
$required = @(
  '几何拓扑输入','多源表征','融合编码','关系感知推理','任务输出',
  '创新点 1A','创新点 1B','创新点 2',
  '输入表征与编码细节见图二','IAAB 与后层注入细节见图三',
  'X⁰=[g⁰;H⁰]','推理无需实例标签'
)
$missing = @($required | Where-Object { -not $svg.Contains($_) })
if ($majorCount -ne 9 -or $missing.Count -gt 0) {
  throw "expected simplified figure; major=$majorCount; missing=$($missing -join ',')"
}
```

Expected: FAIL，因为旧图没有九个 `data-role="major-box"`，且仍是实现细节密集布局。

- [ ] **Step 2: 记录旧图的实现级细节基线**

```powershell
$forbidden = @('32～96／面','10维样本属性','16近邻','第1～4层','第5～6层','第7～8层','正样本','困难负样本')
$present = @($forbidden | Where-Object { $svg.Contains($_) })
$present
```

Expected: 至少命中一个被下放的细节，证明测试针对的是“简化”而非普通 XML 有效性。

### Task 2: 重绘五列九框总体算法图

**Files:**
- Modify: `docs/img/overall_algorithm_framework.svg`

**Interfaces:**
- Consumes: Task 1 的结构断言和设计规格。
- Produces: `1680 × 900` 的 Visio 兼容 SVG，总共九个标记为 `data-role="major-box"` 的主要内容框。

- [ ] **Step 1: 重建画布、样式和五列背景**

保留白色画布和五列低饱和背景。定义三类箭头标记，全部使用：

```xml
<marker ... orient="auto">...</marker>
```

CSS 必须声明类型：

```xml
<style type="text/css">...</style>
```

- [ ] **Step 2: 绘制第一至第三列的五个主要框**

按以下内容绘制并添加 `data-role="major-box"`：

```text
1. 完整 B-rep 零件＋属性邻接图 AAG
2. 创新点 1A：基于内蕴度量的参数曲面表征
3. 创新点 1B：完整零件柱坐标位置＋AAG 属性
4. 参数曲面、AAG 属性与位置三分支编码
5. 以 AAG 表示为基础的门控融合 → H⁰；独立 g⁰；X⁰=[g⁰;H⁰]
```

第二列底部写“输入表征与编码细节见图二”，不绘制采样数、10 维属性、第一基本形式、邻域数和网络维数。

- [ ] **Step 3: 绘制第四至第五列的四个主要框**

按以下内容绘制并添加 `data-role="major-box"`：

```text
6. 图变换器主干：学习语义与拓扑上下文
7. 创新点 2：IAAB 预测动态实例关系并门控注入后层
8. 局部—全局融合与面级加工特征识别
9. 语义损失＋实例亲和辅助损失；推理无需实例标签
```

第四列底部写“IAAB 与后层注入细节见图三”，不拆分八层主干或双路径注意力内部结构。

- [ ] **Step 4: 用正交箭头连接主数据流和监督支路**

主数据流从左到右依次连接五列；同列内部使用垂直箭头。监督支路使用紫色虚线，从 IAAB 指向辅助损失。所有功能路径的 `d` 属性只允许 `M/H/V` 指令。

- [ ] **Step 5: 运行结构验收检查**

重新运行 Task 1 Step 1，并增加以下断言：

```powershell
[xml]$xml = Get-Content -Raw -Encoding UTF8 $path
if ($xml.svg.width -ne '1680' -or $xml.svg.height -ne '900' -or $xml.svg.viewBox -ne '0 0 1680 900') { throw 'bad canvas' }
if ($svg.Contains('auto-start-reverse')) { throw 'Visio-incompatible marker orientation' }
if (-not $svg.Contains('<style type="text/css">')) { throw 'missing CSS MIME type' }
$forbidden = @('点云','Point Cloud','32～96／面','10维样本属性','16近邻','第1～4层','第5～6层','第7～8层','正样本','困难负样本')
$found = @($forbidden | Where-Object { $svg.Contains($_) })
if ($found.Count -gt 0) { throw "implementation details remain: $($found -join ',')" }
$paths = [regex]::Matches($svg, '<path\s+class="(?:flow|dynamic|supervision)"\s+d="([^"]+)"')
$bad = @($paths | Where-Object { $_.Groups[1].Value -match '[LCQASTlcqast]' })
if ($bad.Count -gt 0) { throw 'non-orthogonal functional arrow' }
```

Expected: PASS。

### Task 3: 渲染和 Visio 验收

**Files:**
- Verify: `docs/img/overall_algorithm_framework.svg`
- Temporary preview: `tmp/paper-figures-preview/overall_algorithm_framework_simplified.png`

**Interfaces:**
- Consumes: Task 2 的最终 SVG。
- Produces: 浏览器可读性证据和 Visio 导入兼容性证据。

- [ ] **Step 1: 使用 Edge 渲染 1680 × 900 预览图**

```powershell
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$profile = 'D:\Code\BrepMFR\tmp\edge-overall-framework-simplified'
New-Item -ItemType Directory -Force $profile | Out-Null
& $edge --headless --disable-gpu --hide-scrollbars --user-data-dir=$profile `
  --window-size=1680,900 `
  --screenshot='D:\Code\BrepMFR\tmp\paper-figures-preview\overall_algorithm_framework_simplified.png' `
  'file:///D:/Code/BrepMFR/docs/img/overall_algorithm_framework.svg'
```

Expected: 生成非空 PNG，尺寸为 `1680 × 900`。

- [ ] **Step 2: 目视检查论文缩印可读性**

检查：九个主框无重叠、文字不越界、三项创新标签突出、主线箭头无歧义、图二/图三引用可见。若发现问题，只调整图一的间距、字号和箭头路由。

- [ ] **Step 3: 使用本机 Visio 实际导入 SVG**

```powershell
$visio = New-Object -ComObject Visio.InvisibleApp
try {
  $visio.AlertResponse = 7
  $doc = $visio.Documents.Add('')
  $shape = $visio.ActivePage.Import('D:\Code\BrepMFR\docs\img\overall_algorithm_framework.svg')
  if (-not $shape) { throw 'Visio returned no imported shape' }
  "SUCCESS shape=$($shape.ID) subshapes=$($shape.Shapes.Count)"
} finally {
  if ($doc) { $doc.Close() }
  $visio.Quit()
}
```

Expected: `SUCCESS`，不出现“导入/导出操作未能完成”。

- [ ] **Step 4: 检查 Visio 读取到的颜色**

递归枚举导入组合图元的 `FillForegnd` 和 `LineColor`，断言至少包含橙色 `RGB(223,113,25)`、紫色 `RGB(116,68,167)`、绿色 `RGB(91,145,83)` 和蓝色系颜色，并且九个主要框没有默认黑色填充。

- [ ] **Step 5: 检查变更边界**

```powershell
git diff --check -- docs/img/overall_algorithm_framework.svg
git status --short
```

Expected: 本轮新增变更只涉及 `docs/img/overall_algorithm_framework.svg`；图二、图三维持本轮开始时的原有状态，用户的 `docs/experiment_summary.md` 仍保持原先暂存状态。

- [ ] **Step 6: 提交图一重绘**

只暂存并提交图一：

```powershell
git add -- docs/img/overall_algorithm_framework.svg
git commit -m "docs: simplify overall algorithm framework figure" -- docs/img/overall_algorithm_framework.svg
```

提交前再次核对路径限定，禁止把其他已暂存或未跟踪文件带入提交。
