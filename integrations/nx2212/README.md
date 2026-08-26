# BrepMFR NX2212 识别插件

该插件把 NX2212 用作当前 STEP 的入口和识别结果可视化界面。STEP 转 BIN、缓存
校验、模型加载、推理、后处理和 `result.json` 写入全部由独立 Python 进程完成；
DLL 不会把 Python、PyTorch 或 CUDA 加载进 NX 进程。

Python 输出 JSON 1.1 时，插件从当前 Work Part 提取 NX 面几何/拓扑签名，进行
OCC/AAG 面到 NX Face Tag 的全局一一映射，再按特征实例为对应面着色。JSON 中的
`face_index` 始终是 OCC/AAG 构图时的 0-based 面序号，不是 NX Face Tag；二者只
能通过已验证的映射关系对应，不能直接互换。

## 环境要求

- Siemens NX 2212，并设置 `UGII_BASE_DIR`。
- VS2019/v142/x64 C++ 工具和 Windows 10 SDK。
- 能独立运行 `tools/infer_step.py` 的 Python/Conda 环境。
- 与当前模型结构兼容的 `.ckpt` checkpoint。

在仓库根目录打开 PowerShell，先确认 NXOpen 开发文件可用：

```powershell
if (-not $env:UGII_BASE_DIR) {
    throw 'UGII_BASE_DIR is not set.'
}
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\NXOpen\UI.hxx')
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\libnxopencpp.lib')
```

两项都应返回 `True`。

## 构建和测试

以下命令通过 `vswhere.exe` 定位 Visual Studio，不依赖固定安装盘符：

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if ([string]::IsNullOrWhiteSpace($vsRoot)) {
    throw 'A Visual Studio C++ installation was not found.'
}
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'

& $msbuild integrations\nx2212\BrepMFRNx2212.sln `
    /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
if ($LASTEXITCODE -ne 0) {
    throw "MSBuild failed with exit code $LASTEXITCODE."
}

& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'
if ($LASTEXITCODE -ne 0) {
    throw "Runtime tests failed with exit code $LASTEXITCODE."
}

python -m pytest tests\test_nx2212_project.py -q
```

主要构建产物位于：

```text
integrations/nx2212/build/Release/BrepMFRNx2212.dll
integrations/nx2212/build/Release/BrepMFRNx2212.config.example.json
```

## 配置 Python 推理后台

构建后，把 DLL 旁边的示例复制为真实配置：

```powershell
$release = 'integrations\nx2212\build\Release'
Copy-Item `
    "$release\BrepMFRNx2212.config.example.json" `
    "$release\BrepMFRNx2212.config.json"
```

编辑 `BrepMFRNx2212.config.json`：

```json
{
  "schema_version": "1.0",
  "python_executable": "C:/absolute/path/to/python.exe",
  "inference_script": "D:/absolute/path/to/BrepMFR/tools/infer_step.py",
  "checkpoint": "D:/absolute/path/to/best-model.ckpt",
  "output_root": "D:/absolute/path/to/recognition-output",
  "device": "auto",
  "affinity_threshold": 0.5
}
```

四个路径必须是绝对路径；Python、推理脚本和 checkpoint 必须已经存在，
checkpoint 必须以 `.ckpt` 结尾。`device` 只能是 `auto`、`cpu` 或 `cuda`，
阈值必须在 0 到 1 之间。真实配置位于被忽略的 `build/` 目录，不应提交到 Git。

建议先在 PowerShell 里用同一 Python、STEP 和 checkpoint 直接运行 CLI；确认
生成 `status=success`、`schema_version=1.1`、`face_mapping.status=ready` 的
`result.json` 后，再从 NX 加载 DLL。这样可以把 Python 环境问题与 NX 集成问题
分开定位。转换输出还包含 `input.face_signatures.json`；它与 `input.bin` 共享缓存
身份，供 JSON 1.1 面映射使用。

## 在 NX2212 中运行

1. 启动 NX2212，用 `File -> Open` 打开要识别的 `.step` 或 `.stp`。
2. 第一版要求当前 Work Part 未经修改，并且只包含一个非片体、非收敛体的精确
   solid body；打开 STEP 后不要编辑、切换 Part 或增删面。
3. 使用 `File -> Execute -> NX Open`（快捷键通常为 `Ctrl+U`），选择
   `build/Release/BrepMFRNx2212.dll`。插件不再显示 STEP 文件选择器，输入就是当前
   Work Part 对应的原始 STEP 路径。
4. Python 启动后会显示当前 STEP、真实处理阶段、不确定进度条和已用时间。首版
   不提供强制取消，避免损坏正在写入的转换缓存。
5. 成功后，信息框显示面数量、特征实例数、BIN 缓存是否命中、Python 进程总用时、
   流水线用时和映射状态；NX Listing Window 同时保留每个实例的 OCC/AAG 面编号、
   NX Face Tag、颜色和置信度。
6. JSON 1.1 且完整映射成功时，识别到的特征面按实例着色，stock 面保持原色。以
   41 面黄金 STEP 为例，成功摘要应显示 `41/41` 已映射。着色创建名为
   `BrepMFR feature colors` 的可见撤销标记，但不自动保存 Part；需要保留颜色时
   由用户手动保存，需要撤销时使用 NX Undo。

NX2212 打开中性 STEP 后，通常会把当前 Work Part 表示为同目录下自动生成的
`<名称>_step.prt`，并生成对应的 `<名称>.log`。插件会校验日志中的 `Converting Part` 与
`Filing Part` 配对，唯一还原原始 `.step/.stp` 供 Python 推理；取面、Face Tag
匹配和着色仍针对当前 NX Work Part。日志缺失、目标 PRT 不匹配或来源不唯一时会
终止识别，不会猜测同名 STEP，也不会自动导出新 STEP。

JSON 1.0 仍可显示原有文字结果，但不会读取 NX 几何或着色。JSON 1.1 的签名
不可用、非法、匹配歧义或着色异常时，识别文字结果仍然保留，摘要明确显示映射失败；
任何映射失败都实行零着色，不留下部分修改，也不会被误报为 Python 推理失败。

同一个 STEP 路径会使用稳定的独立输出目录：

```text
output_root/<STEP stem>-<path hash>/
  input.bin
  input.bin.meta.json
  input.face_signatures.json
  result.json
  run.log
```

第一次识别通常生成 BIN，摘要显示缓存未命中。第二次识别同一路径时，只要 STEP
内容、转换指纹、采样参数和 checkpoint 几何需求仍匹配，就会复用 BIN，并显示
缓存命中。每次运行前只删除旧 `result.json`，不会删除有效的 BIN 缓存。

插件用配置中的 Python 环境启动独立子进程，并显式使用仓库根目录作为工作目录。
启动时会清除 NX 注入的 `UGII_*`、`UGS_*`、`PYTHONHOME` 和 `PYTHONPATH`，从
`PATH` 中移除 NX 安装目录，再把 Conda 环境自身的目录放到最前面。这可以避免
NX 自带 DLL 或 Python 设置污染 OCC/深度学习依赖；不会把 Python 嵌入 NX 进程。

## 故障排查

- 报错包含 `BrepMFRNx2212.config.json`：检查真实配置是否与 DLL 同目录，以及
  所有路径、设备名和阈值是否合法。
- 报错指出 checkpoint 不存在或后缀错误：修正配置后重新加载 DLL。
- Python 非零退出、JSON 缺失或损坏：先查看错误框给出的 `run.log` 和
  `result.json` 路径，再在 PowerShell 中直接运行同一 CLI 命令。
- CUDA 环境异常：先把 `device` 改为 `cpu` 验证流程，或修复 CUDA/PyTorch
  环境后继续使用 `auto`/`cuda`。
- NX 首次 STEP→BIN 明显慢于相同终端命令：查看 `run.log` 中最后出现的
  `BREPMFR_PROGRESS:<stage>`，并比较成功摘要里的“Python 进程总用时”和 JSON
  里的流水线用时。前者额外包含 Python 冷启动和模块导入。
- 重复识别未命中缓存：查看 `input.bin.meta.json`；STEP 内容、转换实现、采样
  配置或 checkpoint 几何需求任一改变都应触发重建。
- 当前 Work Part 校验失败：确认 NX 中已直接打开现有 `.step`/`.stp`，原始 STEP、
  NX 自动生成的 `*_step.prt` 和转换 `.log` 仍位于原目录，且模型仍是未修改的单个
  精确 solid body。不要在 Python 推理期间切换 Part、重命名转换文件或改变拓扑。
- 映射失败但识别文字正常：查看 Listing Window 的 `Face mapping failed` 诊断。
  常见原因是 JSON 1.0、`input.face_signatures.json` 不可用、NX/OCC 面数不同、几何
  超出容差或存在多个同样合理的拓扑映射；失败路径不会应用任何颜色。
- 面已着色但关闭后颜色消失：这是预期行为，插件不自动保存。确认结果后由用户决定
  是否保存 NX Part；也可以通过 `BrepMFR feature colors` 撤销标记恢复原色。
- Listing Window 同时显示 `OCC faces [...]` 与 `NX Face Tags [...]`：前者属于
  Python/OCC AAG，后者才是当前 NX 会话中的可视化对象标识。

DLL 导出 `ufusr` 和 `ufusr_ask_unload`，入口完成后请求 NX 立即卸载，因此同一
NX 会话可以再次通过 `Ctrl+U` 加载并执行。
