# BrepMFR NX2212 识别插件

该插件把 NX2212 用作 STEP 选择和识别结果展示界面。STEP 转 BIN、缓存校验、
模型加载、推理、后处理和 `result.json` 写入全部由独立 Python 进程完成；DLL
不会把 Python、PyTorch 或 CUDA 加载进 NX 进程。

当前版本展示识别摘要和特征实例列表，但不高亮 NX 面。JSON 中的
`face_index` 是 OCC/AAG 构图时的 0-based 面序号，不是 NX Face Tag；在建立
经过验证的 OCC 面到 NX 面映射之前，不能把该编号直接用于 NX 高亮。

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
生成 `status=success` 的 `result.json` 后，再从 NX 加载 DLL。这样可以把 Python
环境问题与 NX 集成问题分开定位。

## 在 NX2212 中运行

1. 启动 NX2212。
2. 使用 `File -> Execute -> NX Open`，快捷键通常为 `Ctrl+U`。
3. 选择 `build/Release/BrepMFRNx2212.dll`。
4. 在 Unicode 文件对话框中选择现有 `.step` 或 `.stp` 文件；取消选择不会报错。
5. Python 启动后会显示不确定进度条和已用时间。首版不提供强制取消，避免损坏
   正在写入的转换缓存。
6. 成功后，信息框显示面数量、特征实例数、BIN 缓存是否命中和总用时；NX
   Listing Window 显示每个实例及其 OCC/AAG 面编号。

同一个 STEP 路径会使用稳定的独立输出目录：

```text
output_root/<STEP stem>-<path hash>/
  input.bin
  input.bin.meta.json
  result.json
  run.log
```

第一次识别通常生成 BIN，摘要显示缓存未命中。第二次识别同一路径时，只要 STEP
内容、转换指纹、采样参数和 checkpoint 几何需求仍匹配，就会复用 BIN，并显示
缓存命中。每次运行前只删除旧 `result.json`，不会删除有效的 BIN 缓存。

## 故障排查

- 报错包含 `BrepMFRNx2212.config.json`：检查真实配置是否与 DLL 同目录，以及
  所有路径、设备名和阈值是否合法。
- 报错指出 checkpoint 不存在或后缀错误：修正配置后重新加载 DLL。
- Python 非零退出、JSON 缺失或损坏：先查看错误框给出的 `run.log` 和
  `result.json` 路径，再在 PowerShell 中直接运行同一 CLI 命令。
- CUDA 环境异常：先把 `device` 改为 `cpu` 验证流程，或修复 CUDA/PyTorch
  环境后继续使用 `auto`/`cuda`。
- 重复识别未命中缓存：查看 `input.bin.meta.json`；STEP 内容、转换实现、采样
  配置或 checkpoint 几何需求任一改变都应触发重建。
- 面编号不能在 NX 中直接定位：这是当前明确边界。Listing Window 已统一标注
  `OCC/AAG face_index (not NX Face Tag)`，面高亮要等可靠映射完成后再接入。

DLL 导出 `ufusr` 和 `ufusr_ask_unload`，入口完成后请求 NX 立即卸载，因此同一
NX 会话可以再次通过 `Ctrl+U` 加载并执行。
