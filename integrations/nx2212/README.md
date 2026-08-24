# BrepMFR NX2212 插件骨架

该目录用于验证 NX2212 C++ 插件的构建和加载边界。当前 DLL 只显示一条加载成功
消息，不调用 Python，也不读取或修改当前零件。

## 环境要求

- Siemens NX 2212，并设置 `UGII_BASE_DIR` 环境变量。
- MSVC `PlatformToolset=v142` 和 x64 C++ 工具。
- Windows 10 SDK。

在仓库根目录打开 PowerShell，先检查 NXOpen 开发文件：

```powershell
if (-not $env:UGII_BASE_DIR) {
    throw 'UGII_BASE_DIR is not set.'
}
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\NXOpen\UI.hxx')
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\libnxopencpp.lib')
```

两项检查都必须返回 `True`。

## 构建 DLL

以下命令通过 Visual Studio Installer 的 `vswhere.exe` 定位 MSBuild，因此不依赖
固定的 Visual Studio 安装盘符：

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
& $msbuild integrations\nx2212\BrepMFRNx2212.vcxproj `
    /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
if ($LASTEXITCODE -ne 0) {
    throw "MSBuild failed with exit code $LASTEXITCODE."
}
```

构建结果位于：

```text
integrations/nx2212/build/Release/BrepMFRNx2212.dll
```

## 运行原生行为测试

```powershell
& $msbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
    /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
if ($LASTEXITCODE -ne 0) {
    throw "Runtime test build failed with exit code $LASTEXITCODE."
}
& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'
if ($LASTEXITCODE -ne 0) {
    throw "Runtime tests failed with exit code $LASTEXITCODE."
}
```

两个命令都必须以退出码 `0` 结束。测试覆盖成功消息、标准异常和未知异常路径。

## 在 NX2212 中手动验收

1. 启动 NX2212，打开或新建一个零件。
2. 使用 `File -> Execute -> NX Open`；常用快捷键为 `Ctrl+U`。
3. 选择 `build/Release/BrepMFRNx2212.dll`。
4. 确认 NX 显示：

```text
BrepMFR NX2212 plug-in loaded successfully.
```

DLL 导出 `ufusr` 和 `ufusr_ask_unload`。入口结束后，后者请求 NX 立即卸载 DLL，
因此同一 NX 会话中应当可以再次加载该 DLL。

如果 NX 报告镜像不兼容，请确认 DLL 使用
`Configuration=Release`、`Platform=x64` 和 `PlatformToolset=v142` 构建。
