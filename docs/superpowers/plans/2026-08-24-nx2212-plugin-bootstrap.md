# NX2212 C++ 插件最小工程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `integrations/nx2212/` 建立一个全新的 NX2212 C++ DLL 工程，使其能够用 v142/x64 构建、导出 NX 用户函数入口，并在 NX2212 手动加载后显示成功消息。

**Architecture:** 新工程是 Python 推理系统之外的薄 NX 适配层，本阶段只证明 DLL 构建与 NX 进程加载边界。项目设置来源于本机 NX2212 官方模板，使用 `$(UGII_BASE_DIR)` 定位 NXOpen 头文件和库；自动测试校验工程契约和入口源码，真实 NX 加载由用户做最终验收。

**Tech Stack:** NX2212 NXOpen C++、UF 用户函数入口、MSVC v142、x64、Windows 10 SDK、Visual Studio/MSBuild、pytest 静态契约测试。

**Spec:** `docs/superpowers/specs/2026-08-24-nx2212-plugin-bootstrap-design.md`

## Global Constraints

- 新实现不得引用或复制 `MFR_Recognition_Project/` 中的源代码。
- 工程只提供 `Release|x64`，并固定使用 `PlatformToolset=v142`。
- 工程使用 `WindowsTargetPlatformVersion=10.0`，由 MSBuild 选择已安装 SDK。
- NX include 和 library 路径必须使用 `$(UGII_BASE_DIR)\UGOPEN`，不得写入本机 NX 绝对路径。
- DLL 只链接第一阶段所需的 `libnxopencpp.lib` 和 `libnxopenuicpp.lib`。
- 第一阶段不得加入 DLX、STEP 导出、Python、JSON、临时文件或面高亮代码。
- 所有 C ABI 入口必须捕获异常，异常不得越过 DLL 边界。
- 当前 Python 测试解释器为 `F:\ProgramData\anaconda3\envs\aagmspenv\python.exe`。
- 当前 MSBuild 为 `F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe`。
- MSBuild 在 Codex 沙箱内可能被 `FileTracker` 拒绝访问；遇到该错误时应使用经过批准的沙箱外构建，不得通过关闭依赖跟踪掩盖问题。

## File Structure

- Create: `integrations/nx2212/BrepMFRNx2212.sln` — Visual Studio 解决方案，仅暴露 Release/x64。
- Create: `integrations/nx2212/BrepMFRNx2212.vcxproj` — NX2212 v142 DLL 构建契约。
- Create: `integrations/nx2212/include/plugin_exports.hpp` — 两个 NX 用户函数的 C ABI 导出声明。
- Create: `integrations/nx2212/src/plugin_entry.cpp` — DLL 入口、成功消息和异常边界。
- Create: `integrations/nx2212/README.md` — 可复现构建和 NX2212 手动加载说明。
- Create: `tests/test_nx2212_project.py` — 工程、入口和文档的静态契约测试。

---

### Task 1: 建立可构建且正确导出的 NX2212 DLL 骨架

**Files:**
- Create: `tests/test_nx2212_project.py`
- Create: `integrations/nx2212/BrepMFRNx2212.sln`
- Create: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Create: `integrations/nx2212/include/plugin_exports.hpp`
- Create: `integrations/nx2212/src/plugin_entry.cpp`

**Interfaces:**
- Consumes: `UGII_BASE_DIR` 环境变量和 NX2212 `UGOPEN` 目录。
- Produces: `ufusr(char*, int*, int)`、`ufusr_ask_unload()`，以及 `integrations/nx2212/build/Release/BrepMFRNx2212.dll`。

- [ ] **Step 1: 写入失败的工程契约测试**

创建 `tests/test_nx2212_project.py`：

```python
from pathlib import Path
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NX_ROOT = REPOSITORY_ROOT / "integrations" / "nx2212"
SOLUTION_PATH = NX_ROOT / "BrepMFRNx2212.sln"
PROJECT_PATH = NX_ROOT / "BrepMFRNx2212.vcxproj"
EXPORT_HEADER_PATH = NX_ROOT / "include" / "plugin_exports.hpp"
ENTRY_SOURCE_PATH = NX_ROOT / "src" / "plugin_entry.cpp"
MSBUILD = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}


def _project_root():
    return ET.parse(str(PROJECT_PATH)).getroot()


def _project_values(root, name):
    return [
        element.text or ""
        for element in root.findall(".//m:{0}".format(name), MSBUILD)
    ]


def test_nx2212_solution_and_source_layout_exists():
    assert SOLUTION_PATH.is_file()
    assert PROJECT_PATH.is_file()
    assert EXPORT_HEADER_PATH.is_file()
    assert ENTRY_SOURCE_PATH.is_file()

    solution = SOLUTION_PATH.read_text(encoding="utf-8-sig")
    assert '"BrepMFRNx2212", "BrepMFRNx2212.vcxproj"' in solution
    assert "Release|x64" in solution


def test_nx2212_project_uses_portable_release_x64_contract():
    root = _project_root()
    configurations = {
        node.attrib["Include"]
        for node in root.findall("m:ItemGroup/m:ProjectConfiguration", MSBUILD)
    }
    assert configurations == {"Release|x64"}
    assert _project_values(root, "ConfigurationType") == ["DynamicLibrary"]
    assert _project_values(root, "PlatformToolset") == ["v142"]
    assert _project_values(root, "WindowsTargetPlatformVersion") == ["10.0"]
    assert _project_values(root, "RuntimeLibrary") == ["MultiThreadedDLL"]

    project_text = PROJECT_PATH.read_text(encoding="utf-8-sig")
    normalized = project_text.replace("/", "\\").casefold()
    assert "$(ugii_base_dir)\\ugopen" in normalized
    assert "$(projectdir)include" in normalized
    assert "libnxopencpp.lib" in normalized
    assert "libnxopenuicpp.lib" in normalized
    assert "libufun.lib" not in normalized
    assert "nx 10.0" not in normalized
    assert "f:\\program files\\siemens" not in normalized


def test_nx2212_project_includes_only_the_bootstrap_sources():
    root = _project_root()
    compiled = {
        node.attrib["Include"].replace("/", "\\").casefold()
        for node in root.findall("m:ItemGroup/m:ClCompile", MSBUILD)
    }
    included = {
        node.attrib["Include"].replace("/", "\\").casefold()
        for node in root.findall("m:ItemGroup/m:ClInclude", MSBUILD)
    }
    assert compiled == {"src\\plugin_entry.cpp"}
    assert included == {"include\\plugin_exports.hpp"}
```

- [ ] **Step 2: 运行测试并确认因为新工程不存在而失败**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: FAIL，首个失败指出 `BrepMFRNx2212.sln` 不存在。

- [ ] **Step 3: 创建导出声明和无 UI 的最小入口**

创建 `integrations/nx2212/include/plugin_exports.hpp`：

```cpp
#pragma once

#define BREPMFR_NX_EXPORT extern "C" __declspec(dllexport)

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len);
BREPMFR_NX_EXPORT int ufusr_ask_unload();
```

创建 `integrations/nx2212/src/plugin_entry.cpp`：

```cpp
#include "plugin_exports.hpp"

#include <uf_defs.h>

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    if (retcod != nullptr)
    {
        *retcod = 0;
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
```

- [ ] **Step 4: 创建 Release/x64 Visual Studio 工程**

创建 `integrations/nx2212/BrepMFRNx2212.vcxproj`：

```xml
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>16.0</VCProjectVersion>
    <ProjectGuid>{732364C3-3E02-4D46-8D58-D90CF71767F4}</ProjectGuid>
    <RootNamespace>BrepMFRNx2212</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v142</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings" />
  <ImportGroup Label="Shared" />
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props"
            Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')"
            Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <OutDir>$(ProjectDir)build\Release\</OutDir>
    <IntDir>$(ProjectDir)build\Release\obj\</IntDir>
    <TargetName>BrepMFRNx2212</TargetName>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level4</WarningLevel>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <ConformanceMode>false</ConformanceMode>
      <PreprocessorDefinitions>_CRT_SECURE_NO_WARNINGS;_USRDLL;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <AdditionalIncludeDirectories>$(ProjectDir)include;$(UGII_BASE_DIR)\UGOPEN;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <RuntimeTypeInfo>true</RuntimeTypeInfo>
      <AdditionalOptions>/d1scalableinclude- %(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <AdditionalLibraryDirectories>$(UGII_BASE_DIR)\UGOPEN;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <AdditionalDependencies>libnxopencpp.lib;libnxopenuicpp.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="src\plugin_entry.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="include\plugin_exports.hpp" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets" />
</Project>
```

创建 `integrations/nx2212/BrepMFRNx2212.sln`：

```text
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 16
VisualStudioVersion = 16.0.28701.123
MinimumVisualStudioVersion = 10.0.40219.1
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "BrepMFRNx2212", "BrepMFRNx2212.vcxproj", "{732364C3-3E02-4D46-8D58-D90CF71767F4}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Release|x64 = Release|x64
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{732364C3-3E02-4D46-8D58-D90CF71767F4}.Release|x64.ActiveCfg = Release|x64
		{732364C3-3E02-4D46-8D58-D90CF71767F4}.Release|x64.Build.0 = Release|x64
	EndGlobalSection
EndGlobal
```

- [ ] **Step 5: 运行工程契约测试并确认通过**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: `3 passed`。

- [ ] **Step 6: 实际构建 DLL**

Run:

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal
```

Expected: exit code `0`，并生成：

```text
integrations/nx2212/build/Release/BrepMFRNx2212.dll
integrations/nx2212/build/Release/BrepMFRNx2212.lib
integrations/nx2212/build/Release/BrepMFRNx2212.exp
```

- [ ] **Step 7: 检查 DLL 导出表**

Run:

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\dumpbin.exe' `
  /exports integrations\nx2212\build\Release\BrepMFRNx2212.dll
```

Expected: 输出同时包含未修饰名称 `ufusr` 和 `ufusr_ask_unload`。

- [ ] **Step 8: 提交 DLL 骨架**

```powershell
git add tests/test_nx2212_project.py integrations/nx2212
git commit -m "build: add NX2212 plugin scaffold"
```

---

### Task 2: 在 NX 入口显示成功消息并封闭异常

**Files:**
- Modify: `tests/test_nx2212_project.py`
- Modify: `integrations/nx2212/src/plugin_entry.cpp`

**Interfaces:**
- Consumes: Task 1 导出的 `ufusr` 和 `ufusr_ask_unload`。
- Produces: NX 内可见的加载成功消息、成功/失败 `retcod`，以及不会越过 DLL 边界的异常处理。

- [ ] **Step 1: 添加失败的入口行为契约测试**

把下面测试追加到 `tests/test_nx2212_project.py`：

```python
def test_nx2212_entrypoint_has_visible_success_and_exception_boundary():
    source = ENTRY_SOURCE_PATH.read_text(encoding="utf-8")

    assert "#include <NXOpen/UI.hxx>" in source
    assert "#include <NXOpen/NXMessageBox.hxx>" in source
    assert "#include <NXOpen/NXException.hxx>" in source
    assert "BrepMFR NX2212 plug-in loaded successfully." in source
    assert "NXOpen::NXMessageBox::DialogTypeInformation" in source
    assert "NXOpen::NXMessageBox::DialogTypeError" in source
    assert "SetReturnCode(retcod, kFailure);" in source
    assert "SetReturnCode(retcod, kSuccess);" in source
    assert "catch (const NXOpen::NXException& exception)" in source
    assert "catch (const std::exception& exception)" in source
    assert "catch (...)" in source
    assert source.index("catch (const NXOpen::NXException& exception)") < source.index(
        "catch (const std::exception& exception)"
    )
    for forbidden in (
        "system(",
        "CreateProcess",
        "StepCreator",
        ".json",
        ".dlx",
        "UF_DISP_set_highlight",
    ):
        assert forbidden not in source
```

- [ ] **Step 2: 运行新增测试并确认因为入口尚未使用 NX UI 而失败**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest `
  tests/test_nx2212_project.py::test_nx2212_entrypoint_has_visible_success_and_exception_boundary `
  -q
```

Expected: FAIL，指出缺少 `#include <NXOpen/UI.hxx>`。

- [ ] **Step 3: 实现 NX 消息和异常边界**

用以下内容替换 `integrations/nx2212/src/plugin_entry.cpp`：

```cpp
#include "plugin_exports.hpp"

#include <NXOpen/NXException.hxx>
#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/UI.hxx>
#include <uf_defs.h>

#include <exception>

namespace
{
constexpr int kSuccess = 0;
constexpr int kFailure = 1;
constexpr char kDialogTitle[] = "BrepMFR NX2212";
constexpr char kSuccessMessage[] =
    "BrepMFR NX2212 plug-in loaded successfully.";
constexpr char kUnknownErrorMessage[] =
    "BrepMFR NX2212 plug-in failed with an unknown error.";

void SetReturnCode(int* retcod, int value) noexcept
{
    if (retcod != nullptr)
    {
        *retcod = value;
    }
}

void TryShowError(const char* message) noexcept
{
    try
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui != nullptr && ui->NXMessageBox() != nullptr)
        {
            ui->NXMessageBox()->Show(
                kDialogTitle,
                NXOpen::NXMessageBox::DialogTypeError,
                message);
        }
    }
    catch (...)
    {
    }
}
}

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    SetReturnCode(retcod, kFailure);

    try
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui == nullptr || ui->NXMessageBox() == nullptr)
        {
            return;
        }

        ui->NXMessageBox()->Show(
            kDialogTitle,
            NXOpen::NXMessageBox::DialogTypeInformation,
            kSuccessMessage);
        SetReturnCode(retcod, kSuccess);
    }
    catch (const NXOpen::NXException& exception)
    {
        TryShowError(exception.what());
    }
    catch (const std::exception& exception)
    {
        TryShowError(exception.what());
    }
    catch (...)
    {
        TryShowError(kUnknownErrorMessage);
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
```

- [ ] **Step 4: 运行入口和工程契约测试**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: `4 passed`。

- [ ] **Step 5: 重建并重新检查导出表**

Run:

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal

& 'F:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\dumpbin.exe' `
  /exports integrations\nx2212\build\Release\BrepMFRNx2212.dll
```

Expected: 编译链接成功，导出表仍包含 `ufusr` 和 `ufusr_ask_unload`。

- [ ] **Step 6: 提交可见加载行为**

```powershell
git add tests/test_nx2212_project.py integrations/nx2212/src/plugin_entry.cpp
git commit -m "feat: add NX2212 load confirmation"
```

---

### Task 3: 添加可复现的构建与手动验收文档

**Files:**
- Modify: `tests/test_nx2212_project.py`
- Create: `integrations/nx2212/README.md`

**Interfaces:**
- Consumes: Task 2 生成的 `BrepMFRNx2212.dll`。
- Produces: 不依赖固定 Visual Studio 安装目录的构建命令，以及 NX2212 内的手动加载步骤。

- [ ] **Step 1: 添加失败的 README 契约测试**

把下面常量和测试追加到 `tests/test_nx2212_project.py`：

```python
README_PATH = NX_ROOT / "README.md"


def test_nx2212_readme_documents_build_output_and_manual_acceptance():
    assert README_PATH.is_file()
    readme = README_PATH.read_text(encoding="utf-8")

    required = [
        "UGII_BASE_DIR",
        "PlatformToolset=v142",
        "Configuration=Release",
        "Platform=x64",
        "build/Release/BrepMFRNx2212.dll",
        "ufusr",
        "NX Open",
        "BrepMFR NX2212 plug-in loaded successfully.",
    ]
    for text in required:
        assert text in readme

    assert "MFR_Recognition_Project" not in readme
    assert "recognize2.py" not in readme
```

- [ ] **Step 2: 运行 README 测试并确认失败**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest `
  tests/test_nx2212_project.py::test_nx2212_readme_documents_build_output_and_manual_acceptance `
  -q
```

Expected: FAIL，指出 `integrations/nx2212/README.md` 不存在。

- [ ] **Step 3: 编写构建与手动验收 README**

创建 `integrations/nx2212/README.md`：

````markdown
# BrepMFR NX2212 plug-in bootstrap

This directory contains the minimal NXOpen C++ plug-in used to verify the
NX2212 build and load boundary. It does not invoke Python or modify the active
part.

## Requirements

- Siemens NX 2212 with `UGII_BASE_DIR` set to the NX installation directory.
- MSVC `PlatformToolset=v142` with x64 C++ tools.
- A Windows 10 SDK.

Verify the NX environment in PowerShell:

```powershell
if (-not $env:UGII_BASE_DIR) {
    throw 'UGII_BASE_DIR is not set.'
}
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\NXOpen\UI.hxx')
Test-Path (Join-Path $env:UGII_BASE_DIR 'UGOPEN\libnxopencpp.lib')
```

Both checks must return `True`.

## Build

Locate MSBuild through Visual Studio Installer metadata and build Release/x64:

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

The DLL is written to:

```text
integrations/nx2212/build/Release/BrepMFRNx2212.dll
```

## Manual NX2212 acceptance

1. Start NX2212 and open or create a part.
2. Use the NX Open application execution command, commonly available through
   `File -> Execute -> NX Open` or `Ctrl+U`.
3. Select `build/Release/BrepMFRNx2212.dll`.
4. Confirm that NX displays:

```text
BrepMFR NX2212 plug-in loaded successfully.
```

The exported entry point is `ufusr`; the DLL requests immediate unload through
`ufusr_ask_unload` after the entry point returns.

If NX reports an incompatible image, confirm that the DLL was built with
`Configuration=Release`, `Platform=x64`, and `PlatformToolset=v142`.
````

- [ ] **Step 4: 运行全部 NX 工程契约测试**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: `5 passed`。

- [ ] **Step 5: 运行完整仓库测试和最终构建检查**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q

& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal

git diff --check
```

Expected: Python 基线仍为 `493 passed, 1 skipped`，MSBuild exit code 为 `0`，`git diff --check` 无输出。

- [ ] **Step 6: 提交构建和验收文档**

```powershell
git add tests/test_nx2212_project.py integrations/nx2212/README.md
git commit -m "docs: add NX2212 build and load guide"
```

---

### Task 4: 交付 NX2212 手动加载验收

**Files:**
- Verify: `integrations/nx2212/build/Release/BrepMFRNx2212.dll`
- Verify: `integrations/nx2212/README.md`

**Interfaces:**
- Consumes: Task 3 最终构建的 DLL。
- Produces: 用户对 NX2212 实际加载行为的验收结果。

- [ ] **Step 1: 再次确认分支状态和 DLL 信息**

Run:

```powershell
git status --short --branch
Get-Item integrations\nx2212\build\Release\BrepMFRNx2212.dll |
    Select-Object FullName, Length, LastWriteTime
```

Expected: 分支无未提交源代码修改，DLL 存在且长度大于零。

- [ ] **Step 2: 向用户提供 DLL 绝对路径和 README 链接**

报告 DLL：

```text
D:\Code\BrepMFR\.worktrees\nx2212-integration\integrations\nx2212\build\Release\BrepMFRNx2212.dll
```

并要求用户按 README 的 “Manual NX2212 acceptance” 步骤在 NX2212 中加载。

- [ ] **Step 3: 记录用户验收结果**

成功标准：NX2212 显示 `BrepMFR NX2212 plug-in loaded successfully.`，NX 不崩溃，入口返回后可以再次加载 DLL。

如果 NX 拒绝加载，保留完整错误文本并进入系统化调试；在取得错误证据前不开始 CLI 接入阶段。
