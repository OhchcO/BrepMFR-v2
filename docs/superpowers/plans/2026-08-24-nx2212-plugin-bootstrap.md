# NX2212 C++ 插件最小工程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `integrations/nx2212/` 建立一个全新的 NX2212 C++ DLL 工程，使其能够用 v142/x64 构建、导出 NX 用户函数入口，并在 NX2212 手动加载后显示成功消息。

**Architecture:** NX 入口只负责把 NXOpen 消息框适配为 `LoadNotifier`，可测试的 `RunLoadConfirmation` 负责成功消息、返回码和异常边界。原生 C++ 测试直接执行生产运行时逻辑；pytest 通过真实 MSBuild 和 `dumpbin` 验证 DLL 构建与导出；NX 主进程内的加载由用户最终验收。

**Tech Stack:** NX2212 NXOpen C++、MSVC v142、x64、Windows 10 SDK、Visual Studio/MSBuild、原生 C++ 测试程序、pytest 构建集成测试。

**Spec:** `docs/superpowers/specs/2026-08-24-nx2212-plugin-bootstrap-design.md`

## Global Constraints

- 新实现不得引用或复制 `MFR_Recognition_Project/` 中的源代码。
- 工程只提供 `Release|x64`，并固定使用 `PlatformToolset=v142`。
- 工程使用 `WindowsTargetPlatformVersion=10.0`，由 MSBuild 选择已安装 SDK。
- NX include 和 library 路径必须使用 `$(UGII_BASE_DIR)\UGOPEN`，不得写入本机 NX 绝对路径。
- DLL 只链接第一阶段所需的 `libnxopencpp.lib` 和 `libnxopenuicpp.lib`。
- 第一阶段不得加入 DLX、STEP 导出、Python 调用、JSON、临时文件或面高亮代码。
- `RunLoadConfirmation(LoadNotifier&) noexcept` 必须覆盖成功、标准异常和未知异常。
- C ABI 入口必须捕获异常，异常不得越过 DLL 边界。
- 人工说明文档不做源码文本断言；构建产物和运行时行为必须通过真实执行验证。
- Python 测试解释器：`F:\ProgramData\anaconda3\envs\aagmspenv\python.exe`。
- MSBuild：`F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe`。
- Codex 沙箱可能阻止 MSBuild `FileTracker`；遇到该错误时使用经过批准的沙箱外执行，不修改构建语义。

## File Structure

- Create: `integrations/nx2212/BrepMFRNx2212.sln` — Release/x64 解决方案。
- Create: `integrations/nx2212/BrepMFRNx2212.vcxproj` — NX2212 DLL 工程。
- Create: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj` — 无 NX 运行时依赖的 C++ 行为测试工程。
- Create: `integrations/nx2212/include/plugin_exports.hpp` — NX C ABI 导出声明。
- Create: `integrations/nx2212/include/plugin_runtime.hpp` — `LoadNotifier` 和 `RunLoadConfirmation` 契约。
- Create: `integrations/nx2212/src/plugin_entry.cpp` — NXOpen 消息框适配和 DLL 入口。
- Create: `integrations/nx2212/src/plugin_runtime.cpp` — 可测试的加载确认逻辑。
- Create: `integrations/nx2212/tests/plugin_runtime_test.cpp` — 成功与异常路径的原生行为测试。
- Create: `integrations/nx2212/README.md` — 构建与 NX 手动加载说明。
- Create: `tests/test_nx2212_project.py` — 真实构建、DLL 生成和导出契约测试。

---

### Task 1: 建立可构建且正确导出的 NX2212 DLL 骨架

**Files:**
- Create: `tests/test_nx2212_project.py`
- Create: `integrations/nx2212/BrepMFRNx2212.sln`
- Create: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Create: `integrations/nx2212/include/plugin_exports.hpp`
- Create: `integrations/nx2212/src/plugin_entry.cpp`

**Interfaces:**
- Consumes: `UGII_BASE_DIR`、Visual Studio Installer 元数据、NX2212 头文件和导入库。
- Produces: `BrepMFRNx2212.dll`，并导出未修饰的 `ufusr` 与 `ufusr_ask_unload`。

- [ ] **Step 1: 写入失败的真实构建与导出测试**

创建 `tests/test_nx2212_project.py`：

```python
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NX_ROOT = REPOSITORY_ROOT / "integrations" / "nx2212"
PROJECT_PATH = NX_ROOT / "BrepMFRNx2212.vcxproj"
DLL_PATH = NX_ROOT / "build" / "Release" / "BrepMFRNx2212.dll"
MSBUILD = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}


def _project_values(root, name):
    return [
        element.text or ""
        for element in root.findall(".//m:{0}".format(name), MSBUILD)
    ]


def _windows_build_tools():
    if sys.platform != "win32":
        pytest.skip("NX2212 build verification requires Windows")

    nx_dir = os.environ.get("UGII_BASE_DIR")
    if not nx_dir:
        pytest.skip("UGII_BASE_DIR is not configured")
    assert (Path(nx_dir) / "UGOPEN" / "NXOpen" / "UI.hxx").is_file()
    assert (Path(nx_dir) / "UGOPEN" / "libnxopencpp.lib").is_file()

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    assert program_files_x86
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    assert vswhere.is_file()
    discovered = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert discovered
    visual_studio = Path(discovered.splitlines()[-1])

    msbuild = visual_studio / "MSBuild" / "Current" / "Bin" / "amd64" / "MSBuild.exe"
    v142_version_file = (
        visual_studio
        / "VC"
        / "Auxiliary"
        / "Build"
        / "Microsoft.VCToolsVersion.v142.default.txt"
    )
    assert msbuild.is_file()
    assert v142_version_file.is_file()
    v142_version = v142_version_file.read_text(encoding="utf-8-sig").strip()
    dumpbin = (
        visual_studio
        / "VC"
        / "Tools"
        / "MSVC"
        / v142_version
        / "bin"
        / "HostX64"
        / "x64"
        / "dumpbin.exe"
    )
    assert dumpbin.is_file()
    return msbuild, dumpbin


def test_nx2212_project_declares_compatible_portable_contract():
    assert PROJECT_PATH.is_file()
    root = ET.parse(str(PROJECT_PATH)).getroot()
    configurations = {
        node.attrib["Include"]
        for node in root.findall("m:ItemGroup/m:ProjectConfiguration", MSBUILD)
    }
    assert configurations == {"Release|x64"}
    assert _project_values(root, "ConfigurationType") == ["DynamicLibrary"]
    assert _project_values(root, "PlatformToolset") == ["v142"]
    assert _project_values(root, "WindowsTargetPlatformVersion") == ["10.0"]
    assert _project_values(root, "RuntimeLibrary") == ["MultiThreadedDLL"]

    normalized = PROJECT_PATH.read_text(encoding="utf-8-sig").replace("/", "\\").casefold()
    assert "$(ugii_base_dir)\\ugopen" in normalized
    assert "libnxopencpp.lib" in normalized
    assert "libnxopenuicpp.lib" in normalized
    assert "libufun.lib" not in normalized
    assert "nx 10.0" not in normalized
    assert "f:\\program files\\siemens" not in normalized


def test_nx2212_plugin_builds_and_exports_user_functions():
    msbuild, dumpbin = _windows_build_tools()
    completed = subprocess.run(
        [
            str(msbuild),
            str(PROJECT_PATH),
            "/t:Rebuild",
            "/p:Configuration=Release",
            "/p:Platform=x64",
            "/m",
            "/verbosity:minimal",
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert DLL_PATH.is_file()
    assert DLL_PATH.stat().st_size > 0

    exports = subprocess.run(
        [str(dumpbin), "/exports", str(DLL_PATH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert re.search(r"\bufusr\b", exports)
    assert re.search(r"\bufusr_ask_unload\b", exports)
```

- [ ] **Step 2: 运行契约测试并确认因工程不存在而失败**

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest `
  tests/test_nx2212_project.py::test_nx2212_project_declares_compatible_portable_contract `
  -q
```

Expected: FAIL，指出 `BrepMFRNx2212.vcxproj` 不存在。

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

- [ ] **Step 4: 创建 Release/x64 工程与解决方案**

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

- [ ] **Step 5: 运行真实构建与导出测试**

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: `2 passed`，并生成非空 DLL；沙箱出现 `FileTracker` 拒绝访问时以批准的沙箱外执行重跑同一命令。

- [ ] **Step 6: 提交 DLL 骨架**

```powershell
git add tests/test_nx2212_project.py integrations/nx2212
git commit -m "build: add NX2212 plugin scaffold"
```

---

### Task 2: 用原生 C++ 测试驱动加载确认运行时和 NX 适配入口

**Files:**
- Create: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`
- Create: `integrations/nx2212/include/plugin_runtime.hpp`
- Create: `integrations/nx2212/src/plugin_runtime.cpp`
- Create: `integrations/nx2212/tests/plugin_runtime_test.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.sln`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `integrations/nx2212/src/plugin_entry.cpp`

**Interfaces:**
- Consumes: Task 1 的 C ABI 入口和 DLL 工程。
- Produces: `LoadNotifier`、`RunLoadConfirmation(LoadNotifier&) noexcept`、真实 NXOpen notifier，以及 `BrepMFRNx2212RuntimeTests.exe`。

- [ ] **Step 1: 先创建原生行为测试和测试工程**

创建 `integrations/nx2212/tests/plugin_runtime_test.cpp`：

```cpp
#include "plugin_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
enum class FailureMode
{
    None,
    Standard,
    Unknown
};

class RecordingNotifier final : public brepmfr::nx2212::LoadNotifier
{
public:
    explicit RecordingNotifier(FailureMode mode) : mode_(mode) {}

    void ShowInformation(const char* title, const char* message) override
    {
        information_title = title;
        information_message = message;
        ++information_calls;
        if (mode_ == FailureMode::Standard)
        {
            throw std::runtime_error("synthetic standard failure");
        }
        if (mode_ == FailureMode::Unknown)
        {
            throw 42;
        }
    }

    void ShowError(const char* title, const char* message) noexcept override
    {
        error_title = title;
        error_message = message;
        ++error_calls;
    }

    FailureMode mode_;
    int information_calls = 0;
    int error_calls = 0;
    std::string information_title;
    std::string information_message;
    std::string error_title;
    std::string error_message;
};

void Expect(bool condition, const char* message, int& failures)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        ++failures;
    }
}
}

int main()
{
    int failures = 0;

    RecordingNotifier success(FailureMode::None);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(success) == brepmfr::nx2212::kSuccess,
        "success path returned failure",
        failures);
    Expect(success.information_calls == 1, "success message count was not one", failures);
    Expect(success.error_calls == 0, "success path displayed an error", failures);
    Expect(success.information_title == "BrepMFR NX2212", "success title changed", failures);
    Expect(
        success.information_message == "BrepMFR NX2212 plug-in loaded successfully.",
        "success message changed",
        failures);

    RecordingNotifier standard_failure(FailureMode::Standard);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(standard_failure) == brepmfr::nx2212::kFailure,
        "standard exception returned success",
        failures);
    Expect(standard_failure.error_calls == 1, "standard exception was not reported", failures);
    Expect(
        standard_failure.error_message == "synthetic standard failure",
        "standard exception message was not preserved",
        failures);

    RecordingNotifier unknown_failure(FailureMode::Unknown);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(unknown_failure) == brepmfr::nx2212::kFailure,
        "unknown exception returned success",
        failures);
    Expect(unknown_failure.error_calls == 1, "unknown exception was not reported", failures);
    Expect(
        unknown_failure.error_message ==
            "BrepMFR NX2212 plug-in failed with an unknown error.",
        "unknown exception message changed",
        failures);

    return failures == 0 ? 0 : 1;
}
```

创建 `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`：

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
    <ProjectGuid>{F078766D-F325-40F6-86B8-EAEF6B637457}</ProjectGuid>
    <RootNamespace>BrepMFRNx2212RuntimeTests</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v142</PlatformToolset>
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
    <OutDir>$(ProjectDir)build\Tests\</OutDir>
    <IntDir>$(ProjectDir)build\Tests\obj\</IntDir>
    <TargetName>BrepMFRNx2212RuntimeTests</TargetName>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level4</WarningLevel>
      <Optimization>Disabled</Optimization>
      <SDLCheck>true</SDLCheck>
      <ConformanceMode>false</ConformanceMode>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <AdditionalIncludeDirectories>$(ProjectDir)include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="src\plugin_runtime.cpp" />
    <ClCompile Include="tests\plugin_runtime_test.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="include\plugin_runtime.hpp" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets" />
</Project>
```

- [ ] **Step 2: 构建原生测试并确认因生产运行时不存在而失败**

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal
```

Expected: FAIL，编译器报告缺少 `plugin_runtime.hpp` 或 `plugin_runtime.cpp`。

- [ ] **Step 3: 实现最小可测试运行时**

创建 `integrations/nx2212/include/plugin_runtime.hpp`：

```cpp
#pragma once

namespace brepmfr
{
namespace nx2212
{
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

class LoadNotifier
{
public:
    virtual ~LoadNotifier() = default;
    virtual void ShowInformation(const char* title, const char* message) = 0;
    virtual void ShowError(const char* title, const char* message) noexcept = 0;
};

int RunLoadConfirmation(LoadNotifier& notifier) noexcept;
}
}
```

创建 `integrations/nx2212/src/plugin_runtime.cpp`：

```cpp
#include "plugin_runtime.hpp"

#include <exception>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr char kDialogTitle[] = "BrepMFR NX2212";
constexpr char kSuccessMessage[] =
    "BrepMFR NX2212 plug-in loaded successfully.";
constexpr char kUnknownErrorMessage[] =
    "BrepMFR NX2212 plug-in failed with an unknown error.";
}

int RunLoadConfirmation(LoadNotifier& notifier) noexcept
{
    try
    {
        notifier.ShowInformation(kDialogTitle, kSuccessMessage);
        return kSuccess;
    }
    catch (const std::exception& exception)
    {
        notifier.ShowError(kDialogTitle, exception.what());
    }
    catch (...)
    {
        notifier.ShowError(kDialogTitle, kUnknownErrorMessage);
    }
    return kFailure;
}
}
}
```

- [ ] **Step 4: 构建并运行原生行为测试**

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal

& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'
```

Expected: MSBuild 和测试程序的 exit code 都为 `0`。

- [ ] **Step 5: 将生产入口适配到 NXOpen 消息框**

用以下内容替换 `integrations/nx2212/src/plugin_entry.cpp`：

```cpp
#include "plugin_exports.hpp"
#include "plugin_runtime.hpp"

#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/UI.hxx>
#include <uf_defs.h>

#include <stdexcept>

namespace
{
class NxLoadNotifier final : public brepmfr::nx2212::LoadNotifier
{
public:
    void ShowInformation(const char* title, const char* message) override
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui == nullptr || ui->NXMessageBox() == nullptr)
        {
            throw std::runtime_error("NX UI is unavailable.");
        }
        ui->NXMessageBox()->Show(
            title,
            NXOpen::NXMessageBox::DialogTypeInformation,
            message);
    }

    void ShowError(const char* title, const char* message) noexcept override
    {
        try
        {
            NXOpen::UI* ui = NXOpen::UI::GetUI();
            if (ui != nullptr && ui->NXMessageBox() != nullptr)
            {
                ui->NXMessageBox()->Show(
                    title,
                    NXOpen::NXMessageBox::DialogTypeError,
                    message);
            }
        }
        catch (...)
        {
        }
    }
};

void SetReturnCode(int* retcod, int value) noexcept
{
    if (retcod != nullptr)
    {
        *retcod = value;
    }
}
}

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    SetReturnCode(retcod, brepmfr::nx2212::kFailure);

    try
    {
        NxLoadNotifier notifier;
        SetReturnCode(retcod, brepmfr::nx2212::RunLoadConfirmation(notifier));
    }
    catch (...)
    {
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
```

在 `BrepMFRNx2212.vcxproj` 中把 `src\plugin_runtime.cpp` 和
`include\plugin_runtime.hpp` 分别加入 `ClCompile` 与 `ClInclude`。

在 `BrepMFRNx2212.sln` 中加入测试工程 GUID
`{F078766D-F325-40F6-86B8-EAEF6B637457}` 及其 Release/x64 映射。

- [ ] **Step 6: 运行运行时测试与 DLL 集成测试**

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal
& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'

& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_nx2212_project.py -q
```

Expected: 原生测试程序 exit code 为 `0`，pytest 为 `2 passed`。

- [ ] **Step 7: 提交加载确认运行时和 NX 适配**

```powershell
git add integrations/nx2212
git commit -m "feat: add tested NX2212 load confirmation"
```

---

### Task 3: 添加构建说明并运行完整回归

**Files:**
- Create: `integrations/nx2212/README.md`

**Interfaces:**
- Consumes: Task 2 生成的 DLL 与原生测试程序。
- Produces: 可复现构建、测试及 NX2212 手动加载步骤。

- [ ] **Step 1: 编写 README**

创建 `integrations/nx2212/README.md`，准确记录：

```markdown
# BrepMFR NX2212 plug-in bootstrap

This directory verifies the NX2212 C++ build and load boundary. The plug-in
does not invoke Python or modify the active part.

## Requirements

- Siemens NX 2212 with `UGII_BASE_DIR` set to the NX installation directory.
- MSVC `PlatformToolset=v142` with x64 C++ tools.
- A Windows 10 SDK.

## Build

From the repository root, locate MSBuild with Visual Studio Installer's
`vswhere.exe`, then build:

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.vcxproj `
    /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
```

Output:

```text
integrations/nx2212/build/Release/BrepMFRNx2212.dll
```

## Native runtime test

```powershell
& $msbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
    /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'
```

Both commands must return exit code `0`.

## Manual NX2212 acceptance

1. Start NX2212 and open or create a part.
2. Use `File -> Execute -> NX Open`, commonly available as `Ctrl+U`.
3. Select `build/Release/BrepMFRNx2212.dll`.
4. Confirm that NX displays:

```text
BrepMFR NX2212 plug-in loaded successfully.
```

The exported functions are `ufusr` and `ufusr_ask_unload`. The latter requests
immediate unload after the entry point returns.
```

- [ ] **Step 2: 运行完整自动验证**

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj `
  /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m /verbosity:minimal
& 'integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe'

& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q
git diff --check
```

Expected: 原生测试 exit code 为 `0`；Python 回归为 `495 passed, 1 skipped`；`git diff --check` 无输出。

- [ ] **Step 3: 提交构建和验收文档**

```powershell
git add integrations/nx2212/README.md
git commit -m "docs: add NX2212 build and load guide"
```

---

### Task 4: 交付 NX2212 手动加载验收

**Files:**
- Verify: `integrations/nx2212/build/Release/BrepMFRNx2212.dll`
- Verify: `integrations/nx2212/README.md`

**Interfaces:**
- Consumes: Task 3 的最终 DLL。
- Produces: 用户在 NX2212 主进程中的加载验收结果。

- [ ] **Step 1: 确认分支状态与 DLL 信息**

```powershell
git status --short --branch
Get-Item integrations\nx2212\build\Release\BrepMFRNx2212.dll |
    Select-Object FullName, Length, LastWriteTime
```

Expected: 无未提交源代码修改，DLL 存在且长度大于零。

- [ ] **Step 2: 向用户交付 DLL**

DLL 绝对路径：

```text
D:\Code\BrepMFR\.worktrees\nx2212-integration\integrations\nx2212\build\Release\BrepMFRNx2212.dll
```

要求用户按 README 的 “Manual NX2212 acceptance” 步骤加载。

- [ ] **Step 3: 记录用户验收结果**

成功标准：NX2212 显示 `BrepMFR NX2212 plug-in loaded successfully.`，NX 不崩溃，入口返回后 DLL 可以再次加载。

如果 NX 拒绝加载，保留完整错误文本并进入系统化调试；在取得证据前不开始 CLI 接入。
