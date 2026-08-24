# NX2212 C++ 插件最小工程设计

## 背景

现有 `MFR_Recognition_Project/` 是一个由 NX 10 Block UI Styler 生成、使用
Visual Studio 2012 `v110` 工具集构建的本地原型。它包含硬编码路径、旧的
`Step203Creator` API、旧 `recognize2.py` 调用和 `result.txt` 解析逻辑，不能作为
NX2212 集成的可靠基础。

当前仓库已经具备经过测试的 Python STEP 推理 CLI。NX 集成应成为独立的薄适配
层：NX 负责 CAD 交互和可视化，Python 负责 STEP 转换、BIN 缓存、模型推理和结果
后处理。

本设计只覆盖第一阶段的最小 NX2212 插件工程。CLI、JSON、STEP 导出和面高亮将在
后续阶段逐项接入。

## 已验证的开发环境

- NX 安装目录：`F:\Program Files\Siemens\NX 2212`
- NXOpen/UF 开发文件：`$(UGII_BASE_DIR)\UGOPEN`
- 目标架构：x64
- 编译工具集：MSVC v142，编译器版本 `19.29.30159`
- Windows SDK：`10.0.26100.0`
- MSBuild：Visual Studio 2026 中安装的 MSBuild 18

虽然当前 IDE 不是独立的 Visual Studio 2019，但其中已经安装完整的 v142 x64
工具链。最小 NXOpen DLL 已经使用上述工具链和 NX2212 库实际编译、链接成功。

## 目标

在当前仓库中建立一个全新、受 Git 管理的 NX2212 C++ DLL 工程，并满足：

1. 不引用或复制 `MFR_Recognition_Project/` 的源代码。
2. 使用 NX2212 头文件和导入库编译 Release x64 DLL。
3. 导出 NX 用户函数入口 `ufusr` 和卸载入口 `ufusr_ask_unload`。
4. 用户从 NX2212 手动加载 DLL 后，显示一条明确的成功消息。
5. 工程不包含机器相关的 NX 绝对路径，统一使用 `$(UGII_BASE_DIR)`。
6. 提供可复现的命令行构建和 NX 内手动验收说明。

## 非目标

第一阶段不实现：

- Block UI Styler 对话框或 `.dlx` 文件；
- STEP 导出；
- Python CLI 进程调用；
- `result.json` 读取；
- OCC/AAG 面与 NX Face Tag 的映射；
- 特征列表、着色或高亮；
- 安装器、菜单文件或自动部署；
- Debug 配置下的 NX 进程调试。

把这些能力同时加入第一版会使“工具链、DLL 加载、NX API、CLI、JSON、面映射”多
个边界一起变化，无法快速判断失败来自哪一层。第一阶段只证明 C++ 插件能够被
NX2212 正确加载和执行。

## 方案比较

### 方案 A：继续修改 NX10 原型

优点是表面上已有界面和回调代码。缺点是工程、API、路径和输出契约均已过时，且旧
目录整体被 Git 忽略。迁移时很难区分仍有价值的逻辑和历史试验代码。

结论：不采用。

### 方案 B：基于 NX2212 官方 C++ 模板建立新工程

从 NX2212 自带模板的编译选项和依赖约定出发，只实现最小入口。工程使用 v142、
x64、动态运行库和 `$(UGII_BASE_DIR)\UGOPEN`，然后逐阶段增加集成功能。

优点是版本边界清楚、可编译验证、可以纳入 Git，并且不会把旧原型的问题带入新
实现。

结论：采用。

### 方案 C：改用 NXOpen C#

C# 在 JSON 和进程调用方面更方便，但用户已经确定采用 C++，后续也需要较直接地
使用 UF/NXOpen 面对象与显示接口。更换语言会扩大当前决策范围。

结论：本阶段不采用。

## 工程结构

新增目录：

```text
integrations/
└── nx2212/
    ├── BrepMFRNx2212.sln
    ├── BrepMFRNx2212.vcxproj
    ├── BrepMFRNx2212RuntimeTests.vcxproj
    ├── include/
    │   ├── plugin_exports.hpp
    │   └── plugin_runtime.hpp
    ├── src/
    │   ├── plugin_entry.cpp
    │   └── plugin_runtime.cpp
    ├── tests/
    │   └── plugin_runtime_test.cpp
    └── README.md
```

第一阶段不建立空的 `ui/`、`config/` 或 `third_party/` 目录。后续功能真正需要时再
加入，避免留下没有契约的占位结构。

## 构建设计

工程仅包含 `Release|x64` 配置，关键设置为：

- `ConfigurationType=DynamicLibrary`
- `PlatformToolset=v142`
- `WindowsTargetPlatformVersion=10.0`，由 MSBuild 选择已安装的 Windows 10 SDK
- `RuntimeLibrary=MultiThreadedDLL`
- include：`$(UGII_BASE_DIR)\UGOPEN`
- library：`$(UGII_BASE_DIR)\UGOPEN`
- 最小链接库：`libnxopencpp.lib`、`libnxopenuicpp.lib`

输出目录固定在工程内的 `build\Release\`，并由仓库 `.gitignore` 排除构建产物。
源代码和工程文件不得出现本机 NX 安装绝对路径。

由于 NX2212 自带 VSIX 只声明支持 Visual Studio 2019，当前不依赖向 Visual Studio
2026 安装该向导。工程文件按 NX2212 自带模板的设置建立，并直接选择已安装的
v142 工具集。

## 运行时设计

DLL 导出两个 C ABI 函数：

- `ufusr(char*, int*, int)`：NX 手动执行插件时的入口。
- `ufusr_ask_unload()`：返回立即卸载选项。

`ufusr` 获取 `NXOpen::UI`，通过 NX 消息框显示：

```text
BrepMFR NX2212 plug-in loaded successfully.
```

入口捕获标准异常和未知异常，失败时尝试显示错误消息，不允许异常越过 DLL 边界。
第一阶段不初始化全局状态、不创建线程、不访问 Python，也不读写用户模型。

为了自动验证上述行为，加载确认逻辑位于不依赖 NX API 的
`RunLoadConfirmation(LoadNotifier&) noexcept` 中。`LoadNotifier` 只抽象“显示信息”
和“显示错误”两个动作；NX 入口提供真实 NXOpen 实现，原生 C++ 测试提供记录实现。
测试直接执行同一个生产运行时函数，覆盖成功、标准异常和未知异常，不通过搜索源码
文本推断行为。

## 错误处理

- 无法取得 NX UI 时，入口安全返回，并通过 `retcod` 表示失败。
- NXOpen 异常和标准异常转换为简短的用户消息。
- 成功时将 `retcod` 设为 `0`；失败时设为非零。
- 不使用 `system()`，不产生临时文件，不依赖当前工作目录。

## 验证

### 自动验证

1. 用 MSBuild、`Release|x64` 和 v142 完整重建工程。
2. 确认 DLL、LIB 和 EXP 生成。
3. 检查 DLL 导出表包含 `ufusr` 和 `ufusr_ask_unload`。
4. 构建并运行 `BrepMFRNx2212RuntimeTests.exe`，验证成功消息和两类异常路径。
5. 运行仓库 Python 测试，确认新工程没有影响现有 CLI 和训练代码。
6. 运行 `git diff --check`，确认没有格式错误。

### NX2212 手动验收

1. 启动 NX2212，并打开或新建一个零件。
2. 从 NX Open 执行入口选择生成的 DLL。
3. 确认出现成功消息框，NX 不崩溃且 DLL 能正常卸载。

NX 内加载属于必须由用户执行的最终验收，因为命令行构建无法证明 NX 主进程实际
加载行为。

## 后续阶段

最小 DLL 验收通过后，按以下顺序继续：

1. 新建 NX2212 Block UI Styler 对话框。
2. 使用 `StepCreator` 导出当前工作部件。
3. 使用 Windows 进程 API 调用 `tools/infer_step.py`。
4. 校验退出码并解析版本化 `result.json`。
5. 在 NX 中展示特征实例列表。
6. 单独设计并验证 OCC/AAG 面到 NX Face Tag 的映射。
7. 映射明确且无歧义后才启用选择、高亮和着色。

## 完成标准

- 新工程位于 `integrations/nx2212/`，并纳入 Git。
- 工程不依赖 `MFR_Recognition_Project/`。
- Release x64 DLL 使用 NX2212/v142 成功构建。
- DLL 导出所需 NX 用户函数入口。
- README 给出准确的构建和手动加载命令。
- 用户在 NX2212 中看到成功消息框，确认第一阶段完成。
