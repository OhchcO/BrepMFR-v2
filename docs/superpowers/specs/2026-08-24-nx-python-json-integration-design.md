# NX2212 调用 Python CLI 与读取识别 JSON 设计

## 目标

在现有 NX2212 C++ 插件骨架上增加首个可用的识别入口。用户在 NX
中选择一个已有的 STEP/STP 文件，插件调用现有 Python CLI 完成
STEP 到 BIN、模型推理、后处理和标准 JSON 输出，然后由 NX 展示识别摘要和
按实例组织的 OCC 面编号。

本阶段不在 NX 进程内加载 Python、PyTorch、CUDA 或 checkpoint。NX 只负责
文件选择、进程编排、进度反馈、结果校验和显示，全部识别计算继续由独立的
Python 环境完成。

## 范围

本阶段包含：

- 从 NX 中选择一个现有的 .step 或 .stp 文件；
- 从 DLL 同目录读取本机配置；
- 以独立隐藏进程运行 tools/infer_step.py；
- 推理期间显示不确定进度条和已用时间；
- 将 Python 标准输出和错误输出保存到 run.log；
- 读取并严格校验 result.json 1.0；
- 在 NX 消息框显示摘要，在 Listing Window 显示实例明细；
- 对取消、配置错误、进程错误、推理错误和 JSON 错误给出明确反馈；
- 保留现有 STEP 对应的 BIN 缓存复用行为。

本阶段不包含：

- 从当前 NX Part 自动导出 STEP；
- 把 OCC/AAG face_index 直接当作 NX Face Tag；
- NX 面匹配、着色或高亮；
- 多任务并发、后台任务队列、百分比进度协议或取消推理；
- 在 NX 进程内嵌入 Python 解释器；
- 修改现有 Python 推理算法或 result.json 1.0 契约。

## 总体架构

采用“NX 原生编排器 + 外部 Python CLI”结构：

    NX2212 / C++ DLL
      -> 读取配置
      -> 选择 STEP
      -> 创建输出目录
      -> 启动 Python 子进程并显示进度
      -> 读取和校验 result.json
      -> 显示摘要与实例列表

    Python CLI
      -> STEP 转 BIN 或复用有效缓存
      -> 加载 checkpoint
      -> 模型推理
      -> 语义与实例后处理
      -> 原子写入 result.json

进程隔离是固定边界。C++ 不链接 Python DLL，不 import Python 模块，也不直接
链接 PyTorch。这样 Conda、CUDA 和深度学习依赖异常不会进入 NX 地址空间。

## 本机配置

插件根据自身 DLL 路径，在 DLL 同目录查找：

    BrepMFRNx2212.config.json

配置契约为：

    {
      "schema_version": "1.0",
      "python_executable": "F:/ProgramData/anaconda3/envs/aagmspenv/python.exe",
      "inference_script": "D:/Code/BrepMFR/tools/infer_step.py",
      "checkpoint": "D:/path/best-model.ckpt",
      "output_root": "D:/Code/BrepMFR/recognition-output",
      "device": "auto",
      "affinity_threshold": 0.5
    }

所有路径字段必须是绝对路径。python_executable、inference_script 和 checkpoint
必须是普通文件；checkpoint 必须使用 .ckpt 后缀；output_root 必须能够创建或
已经是目录；device 只能是 auto、cpu 或 cuda；affinity_threshold 必须是
0 到 1 之间的有限数；未知 schema_version 拒绝执行。

仓库提交 BrepMFRNx2212.config.example.json，并在构建时复制到 DLL 输出目录。
真实的 BrepMFRNx2212.config.json 含本机绝对路径，只保留在忽略的 build
目录中，不提交 Git。

## STEP 选择与输出身份

STEP 选择使用 Windows Unicode 文件对话框，只允许 .step 和 .stp。用户取消
时返回“未执行”，不显示错误，也不启动 Python。

每个输入使用独立输出目录：

    output_root/<安全的 STEP stem>-<规范化绝对路径 FNV-1a 64 位短哈希>/

stem 中不允许用于 Windows 文件名的字符会替换为下划线。路径短哈希用于区分
不同目录下的同名 STEP，不作为安全或缓存校验哈希。真正的缓存正确性仍由
Python 已有的 STEP 内容 SHA-256、转换实现指纹、采样配置、checkpoint 几何
需求和 BIN 校验共同保证。

输出目录固定包含：

    input.bin
    input.bin.meta.json
    result.json
    run.log

重复选择同一路径会进入同一输出目录，从而复用有效 BIN。每次启动 Python 前
必须移除该目录中的旧 result.json，避免进程在生成新 JSON 前失败时误读历史
成功结果；input.bin 和 input.bin.meta.json 不删除。

## Python 进程调用

插件使用 CreateProcessW 直接执行配置中的 python_executable，不经过
cmd.exe、PowerShell 或 shell。参数顺序固定为：

    <python_executable>
    <inference_script>
    --step <selected STEP>
    --checkpoint <checkpoint>
    --output-dir <derived output directory>
    --device <device>
    --affinity-threshold <threshold>

每个参数按 Windows CreateProcess 命令行规则独立引用，正确处理空格、中文、
双引号和尾随反斜杠。子进程使用 CREATE_NO_WINDOW。标准输出和标准错误共同
重定向到本次输出目录的 run.log，日志以新文件方式开始，避免把多次运行混在
一起。

插件记录进程退出码。CreateProcessW 失败时报告 Windows 错误文本；等待失败
时报告等待错误；无论进程退出码如何，只读取本次新生成的 result.json，且
必须检查退出码与 JSON status 一致：

- 退出码 0 只接受 status=success；
- 非零退出码只接受 status=error；
- 状态矛盾、没有新 JSON 或 JSON 无法校验都视为集成失败。

## 进度窗口

Python 启动成功后显示一个小型 Windows 原生进度窗口，包含：

- “正在识别，请稍候……”；
- marquee 风格的不确定进度条；
- 每秒更新的已用时间。

当前 CLI 没有阶段百分比协议，因此不能显示虚假的完成比例。插件在等待子进程
时使用 MsgWaitForMultipleObjects 或等价消息泵持续处理窗口消息，使进度条和
计时正常刷新。识别期间锁定 NX 编辑入口，使用 RAII 保证任意成功或异常路径
都关闭进度窗口并恢复 NX UI。

首版不提供取消按钮。关闭按钮禁用，避免强制终止 Python 时留下临时转换文件
或不完整状态。后续若需要取消，必须先为 Python CLI 设计显式取消协议和缓存
清理契约。

## JSON 依赖与解析边界

C++ 标准库不提供 JSON 解析。插件固定使用官方 nlohmann/json 单头文件，并在
third_party 目录保留对应 MIT 许可证。依赖随源码构建，不要求 NX 使用者联网、
安装包管理器或设置额外 include 路径。

result_reader 只接受现有成功或错误文档 schema_version 1.0。禁止用字符串搜索
读取 JSON。解析后转换为插件自己的小型数据结构，NX 适配层不直接依赖 JSON
对象。

成功文档至少校验：

- schema_version 精确等于 1.0；
- status 精确等于 success；
- input.step_path 与所选 STEP 指向同一 Windows 路径；
- input.face_count 是非负整数；
- cache.hit 是布尔值；
- faces 是数组，数量等于 input.face_count；
- 每个 face_index 是唯一、连续的 0-based 整数；
- 每个 face 含 class_id、feature_name、有限 confidence 和可空 instance_id；
- instances 是数组，instance_id 唯一；
- 每个实例含 class_id、feature_name、有限 confidence 和升序且唯一的
  face_indices；
- 实例引用的 face_index 存在，并与逐面 instance_id 一致；
- stock 面不要求生成实例。

错误文档至少校验：

- schema_version 精确等于 1.0；
- status 精确等于 error；
- stage、error.type 和 error.message 是非空字符串；
- input.step_path 与所选 STEP 指向同一 Windows 路径。

未知字段允许存在，以便 1.0 契约做向后兼容扩展；缺少必需字段、字段类型错误、
非有限数字、路径不匹配或内部引用不一致均拒绝展示。

## NX 展示

成功后先显示简短信息框：

    识别完成
    面数量：41
    特征实例：9
    BIN 缓存：命中
    用时：12.3 秒

随后打开 NX Listing Window，使用不截断长行的接口输出：

    BrepMFR recognition result
    STEP: D:/path/part.step
    Result: D:/path/output/result.json
    Face numbering: OCC/AAG face_index (not NX Face Tag)

    [0] triangular_passage: OCC faces [1, 3, 13], confidence 0.9947
    [2] rectangular_blind_slot: OCC faces [6, 8, 15], confidence 0.9582
    [7] blind_hole: OCC faces [31, 39], confidence 0.9999

实例按 instance_id 升序显示。没有实例但 status=success 时仍显示成功，并明确
提示“未识别到非 stock 特征”。所有面编号都标注为 OCC/AAG face_index，避免
用户误认为可以直接在 NX 中按数组下标选面。

错误 JSON 显示失败阶段和错误消息。集成层错误显示具体原因、配置文件路径、
预期 result.json 路径和 run.log 路径；不把 C++ 异常传播越过 ufusr ABI 边界。

## C++ 模块边界

新增或调整的模块职责如下：

- plugin_config：配置数据结构、UTF-8 JSON 读取和字段/路径校验；
- output_paths：安全 stem、稳定路径哈希和输出文件位置；
- inference_command：CreateProcessW 参数引用与命令行构造；
- inference_process：日志重定向、子进程生命周期和退出码；
- progress_window：Win32 不确定进度窗口、计时和消息泵；
- result_reader：result.json 1.0 解析、语义校验和插件数据结构；
- result_formatter：成功摘要、Listing Window 文本和错误文本；
- plugin_runtime：依赖接口和完整工作流编排；
- plugin_entry：NXOpen/Win32 适配，包括文件选择、消息框和 Listing Window。

纯逻辑模块不得 include NXOpen 头文件。NX 依赖集中在 plugin_entry 及很薄的
适配类中，从而允许原生测试程序在不启动 NX 的情况下覆盖绝大多数行为。

## 测试策略

所有新增生产行为遵循测试先行。

原生自动化测试覆盖：

- 有效配置与每个字段的缺失、类型、取值和路径错误；
- 中文、空格、双引号和尾随反斜杠参数的 Windows 引用；
- 安全 stem、稳定路径哈希和同名不同路径输出隔离；
- 成功 JSON、错误 JSON、损坏 JSON、未知版本、路径不匹配；
- face_index 连续性、实例引用一致性和有限浮点数；
- 成功摘要、无实例摘要、实例排序与 OCC 编号警告；
- 用户取消、进程启动失败、退出码冲突和缺失新 JSON；
- 任意失败路径都关闭进度并恢复 UI 的编排行为。

仓库级测试覆盖：

- VS 项目包含所有新增源文件和第三方许可证；
- Release/x64/v142 能构建运行时测试和 NX DLL；
- DLL 继续导出 ufusr 与 ufusr_ask_unload；
- 现有 Python 推理和其他 pytest 回归保持通过。

NX2212 手动验收覆盖：

1. 配置真实 Python、checkpoint 和 output_root；
2. 加载 DLL 并选择一个真实 STEP；
3. 确认进度窗口可见、滚动且计时更新；
4. 确认 Python 结束后 NX 显示摘要和 Listing Window 实例明细；
5. 确认 output 目录包含 result.json、run.log 和 BIN 缓存；
6. 再次选择同一 STEP，确认 result.json 中 cache.hit 为 true；
7. 取消文件选择，确认不显示错误；
8. 临时使用无效 checkpoint，确认 NX 展示可操作的失败信息和日志路径。

## 后续面高亮

下一阶段必须单独解决 OCC/AAG face_index 到 NX Face Tag 的稳定映射。当前
result.json 明确不承诺两者顺序相同，因此禁止在本阶段按 NX face 数组位置
高亮。后续方案应由 Python 输出可跨 CAD 内核匹配的几何签名，或建立导入过程
中的显式映射，再由 NX 负责颜色、高亮和选择交互。该工作不改变本设计中
“Python 负责计算，NX 负责展示”的总体边界。
