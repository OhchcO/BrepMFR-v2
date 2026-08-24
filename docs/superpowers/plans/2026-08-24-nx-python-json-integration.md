# NX2212 Python CLI and JSON Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an NX2212 C++ plug-in workflow that selects an existing STEP file, runs the existing Python inference CLI with visible progress, validates result.json 1.0, and displays the recognition summary and OCC face-instance list in NX.

**Architecture:** Keep Python inference in a separate `CreateProcessW` child process. Pure C++17 modules own configuration, output identity, command construction, process execution, JSON validation, formatting, and orchestration; a thin NX/Win32 host owns file selection, progress UI, NX messages, and Listing Window output.

**Tech Stack:** Siemens NX2212 NXOpen C++, MSVC v142/x64, C++17, Win32, nlohmann/json 3.12.0, Python 3.8 CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-nx-python-json-integration-design.md`

## Global Constraints

- Build only `Release|x64` with `PlatformToolset=v142` and `LanguageStandard=stdcpp17`.
- Keep `tools/infer_step.py` and Python `result.json` schema version `1.0` unchanged.
- Do not load Python, PyTorch, CUDA, or checkpoints into the NX process.
- Launch Python with `CreateProcessW` directly; never use cmd.exe, PowerShell, or another shell.
- Vendor official nlohmann/json `v3.12.0` `json.hpp` with SHA-256 `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` and retain `LICENSE.MIT`.
- Treat paths as Unicode Windows paths; treat JSON and NX display text explicitly as UTF-8.
- Cancelling the STEP dialog returns success without launching Python or showing an error.
- Never interpret OCC/AAG `face_index` as an NX face index or Face Tag; do not highlight faces in this phase.
- Use strict TDD: observe the intended test failure before adding each production behavior.

---

### Task 1: Native Test Harness, JSON Dependency, and Plug-in Configuration

**Files:**
- Create: `integrations/nx2212/tests/test_support.hpp`
- Create: `integrations/nx2212/tests/test_main.cpp`
- Modify: `integrations/nx2212/tests/plugin_runtime_test.cpp`
- Create: `integrations/nx2212/tests/plugin_config_test.cpp`
- Create: `integrations/nx2212/include/plugin_config.hpp`
- Create: `integrations/nx2212/src/plugin_config.cpp`
- Create: `integrations/nx2212/third_party/nlohmann/json.hpp`
- Create: `integrations/nx2212/third_party/nlohmann/LICENSE.MIT`
- Create: `integrations/nx2212/third_party/nlohmann/README.md`
- Modify: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
struct PluginConfig {
    std::filesystem::path python_executable;
    std::filesystem::path inference_script;
    std::filesystem::path checkpoint;
    std::filesystem::path output_root;
    std::string device;
    double affinity_threshold;
};

PluginConfig LoadPluginConfig(const std::filesystem::path& config_path);
}
```

- Both Visual Studio projects compile as C++17 and include `third_party`.
- Tests register named functions with `BREPMFR_TEST(name)`; `test_main.cpp` executes the registry and returns nonzero on any assertion or unexpected exception.
- Test utilities expose these exact helpers:

```cpp
namespace brepmfr::nx2212::test {
class TemporaryDirectory {
public:
    TemporaryDirectory();
    ~TemporaryDirectory();
    const std::filesystem::path& Path() const noexcept;
    std::filesystem::path WriteFile(
        const std::filesystem::path& name,
        std::string_view bytes);
    std::filesystem::path WriteUtf8(
        const std::filesystem::path& name,
        std::string_view text);
};

template <typename Actual, typename Expected>
void ExpectEqual(
    const Actual& actual,
    const Expected& expected,
    const std::string& message);

std::filesystem::path CurrentTestExecutable();
std::string ReadUtf8(const std::filesystem::path& path);
std::size_t CountOccurrences(std::string_view text, std::string_view needle);
}
```

- `plugin_config_test.cpp` owns:

```cpp
std::string CompleteConfigJson(
    const std::filesystem::path& python,
    const std::filesystem::path& script,
    const std::filesystem::path& checkpoint,
    const std::filesystem::path& output,
    const std::string& device,
    double threshold);
```

- [ ] **Step 1: Replace the one-file test main with a registry**

Create `test_support.hpp`:

```cpp
namespace brepmfr::nx2212::test {
using TestFunction = void (*)();
std::vector<std::pair<std::string, TestFunction>>& Registry();

class Registration {
public:
    Registration(const char* name, TestFunction function);
};

void Expect(bool condition, const std::string& message);
}

#define BREPMFR_TEST(name)                                   \
    void name();                                             \
    static brepmfr::nx2212::test::Registration name##_entry( \
        #name, &name);                                       \
    void name()
```

Move `main` to `test_main.cpp`. Rewrite the current success, standard-exception, and unknown-exception load-confirmation cases as three registered tests without changing production code.

- [ ] **Step 2: Verify the test-only refactor stays green**

Run:

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: build and executable both exit `0`.

- [ ] **Step 3: Write the first failing real configuration test**

Create real temporary Python, script, and checkpoint files, then write a complete configuration:

```cpp
BREPMFR_TEST(load_plugin_config_accepts_complete_version_1_document)
{
    TemporaryDirectory temporary;
    const auto python = temporary.WriteFile(L"python.exe", "");
    const auto script = temporary.WriteFile(L"infer_step.py", "");
    const auto checkpoint = temporary.WriteFile(L"model.ckpt", "");
    const auto output = temporary.Path() / L"output";
    const auto config = temporary.WriteUtf8(
        L"BrepMFRNx2212.config.json",
        CompleteConfigJson(python, script, checkpoint, output, "auto", 0.5));

    const auto loaded = brepmfr::nx2212::LoadPluginConfig(config);

    ExpectEqual(loaded.python_executable, python, "python path changed");
    ExpectEqual(loaded.checkpoint, checkpoint, "checkpoint path changed");
    ExpectEqual(loaded.device, std::string("auto"), "device changed");
    Expect(loaded.affinity_threshold == 0.5, "threshold changed");
}
```

`CompleteConfigJson` is a test-only helper that emits every real field. Expectations remain hand-written literals.

- [ ] **Step 4: Run to verify RED**

Run the Task 1 Step 2 build.

Expected: compilation fails because `plugin_config.hpp` or `LoadPluginConfig` is missing.

- [ ] **Step 5: Vendor and verify nlohmann/json 3.12.0**

```powershell
$vendor = 'integrations\nx2212\third_party\nlohmann'
New-Item -ItemType Directory -Force -Path $vendor | Out-Null
Invoke-WebRequest 'https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp' -OutFile (Join-Path $vendor 'json.hpp')
Invoke-WebRequest 'https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT' -OutFile (Join-Path $vendor 'LICENSE.MIT')
$actual = (Get-FileHash (Join-Path $vendor 'json.hpp') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne 'aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63') { throw "Unexpected nlohmann/json hash: $actual" }
```

Record the version, release URL, upstream repository, SHA-256, and retained MIT license in the third-party README. Do not edit downloaded files.

- [ ] **Step 6: Implement the minimum valid configuration reader**

Parse a binary UTF-8 stream with nlohmann/json. Require exact version `1.0` and typed fields. Reject non-absolute paths; require Python, script, and checkpoint to be files; require `.ckpt`; allow a missing output root but reject an existing non-directory. Translate JSON/filesystem errors into `std::runtime_error` messages that name the config path and field.

Set `LanguageStandard=stdcpp17` and add `third_party` to both project include paths. Compile `plugin_config.cpp` in both targets.

- [ ] **Step 7: Verify GREEN**

Run the native build and executable.

Expected: the configuration test and original three behavior tests pass.

- [ ] **Step 8: Add table-driven validation tests and observe RED**

Cover missing schema version, version `2.0`, relative Python, missing Python file, non-.ckpt checkpoint, device `tpu`, thresholds `-0.01` and `1.01`, string threshold `"0.5"`, and output_root that is a file. Each case asserts the failing field appears in the error.

- [ ] **Step 9: Complete validation, rerun, and commit**

Add only missing checks, including `std::isfinite`. Run native tests.

```powershell
git add integrations/nx2212
git commit -m "feat: load validated NX inference configuration"
```

---

### Task 2: Stable Output Paths and Windows Command Arguments

**Files:**
- Create: `integrations/nx2212/include/output_paths.hpp`
- Create: `integrations/nx2212/src/output_paths.cpp`
- Create: `integrations/nx2212/include/inference_command.hpp`
- Create: `integrations/nx2212/src/inference_command.cpp`
- Create: `integrations/nx2212/tests/output_paths_test.cpp`
- Create: `integrations/nx2212/tests/inference_command_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
struct OutputPaths {
    std::filesystem::path directory;
    std::filesystem::path bin_path;
    std::filesystem::path metadata_path;
    std::filesystem::path result_path;
    std::filesystem::path log_path;
};

std::wstring SanitizeStepStem(std::wstring_view stem);
std::uint64_t StablePathHash(const std::filesystem::path& absolute_path);
OutputPaths BuildOutputPaths(
    const std::filesystem::path& output_root,
    const std::filesystem::path& step_path);

std::wstring QuoteWindowsArgument(std::wstring_view argument);
std::vector<std::wstring> BuildInferenceArguments(
    const PluginConfig& config,
    const std::filesystem::path& step_path,
    const OutputPaths& output);
std::wstring BuildWindowsCommandLine(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments);
}
```

- [ ] **Step 1: Write failing output identity tests**

```cpp
BREPMFR_TEST(stable_path_hash_uses_normalized_lowercase_generic_path)
{
    ExpectEqual(
        brepmfr::nx2212::StablePathHash(L"C:\\Parts\\Widget.step"),
        std::uint64_t{0xefa652266344ab46ULL},
        "normalized path hash changed");
}

BREPMFR_TEST(output_paths_separate_same_stem_from_different_directories)
{
    const auto left = BuildOutputPaths(L"C:\\out", L"C:\\a\\part.step");
    const auto right = BuildOutputPaths(L"C:\\out", L"D:\\b\\part.step");
    Expect(left.directory != right.directory, "same-name STEP paths collided");
    ExpectEqual(left.result_path.filename(), std::filesystem::path(L"result.json"),
                "result filename changed");
}
```

Also assert `SanitizeStepStem(L"a<b>:c") == L"a_b__c"` and all five files share the derived directory.

- [ ] **Step 2: Run to verify RED**

Expected: compile failure because `output_paths.hpp` is missing.

- [ ] **Step 3: Implement output identity and verify GREEN**

Normalize an absolute path to generic separators, lowercase ASCII, encode as UTF-8, and apply FNV-1a 64 with offset `0xcbf29ce484222325` and prime `0x100000001b3`. Render exactly 16 lowercase hex digits. Reject non-absolute inputs and do not create directories in this pure function.

- [ ] **Step 4: Write failing Windows quoting and CLI-order tests**

```cpp
BREPMFR_TEST(windows_argument_quoting_handles_spaces_quotes_and_trailing_slashes)
{
    ExpectEqual(QuoteWindowsArgument(L"plain"), L"plain", "plain changed");
    ExpectEqual(QuoteWindowsArgument(L"two words"), L"\"two words\"",
                "spaces not quoted");
    ExpectEqual(
        QuoteWindowsArgument(L"C:\\Program Files\\"),
        L"\"C:\\Program Files\\\\\"",
        "trailing slash not doubled");
    ExpectEqual(QuoteWindowsArgument(L"a\"b"), L"\"a\\\"b\"",
                "embedded quote not escaped");
}
```

Assert `BuildInferenceArguments` returns exactly: script, `--step`, STEP, `--checkpoint`, checkpoint, `--output-dir`, derived directory, `--device`, device, `--affinity-threshold`, and locale-independent threshold.

- [ ] **Step 5: Run to verify RED**

Expected: compile failure because `inference_command.hpp` is missing.

- [ ] **Step 6: Implement quoting, verify GREEN, and commit**

Implement the CommandLineToArgvW-compatible backslash/quote algorithm. Prepend the quoted executable in `BuildWindowsCommandLine`. Format the threshold with classic locale and `max_digits10`. Run native tests.

```powershell
git add integrations/nx2212
git commit -m "feat: derive NX inference paths and arguments"
```

---

### Task 3: Real Hidden Child Process and Log Capture

**Files:**
- Create: `integrations/nx2212/include/inference_process.hpp`
- Create: `integrations/nx2212/src/inference_process.cpp`
- Create: `integrations/nx2212/tests/inference_process_test.cpp`
- Modify: `integrations/nx2212/tests/test_main.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
class ProcessWaitObserver {
public:
    virtual ~ProcessWaitObserver() = default;
    virtual void OnProcessStarted() = 0;
    virtual void OnWaitTick(std::chrono::milliseconds elapsed) = 0;
    virtual void OnProcessFinished() noexcept = 0;
};

struct ProcessExecution {
    unsigned long exit_code;
    std::chrono::milliseconds elapsed;
};

ProcessExecution RunHiddenProcess(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::filesystem::path& log_path,
    ProcessWaitObserver& observer);
}
```

- [ ] **Step 1: Add real process-fixture mode to the test executable**

Before registry execution, `wmain` recognizes `--process-fixture <exit-code> <payload>`. It writes payload once to stdout and once to stderr, flushes, and returns the requested exit code.

- [ ] **Step 2: Write the failing real-process test**

Create a test-only observer whose callbacks only record the observable lifecycle:

```cpp
class RecordingWaitObserver final : public ProcessWaitObserver {
public:
    void OnProcessStarted() override { ++started; }
    void OnWaitTick(std::chrono::milliseconds) override { ++ticks; }
    void OnProcessFinished() noexcept override { ++finished; }
    int started = 0;
    int ticks = 0;
    int finished = 0;
};
```

```cpp
BREPMFR_TEST(hidden_process_preserves_arguments_captures_streams_and_exit_code)
{
    TemporaryDirectory temporary;
    RecordingWaitObserver observer;
    const auto log = temporary.Path() / L"run.log";

    const auto execution = RunHiddenProcess(
        CurrentTestExecutable(),
        {L"--process-fixture", L"7", L"value with spaces"},
        log,
        observer);

    ExpectEqual(execution.exit_code, 7UL, "exit code changed");
    Expect(CountOccurrences(ReadUtf8(log), "value with spaces") == 2,
           "stdout and stderr were not both captured");
    Expect(observer.started == 1, "start missing");
    Expect(observer.finished == 1, "finish missing");
}
```

- [ ] **Step 3: Run to verify RED**

Expected: compile failure because `RunHiddenProcess` is missing.

- [ ] **Step 4: Implement process lifecycle**

Create the log with:

```cpp
CreateFileW(
    log_path.c_str(),
    GENERIC_WRITE,
    FILE_SHARE_READ,
    &inheritable_security_attributes,
    CREATE_ALWAYS,
    FILE_ATTRIBUTE_NORMAL,
    nullptr);
```

Configure `STARTF_USESTDHANDLES`, pass the same log handle as stdout and stderr, and call `CreateProcessW` with `CREATE_NO_WINDOW` and inherited handles. Use RAII for log, thread, and process handles.

After successful creation call `OnProcessStarted`, wait in 100 ms intervals, call `OnWaitTick` after timeouts, read `GetExitCodeProcess`, and guarantee one `OnProcessFinished` call on every post-start exit path. Format Win32 errors with `FormatMessageW`.

- [ ] **Step 5: Verify GREEN and add launch-failure coverage**

Run native tests. Then test a nonexistent executable: error includes its path, observer start/finish stay zero, and stale log content is not accepted. Add only missing failure handling and rerun.

- [ ] **Step 6: Commit**

```powershell
git add integrations/nx2212
git commit -m "feat: run hidden Python process with log capture"
```

---

### Task 4: Strict result.json 1.0 Reader

**Files:**
- Create: `integrations/nx2212/include/result_reader.hpp`
- Create: `integrations/nx2212/src/result_reader.cpp`
- Create: `integrations/nx2212/tests/fixtures/result_success.json`
- Create: `integrations/nx2212/tests/fixtures/result_error.json`
- Create: `integrations/nx2212/tests/result_reader_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
struct FaceResult {
    int face_index;
    int class_id;
    std::string feature_name;
    double confidence;
    std::optional<int> instance_id;
};

struct InstanceResult {
    int instance_id;
    int class_id;
    std::string feature_name;
    std::vector<int> face_indices;
    double confidence;
};

struct SuccessResult {
    std::filesystem::path step_path;
    int face_count;
    bool cache_hit;
    double total_seconds;
    std::vector<FaceResult> faces;
    std::vector<InstanceResult> instances;
};

struct ErrorResult {
    std::filesystem::path step_path;
    std::string stage;
    std::string error_type;
    std::string message;
};

using InferenceResult = std::variant<SuccessResult, ErrorResult>;
InferenceResult ReadInferenceResult(
    const std::filesystem::path& result_path,
    const std::filesystem::path& selected_step);
}
```

- [ ] **Step 1: Create complete success and error fixtures**

The success fixture must contain schema version, status, full input/cache/model, two faces, one instance, timings, and warnings:

```json
{
  "schema_version": "1.0",
  "status": "success",
  "input": {"step_path": "__STEP_PATH__", "step_sha256": "abc", "face_count": 2},
  "cache": {"bin_path": "D:/output/input.bin", "metadata_path": "D:/output/input.bin.meta.json", "hit": true, "fingerprint": "fp"},
  "model": {"checkpoint_path": "D:/model.ckpt", "checkpoint_sha256": "def", "checkpoint_epoch": 187, "experiment_mode": "dynamic_affinity", "device": "cuda", "num_classes": 25, "affinity_threshold": 0.5},
  "faces": [
    {"face_index": 0, "class_id": 24, "feature_name": "stock", "confidence": 0.99, "instance_id": null},
    {"face_index": 1, "class_id": 12, "feature_name": "blind_hole", "confidence": 0.95, "instance_id": 0}
  ],
  "instances": [
    {"instance_id": 0, "class_id": 12, "feature_name": "blind_hole", "face_indices": [1], "confidence": 0.95}
  ],
  "timings_seconds": {"checkpoint_load": 1.0, "conversion": 2.0, "inference": 3.0, "total": 6.0},
  "warnings": []
}
```

The error fixture has status error, stage `conversion`, type `ValueError`, message `shape mismatch`, and complete input step path.

- [ ] **Step 2: Write failing typed-reader tests**

Replace `__STEP_PATH__` with the selected path in generic UTF-8 form. Assert success has two faces, one instance, cache hit, total 6.0, and instance faces `[1]`. Assert error preserves stage, type, and message.

- [ ] **Step 3: Run to verify RED**

Expected: compile failure because `result_reader.hpp` is missing.

- [ ] **Step 4: Implement minimum typed parsing and verify GREEN**

Parse UTF-8, dispatch only literal `success` or `error`, and build the typed structures. Compare document/selected paths as absolute, lexically normalized, ASCII-case-insensitive Windows paths. Require finite nonnegative total time.

- [ ] **Step 5: Add semantic validation tests and observe RED**

Mutate one field per case: version 2.0, missing status, STEP mismatch, face_count mismatch, duplicate/non-contiguous face_index, wrong/non-finite confidence, duplicate instance_id, unsorted/duplicate face_indices, nonexistent referenced face, instance_id mismatch, and instance class/name mismatch. For errors cover empty stage, missing message, and path mismatch.

- [ ] **Step 6: Complete validation, rerun, and commit**

Validate via maps keyed by face_index and instance_id. Permit unknown fields. Sort output instances only after uniqueness checks. Run native tests.

```powershell
git add integrations/nx2212
git commit -m "feat: validate NX inference JSON results"
```

---

### Task 5: Deterministic NX Display Formatting

**Files:**
- Create: `integrations/nx2212/include/result_formatter.hpp`
- Create: `integrations/nx2212/src/result_formatter.cpp`
- Create: `integrations/nx2212/tests/result_formatter_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
struct DisplayText {
    std::string summary_utf8;
    std::vector<std::string> listing_lines_utf8;
};

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path);
std::string FormatErrorDisplay(
    const ErrorResult& error,
    const std::filesystem::path& result_path,
    const std::filesystem::path& log_path);
}
```

- [ ] **Step 1: Write failing literal formatter tests**

For the complete two-face fixture assert summary facts: 2 faces, 1 instance, cache hit, 6.0 seconds. Assert listing contains:

```text
Face numbering: OCC/AAG face_index (not NX Face Tag)
[0] blind_hole: OCC faces [1], confidence 0.9500
```

For no instances require `No non-stock feature instances were recognized.`. Error display must contain conversion, ValueError, shape mismatch, result path, and log path.

- [ ] **Step 2: Run to verify RED**

Expected: compile failure because `result_formatter.hpp` is missing.

- [ ] **Step 3: Implement, verify GREEN, and commit**

Use classic locale, fixed four decimals for confidence, and one decimal for seconds. Sort a copy by instance_id. Emit generic UTF-8 STEP/result paths and the mandatory OCC warning. Run native tests.

```powershell
git add integrations/nx2212
git commit -m "feat: format NX inference result display"
```

---

### Task 6: Testable End-to-End C++ Workflow

**Files:**
- Replace behavior in: `integrations/nx2212/include/plugin_runtime.hpp`
- Replace behavior in: `integrations/nx2212/src/plugin_runtime.cpp`
- Rewrite: `integrations/nx2212/tests/plugin_runtime_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

class PluginHost {
public:
    virtual ~PluginHost() = default;
    virtual std::filesystem::path ModuleDirectory() const = 0;
    virtual std::optional<std::filesystem::path> SelectStepFile() = 0;
    virtual unsigned long RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path) = 0;
    virtual void ShowSuccess(const DisplayText& display) = 0;
    virtual void ShowError(const std::string& message_utf8) noexcept = 0;
};

int RunInferenceWorkflow(PluginHost& host) noexcept;
}
```

- [ ] **Step 1: Write failing cancellation workflow test**

A recording host supplies a valid module config then returns `std::nullopt` from selection. Assert success return, no process, no success display, and no error display.

- [ ] **Step 2: Implement config loading plus cancellation**

Load `ModuleDirectory()/BrepMFRNx2212.config.json`, select STEP, return success on cancellation, and contain all exceptions with one noexcept error report.

- [ ] **Step 3: Write failing real success-orchestration test**

The host selects a real temporary STEP. Its RunProcess asserts the executable and exact arguments, verifies stale result.json was removed, writes the complete success fixture to the requested output directory, and returns 0. Assert output directory creation, one success display with expected text, no error, and `kSuccess`.

- [ ] **Step 4: Implement success path and verify GREEN**

Build output paths/arguments, create the directory, remove only derived stale `result.json`, run, parse the fresh result, require exit 0 plus SuccessResult, format, and display.

- [ ] **Step 5: Write failing failure-boundary tests**

Cover nonzero plus ErrorResult, zero plus ErrorResult, nonzero plus SuccessResult, missing fresh JSON, malformed JSON, standard exception, and unknown exception. Require result/log paths after output derivation and config path before it. Assert the user-visible outcome, not only fake call counts.

- [ ] **Step 6: Complete failure handling, rerun, and commit**

Call `ShowError` at most once; never let an exception escape. Run all native tests.

```powershell
git add integrations/nx2212
git commit -m "feat: orchestrate NX Python inference workflow"
```

---

### Task 7: NX2212 File Dialog, Progress Window, and Result Host

**Files:**
- Create: `integrations/nx2212/include/progress_window.hpp`
- Create: `integrations/nx2212/src/progress_window.cpp`
- Create: `integrations/nx2212/include/nx_plugin_host.hpp`
- Create: `integrations/nx2212/src/nx_plugin_host.cpp`
- Modify: `integrations/nx2212/src/plugin_entry.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
class ProgressWindow final : public ProcessWaitObserver {
public:
    explicit ProgressWindow(void* owner_window);
    ~ProgressWindow() override;
    void OnProcessStarted() override;
    void OnWaitTick(std::chrono::milliseconds elapsed) override;
    void OnProcessFinished() noexcept override;
};

class NxPluginHost final : public PluginHost {
public:
    std::filesystem::path ModuleDirectory() const override;
    std::optional<std::filesystem::path> SelectStepFile() override;
    unsigned long RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path) override;
    void ShowSuccess(const DisplayText& display) override;
    void ShowError(const std::string& message_utf8) noexcept override;
};
}
```

- [ ] **Step 1: Add a failing build-level test**

Extend `tests/test_nx2212_project.py` to rebuild the runtime test project and execute `BrepMFRNx2212RuntimeTests.exe`, asserting exit 0 with captured output on failure. Add nonexistent `nx_plugin_host.cpp` and `progress_window.cpp` to the DLL project, then run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_nx2212_project.py -q
```

Expected: actual DLL build fails because the production adapters are absent.

- [ ] **Step 2: Implement Unicode STEP selection and module discovery**

Use `GetOpenFileNameW` with a 32,768-wide-character buffer, `OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST`, and filter `*.step;*.stp`. Return nullopt only when `CommDlgExtendedError()==0`.

Resolve the loaded DLL rather than the NX executable:

```cpp
extern "C" IMAGE_DOS_HEADER __ImageBase;
const auto module = reinterpret_cast<HMODULE>(&__ImageBase);
```

Call `GetModuleFileNameW(module, ...)` with a growing buffer until the result is not truncated, then return its parent directory.

- [ ] **Step 3: Implement marquee progress**

Initialize `ICC_PROGRESS_CLASS`. Create an owned Win32 window with Chinese status text, a `PROGRESS_CLASSW | PBS_MARQUEE` control, and elapsed seconds. Disable close. On each wait tick use `PeekMessageW` to pump messages and update the integer-second label. Make finish and destructor idempotently stop/destroy the window.

In `NxPluginHost::RunProcess` lock NX UI, create the observer, call `RunHiddenProcess`, and guarantee `UnlockAccess` with RAII.

- [ ] **Step 4: Implement NX UTF-8 output**

Use `NXOpen::NXString(text, NXOpen::NXString::UTF8)`. Open `Session::GetSession()->ListingWindow()`, write full lines with `WriteFullline`, then show the summary as information. Show errors with `DialogTypeError` inside a catch-all noexcept method.

- [ ] **Step 5: Replace load-only entry and link Win32 dependencies**

`ufusr` constructs `NxPluginHost` and calls `RunInferenceWorkflow` while preserving safe retcod initialization and catch-all ABI containment. Keep immediate unload. Link `comdlg32.lib` and `comctl32.lib`; do not add libufun unless an implemented symbol requires it.

- [ ] **Step 6: Verify build/exports and commit**

Run focused pytest. Require runtime test pass, DLL build, and dumpbin exports `ufusr` and `ufusr_ask_unload`.

```powershell
git add integrations/nx2212 tests/test_nx2212_project.py
git commit -m "feat: connect NX2212 UI to Python inference"
```

---

### Task 8: Configuration Example, Operator Guide, and Full Verification

**Files:**
- Create: `integrations/nx2212/BrepMFRNx2212.config.example.json`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `integrations/nx2212/README.md`
- Modify: `README.md`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**
- A clean DLL build copies `BrepMFRNx2212.config.example.json` beside the DLL.
- The user copies it to ignored `BrepMFRNx2212.config.json` and supplies local paths.

- [ ] **Step 1: Write failing build-artifact test**

After the real DLL build assert the adjacent example exists and parses to:

```python
{
    "schema_version": "1.0",
    "python_executable": "C:/path/to/python.exe",
    "inference_script": "C:/path/to/BrepMFR/tools/infer_step.py",
    "checkpoint": "C:/path/to/best-model.ckpt",
    "output_root": "C:/path/to/recognition-output",
    "device": "auto",
    "affinity_threshold": 0.5,
}
```

Run focused pytest and observe failure because the output example is absent.

- [ ] **Step 2: Add example and post-build copy**

Create exactly the JSON above. Add a PostBuildEvent that copies it beside the DLL and fails on copy error. Do not create or commit the real config.

- [ ] **Step 3: Rewrite operator documentation**

Document build/test commands, DLL/config paths, copying the example, Ctrl+U STEP selection, progress/summary/Listing Window, result.json/run.log, cache miss then hit, troubleshooting, and the OCC face_index warning. Link this guide from repository README.

- [ ] **Step 4: Run real Python CLI twice before NX**

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' tools\infer_step.py --step tests\assets\step_inference\20221121_154647_0.step --checkpoint 'D:\Code\BrepMFR\results\paper_core_experiments\MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_untrimmed_gated_seed42\0821\092832\best-mIoU-187-0.939778.ckpt' --output-dir 'D:\Code\BrepMFR\results\nx_recognition\cli-smoke' --device auto --affinity-threshold 0.5
```

Expected first run: exit 0, success, epoch 187, 41 faces. Run again and require `cache.hit=true`.

- [ ] **Step 5: Create ignored local DLL config**

After building, copy the example beside the worktree DLL and use:

```json
{
  "schema_version": "1.0",
  "python_executable": "F:/ProgramData/anaconda3/envs/aagmspenv/python.exe",
  "inference_script": "D:/Code/BrepMFR/.worktrees/nx-python-json-integration/tools/infer_step.py",
  "checkpoint": "D:/Code/BrepMFR/results/paper_core_experiments/MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_untrimmed_gated_seed42/0821/092832/best-mIoU-187-0.939778.ckpt",
  "output_root": "D:/Code/BrepMFR/results/nx_recognition",
  "device": "auto",
  "affinity_threshold": 0.5
}
```

Confirm `git status --short` does not list it because build output is ignored.

- [ ] **Step 6: Run complete automated verification**

```powershell
& 'F:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' integrations\nx2212\BrepMFRNx2212.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q
git diff --check
git status --short
```

Expected: solution/native/full pytest all pass; diff check clean; only intentional tracked changes remain before commit.

- [ ] **Step 7: Commit docs and example**

```powershell
git add README.md integrations/nx2212 tests/test_nx2212_project.py
git commit -m "docs: explain NX Python inference workflow"
```

- [ ] **Step 8: Perform NX2212 manual acceptance**

1. Load the worktree DLL with Ctrl+U.
2. Select `tests/assets/step_inference/20221121_154647_0.step`.
3. Confirm marquee motion and elapsed time updates.
4. Confirm summary reports 41 faces and the instance count from result.json.
5. Confirm Listing Window labels numbering as OCC/AAG face_index.
6. Repeat and confirm cache hit.
7. Cancel selection and confirm no error.
8. Temporarily set a nonexistent checkpoint, confirm the error names checkpoint and config path, then restore it.

Record observed outcomes. Do not claim completion until the user confirms NX-visible behavior.

---

## Final Review Checklist

- [ ] Every production function followed an observed failing test.
- [ ] nlohmann/json header hash and MIT license match v3.12.0.
- [ ] No real machine-specific configuration is tracked.
- [ ] No Python/model runtime is loaded into NX.
- [ ] Stale result.json can never be displayed.
- [ ] Exit code and JSON status contradictions fail safely.
- [ ] Unicode/spaced paths survive config, process, JSON, and NX output.
- [ ] Progress cleanup and NX unlock happen on every post-launch path.
- [ ] Native tests, pytest, Release/x64 build, and exports pass.
- [ ] NX manual success, cache hit, cancellation, and config error are observed.
- [ ] No code treats OCC/AAG face_index as NX Face Tag.
