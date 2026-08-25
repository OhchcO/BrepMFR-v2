#include "inference_process.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
#include <windows.h>

namespace
{
class RecordingObserver final : public brepmfr::nx2212::ProcessWaitObserver
{
public:
    void OnProcessStarted() override
    {
        ++started_count;
    }

    void OnWaitTick(std::chrono::milliseconds elapsed) override
    {
        ++tick_count;
        last_elapsed = elapsed;
    }

    void OnProgressStage(const std::string& stage) override
    {
        stages.push_back(stage);
    }

    void OnProcessFinished() noexcept override
    {
        ++finished_count;
    }

    int started_count = 0;
    int tick_count = 0;
    int finished_count = 0;
    std::chrono::milliseconds last_elapsed{0};
    std::vector<std::string> stages;
};

class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const wchar_t* name, const std::wstring& value)
        : name_(name)
    {
        const DWORD required = GetEnvironmentVariableW(name, nullptr, 0);
        if (required != 0)
        {
            std::wstring existing(required, L'\0');
            GetEnvironmentVariableW(name, existing.data(), required);
            existing.resize(required - 1);
            original_ = std::move(existing);
        }
        SetEnvironmentVariableW(name, value.c_str());
    }

    ~ScopedEnvironmentVariable()
    {
        SetEnvironmentVariableW(
            name_.c_str(), original_ ? original_->c_str() : nullptr);
    }

private:
    std::wstring name_;
    std::optional<std::wstring> original_;
};
}

BREPMFR_TEST(RunHiddenProcessCapturesOutputAndExitCode)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto log_path = temporary.Path() / "process.log";
    RecordingObserver observer;

    const auto execution = RunHiddenProcess(
        CurrentTestExecutable(),
        {L"--process-fixture", L"7", L"value with spaces"},
        log_path,
        observer);

    ExpectEqual(execution.exit_code, 7UL, "child exit code was not preserved");
    const auto log = ReadUtf8(log_path);
    ExpectEqual(
        CountOccurrences(log, "value with spaces"),
        static_cast<std::size_t>(2),
        "stdout and stderr were not both captured");
    ExpectEqual(observer.started_count, 1, "start notification count mismatch");
    ExpectEqual(observer.finished_count, 1, "finish notification count mismatch");
    Expect(execution.elapsed.count() >= 0, "elapsed duration was negative");
}

BREPMFR_TEST(RunHiddenProcessReportsLaunchFailureWithoutNotifications)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    RecordingObserver observer;
    const auto executable = temporary.Path() / "missing-python.exe";
    const auto log_path = temporary.WriteUtf8(
        "launch-failure.log", "stale output must be removed");
    bool threw = false;
    std::string error_text;
    try
    {
        (void)RunHiddenProcess(
            executable,
            {},
            log_path,
            observer);
    }
    catch (const std::system_error& error)
    {
        threw = true;
        error_text = error.what();
    }

    Expect(threw, "missing executable did not report a launch failure");
    Expect(
        error_text.find(executable.string()) != std::string::npos,
        "launch error did not identify the executable");
    Expect(ReadUtf8(log_path).empty(), "failed launch retained stale log content");
    ExpectEqual(observer.started_count, 0, "failed launch emitted start notification");
    ExpectEqual(observer.finished_count, 0, "failed launch emitted finish notification");
}

BREPMFR_TEST(RunHiddenProcessUsesIsolatedPythonWorkingDirectoryAndEnvironment)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto python_root = temporary.Path() / "python-env";
    const auto executable = python_root / "python.exe";
    const auto working_directory = temporary.Path() / "project";
    const auto nx_root = temporary.Path() / "nx2212";
    std::filesystem::create_directories(python_root / "Library/bin");
    std::filesystem::create_directories(python_root / "DLLs");
    std::filesystem::create_directories(python_root / "Scripts");
    std::filesystem::create_directories(working_directory);
    std::filesystem::create_directories(nx_root / "UGII");
    std::filesystem::copy_file(CurrentTestExecutable(), executable);

    ScopedEnvironmentVariable ugii_base(L"UGII_BASE_DIR", nx_root.wstring());
    ScopedEnvironmentVariable path(
        L"PATH", (nx_root / "UGII").wstring() + L";C:\\Windows\\System32");
    ScopedEnvironmentVariable python_home(L"PythonHome", L"NX-PYTHON-HOME");
    ScopedEnvironmentVariable python_path(L"PythonPath", L"NX-PYTHON-PATH");
    ScopedEnvironmentVariable nx_marker(L"ugii_stage1_marker", L"NX-MARKER");

    RecordingObserver observer;
    const auto log_path = temporary.Path() / "environment.log";
    ProcessLaunchOptions options;
    options.working_directory = working_directory;
    options.isolate_python_environment = true;
    const auto execution = RunHiddenProcess(
        executable,
        {L"--environment-fixture"},
        log_path,
        options,
        observer);

    ExpectEqual(execution.exit_code, 0UL, "environment fixture failed");
    const auto log = ReadUtf8(log_path);
    auto normalized_log = log;
    std::replace(normalized_log.begin(), normalized_log.end(), '\\', '/');
    Expect(
        log.find("CWD=" + working_directory.generic_u8string()) !=
            std::string::npos,
        "child did not use the explicit working directory");
    Expect(
        normalized_log.find(nx_root.generic_u8string()) == std::string::npos,
        "child PATH retained an NX installation directory");
    Expect(
        log.find("PYTHONHOME=<unset>") != std::string::npos,
        "child inherited PYTHONHOME");
    Expect(
        log.find("PYTHONPATH=<unset>") != std::string::npos,
        "child inherited PYTHONPATH");
    Expect(
        log.find("UGII_STAGE1_MARKER=<unset>") != std::string::npos,
        "child inherited a mixed-case UGII variable");
    Expect(
        log.find("PYTHONNOUSERSITE=1") != std::string::npos,
        "child did not disable user site packages");
    Expect(
        log.find("PYTHONUNBUFFERED=1") != std::string::npos,
        "child did not enable unbuffered output");
    Expect(
        normalized_log.find((python_root / "Library/bin").generic_u8string()) !=
            std::string::npos,
        "child PATH did not prioritize the Python environment");
}

BREPMFR_TEST(RunHiddenProcessReportsUnbufferedProgressStagesWhileWaiting)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    RecordingObserver observer;
    const auto execution = RunHiddenProcess(
        CurrentTestExecutable(),
        {L"--progress-fixture"},
        temporary.Path() / "progress.log",
        observer);

    ExpectEqual(execution.exit_code, 0UL, "progress fixture failed");
    std::string actual_stages;
    for (const auto& stage : observer.stages)
    {
        if (!actual_stages.empty())
        {
            actual_stages += ',';
        }
        actual_stages += stage;
    }
    ExpectEqual(
        observer.stages,
        std::vector<std::string>({"startup", "conversion"}),
        "progress stages were not reported in order; actual=" + actual_stages);
}
