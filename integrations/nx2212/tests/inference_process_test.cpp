#include "inference_process.hpp"

#include "test_support.hpp"

#include <chrono>
#include <string>
#include <system_error>
#include <vector>

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

    void OnProcessFinished() noexcept override
    {
        ++finished_count;
    }

    int started_count = 0;
    int tick_count = 0;
    int finished_count = 0;
    std::chrono::milliseconds last_elapsed{0};
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
