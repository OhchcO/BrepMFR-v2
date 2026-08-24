#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
class ProcessWaitObserver
{
public:
    virtual ~ProcessWaitObserver() = default;
    virtual void OnProcessStarted() = 0;
    virtual void OnWaitTick(std::chrono::milliseconds elapsed) = 0;
    virtual void OnProcessFinished() noexcept = 0;
};

struct ProcessExecution
{
    unsigned long exit_code = 0;
    std::chrono::milliseconds elapsed{0};
};

ProcessExecution RunHiddenProcess(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::filesystem::path& log_path,
    ProcessWaitObserver& observer);
}
}
