#pragma once

#include "inference_process.hpp"

#include <chrono>
#include <filesystem>
#include <string>

namespace brepmfr
{
namespace nx2212
{
class ProgressWindow final : public ProcessWaitObserver
{
public:
    ProgressWindow(
        void* owner_window,
        const std::filesystem::path& input_path);
    ~ProgressWindow() override;

    ProgressWindow(const ProgressWindow&) = delete;
    ProgressWindow& operator=(const ProgressWindow&) = delete;

    void OnProcessStarted() override;
    void OnWaitTick(std::chrono::milliseconds elapsed) override;
    void OnProgressStage(const std::string& stage) override;
    void OnProcessFinished() noexcept override;

private:
    void* window_ = nullptr;
    void* progress_ = nullptr;
    void* stage_label_ = nullptr;
    void* elapsed_label_ = nullptr;
    long long last_elapsed_seconds_ = -1;
};
}
}
