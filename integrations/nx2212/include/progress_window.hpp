#pragma once

#include "inference_process.hpp"

#include <chrono>

namespace brepmfr
{
namespace nx2212
{
class ProgressWindow final : public ProcessWaitObserver
{
public:
    explicit ProgressWindow(void* owner_window);
    ~ProgressWindow() override;

    ProgressWindow(const ProgressWindow&) = delete;
    ProgressWindow& operator=(const ProgressWindow&) = delete;

    void OnProcessStarted() override;
    void OnWaitTick(std::chrono::milliseconds elapsed) override;
    void OnProcessFinished() noexcept override;

private:
    void* window_ = nullptr;
    void* progress_ = nullptr;
    void* elapsed_label_ = nullptr;
    long long last_elapsed_seconds_ = -1;
};
}
}
