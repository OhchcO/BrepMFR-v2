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
