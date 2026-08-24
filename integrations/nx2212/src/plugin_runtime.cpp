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
