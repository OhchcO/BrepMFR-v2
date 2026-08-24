#include "plugin_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
enum class FailureMode
{
    None,
    Standard,
    Unknown
};

class RecordingNotifier final : public brepmfr::nx2212::LoadNotifier
{
public:
    explicit RecordingNotifier(FailureMode mode) : mode_(mode) {}

    void ShowInformation(const char* title, const char* message) override
    {
        information_title = title;
        information_message = message;
        ++information_calls;
        if (mode_ == FailureMode::Standard)
        {
            throw std::runtime_error("synthetic standard failure");
        }
        if (mode_ == FailureMode::Unknown)
        {
            throw 42;
        }
    }

    void ShowError(const char* title, const char* message) noexcept override
    {
        error_title = title;
        error_message = message;
        ++error_calls;
    }

    FailureMode mode_;
    int information_calls = 0;
    int error_calls = 0;
    std::string information_title;
    std::string information_message;
    std::string error_title;
    std::string error_message;
};

void Expect(bool condition, const char* message, int& failures)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        ++failures;
    }
}
}

int main()
{
    int failures = 0;

    RecordingNotifier success(FailureMode::None);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(success) ==
            brepmfr::nx2212::kSuccess,
        "success path returned failure",
        failures);
    Expect(
        success.information_calls == 1,
        "success message count was not one",
        failures);
    Expect(success.error_calls == 0, "success path displayed an error", failures);
    Expect(
        success.information_title == "BrepMFR NX2212",
        "success title changed",
        failures);
    Expect(
        success.information_message ==
            "BrepMFR NX2212 plug-in loaded successfully.",
        "success message changed",
        failures);

    RecordingNotifier standard_failure(FailureMode::Standard);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(standard_failure) ==
            brepmfr::nx2212::kFailure,
        "standard exception returned success",
        failures);
    Expect(
        standard_failure.error_calls == 1,
        "standard exception was not reported",
        failures);
    Expect(
        standard_failure.error_message == "synthetic standard failure",
        "standard exception message was not preserved",
        failures);

    RecordingNotifier unknown_failure(FailureMode::Unknown);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(unknown_failure) ==
            brepmfr::nx2212::kFailure,
        "unknown exception returned success",
        failures);
    Expect(
        unknown_failure.error_calls == 1,
        "unknown exception was not reported",
        failures);
    Expect(
        unknown_failure.error_message ==
            "BrepMFR NX2212 plug-in failed with an unknown error.",
        "unknown exception message changed",
        failures);

    return failures == 0 ? 0 : 1;
}
