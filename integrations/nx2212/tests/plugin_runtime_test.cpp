#include "plugin_runtime.hpp"
#include "test_support.hpp"

#include <stdexcept>
#include <string>

namespace
{
using brepmfr::nx2212::test::Expect;
using brepmfr::nx2212::test::ExpectEqual;

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
}

BREPMFR_TEST(load_confirmation_reports_success)
{
    RecordingNotifier success(FailureMode::None);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(success) ==
            brepmfr::nx2212::kSuccess,
        "success path returned failure");
    ExpectEqual(success.information_calls, 1, "success message count changed");
    ExpectEqual(success.error_calls, 0, "success path displayed an error");
    ExpectEqual(
        success.information_title,
        std::string("BrepMFR NX2212"),
        "success title changed");
    ExpectEqual(
        success.information_message ==
            "BrepMFR NX2212 plug-in loaded successfully.",
        true,
        "success message changed");
}

BREPMFR_TEST(load_confirmation_reports_standard_exception)
{
    RecordingNotifier standard_failure(FailureMode::Standard);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(standard_failure) ==
            brepmfr::nx2212::kFailure,
        "standard exception returned success");
    ExpectEqual(
        standard_failure.error_calls,
        1,
        "standard exception was not reported");
    ExpectEqual(
        standard_failure.error_message,
        std::string("synthetic standard failure"),
        "standard exception message was not preserved");
}

BREPMFR_TEST(load_confirmation_reports_unknown_exception)
{
    RecordingNotifier unknown_failure(FailureMode::Unknown);
    Expect(
        brepmfr::nx2212::RunLoadConfirmation(unknown_failure) ==
            brepmfr::nx2212::kFailure,
        "unknown exception returned success");
    ExpectEqual(
        unknown_failure.error_calls,
        1,
        "unknown exception was not reported");
    ExpectEqual(
        unknown_failure.error_message,
        std::string(
            "BrepMFR NX2212 plug-in failed with an unknown error."),
        "unknown exception message changed");
}
