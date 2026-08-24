#include "inference_process.hpp"

#include "inference_command.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <windows.h>

namespace brepmfr
{
namespace nx2212
{
namespace
{
class UniqueHandle
{
public:
    explicit UniqueHandle(HANDLE handle = nullptr) noexcept : handle_(handle)
    {
    }

    ~UniqueHandle()
    {
        Reset();
    }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    HANDLE Get() const noexcept
    {
        return handle_;
    }

    void Reset(HANDLE replacement = nullptr) noexcept
    {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE)
        {
            CloseHandle(handle_);
        }
        handle_ = replacement;
    }

private:
    HANDLE handle_;
};

class FinishNotification
{
public:
    explicit FinishNotification(ProcessWaitObserver& observer) noexcept
        : observer_(observer)
    {
    }

    ~FinishNotification()
    {
        observer_.OnProcessFinished();
    }

    FinishNotification(const FinishNotification&) = delete;
    FinishNotification& operator=(const FinishNotification&) = delete;

private:
    ProcessWaitObserver& observer_;
};

std::wstring FormatWindowsErrorMessage(DWORD error_code)
{
    wchar_t* allocated = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error_code,
        0,
        reinterpret_cast<wchar_t*>(&allocated),
        0,
        nullptr);
    if (length == 0 || allocated == nullptr)
    {
        return L"unknown Windows error";
    }

    std::wstring message(allocated, length);
    LocalFree(allocated);
    while (!message.empty() &&
           (message.back() == L'\r' || message.back() == L'\n'))
    {
        message.pop_back();
    }
    return message;
}

std::string WideToUtf8(const std::wstring& text)
{
    if (text.empty())
    {
        return {};
    }
    const int required = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        text.data(),
        static_cast<int>(text.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (required <= 0)
    {
        return "could not encode Windows error text";
    }
    std::string utf8(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        text.data(),
        static_cast<int>(text.size()),
        utf8.data(),
        required,
        nullptr,
        nullptr);
    return utf8;
}

[[noreturn]] void ThrowWindowsError(
    DWORD error_code,
    const std::wstring& operation)
{
    const auto context = operation + L": " +
        FormatWindowsErrorMessage(error_code);
    throw std::system_error(
        static_cast<int>(error_code),
        std::system_category(),
        WideToUtf8(context));
}
}

ProcessExecution RunHiddenProcess(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::filesystem::path& log_path,
    ProcessWaitObserver& observer)
{
    if (!log_path.parent_path().empty())
    {
        std::filesystem::create_directories(log_path.parent_path());
    }

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    UniqueHandle log_file(CreateFileW(
        log_path.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        &security,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr));
    if (log_file.Get() == INVALID_HANDLE_VALUE)
    {
        const DWORD error_code = GetLastError();
        ThrowWindowsError(
            error_code,
            L"CreateFileW failed for inference log " + log_path.wstring());
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = log_file.Get();
    startup.hStdError = log_file.Get();

    PROCESS_INFORMATION process_info{};
    auto command_line = BuildWindowsCommandLine(executable, arguments);
    std::vector<wchar_t> mutable_command(
        command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    const auto started_at = std::chrono::steady_clock::now();
    if (!CreateProcessW(
            executable.c_str(),
            mutable_command.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup,
            &process_info))
    {
        const DWORD error_code = GetLastError();
        ThrowWindowsError(
            error_code,
            L"CreateProcessW failed for " + executable.wstring());
    }

    UniqueHandle process(process_info.hProcess);
    UniqueHandle thread(process_info.hThread);
    log_file.Reset();

    FinishNotification finish_notification(observer);
    observer.OnProcessStarted();

    for (;;)
    {
        const DWORD wait_result = WaitForSingleObject(process.Get(), 100);
        if (wait_result == WAIT_OBJECT_0)
        {
            break;
        }
        if (wait_result != WAIT_TIMEOUT)
        {
            const DWORD error_code = GetLastError();
            ThrowWindowsError(error_code, L"WaitForSingleObject failed");
        }
        observer.OnWaitTick(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at));
    }

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(process.Get(), &exit_code))
    {
        const DWORD error_code = GetLastError();
        ThrowWindowsError(error_code, L"GetExitCodeProcess failed");
    }

    return {
        exit_code,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at)};
}
}
}
