#include "inference_process.hpp"

#include "inference_command.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
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

struct CaseInsensitiveLess
{
    bool operator()(const std::wstring& left, const std::wstring& right) const
    {
        return CompareStringOrdinal(
                   left.c_str(),
                   static_cast<int>(left.size()),
                   right.c_str(),
                   static_cast<int>(right.size()),
                   TRUE) == CSTR_LESS_THAN;
    }
};

using Environment =
    std::map<std::wstring, std::wstring, CaseInsensitiveLess>;

bool StartsWithCaseInsensitive(
    const std::wstring& value,
    const std::wstring& prefix)
{
    return value.size() >= prefix.size() &&
        CompareStringOrdinal(
            value.c_str(),
            static_cast<int>(prefix.size()),
            prefix.c_str(),
            static_cast<int>(prefix.size()),
            TRUE) == CSTR_EQUAL;
}

bool EqualsCaseInsensitive(
    const std::wstring& left,
    const std::wstring& right)
{
    return left.size() == right.size() &&
        StartsWithCaseInsensitive(left, right);
}

[[noreturn]] void ThrowWindowsError(
    DWORD error_code,
    const std::wstring& operation);

Environment ReadCurrentEnvironment()
{
    wchar_t* block = GetEnvironmentStringsW();
    if (block == nullptr)
    {
        ThrowWindowsError(GetLastError(), L"GetEnvironmentStringsW failed");
    }

    Environment environment;
    for (const wchar_t* entry = block; *entry != L'\0';)
    {
        std::wstring item(entry);
        entry += item.size() + 1;
        const std::size_t separator = item.find(
            L'=', item.empty() || item.front() != L'=' ? 0 : 1);
        if (separator != std::wstring::npos)
        {
            environment[item.substr(0, separator)] =
                item.substr(separator + 1);
        }
    }
    FreeEnvironmentStringsW(block);
    return environment;
}

std::wstring LowerWindowsPath(std::filesystem::path path)
{
    auto text = path.lexically_normal().wstring();
    std::replace(text.begin(), text.end(), L'/', L'\\');
    std::transform(text.begin(), text.end(), text.begin(), [](wchar_t value) {
        if (value >= L'A' && value <= L'Z')
        {
            return static_cast<wchar_t>(value - L'A' + L'a');
        }
        return value;
    });
    while (text.size() > 3 && text.back() == L'\\')
    {
        text.pop_back();
    }
    return text;
}

bool IsPathInside(
    const std::filesystem::path& candidate,
    const std::filesystem::path& root)
{
    if (candidate.empty() || root.empty())
    {
        return false;
    }
    const auto candidate_text = LowerWindowsPath(candidate);
    const auto root_text = LowerWindowsPath(root);
    return candidate_text == root_text ||
        (candidate_text.size() > root_text.size() &&
         candidate_text.compare(0, root_text.size(), root_text) == 0 &&
         candidate_text[root_text.size()] == L'\\');
}

std::vector<std::wstring> SplitPathList(const std::wstring& value)
{
    std::vector<std::wstring> entries;
    std::size_t start = 0;
    while (start <= value.size())
    {
        const std::size_t end = value.find(L';', start);
        std::wstring entry = value.substr(
            start,
            end == std::wstring::npos ? std::wstring::npos : end - start);
        if (entry.size() >= 2 && entry.front() == L'"' && entry.back() == L'"')
        {
            entry = entry.substr(1, entry.size() - 2);
        }
        if (!entry.empty())
        {
            entries.push_back(std::move(entry));
        }
        if (end == std::wstring::npos)
        {
            break;
        }
        start = end + 1;
    }
    return entries;
}

void AppendUniquePath(
    std::vector<std::wstring>& entries,
    const std::filesystem::path& path)
{
    const auto key = LowerWindowsPath(path);
    const auto duplicate = std::find_if(
        entries.begin(), entries.end(), [&](const std::wstring& existing) {
            return LowerWindowsPath(existing) == key;
        });
    if (duplicate == entries.end())
    {
        entries.push_back(path.wstring());
    }
}

std::vector<wchar_t> BuildIsolatedPythonEnvironment(
    const std::filesystem::path& executable)
{
    auto environment = ReadCurrentEnvironment();
    const auto ugii = environment.find(L"UGII_BASE_DIR");
    const std::filesystem::path nx_root =
        ugii == environment.end() ? std::filesystem::path{} : ugii->second;
    const std::wstring inherited_path =
        environment.count(L"PATH") == 0 ? L"" : environment[L"PATH"];

    for (auto iterator = environment.begin(); iterator != environment.end();)
    {
        const std::wstring& name = iterator->first;
        if (EqualsCaseInsensitive(name, L"PYTHONHOME") ||
            EqualsCaseInsensitive(name, L"PYTHONPATH") ||
            StartsWithCaseInsensitive(name, L"UGII_") ||
            StartsWithCaseInsensitive(name, L"UGS_"))
        {
            iterator = environment.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }

    const auto python_root = executable.parent_path();
    std::vector<std::wstring> path_entries;
    AppendUniquePath(path_entries, python_root);
    AppendUniquePath(path_entries, python_root / "Library" / "bin");
    AppendUniquePath(path_entries, python_root / "DLLs");
    AppendUniquePath(path_entries, python_root / "Scripts");
    for (const auto& entry : SplitPathList(inherited_path))
    {
        if (!IsPathInside(entry, nx_root))
        {
            AppendUniquePath(path_entries, entry);
        }
    }

    std::wstring path_value;
    for (const auto& entry : path_entries)
    {
        if (!path_value.empty())
        {
            path_value.push_back(L';');
        }
        path_value += entry;
    }
    environment[L"PATH"] = std::move(path_value);
    environment[L"CONDA_PREFIX"] = python_root.wstring();
    environment[L"PYTHONNOUSERSITE"] = L"1";
    environment[L"PYTHONUNBUFFERED"] = L"1";
    environment[L"PYTHONUTF8"] = L"1";

    std::vector<wchar_t> block;
    for (const auto& item : environment)
    {
        const std::wstring line = item.first + L"=" + item.second;
        block.insert(block.end(), line.begin(), line.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

void ReportProgressStages(
    const std::filesystem::path& log_path,
    std::size_t& reported_stage_count,
    ProcessWaitObserver& observer)
{
    std::ifstream stream(log_path, std::ios::binary);
    if (!stream)
    {
        return;
    }
    constexpr char prefix[] = "BREPMFR_PROGRESS:";
    std::vector<std::string> stages;
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.rfind(prefix, 0) != 0)
        {
            continue;
        }
        const std::string stage = line.substr(sizeof(prefix) - 1);
        if (!stage.empty())
        {
            stages.push_back(stage);
        }
    }
    while (reported_stage_count < stages.size())
    {
        observer.OnProgressStage(stages[reported_stage_count]);
        ++reported_stage_count;
    }
}

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
    const ProcessLaunchOptions& options,
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
    std::vector<wchar_t> environment;
    if (options.isolate_python_environment)
    {
        environment = BuildIsolatedPythonEnvironment(executable);
    }
    const wchar_t* working_directory = options.working_directory.empty()
        ? nullptr
        : options.working_directory.c_str();
    const DWORD creation_flags = CREATE_NO_WINDOW |
        (environment.empty() ? 0 : CREATE_UNICODE_ENVIRONMENT);
    if (!CreateProcessW(
            executable.c_str(),
            mutable_command.data(),
            nullptr,
            nullptr,
            TRUE,
            creation_flags,
            environment.empty() ? nullptr : environment.data(),
            working_directory,
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
    std::size_t reported_stage_count = 0;

    for (;;)
    {
        const DWORD wait_result = WaitForSingleObject(process.Get(), 100);
        if (wait_result == WAIT_OBJECT_0)
        {
            ReportProgressStages(log_path, reported_stage_count, observer);
            break;
        }
        if (wait_result != WAIT_TIMEOUT)
        {
            const DWORD error_code = GetLastError();
            ThrowWindowsError(error_code, L"WaitForSingleObject failed");
        }
        ReportProgressStages(log_path, reported_stage_count, observer);
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

ProcessExecution RunHiddenProcess(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::filesystem::path& log_path,
    ProcessWaitObserver& observer)
{
    return RunHiddenProcess(
        executable, arguments, log_path, ProcessLaunchOptions{}, observer);
}
}
}
