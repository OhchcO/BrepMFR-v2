#include "nx_plugin_host.hpp"

#include "inference_process.hpp"
#include "nx_coloring.hpp"
#include "nx_part_adapter.hpp"
#include "progress_window.hpp"

#include <windows.h>

#ifdef CreateDialog
#undef CreateDialog
#endif

#include <NXOpen/ListingWindow.hxx>
#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/NXString.hxx>
#include <NXOpen/Session.hxx>
#include <NXOpen/UI.hxx>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr char kDialogTitleUtf8[] = u8"BrepMFR 特征识别";

class NxUiAccessLock
{
public:
    explicit NxUiAccessLock(NXOpen::UI* ui) : ui_(ui)
    {
        if (ui_ == nullptr)
        {
            throw std::runtime_error("NX UI is unavailable.");
        }
        ui_->LockAccess();
    }

    ~NxUiAccessLock()
    {
        try
        {
            ui_->UnlockAccess();
        }
        catch (...)
        {
        }
    }

    NxUiAccessLock(const NxUiAccessLock&) = delete;
    NxUiAccessLock& operator=(const NxUiAccessLock&) = delete;

private:
    NXOpen::UI* ui_;
};

}

std::filesystem::path NxPluginHost::ModuleDirectory() const
{
    const HMODULE module = reinterpret_cast<HMODULE>(&__ImageBase);
    std::vector<wchar_t> buffer(1024);
    for (;;)
    {
        SetLastError(ERROR_SUCCESS);
        const DWORD length = GetModuleFileNameW(
            module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "GetModuleFileNameW failed for the NX plug-in");
        }
        if (length < buffer.size())
        {
            return std::filesystem::path(
                std::wstring(buffer.data(), length)).parent_path();
        }
        if (buffer.size() >= 32768)
        {
            throw std::runtime_error("NX plug-in module path is too long.");
        }
        buffer.resize((std::min)(buffer.size() * 2, static_cast<std::size_t>(32768)));
    }
}

PartSnapshot NxPluginHost::CaptureCurrentPart()
{
    return CaptureNxCurrentPart();
}

FaceGraph NxPluginHost::ReadCurrentFaceGraph(const PartSnapshot& expected)
{
    return ReadNxFaceGraph(expected);
}

void NxPluginHost::ApplyFaceColorsAtomically(
    const PartSnapshot& expected,
    const std::vector<FaceColorAssignment>& assignments)
{
    ApplyNxFaceColorsAtomically(expected, assignments);
}

ProcessExecution NxPluginHost::RunProcess(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::filesystem::path& log_path,
    const ProcessLaunchOptions& options)
{
    void* owner = GetForegroundWindow();
    NxUiAccessLock lock(NXOpen::UI::GetUI());
    ProgressWindow progress(owner, options.display_input_path);
    return RunHiddenProcess(
        executable, arguments, log_path, options, progress);
}

void NxPluginHost::ShowSuccess(const DisplayText& display)
{
    NXOpen::Session* session = NXOpen::Session::GetSession();
    if (session == nullptr || session->ListingWindow() == nullptr)
    {
        throw std::runtime_error("NX Listing Window is unavailable.");
    }
    NXOpen::ListingWindow* listing = session->ListingWindow();
    listing->Open();
    for (const auto& line : display.listing_lines_utf8)
    {
        listing->WriteFullline(NXOpen::NXString(line, NXOpen::NXString::UTF8));
    }

    NXOpen::UI* ui = NXOpen::UI::GetUI();
    if (ui == nullptr || ui->NXMessageBox() == nullptr)
    {
        throw std::runtime_error("NX message box is unavailable.");
    }
    ui->NXMessageBox()->Show(
        NXOpen::NXString(kDialogTitleUtf8, NXOpen::NXString::UTF8),
        NXOpen::NXMessageBox::DialogTypeInformation,
        NXOpen::NXString(display.summary_utf8, NXOpen::NXString::UTF8));
}

void NxPluginHost::ShowError(const std::string& message_utf8) noexcept
{
    try
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui != nullptr && ui->NXMessageBox() != nullptr)
        {
            ui->NXMessageBox()->Show(
                NXOpen::NXString(kDialogTitleUtf8, NXOpen::NXString::UTF8),
                NXOpen::NXMessageBox::DialogTypeError,
                NXOpen::NXString(message_utf8, NXOpen::NXString::UTF8));
        }
    }
    catch (...)
    {
    }
}
}
}
