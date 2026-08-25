#include "progress_window.hpp"

#include "progress_layout.hpp"

#include <windows.h>
#include <commctrl.h>
#include <stdexcept>
#include <string>

namespace brepmfr
{
namespace nx2212
{
namespace
{
HWND AsWindow(void* value) noexcept
{
    return reinterpret_cast<HWND>(value);
}

void ApplyDefaultFont(HWND control)
{
    SendMessageW(
        control,
        WM_SETFONT,
        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)),
        TRUE);
}

POINT CenteredPosition(HWND owner, int width, int height)
{
    RECT bounds{};
    if (owner != nullptr && IsWindow(owner) && GetWindowRect(owner, &bounds))
    {
        return {
            bounds.left + ((bounds.right - bounds.left) - width) / 2,
            bounds.top + ((bounds.bottom - bounds.top) - height) / 2};
    }
    return {
        (GetSystemMetrics(SM_CXSCREEN) - width) / 2,
        (GetSystemMetrics(SM_CYSCREEN) - height) / 2};
}

UINT WindowDpi(HWND owner)
{
    using GetDpiForWindowFunction = UINT(WINAPI*)(HWND);
    const auto get_dpi_for_window =
        reinterpret_cast<GetDpiForWindowFunction>(GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "GetDpiForWindow"));
    if (owner != nullptr && get_dpi_for_window != nullptr)
    {
        const UINT dpi = get_dpi_for_window(owner);
        if (dpi != 0)
        {
            return dpi;
        }
    }

    HDC screen = GetDC(nullptr);
    if (screen == nullptr)
    {
        return 96;
    }
    const int dpi = GetDeviceCaps(screen, LOGPIXELSX);
    ReleaseDC(nullptr, screen);
    return dpi > 0 ? static_cast<UINT>(dpi) : 96;
}

SIZE OuterWindowSize(
    int client_width,
    int client_height,
    DWORD style,
    DWORD extended_style,
    UINT dpi)
{
    RECT bounds{0, 0, client_width, client_height};
    using AdjustWindowRectExForDpiFunction = BOOL(WINAPI*)(
        LPRECT, DWORD, BOOL, DWORD, UINT);
    const auto adjust_for_dpi =
        reinterpret_cast<AdjustWindowRectExForDpiFunction>(GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "AdjustWindowRectExForDpi"));
    const BOOL adjusted = adjust_for_dpi != nullptr
        ? adjust_for_dpi(&bounds, style, FALSE, extended_style, dpi)
        : AdjustWindowRectEx(&bounds, style, FALSE, extended_style);
    if (!adjusted)
    {
        return {client_width, client_height};
    }
    return {bounds.right - bounds.left, bounds.bottom - bounds.top};
}

HWND CreateLabel(
    HWND parent,
    const wchar_t* text,
    const ProgressRect& rect,
    DWORD additional_style = 0)
{
    return CreateWindowExW(
        0,
        L"STATIC",
        text,
        WS_CHILD | WS_VISIBLE | SS_LEFT | additional_style,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        parent,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
}
}

ProgressWindow::ProgressWindow(
    void* owner_window,
    const std::filesystem::path& input_path)
{
    INITCOMMONCONTROLSEX controls{};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_PROGRESS_CLASS;
    if (!InitCommonControlsEx(&controls))
    {
        throw std::runtime_error("Could not initialize the Windows progress control.");
    }

    HWND owner = AsWindow(owner_window);
    if (owner != nullptr && !IsWindow(owner))
    {
        owner = nullptr;
    }

    const UINT dpi = WindowDpi(owner);
    const ProgressWindowLayout layout = CalculateProgressWindowLayout(dpi);
    constexpr DWORD extended_style = WS_EX_DLGMODALFRAME;
    constexpr DWORD style = WS_POPUP | WS_CAPTION | WS_CLIPCHILDREN;
    const SIZE outer = OuterWindowSize(
        layout.client_width,
        layout.client_height,
        style,
        extended_style,
        dpi);
    const POINT position = CenteredPosition(owner, outer.cx, outer.cy);
    HWND window = CreateWindowExW(
        extended_style,
        L"STATIC",
        L"BrepMFR \u7279\u5f81\u8bc6\u522b",
        style,
        position.x,
        position.y,
        outer.cx,
        outer.cy,
        owner,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    if (window == nullptr)
    {
        throw std::runtime_error("Could not create the inference progress window.");
    }

    HWND status = CreateLabel(
        window,
        L"\u6b63\u5728\u8bc6\u522b\uff0c\u8bf7\u7a0d\u5019\u2026\u2026",
        layout.status);
    const std::wstring input_text = L"STEP: " + input_path.wstring();
    HWND input_label = CreateLabel(
        window,
        input_text.c_str(),
        layout.input,
        SS_PATHELLIPSIS);
    const std::wstring initial_stage = ProgressStageText("startup");
    HWND stage_label = CreateLabel(
        window,
        initial_stage.c_str(),
        layout.stage);
    HWND progress = CreateWindowExW(
        0,
        PROGRESS_CLASSW,
        nullptr,
        WS_CHILD | WS_VISIBLE | PBS_MARQUEE,
        layout.progress.x,
        layout.progress.y,
        layout.progress.width,
        layout.progress.height,
        window,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    HWND elapsed_label = CreateLabel(
        window,
        L"\u5df2\u7528\u65f6\u95f4\uff1a0 \u79d2",
        layout.elapsed);
    if (status == nullptr || input_label == nullptr ||
        stage_label == nullptr || progress == nullptr ||
        elapsed_label == nullptr)
    {
        DestroyWindow(window);
        throw std::runtime_error("Could not create the inference progress controls.");
    }

    ApplyDefaultFont(status);
    ApplyDefaultFont(input_label);
    ApplyDefaultFont(stage_label);
    ApplyDefaultFont(elapsed_label);
    SendMessageW(progress, PBM_SETMARQUEE, TRUE, 35);
    window_ = window;
    progress_ = progress;
    stage_label_ = stage_label;
    elapsed_label_ = elapsed_label;
}

ProgressWindow::~ProgressWindow()
{
    OnProcessFinished();
}

void ProgressWindow::OnProcessStarted()
{
    HWND window = AsWindow(window_);
    if (window != nullptr)
    {
        ShowWindow(window, SW_SHOWNORMAL);
        UpdateWindow(window);
    }
}

void ProgressWindow::OnWaitTick(std::chrono::milliseconds elapsed)
{
    const long long elapsed_seconds = elapsed.count() / 1000;
    if (elapsed_seconds != last_elapsed_seconds_)
    {
        last_elapsed_seconds_ = elapsed_seconds;
        wchar_t text[64]{};
        _snwprintf_s(
            text,
            _countof(text),
            _TRUNCATE,
            L"\u5df2\u7528\u65f6\u95f4\uff1a%lld \u79d2",
            elapsed_seconds);
        SetWindowTextW(AsWindow(elapsed_label_), text);
    }

    MSG message{};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
    {
        if (message.message == WM_QUIT)
        {
            PostQuitMessage(static_cast<int>(message.wParam));
            break;
        }
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

void ProgressWindow::OnProgressStage(const std::string& stage)
{
    HWND stage_label = AsWindow(stage_label_);
    if (stage_label == nullptr)
    {
        return;
    }
    const std::wstring text = ProgressStageText(stage);
    SetWindowTextW(stage_label, text.c_str());
    UpdateWindow(stage_label);
}

void ProgressWindow::OnProcessFinished() noexcept
{
    HWND window = AsWindow(window_);
    if (window == nullptr)
    {
        return;
    }
    SendMessageW(AsWindow(progress_), PBM_SETMARQUEE, FALSE, 0);
    DestroyWindow(window);
    window_ = nullptr;
    progress_ = nullptr;
    stage_label_ = nullptr;
    elapsed_label_ = nullptr;
}
}
}
