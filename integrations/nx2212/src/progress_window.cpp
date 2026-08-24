#include "progress_window.hpp"

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
}

ProgressWindow::ProgressWindow(void* owner_window)
{
    INITCOMMONCONTROLSEX controls{};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_PROGRESS_CLASS;
    if (!InitCommonControlsEx(&controls))
    {
        throw std::runtime_error("Could not initialize the Windows progress control.");
    }

    constexpr int width = 460;
    constexpr int height = 165;
    HWND owner = AsWindow(owner_window);
    if (owner != nullptr && !IsWindow(owner))
    {
        owner = nullptr;
    }
    const POINT position = CenteredPosition(owner, width, height);
    HWND window = CreateWindowExW(
        WS_EX_DLGMODALFRAME,
        L"STATIC",
        L"BrepMFR 特征识别",
        WS_POPUP | WS_CAPTION | WS_CLIPCHILDREN,
        position.x,
        position.y,
        width,
        height,
        owner,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    if (window == nullptr)
    {
        throw std::runtime_error("Could not create the inference progress window.");
    }

    HWND status = CreateWindowExW(
        0,
        L"STATIC",
        L"正在识别，请稍候……",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        22,
        20,
        410,
        24,
        window,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    HWND progress = CreateWindowExW(
        0,
        PROGRESS_CLASSW,
        nullptr,
        WS_CHILD | WS_VISIBLE | PBS_MARQUEE,
        22,
        54,
        410,
        24,
        window,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    HWND elapsed_label = CreateWindowExW(
        0,
        L"STATIC",
        L"已用时间：0 秒",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        22,
        91,
        410,
        24,
        window,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);
    if (status == nullptr || progress == nullptr || elapsed_label == nullptr)
    {
        DestroyWindow(window);
        throw std::runtime_error("Could not create the inference progress controls.");
    }

    ApplyDefaultFont(status);
    ApplyDefaultFont(elapsed_label);
    SendMessageW(progress, PBM_SETMARQUEE, TRUE, 35);
    window_ = window;
    progress_ = progress;
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
            L"已用时间：%lld 秒",
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
    elapsed_label_ = nullptr;
}
}
}
