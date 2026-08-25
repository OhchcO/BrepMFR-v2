#include "progress_layout.hpp"

namespace brepmfr
{
namespace nx2212
{
namespace
{
int Scale(int value, unsigned int dpi)
{
    const unsigned int effective_dpi = dpi == 0 ? 96 : dpi;
    return static_cast<int>(
        static_cast<long long>(value) * effective_dpi / 96);
}

ProgressRect ScaleRect(
    int x,
    int y,
    int width,
    int height,
    unsigned int dpi)
{
    return {
        Scale(x, dpi),
        Scale(y, dpi),
        Scale(width, dpi),
        Scale(height, dpi)};
}
}

ProgressWindowLayout CalculateProgressWindowLayout(unsigned int dpi)
{
    ProgressWindowLayout layout;
    layout.client_width = Scale(520, dpi);
    layout.client_height = Scale(190, dpi);
    layout.status = ScaleRect(22, 18, 476, 24, dpi);
    layout.input = ScaleRect(22, 48, 476, 22, dpi);
    layout.stage = ScaleRect(22, 78, 476, 22, dpi);
    layout.progress = ScaleRect(22, 108, 476, 24, dpi);
    layout.elapsed = ScaleRect(22, 145, 476, 24, dpi);
    return layout;
}

std::wstring ProgressStageText(std::string_view stage)
{
    if (stage == "startup")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u542f\u52a8 Python";
    if (stage == "importing")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u5bfc\u5165\u63a8\u7406\u7ec4\u4ef6";
    if (stage == "checkpoint")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u52a0\u8f7d\u6a21\u578b";
    if (stage == "cache")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u68c0\u67e5 BIN \u7f13\u5b58";
    if (stage == "conversion")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u5c06 STEP \u8f6c\u6362\u4e3a BIN";
    if (stage == "loading")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u52a0\u8f7d BIN \u6570\u636e";
    if (stage == "inference")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u6267\u884c GPU \u63a8\u7406";
    if (stage == "postprocess")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u6574\u7406\u8bc6\u522b\u5b9e\u4f8b";
    if (stage == "serialization")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u5199\u5165 JSON \u7ed3\u679c";
    if (stage == "complete")
        return L"\u5f53\u524d\u9636\u6bb5\uff1a\u8bc6\u522b\u5b8c\u6210";
    return L"\u5f53\u524d\u9636\u6bb5\uff1a\u6b63\u5728\u5904\u7406";
}
}
}
