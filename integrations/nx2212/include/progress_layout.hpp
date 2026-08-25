#pragma once

#include <string>
#include <string_view>

namespace brepmfr
{
namespace nx2212
{
struct ProgressRect
{
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct ProgressWindowLayout
{
    int client_width = 0;
    int client_height = 0;
    ProgressRect status;
    ProgressRect input;
    ProgressRect stage;
    ProgressRect progress;
    ProgressRect elapsed;
};

ProgressWindowLayout CalculateProgressWindowLayout(unsigned int dpi);
std::wstring ProgressStageText(std::string_view stage);
}
}
