#pragma once

#include "plugin_runtime.hpp"

#include <filesystem>

namespace brepmfr
{
namespace nx2212
{
void ShowNxInstanceBrowser(
    const std::filesystem::path& dlx_path,
    const PartSnapshot& part,
    const InstanceBrowserModel& model);
}
}
