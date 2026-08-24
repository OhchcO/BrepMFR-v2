#pragma once

#include "output_paths.hpp"
#include "plugin_config.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
std::wstring QuoteWindowsArgument(std::wstring_view argument);
std::vector<std::wstring> BuildInferenceArguments(
    const PluginConfig& config,
    const std::filesystem::path& step_path,
    const OutputPaths& output);
std::wstring BuildWindowsCommandLine(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments);
}
}
