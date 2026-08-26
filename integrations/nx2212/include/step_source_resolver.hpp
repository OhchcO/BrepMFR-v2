#pragma once

#include <filesystem>

namespace brepmfr
{
namespace nx2212
{
std::filesystem::path ResolveOpenedStepSource(
    const std::filesystem::path& work_part_path);
}
}
