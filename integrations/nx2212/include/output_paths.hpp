#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace brepmfr
{
namespace nx2212
{
struct OutputPaths
{
    std::filesystem::path directory;
    std::filesystem::path bin_path;
    std::filesystem::path metadata_path;
    std::filesystem::path result_path;
    std::filesystem::path log_path;
};

std::wstring SanitizeStepStem(std::wstring_view stem);
std::uint64_t StablePathHash(
    const std::filesystem::path& absolute_path);
OutputPaths BuildOutputPaths(
    const std::filesystem::path& output_root,
    const std::filesystem::path& step_path);
}
}
