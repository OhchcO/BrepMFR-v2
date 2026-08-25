#pragma once

#include "result_reader.hpp"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
struct DisplayText
{
    std::string summary_utf8;
    std::vector<std::string> listing_lines_utf8;
};

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed = std::chrono::milliseconds{0});

std::string FormatErrorDisplay(
    const ErrorResult& error,
    const std::filesystem::path& result_path,
    const std::filesystem::path& log_path);
}
}
