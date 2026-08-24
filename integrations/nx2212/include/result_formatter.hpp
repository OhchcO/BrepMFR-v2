#pragma once

#include "result_reader.hpp"

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
    const std::filesystem::path& result_path);

std::string FormatErrorDisplay(
    const ErrorResult& error,
    const std::filesystem::path& result_path,
    const std::filesystem::path& log_path);
}
}
