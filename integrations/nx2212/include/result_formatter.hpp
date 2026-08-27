#pragma once

#include "color_plan.hpp"
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

enum class MappingDisplayStatus
{
    Unsupported,
    Unavailable,
    Failed,
    Colored
};

struct MappingDisplay
{
    MappingDisplayStatus status = MappingDisplayStatus::Unsupported;
    std::string diagnostic;
    std::size_t mapped_faces = 0;
    std::size_t colored_instances = 0;
    double max_error = 0.0;
    std::chrono::milliseconds elapsed{0};
    std::vector<InstanceColorRecord> instances;
};

std::string LocalizedFeatureNameUtf8(const std::string& feature_name);

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed = std::chrono::milliseconds{0});

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed,
    const MappingDisplay& mapping);

std::string FormatErrorDisplay(
    const ErrorResult& error,
    const std::filesystem::path& result_path,
    const std::filesystem::path& log_path);
}
}
