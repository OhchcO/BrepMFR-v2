#pragma once

#include <filesystem>
#include <string>

namespace brepmfr
{
namespace nx2212
{
struct PluginConfig
{
    std::filesystem::path python_executable;
    std::filesystem::path inference_script;
    std::filesystem::path checkpoint;
    std::filesystem::path output_root;
    std::string device;
    double affinity_threshold;
};

PluginConfig LoadPluginConfig(const std::filesystem::path& config_path);
}
}
