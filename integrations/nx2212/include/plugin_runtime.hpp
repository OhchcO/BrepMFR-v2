#pragma once

#include "result_formatter.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

class PluginHost
{
public:
    virtual ~PluginHost() = default;
    virtual std::filesystem::path ModuleDirectory() const = 0;
    virtual std::optional<std::filesystem::path> SelectStepFile() = 0;
    virtual unsigned long RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path) = 0;
    virtual void ShowSuccess(const DisplayText& display) = 0;
    virtual void ShowError(const std::string& message_utf8) noexcept = 0;
};

int RunInferenceWorkflow(PluginHost& host) noexcept;
}
}
