#pragma once

#include "inference_process.hpp"
#include "result_formatter.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
constexpr int kSuccess = 0;
constexpr int kFailure = 1;

struct PartSnapshot
{
    std::filesystem::path step_path;
    FaceKey part_tag = 0;
    FaceKey body_tag = 0;
    std::size_t face_count = 0;
};

class PluginHost
{
public:
    virtual ~PluginHost() = default;
    virtual std::filesystem::path ModuleDirectory() const = 0;
    virtual PartSnapshot CaptureCurrentPart() = 0;
    virtual FaceGraph ReadCurrentFaceGraph(const PartSnapshot&) = 0;
    virtual void ApplyFaceColorsAtomically(
        const PartSnapshot&,
        const std::vector<FaceColorAssignment>&) = 0;
    virtual ProcessExecution RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path,
        const ProcessLaunchOptions& options) = 0;
    virtual void ShowSuccess(const DisplayText& display) = 0;
    virtual void ShowError(const std::string& message_utf8) noexcept = 0;
};

int RunInferenceWorkflow(PluginHost& host) noexcept;
}
}
