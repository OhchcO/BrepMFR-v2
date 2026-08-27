#pragma once

#include "plugin_runtime.hpp"

namespace brepmfr
{
namespace nx2212
{
class NxPluginHost final : public PluginHost
{
public:
    std::filesystem::path ModuleDirectory() const override;
    PartSnapshot CaptureCurrentPart() override;
    FaceGraph ReadCurrentFaceGraph(const PartSnapshot& expected) override;
    void ApplyFaceColorsAtomically(
        const PartSnapshot& expected,
        const std::vector<FaceColorAssignment>& assignments) override;
    ProcessExecution RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path,
        const ProcessLaunchOptions& options) override;
    void ShowSuccess(const DisplayText& display) override;
    void ShowInstanceBrowser(
        const PartSnapshot& part,
        const InstanceBrowserModel& model) override;
    void ShowError(const std::string& message_utf8) noexcept override;
};
}
}
