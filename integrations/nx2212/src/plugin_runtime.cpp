#include "plugin_runtime.hpp"

#include "color_plan.hpp"
#include "face_matcher.hpp"
#include "inference_command.hpp"
#include "instance_browser.hpp"
#include "output_paths.hpp"
#include "plugin_config.hpp"
#include "result_reader.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr char kUnknownErrorMessage[] =
    "BrepMFR NX2212 plug-in failed with an unknown error.";

std::chrono::milliseconds ElapsedSince(
    std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
}

MappingDisplay DisplayForPayload(const FaceMappingPayload& payload)
{
    MappingDisplay display;
    display.diagnostic = payload.diagnostic;
    switch (payload.status)
    {
    case MappingPayloadStatus::Unsupported:
        display.status = MappingDisplayStatus::Unsupported;
        if (display.diagnostic.empty())
        {
            display.diagnostic = "JSON 1.0";
        }
        break;
    case MappingPayloadStatus::Unavailable:
        display.status = MappingDisplayStatus::Unavailable;
        break;
    case MappingPayloadStatus::Invalid:
        display.status = MappingDisplayStatus::Failed;
        break;
    case MappingPayloadStatus::Ready:
        display.status = MappingDisplayStatus::Failed;
        break;
    }
    return display;
}

MappingDisplay MapAndColorCurrentPart(
    PluginHost& host,
    const PartSnapshot& part,
    const SuccessResult& recognition)
{
    MappingDisplay display = DisplayForPayload(recognition.mapping);
    if (recognition.mapping.status != MappingPayloadStatus::Ready)
    {
        return display;
    }

    const auto started = std::chrono::steady_clock::now();
    try
    {
        const auto nx_graph = host.ReadCurrentFaceGraph(part);
        const auto mapping = MatchFaces(recognition.mapping.graph, nx_graph);
        display.mapped_faces = mapping.matches.size();
        display.max_error = mapping.max_error;
        if (!mapping.success)
        {
            display.diagnostic = mapping.diagnostic;
            display.elapsed = ElapsedSince(started);
            return display;
        }

        const auto plan = BuildColorPlan(recognition, mapping);
        if (!plan.success)
        {
            display.diagnostic = plan.diagnostic;
            display.elapsed = ElapsedSince(started);
            return display;
        }

        host.ApplyFaceColorsAtomically(part, plan.assignments);
        display.status = MappingDisplayStatus::Colored;
        display.mapped_faces = mapping.matches.size();
        display.colored_instances = plan.instances.size();
        display.instances = plan.instances;
        display.elapsed = ElapsedSince(started);
        return display;
    }
    catch (const std::exception& error)
    {
        display.status = MappingDisplayStatus::Failed;
        display.diagnostic = error.what();
    }
    catch (...)
    {
        display.status = MappingDisplayStatus::Failed;
        display.diagnostic = "NX face mapping failed with an unknown error";
    }
    display.colored_instances = 0;
    display.instances.clear();
    display.elapsed = ElapsedSince(started);
    return display;
}

std::string AddDiagnosticPaths(
    const std::string& message,
    const std::filesystem::path& config_path,
    const std::optional<OutputPaths>& output,
    const std::optional<unsigned long>& process_exit_code)
{
    std::ostringstream stream;
    stream << message;
    if (process_exit_code &&
        message.find("Python exit code") == std::string::npos)
    {
        stream << "\nPython exit code: " << *process_exit_code;
    }
    if (output)
    {
        stream << "\nResult: " << output->result_path.generic_u8string()
               << "\nLog: " << output->log_path.generic_u8string();
    }
    else if (!config_path.empty())
    {
        stream << "\nConfig: " << config_path.generic_u8string();
    }
    return stream.str();
}
}

int RunInferenceWorkflow(PluginHost& host) noexcept
{
    std::filesystem::path config_path;
    std::optional<OutputPaths> output;
    std::optional<unsigned long> process_exit_code;
    std::optional<ProcessExecution> process_execution;
    try
    {
        config_path = host.ModuleDirectory() /
            "BrepMFRNx2212.config.json";
        const auto config = LoadPluginConfig(config_path);
        const auto part = host.CaptureCurrentPart();

        output = BuildOutputPaths(config.output_root, part.step_path);
        std::filesystem::create_directories(output->directory);
        std::filesystem::remove(output->result_path);
        const auto arguments =
            BuildInferenceArguments(config, part.step_path, *output);
        ProcessLaunchOptions launch_options;
        launch_options.working_directory =
            config.inference_script.parent_path().parent_path();
        launch_options.display_input_path = part.step_path;
        launch_options.isolate_python_environment = true;
        process_execution = host.RunProcess(
            config.python_executable,
            arguments,
            output->log_path,
            launch_options);
        process_exit_code = process_execution->exit_code;
        const unsigned long exit_code = *process_exit_code;
        const auto result =
            ReadInferenceResult(output->result_path, part.step_path);

        if (std::holds_alternative<SuccessResult>(result))
        {
            if (exit_code != 0)
            {
                throw std::runtime_error(
                    "Integration mismatch: Python exit code " +
                    std::to_string(exit_code) +
                    " but result.json has status=success.");
            }
            const auto& success = std::get<SuccessResult>(result);
            const auto mapping = MapAndColorCurrentPart(host, part, success);
            const auto display = FormatSuccessDisplay(
                success,
                output->result_path,
                process_execution->elapsed,
                mapping);
            if (mapping.status == MappingDisplayStatus::Colored)
            {
                host.ShowInstanceBrowser(
                    part,
                    BuildInstanceBrowserModel(
                        success,
                        process_execution->elapsed,
                        mapping,
                        display));
            }
            else
            {
                host.ShowSuccess(display);
            }
            return kSuccess;
        }

        const auto& error = std::get<ErrorResult>(result);
        auto error_message = FormatErrorDisplay(
            error, output->result_path, output->log_path);
        if (exit_code == 0)
        {
            error_message =
                "Integration mismatch: Python exit code 0 but result.json "
                "has status=error.\n" + error_message;
        }
        else
        {
            error_message +=
                "\nPython exit code: " + std::to_string(exit_code);
        }
        host.ShowError(error_message);
        return kFailure;
    }
    catch (const std::exception& exception)
    {
        host.ShowError(AddDiagnosticPaths(
            exception.what(), config_path, output, process_exit_code));
    }
    catch (...)
    {
        host.ShowError(AddDiagnosticPaths(
            kUnknownErrorMessage, config_path, output, process_exit_code));
    }
    return kFailure;
}

}
}
