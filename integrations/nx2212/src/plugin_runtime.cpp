#include "plugin_runtime.hpp"

#include "inference_command.hpp"
#include "output_paths.hpp"
#include "plugin_config.hpp"
#include "result_reader.hpp"

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
        const auto selected_step = host.SelectStepFile();
        if (!selected_step)
        {
            return kSuccess;
        }

        output = BuildOutputPaths(config.output_root, *selected_step);
        std::filesystem::create_directories(output->directory);
        std::filesystem::remove(output->result_path);
        const auto arguments =
            BuildInferenceArguments(config, *selected_step, *output);
        ProcessLaunchOptions launch_options;
        launch_options.working_directory =
            config.inference_script.parent_path().parent_path();
        launch_options.display_input_path = *selected_step;
        launch_options.isolate_python_environment = true;
        process_execution = host.RunProcess(
            config.python_executable,
            arguments,
            output->log_path,
            launch_options);
        process_exit_code = process_execution->exit_code;
        const unsigned long exit_code = *process_exit_code;
        const auto result =
            ReadInferenceResult(output->result_path, *selected_step);

        if (std::holds_alternative<SuccessResult>(result))
        {
            if (exit_code != 0)
            {
                throw std::runtime_error(
                    "Integration mismatch: Python exit code " +
                    std::to_string(exit_code) +
                    " but result.json has status=success.");
            }
            host.ShowSuccess(FormatSuccessDisplay(
                std::get<SuccessResult>(result),
                output->result_path,
                process_execution->elapsed));
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
