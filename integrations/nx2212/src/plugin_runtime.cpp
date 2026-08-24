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
constexpr char kDialogTitle[] = "BrepMFR NX2212";
constexpr char kSuccessMessage[] =
    "BrepMFR NX2212 plug-in loaded successfully.";
constexpr char kUnknownErrorMessage[] =
    "BrepMFR NX2212 plug-in failed with an unknown error.";

std::string AddDiagnosticPaths(
    const std::string& message,
    const std::filesystem::path& config_path,
    const std::optional<OutputPaths>& output)
{
    std::ostringstream stream;
    stream << message;
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
        const unsigned long exit_code = host.RunProcess(
            config.python_executable, arguments, output->log_path);
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
                std::get<SuccessResult>(result), output->result_path));
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
            exception.what(), config_path, output));
    }
    catch (...)
    {
        host.ShowError(AddDiagnosticPaths(
            kUnknownErrorMessage, config_path, output));
    }
    return kFailure;
}

int RunLoadConfirmation(LoadNotifier& notifier) noexcept
{
    try
    {
        notifier.ShowInformation(kDialogTitle, kSuccessMessage);
        return kSuccess;
    }
    catch (const std::exception& exception)
    {
        notifier.ShowError(kDialogTitle, exception.what());
    }
    catch (...)
    {
        notifier.ShowError(kDialogTitle, kUnknownErrorMessage);
    }
    return kFailure;
}
}
}
