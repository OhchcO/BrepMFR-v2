#include "plugin_runtime.hpp"

#include "output_paths.hpp"

#include "test_support.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace
{
nlohmann::json SuccessDocument(const std::filesystem::path& step_path)
{
    return {
        {"schema_version", "1.0"},
        {"status", "success"},
        {"input",
         {{"step_path", step_path.generic_u8string()},
          {"step_sha256", "abc"},
          {"face_count", 2}}},
        {"cache",
         {{"bin_path", "D:/output/input.bin"},
          {"metadata_path", "D:/output/input.bin.meta.json"},
          {"hit", true},
          {"fingerprint", "fp"}}},
        {"model",
         {{"checkpoint_path", "D:/model.ckpt"},
          {"checkpoint_sha256", "def"},
          {"checkpoint_epoch", 187},
          {"experiment_mode", "dynamic_affinity"},
          {"device", "cuda"},
          {"num_classes", 25},
          {"affinity_threshold", 0.5}}},
        {"faces",
         {{{"face_index", 0},
           {"class_id", 24},
           {"feature_name", "stock"},
           {"confidence", 0.99},
           {"instance_id", nullptr}},
          {{"face_index", 1},
           {"class_id", 12},
           {"feature_name", "blind_hole"},
           {"confidence", 0.95},
           {"instance_id", 0}}}},
        {"instances",
         {{{"instance_id", 0},
           {"class_id", 12},
           {"feature_name", "blind_hole"},
           {"face_indices", nlohmann::json::array({1})},
           {"confidence", 0.95}}}},
        {"timings_seconds",
         {{"checkpoint_load", 1.0},
          {"conversion", 2.0},
          {"inference", 3.0},
          {"total", 6.0}}},
        {"warnings", nlohmann::json::array()}};
}

nlohmann::json ErrorDocument(const std::filesystem::path& step_path)
{
    return {
        {"schema_version", "1.0"},
        {"status", "error"},
        {"stage", "conversion"},
        {"error", {{"type", "ValueError"}, {"message", "shape mismatch"}}},
        {"input", {{"step_path", step_path.generic_u8string()}}}};
}

struct ModuleFixture
{
    explicit ModuleFixture(brepmfr::nx2212::test::TemporaryDirectory& temporary)
        : module_directory(temporary.Path() / "module"),
          python(module_directory / "python.exe"),
          script(module_directory / "infer_step.py"),
          checkpoint(module_directory / "model.ckpt"),
          output_root(temporary.Path() / "recognition-output")
    {
        std::filesystem::create_directories(module_directory);
        temporary.WriteFile("module/python.exe", "");
        temporary.WriteFile("module/infer_step.py", "");
        temporary.WriteFile("module/model.ckpt", "");
        const nlohmann::json config = {
            {"schema_version", "1.0"},
            {"python_executable", python.generic_u8string()},
            {"inference_script", script.generic_u8string()},
            {"checkpoint", checkpoint.generic_u8string()},
            {"output_root", output_root.generic_u8string()},
            {"device", "auto"},
            {"affinity_threshold", 0.5}};
        temporary.WriteUtf8(
            "module/BrepMFRNx2212.config.json", config.dump(2));
    }

    std::filesystem::path module_directory;
    std::filesystem::path python;
    std::filesystem::path script;
    std::filesystem::path checkpoint;
    std::filesystem::path output_root;
};

class RecordingHost : public brepmfr::nx2212::PluginHost
{
public:
    explicit RecordingHost(std::filesystem::path module_directory)
        : module_directory_(std::move(module_directory))
    {
    }

    std::filesystem::path ModuleDirectory() const override
    {
        return module_directory_;
    }

    std::optional<std::filesystem::path> SelectStepFile() override
    {
        ++select_calls;
        return selected_step;
    }

    unsigned long RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path) override
    {
        ++process_calls;
        process_executable = executable;
        process_arguments = arguments;
        process_log_path = log_path;
        if (process_behavior)
        {
            return process_behavior(executable, arguments, log_path);
        }
        return process_exit_code;
    }

    void ShowSuccess(const brepmfr::nx2212::DisplayText& display) override
    {
        ++success_calls;
        success_display = display;
    }

    void ShowError(const std::string& message_utf8) noexcept override
    {
        ++error_calls;
        error_message = message_utf8;
    }

    std::filesystem::path module_directory_;
    std::optional<std::filesystem::path> selected_step;
    unsigned long process_exit_code = 0;
    std::function<unsigned long(
        const std::filesystem::path&,
        const std::vector<std::wstring>&,
        const std::filesystem::path&)> process_behavior;
    int select_calls = 0;
    int process_calls = 0;
    int success_calls = 0;
    int error_calls = 0;
    std::filesystem::path process_executable;
    std::vector<std::wstring> process_arguments;
    std::filesystem::path process_log_path;
    brepmfr::nx2212::DisplayText success_display;
    std::string error_message;
};
}

BREPMFR_TEST(InferenceWorkflowTreatsFileSelectionCancellationAsSuccess)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    host.selected_step = std::nullopt;

    const int result = RunInferenceWorkflow(host);

    ExpectEqual(result, kSuccess, "cancellation returned failure");
    ExpectEqual(host.select_calls, 1, "STEP selector call count mismatch");
    ExpectEqual(host.process_calls, 0, "cancellation launched Python");
    ExpectEqual(host.success_calls, 0, "cancellation displayed success");
    ExpectEqual(host.error_calls, 0, "cancellation displayed an error");
}

BREPMFR_TEST(InferenceWorkflowRunsPythonAndDisplaysFreshSuccess)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    const auto selected_step = temporary.WriteUtf8("parts/sample.step", "STEP");
    host.selected_step = selected_step;
    const auto output = BuildOutputPaths(module.output_root, selected_step);
    std::filesystem::create_directories(output.directory);
    temporary.WriteUtf8(
        output.result_path.lexically_relative(temporary.Path()),
        "stale result must be removed");

    const std::vector<std::wstring> expected_arguments = {
        module.script.wstring(),
        L"--step",
        selected_step.wstring(),
        L"--checkpoint",
        module.checkpoint.wstring(),
        L"--output-dir",
        output.directory.wstring(),
        L"--device",
        L"auto",
        L"--affinity-threshold",
        L"0.5"};
    host.process_behavior = [&](const std::filesystem::path& executable,
                                const std::vector<std::wstring>& arguments,
                                const std::filesystem::path& log_path) {
        ExpectEqual(executable, module.python, "Python executable mismatch");
        ExpectEqual(arguments, expected_arguments, "Python arguments mismatch");
        ExpectEqual(log_path, output.log_path, "log path mismatch");
        Expect(
            std::filesystem::is_directory(output.directory),
            "output directory was not created");
        Expect(
            !std::filesystem::exists(output.result_path),
            "stale result.json was not removed before launch");
        temporary.WriteUtf8(
            output.result_path.lexically_relative(temporary.Path()),
            SuccessDocument(selected_step).dump(2));
        return 0UL;
    };

    const int workflow_result = RunInferenceWorkflow(host);

    ExpectEqual(workflow_result, kSuccess, "successful workflow returned failure");
    ExpectEqual(host.process_calls, 1, "Python process call count mismatch");
    ExpectEqual(host.success_calls, 1, "success display call count mismatch");
    ExpectEqual(host.error_calls, 0, "successful workflow displayed an error");
    Expect(
        host.success_display.summary_utf8.find("面数量：2") != std::string::npos,
        "success display did not contain parsed result");
}

BREPMFR_TEST(InferenceWorkflowReportsAllPostLaunchFailureBoundaries)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    enum class Outcome
    {
        NonzeroError,
        ZeroError,
        NonzeroSuccess,
        MissingJson,
        MalformedJson,
        StandardException,
        UnknownException
    };
    struct Case
    {
        const char* name;
        Outcome outcome;
        std::vector<std::string> expected_text;
    };
    const std::vector<Case> cases = {
        {"nonzero plus error",
         Outcome::NonzeroError,
         {"conversion", "ValueError", "shape mismatch", "7"}},
        {"zero plus error",
         Outcome::ZeroError,
         {"status=error", "exit code 0"}},
        {"nonzero plus success",
         Outcome::NonzeroSuccess,
         {"status=success", "exit code 7"}},
        {"missing fresh JSON",
         Outcome::MissingJson,
         {"could not open result.json", "Python exit code: 0"}},
        {"malformed JSON",
         Outcome::MalformedJson,
         {"JSON parse failed", "Python exit code: 0"}},
        {"standard exception",
         Outcome::StandardException,
         {"synthetic process failure"}},
        {"unknown exception",
         Outcome::UnknownException,
         {"unknown error"}}};

    for (const auto& test_case : cases)
    {
        TemporaryDirectory temporary;
        ModuleFixture module(temporary);
        RecordingHost host(module.module_directory);
        const auto selected_step = temporary.WriteUtf8("part.step", "STEP");
        host.selected_step = selected_step;
        const auto output = BuildOutputPaths(module.output_root, selected_step);
        host.process_behavior = [&](const std::filesystem::path&,
                                    const std::vector<std::wstring>&,
                                    const std::filesystem::path&) -> unsigned long {
            const auto relative =
                output.result_path.lexically_relative(temporary.Path());
            switch (test_case.outcome)
            {
            case Outcome::NonzeroError:
                temporary.WriteUtf8(
                    relative, ErrorDocument(selected_step).dump(2));
                return 7;
            case Outcome::ZeroError:
                temporary.WriteUtf8(
                    relative, ErrorDocument(selected_step).dump(2));
                return 0;
            case Outcome::NonzeroSuccess:
                temporary.WriteUtf8(
                    relative, SuccessDocument(selected_step).dump(2));
                return 7;
            case Outcome::MissingJson:
                return 0;
            case Outcome::MalformedJson:
                temporary.WriteUtf8(relative, "{broken json");
                return 0;
            case Outcome::StandardException:
                throw std::runtime_error("synthetic process failure");
            case Outcome::UnknownException:
                throw 42;
            }
            throw std::runtime_error("unhandled test outcome");
        };

        const int workflow_result = RunInferenceWorkflow(host);
        ExpectEqual(
            workflow_result,
            kFailure,
            std::string(test_case.name) + " returned success");
        ExpectEqual(
            host.success_calls,
            0,
            std::string(test_case.name) + " displayed success");
        ExpectEqual(
            host.error_calls,
            1,
            std::string(test_case.name) + " error call count mismatch");
        for (const auto& expected : test_case.expected_text)
        {
            Expect(
                host.error_message.find(expected) != std::string::npos,
                std::string(test_case.name) + " omitted '" + expected + "'");
        }
        Expect(
            host.error_message.find(output.result_path.generic_u8string()) !=
                std::string::npos,
            std::string(test_case.name) + " omitted result path");
        Expect(
            host.error_message.find(output.log_path.generic_u8string()) !=
                std::string::npos,
            std::string(test_case.name) + " omitted log path");
    }
}

BREPMFR_TEST(InferenceWorkflowReportsConfigPathBeforeOutputDerivation)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto module_directory = temporary.Path() / "missing-module";
    RecordingHost host(module_directory);
    const auto expected_config =
        module_directory / "BrepMFRNx2212.config.json";

    const int workflow_result = RunInferenceWorkflow(host);

    ExpectEqual(workflow_result, kFailure, "missing config returned success");
    ExpectEqual(host.process_calls, 0, "missing config launched Python");
    ExpectEqual(host.error_calls, 1, "missing config error call count mismatch");
    Expect(
        host.error_message.find(expected_config.generic_u8string()) !=
            std::string::npos,
        "configuration failure did not identify the config path");
}
