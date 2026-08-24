#include "plugin_config.hpp"
#include "test_support.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace
{
using brepmfr::nx2212::test::Expect;
using brepmfr::nx2212::test::ExpectEqual;
using brepmfr::nx2212::test::TemporaryDirectory;

std::string CompleteConfigJson(
    const std::filesystem::path& python,
    const std::filesystem::path& script,
    const std::filesystem::path& checkpoint,
    const std::filesystem::path& output,
    const std::string& device,
    double threshold)
{
    return std::string("{") +
        "\"schema_version\":\"1.0\"," +
        "\"python_executable\":\"" + python.generic_u8string() + "\"," +
        "\"inference_script\":\"" + script.generic_u8string() + "\"," +
        "\"checkpoint\":\"" + checkpoint.generic_u8string() + "\"," +
        "\"output_root\":\"" + output.generic_u8string() + "\"," +
        "\"device\":\"" + device + "\"," +
        "\"affinity_threshold\":" + std::to_string(threshold) +
        "}";
}

void ExpectRejected(
    TemporaryDirectory& temporary,
    const nlohmann::json& document,
    const std::string& expected_field,
    const std::string& case_name)
{
    const auto config = temporary.WriteUtf8(
        L"BrepMFRNx2212.invalid.config.json", document.dump());
    try
    {
        static_cast<void>(brepmfr::nx2212::LoadPluginConfig(config));
    }
    catch (const std::exception& exception)
    {
        Expect(
            std::string(exception.what()).find(expected_field) !=
                std::string::npos,
            case_name + " did not name field '" + expected_field +
                "': " + exception.what());
        return;
    }
    throw std::runtime_error(case_name + " was accepted");
}
}

BREPMFR_TEST(load_plugin_config_accepts_complete_version_1_document)
{
    TemporaryDirectory temporary;
    const auto python = temporary.WriteFile(L"python.exe", "");
    const auto script = temporary.WriteFile(L"infer_step.py", "");
    const auto checkpoint = temporary.WriteFile(L"model.ckpt", "");
    const auto output = temporary.Path() / L"output";
    const auto config = temporary.WriteUtf8(
        L"BrepMFRNx2212.config.json",
        CompleteConfigJson(
            python, script, checkpoint, output, "auto", 0.5));

    const auto loaded = brepmfr::nx2212::LoadPluginConfig(config);

    ExpectEqual(
        loaded.python_executable, python, "python path changed");
    ExpectEqual(
        loaded.inference_script, script, "inference script changed");
    ExpectEqual(
        loaded.checkpoint, checkpoint, "checkpoint path changed");
    ExpectEqual(
        loaded.output_root, output, "output root changed");
    ExpectEqual(
        loaded.device, std::string("auto"), "device changed");
    Expect(
        loaded.affinity_threshold == 0.5, "threshold changed");
}

BREPMFR_TEST(load_plugin_config_rejects_invalid_fields)
{
    TemporaryDirectory temporary;
    const auto python = temporary.WriteFile(L"python.exe", "");
    const auto script = temporary.WriteFile(L"infer_step.py", "");
    const auto checkpoint = temporary.WriteFile(L"model.ckpt", "");
    const auto output = temporary.Path() / L"output";
    const auto valid = nlohmann::json::parse(CompleteConfigJson(
        python, script, checkpoint, output, "auto", 0.5));

    struct Case
    {
        std::string name;
        std::string field;
        std::function<void(nlohmann::json&)> mutate;
    };

    const auto missing_python =
        (temporary.Path() / L"missing-python.exe").generic_u8string();
    const auto wrong_checkpoint =
        temporary.WriteFile(L"model.bin", "").generic_u8string();
    const auto output_file =
        temporary.WriteFile(L"not-a-directory", "").generic_u8string();

    const std::vector<Case> cases{
        {"missing schema", "schema_version",
         [](nlohmann::json& value) { value.erase("schema_version"); }},
        {"unknown schema", "schema_version",
         [](nlohmann::json& value) { value["schema_version"] = "2.0"; }},
        {"relative python", "python_executable",
         [](nlohmann::json& value)
         { value["python_executable"] = "python.exe"; }},
        {"missing python", "python_executable",
         [missing_python](nlohmann::json& value)
         { value["python_executable"] = missing_python; }},
        {"checkpoint suffix", "checkpoint",
         [wrong_checkpoint](nlohmann::json& value)
         { value["checkpoint"] = wrong_checkpoint; }},
        {"device", "device",
         [](nlohmann::json& value) { value["device"] = "tpu"; }},
        {"negative threshold", "affinity_threshold",
         [](nlohmann::json& value)
         { value["affinity_threshold"] = -0.01; }},
        {"large threshold", "affinity_threshold",
         [](nlohmann::json& value)
         { value["affinity_threshold"] = 1.01; }},
        {"string threshold", "affinity_threshold",
         [](nlohmann::json& value)
         { value["affinity_threshold"] = "0.5"; }},
        {"output file", "output_root",
         [output_file](nlohmann::json& value)
         { value["output_root"] = output_file; }},
    };

    for (const auto& item : cases)
    {
        auto document = valid;
        item.mutate(document);
        ExpectRejected(temporary, document, item.field, item.name);
    }
}
