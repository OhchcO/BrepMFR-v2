#include "plugin_config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace brepmfr
{
namespace nx2212
{
namespace
{
using Json = nlohmann::json;

std::runtime_error ConfigError(
    const std::filesystem::path& config_path,
    const std::string& message)
{
    return std::runtime_error(
        "Configuration '" + config_path.generic_u8string() + "': " + message);
}

std::string RequiredString(
    const Json& document,
    const char* field,
    const std::filesystem::path& config_path)
{
    const auto iterator = document.find(field);
    if (iterator == document.end() || !iterator->is_string())
    {
        throw ConfigError(
            config_path,
            std::string("field '") + field + "' must be a string");
    }
    return iterator->get<std::string>();
}

std::filesystem::path RequiredAbsolutePath(
    const Json& document,
    const char* field,
    const std::filesystem::path& config_path)
{
    const auto value = std::filesystem::u8path(
        RequiredString(document, field, config_path));
    if (!value.is_absolute())
    {
        throw ConfigError(
            config_path,
            std::string("field '") + field + "' must be an absolute path");
    }
    return value.lexically_normal();
}

void RequireRegularFile(
    const std::filesystem::path& value,
    const char* field,
    const std::filesystem::path& config_path)
{
    std::error_code error;
    if (!std::filesystem::is_regular_file(value, error) || error)
    {
        throw ConfigError(
            config_path,
            std::string("field '") + field +
                "' must name an existing file: " + value.generic_u8string());
    }
}

std::wstring LowerAscii(std::wstring value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](wchar_t character)
        {
            return character >= L'A' && character <= L'Z'
                ? static_cast<wchar_t>(character - L'A' + L'a')
                : character;
        });
    return value;
}

double RequiredNumber(
    const Json& document,
    const char* field,
    const std::filesystem::path& config_path)
{
    const auto iterator = document.find(field);
    if (iterator == document.end() || !iterator->is_number())
    {
        throw ConfigError(
            config_path,
            std::string("field '") + field + "' must be a number");
    }
    return iterator->get<double>();
}
}

PluginConfig LoadPluginConfig(const std::filesystem::path& config_path)
{
    Json document;
    try
    {
        std::ifstream stream(config_path, std::ios::binary);
        if (!stream)
        {
            throw ConfigError(config_path, "could not open the file");
        }
        document = Json::parse(stream);
    }
    catch (const nlohmann::json::exception& error)
    {
        throw ConfigError(
            config_path, std::string("invalid JSON: ") + error.what());
    }

    const auto schema_version =
        RequiredString(document, "schema_version", config_path);
    if (schema_version != "1.0")
    {
        throw ConfigError(
            config_path,
            "field 'schema_version' must equal '1.0'");
    }

    PluginConfig config{
        RequiredAbsolutePath(
            document, "python_executable", config_path),
        RequiredAbsolutePath(
            document, "inference_script", config_path),
        RequiredAbsolutePath(document, "checkpoint", config_path),
        RequiredAbsolutePath(document, "output_root", config_path),
        RequiredString(document, "device", config_path),
        RequiredNumber(document, "affinity_threshold", config_path)};

    RequireRegularFile(
        config.python_executable, "python_executable", config_path);
    RequireRegularFile(
        config.inference_script, "inference_script", config_path);
    RequireRegularFile(config.checkpoint, "checkpoint", config_path);

    if (LowerAscii(config.checkpoint.extension().wstring()) != L".ckpt")
    {
        throw ConfigError(
            config_path,
            "field 'checkpoint' must use the .ckpt suffix");
    }

    std::error_code output_error;
    const bool output_exists =
        std::filesystem::exists(config.output_root, output_error);
    if (output_error)
    {
        throw ConfigError(
            config_path, "field 'output_root' could not be inspected");
    }
    if (output_exists)
    {
        const bool output_is_directory =
            std::filesystem::is_directory(config.output_root, output_error);
        if (output_error || !output_is_directory)
        {
            throw ConfigError(
                config_path,
                "field 'output_root' must be a directory when it exists");
        }
    }

    if (config.device != "auto" &&
        config.device != "cpu" &&
        config.device != "cuda")
    {
        throw ConfigError(
            config_path,
            "field 'device' must be one of auto, cpu, or cuda");
    }

    if (!std::isfinite(config.affinity_threshold) ||
        config.affinity_threshold < 0.0 ||
        config.affinity_threshold > 1.0)
    {
        throw ConfigError(
            config_path,
            "field 'affinity_threshold' must be finite and in [0, 1]");
    }
    return config;
}
}
}
