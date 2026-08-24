#include "inference_command.hpp"

#include <cwctype>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace brepmfr
{
namespace nx2212
{
std::wstring QuoteWindowsArgument(std::wstring_view argument)
{
    bool requires_quotes = argument.empty();
    for (const wchar_t character : argument)
    {
        if (std::iswspace(character) || character == L'"')
        {
            requires_quotes = true;
            break;
        }
    }
    if (!requires_quotes)
    {
        return std::wstring(argument);
    }

    std::wstring quoted;
    quoted.push_back(L'"');
    std::size_t backslashes = 0;
    for (const wchar_t character : argument)
    {
        if (character == L'\\')
        {
            ++backslashes;
            continue;
        }
        if (character == L'"')
        {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(L'"');
        }
        else
        {
            quoted.append(backslashes, L'\\');
            quoted.push_back(character);
        }
        backslashes = 0;
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

std::vector<std::wstring> BuildInferenceArguments(
    const PluginConfig& config,
    const std::filesystem::path& step_path,
    const OutputPaths& output)
{
    std::ostringstream threshold_stream;
    threshold_stream.imbue(std::locale::classic());
    threshold_stream
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << config.affinity_threshold;
    const auto threshold_text = threshold_stream.str();

    return {
        config.inference_script.wstring(),
        L"--step",
        step_path.wstring(),
        L"--checkpoint",
        config.checkpoint.wstring(),
        L"--output-dir",
        output.directory.wstring(),
        L"--device",
        std::wstring(config.device.begin(), config.device.end()),
        L"--affinity-threshold",
        std::wstring(threshold_text.begin(), threshold_text.end())};
}

std::wstring BuildWindowsCommandLine(
    const std::filesystem::path& executable,
    const std::vector<std::wstring>& arguments)
{
    std::wstring command_line =
        QuoteWindowsArgument(executable.wstring());
    for (const auto& argument : arguments)
    {
        command_line.push_back(L' ');
        command_line += QuoteWindowsArgument(argument);
    }
    return command_line;
}
}
}
