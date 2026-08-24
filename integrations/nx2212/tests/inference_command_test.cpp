#include "inference_command.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace
{
using brepmfr::nx2212::BuildInferenceArguments;
using brepmfr::nx2212::BuildWindowsCommandLine;
using brepmfr::nx2212::OutputPaths;
using brepmfr::nx2212::PluginConfig;
using brepmfr::nx2212::QuoteWindowsArgument;
using brepmfr::nx2212::test::ExpectEqual;
}

BREPMFR_TEST(
    windows_argument_quoting_handles_empty_spaces_quotes_and_trailing_slashes)
{
    ExpectEqual(
        QuoteWindowsArgument(L""), std::wstring(L"\"\""),
        "empty argument was not quoted");
    ExpectEqual(
        QuoteWindowsArgument(L"plain"), std::wstring(L"plain"),
        "plain argument changed");
    ExpectEqual(
        QuoteWindowsArgument(L"two words"), std::wstring(L"\"two words\""),
        "spaces were not quoted");
    ExpectEqual(
        QuoteWindowsArgument(L"C:\\Program Files\\"),
        std::wstring(L"\"C:\\Program Files\\\\\""),
        "trailing slash was not doubled");
    ExpectEqual(
        QuoteWindowsArgument(L"a\"b"), std::wstring(L"\"a\\\"b\""),
        "embedded quote was not escaped");
}

BREPMFR_TEST(inference_arguments_follow_the_existing_python_cli_contract)
{
    const PluginConfig config{
        L"C:\\Python\\python.exe",
        L"C:\\Repo Path\\tools\\infer_step.py",
        L"C:\\Models\\best.ckpt",
        L"C:\\Output",
        "cuda",
        0.5};
    const OutputPaths output{
        L"C:\\Output\\part-0123456789abcdef",
        L"C:\\Output\\part-0123456789abcdef\\input.bin",
        L"C:\\Output\\part-0123456789abcdef\\input.bin.meta.json",
        L"C:\\Output\\part-0123456789abcdef\\result.json",
        L"C:\\Output\\part-0123456789abcdef\\run.log"};

    const auto arguments = BuildInferenceArguments(
        config, L"C:\\Parts\\part.step", output);

    const std::vector<std::wstring> expected{
        L"C:\\Repo Path\\tools\\infer_step.py",
        L"--step",
        L"C:\\Parts\\part.step",
        L"--checkpoint",
        L"C:\\Models\\best.ckpt",
        L"--output-dir",
        L"C:\\Output\\part-0123456789abcdef",
        L"--device",
        L"cuda",
        L"--affinity-threshold",
        L"0.5"};
    ExpectEqual(arguments, expected, "Python CLI argument order changed");

    ExpectEqual(
        BuildWindowsCommandLine(config.python_executable, arguments),
        std::wstring(
            L"C:\\Python\\python.exe "
            L"\"C:\\Repo Path\\tools\\infer_step.py\" "
            L"--step C:\\Parts\\part.step "
            L"--checkpoint C:\\Models\\best.ckpt "
            L"--output-dir C:\\Output\\part-0123456789abcdef "
            L"--device cuda --affinity-threshold 0.5"),
        "Windows command line changed");
}
