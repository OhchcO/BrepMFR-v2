#include "output_paths.hpp"

#include <windows.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brepmfr
{
namespace nx2212
{
namespace
{
bool IsInvalidFilenameCharacter(wchar_t character)
{
    if (character < 32)
    {
        return true;
    }
    const std::wstring invalid = L"<>:\"/\\|?*";
    return invalid.find(character) != std::wstring::npos;
}

std::string WideToUtf8(const std::wstring& value)
{
    if (value.empty())
    {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0)
    {
        throw std::runtime_error("Could not encode a path as UTF-8");
    }
    std::string encoded(static_cast<std::size_t>(size), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            value.data(),
            static_cast<int>(value.size()),
            encoded.data(),
            size,
            nullptr,
            nullptr) != size)
    {
        throw std::runtime_error("Could not encode a path as UTF-8");
    }
    return encoded;
}

std::wstring HashText(std::uint64_t value)
{
    std::wostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::hex << std::nouppercase
           << std::setfill(L'0') << std::setw(16) << value;
    return stream.str();
}
}

std::wstring SanitizeStepStem(std::wstring_view stem)
{
    if (stem.empty())
    {
        return L"step";
    }

    std::wstring safe(stem);
    for (auto& character : safe)
    {
        if (IsInvalidFilenameCharacter(character))
        {
            character = L'_';
        }
    }
    for (auto iterator = safe.rbegin(); iterator != safe.rend(); ++iterator)
    {
        if (*iterator == L'.' || *iterator == L' ')
        {
            *iterator = L'_';
        }
        else
        {
            break;
        }
    }
    return safe.empty() ? L"step" : safe;
}

std::uint64_t StablePathHash(
    const std::filesystem::path& absolute_path)
{
    if (!absolute_path.is_absolute())
    {
        throw std::invalid_argument(
            "StablePathHash requires an absolute path");
    }

    auto normalized =
        absolute_path.lexically_normal().generic_wstring();
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](wchar_t character)
        {
            return character >= L'A' && character <= L'Z'
                ? static_cast<wchar_t>(character - L'A' + L'a')
                : character;
        });

    constexpr std::uint64_t offset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t prime = 0x100000001b3ULL;
    std::uint64_t value = offset;
    for (const auto byte : WideToUtf8(normalized))
    {
        value ^= static_cast<unsigned char>(byte);
        value *= prime;
    }
    return value;
}

OutputPaths BuildOutputPaths(
    const std::filesystem::path& output_root,
    const std::filesystem::path& step_path)
{
    if (!output_root.is_absolute())
    {
        throw std::invalid_argument(
            "BuildOutputPaths requires an absolute output root");
    }
    if (!step_path.is_absolute())
    {
        throw std::invalid_argument(
            "BuildOutputPaths requires an absolute STEP path");
    }

    const auto directory_name =
        SanitizeStepStem(step_path.stem().wstring()) +
        L"-" + HashText(StablePathHash(step_path));
    const auto directory = output_root / directory_name;
    return OutputPaths{
        directory,
        directory / L"input.bin",
        directory / L"input.bin.meta.json",
        directory / L"result.json",
        directory / L"run.log"};
}
}
}
