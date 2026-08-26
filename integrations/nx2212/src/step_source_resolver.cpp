#include "step_source_resolver.hpp"

#include <algorithm>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <windows.h>

namespace brepmfr
{
namespace nx2212
{
namespace
{
std::wstring LowerAscii(std::wstring value)
{
    for (wchar_t& character : value)
    {
        if (character >= L'A' && character <= L'Z')
        {
            character = static_cast<wchar_t>(character - L'A' + L'a');
        }
    }
    return value;
}

bool IsStepExtension(const std::filesystem::path& path)
{
    const auto extension = LowerAscii(path.extension().wstring());
    return extension == L".step" || extension == L".stp";
}

std::wstring ComparablePath(const std::filesystem::path& path)
{
    return LowerAscii(std::filesystem::weakly_canonical(path).wstring());
}

std::wstring_view Trim(std::wstring_view value)
{
    while (!value.empty() &&
           (value.front() == L' ' || value.front() == L'\t'))
    {
        value.remove_prefix(1);
    }
    while (!value.empty() &&
           (value.back() == L' ' || value.back() == L'\t' ||
            value.back() == L'\r'))
    {
        value.remove_suffix(1);
    }
    return value;
}

std::optional<std::wstring> DecodeBytes(
    const std::string& bytes,
    UINT code_page,
    DWORD flags)
{
    if (bytes.empty())
    {
        return std::wstring{};
    }
    const int byte_count = static_cast<int>(bytes.size());
    const int required = MultiByteToWideChar(
        code_page, flags, bytes.data(), byte_count, nullptr, 0);
    if (required <= 0)
    {
        return std::nullopt;
    }
    std::wstring decoded(static_cast<std::size_t>(required), L'\0');
    if (MultiByteToWideChar(
            code_page, flags, bytes.data(), byte_count,
            decoded.data(), required) != required)
    {
        return std::nullopt;
    }
    return decoded;
}

std::optional<std::wstring> DecodeLogLine(const std::string& bytes)
{
    if (auto utf8 = DecodeBytes(bytes, CP_UTF8, MB_ERR_INVALID_CHARS))
    {
        return utf8;
    }
    return DecodeBytes(bytes, CP_ACP, 0);
}

std::optional<std::filesystem::path> ValueAfterPrefix(
    const std::wstring& line,
    std::wstring_view prefix)
{
    if (line.size() < prefix.size() ||
        std::wstring_view(line).substr(0, prefix.size()) != prefix)
    {
        return std::nullopt;
    }
    const auto value = Trim(std::wstring_view(line).substr(
        prefix.size()));
    if (value.empty())
    {
        return std::nullopt;
    }
    return std::filesystem::path(value);
}

std::vector<std::filesystem::path> SourcesFiledAs(
    const std::filesystem::path& log_path,
    const std::filesystem::path& work_part_path)
{
    std::ifstream stream(log_path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error(
            "could not read the NX STEP translator log: " +
            log_path.generic_u8string());
    }

    constexpr std::wstring_view kConvertingPrefix =
        L"! STATUS-   Converting Part:";
    constexpr std::wstring_view kFilingPrefix =
        L"! INFO-     Filing Part ..:";
    std::optional<std::filesystem::path> pending_source;
    std::vector<std::filesystem::path> matches;
    std::string line;
    while (std::getline(stream, line))
    {
        const auto decoded = DecodeLogLine(line);
        if (!decoded)
        {
            throw std::runtime_error(
                "could not decode the NX STEP translator log");
        }
        if (auto source = ValueAfterPrefix(*decoded, kConvertingPrefix))
        {
            if (pending_source)
            {
                throw std::runtime_error(
                    "NX STEP translator log contains overlapping source "
                    "records");
            }
            pending_source = std::move(source);
            continue;
        }
        auto filed_path = ValueAfterPrefix(*decoded, kFilingPrefix);
        if (!filed_path)
        {
            continue;
        }
        if (!pending_source)
        {
            throw std::runtime_error(
                "NX STEP translator log contains an unpaired filed Part");
        }
        if (!filed_path->is_absolute())
        {
            throw std::runtime_error(
                "NX STEP translator log contains a relative filed Part path");
        }
        if (ComparablePath(*filed_path) == ComparablePath(work_part_path))
        {
            matches.push_back(*pending_source);
        }
        pending_source.reset();
    }
    if (stream.bad())
    {
        throw std::runtime_error(
            "failed while reading the NX STEP translator log");
    }
    if (pending_source)
    {
        throw std::runtime_error(
            "NX STEP translator log contains an unpaired source STEP");
    }
    return matches;
}

std::filesystem::path TranslatorLogPath(
    const std::filesystem::path& work_part_path)
{
    const std::wstring stem = work_part_path.stem().wstring();
    constexpr std::wstring_view kGeneratedSuffix = L"_step";
    const std::wstring lower_stem = LowerAscii(stem);
    if (lower_stem.size() <= kGeneratedSuffix.size() ||
        std::wstring_view(lower_stem).substr(
            lower_stem.size() - kGeneratedSuffix.size()) != kGeneratedSuffix)
    {
        throw std::runtime_error(
            "current PRT does not use the NX STEP translator *_step.prt name");
    }
    const std::wstring log_name =
        stem.substr(0, stem.size() - kGeneratedSuffix.size()) + L".log";
    return work_part_path.parent_path() / log_name;
}

std::filesystem::path ResolveGeneratedPrtSource(
    const std::filesystem::path& work_part_path)
{
    const auto matches = SourcesFiledAs(
        TranslatorLogPath(work_part_path), work_part_path);

    std::vector<std::filesystem::path> valid_sources;
    for (const auto& source : matches)
    {
        if (!source.is_absolute())
        {
            throw std::runtime_error(
                "could not uniquely trace the current NX Work Part to its "
                "source STEP: translator log contains a relative source path");
        }
        const auto canonical = std::filesystem::weakly_canonical(source);
        std::error_code source_error;
        if (!IsStepExtension(canonical) ||
            !std::filesystem::is_regular_file(canonical, source_error) ||
            source_error)
        {
            throw std::runtime_error(
                "could not uniquely trace the current NX Work Part to its "
                "source STEP: translator log contains an invalid source");
        }
        const auto comparable = ComparablePath(canonical);
        const bool duplicate = std::any_of(
            valid_sources.begin(), valid_sources.end(),
            [&comparable](const std::filesystem::path& existing) {
                return ComparablePath(existing) == comparable;
            });
        if (!duplicate)
        {
            valid_sources.push_back(canonical);
        }
    }
    if (valid_sources.size() != 1)
    {
        throw std::runtime_error(
            "could not uniquely trace the current NX Work Part to its "
            "source STEP through the translator log");
    }
    return valid_sources.front();
}
}

std::filesystem::path ResolveOpenedStepSource(
    const std::filesystem::path& work_part_path)
{
    const auto path = std::filesystem::weakly_canonical(work_part_path);
    std::error_code error;
    const bool exists = std::filesystem::is_regular_file(path, error);
    if (error || !exists)
    {
        throw std::runtime_error(
            "current Work Part path is not an existing file: " +
            path.generic_u8string());
    }
    if (IsStepExtension(path))
    {
        return path;
    }
    if (LowerAscii(path.extension().wstring()) == L".prt")
    {
        return ResolveGeneratedPrtSource(path);
    }
    throw std::runtime_error(
        "current Work Part must originate from an opened STEP file");
}
}
}
