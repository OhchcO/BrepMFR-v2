#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <process.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <windows.h>

namespace brepmfr
{
namespace nx2212
{
namespace test
{
using TestFunction = void (*)();

std::vector<std::pair<std::string, TestFunction>>& Registry();

class Registration
{
public:
    Registration(const char* name, TestFunction function);
};

void Expect(bool condition, const std::string& message);

inline void ExpectNear(
    double actual,
    double expected,
    double tolerance,
    const std::string& message)
{
    Expect(
        std::isfinite(actual) && std::isfinite(expected) &&
            std::isfinite(tolerance) && tolerance >= 0.0 &&
            std::abs(actual - expected) <= tolerance,
        message);
}

inline void ExpectContains(
    std::string_view text,
    std::string_view needle,
    const std::string& message)
{
    Expect(text.find(needle) != std::string_view::npos, message);
}

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        static std::atomic<unsigned long> sequence{0};
        const auto unique = std::string("brepmfr-nx2212-") +
            std::to_string(_getpid()) + "-" +
            std::to_string(sequence.fetch_add(1));
        path_ = std::filesystem::temp_directory_path() / unique;
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& Path() const noexcept
    {
        return path_;
    }

    std::filesystem::path WriteFile(
        const std::filesystem::path& name,
        std::string_view bytes)
    {
        const auto target = path_ / name;
        std::filesystem::create_directories(target.parent_path());
        std::ofstream stream(target, std::ios::binary | std::ios::trunc);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!stream)
        {
            throw std::runtime_error("could not write temporary test file");
        }
        return target;
    }

    std::filesystem::path WriteUtf8(
        const std::filesystem::path& name,
        std::string_view text)
    {
        return WriteFile(name, text);
    }

private:
    std::filesystem::path path_;
};

inline std::filesystem::path CurrentTestExecutable()
{
    std::vector<wchar_t> buffer(1024);
    for (;;)
    {
        const DWORD length = GetModuleFileNameW(
            nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            throw std::runtime_error("GetModuleFileNameW failed");
        }
        if (length < buffer.size() - 1)
        {
            return std::filesystem::path(
                std::wstring(buffer.data(), static_cast<std::size_t>(length)));
        }
        buffer.resize(buffer.size() * 2);
    }
}

inline std::string ReadUtf8(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("could not read test file");
    }
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

inline std::size_t CountOccurrences(
    std::string_view text,
    std::string_view needle)
{
    if (needle.empty())
    {
        return 0;
    }
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string_view::npos)
    {
        ++count;
        position += needle.size();
    }
    return count;
}

template <typename Actual, typename Expected>
void ExpectEqual(
    const Actual& actual,
    const Expected& expected,
    const std::string& message)
{
    Expect(actual == expected, message);
}
}
}
}

#define BREPMFR_TEST(name)                                      \
    void name();                                                \
    static brepmfr::nx2212::test::Registration name##_entry(    \
        #name, &name);                                          \
    void name()
