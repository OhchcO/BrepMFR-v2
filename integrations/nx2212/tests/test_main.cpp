#include "test_support.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>

namespace brepmfr
{
namespace nx2212
{
namespace test
{
std::vector<std::pair<std::string, TestFunction>>& Registry()
{
    static std::vector<std::pair<std::string, TestFunction>> tests;
    return tests;
}

Registration::Registration(const char* name, TestFunction function)
{
    Registry().emplace_back(name, function);
}

void Expect(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}
}
}
}

int main(int argc, char* argv[])
{
    if (argc == 2 && std::string(argv[1]) == "--progress-fixture")
    {
        std::cout << "BREPMFR_PROGRESS:startup\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::cout << "BREPMFR_PROGRESS:conversion\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return 0;
    }

    if (argc == 2 && std::string(argv[1]) == "--environment-fixture")
    {
        const auto print_variable = [](const char* name) {
            const DWORD required = GetEnvironmentVariableA(name, nullptr, 0);
            if (required == 0)
            {
                std::cout << name << "=<unset>\n";
                return;
            }
            std::vector<char> value(required, '\0');
            GetEnvironmentVariableA(name, value.data(), required);
            std::cout << name << '=' << value.data() << '\n';
        };
        std::cout << "CWD=" << std::filesystem::current_path().generic_u8string()
                  << '\n';
        print_variable("PATH");
        print_variable("PYTHONHOME");
        print_variable("PYTHONPATH");
        print_variable("UGII_STAGE1_MARKER");
        print_variable("PYTHONNOUSERSITE");
        print_variable("PYTHONUNBUFFERED");
        return 0;
    }

    if (argc == 4 && std::string(argv[1]) == "--process-fixture")
    {
        const int exit_code = std::stoi(argv[2]);
        std::cout << argv[3] << '\n';
        std::cerr << argv[3] << '\n';
        return exit_code;
    }

    int failures = 0;
    for (const auto& registered : brepmfr::nx2212::test::Registry())
    {
        try
        {
            registered.second();
        }
        catch (const std::exception& exception)
        {
            std::cerr << "[FAIL] " << registered.first << ": "
                      << exception.what() << '\n';
            ++failures;
        }
        catch (...)
        {
            std::cerr << "[FAIL] " << registered.first
                      << ": unknown exception\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
