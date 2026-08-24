#include "test_support.hpp"

#include <exception>
#include <iostream>

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

int main()
{
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
