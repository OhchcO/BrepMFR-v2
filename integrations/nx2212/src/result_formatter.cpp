#include "result_formatter.hpp"

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
std::string FormatInstance(const InstanceResult& instance)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << '[' << instance.instance_id << "] "
           << instance.feature_name << ": OCC faces [";
    for (std::size_t index = 0; index < instance.face_indices.size(); ++index)
    {
        if (index != 0)
        {
            stream << ", ";
        }
        stream << instance.face_indices[index];
    }
    stream << "], confidence " << std::fixed << std::setprecision(4)
           << instance.confidence;
    return stream.str();
}
}

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path)
{
    DisplayText display;
    std::ostringstream summary;
    summary.imbue(std::locale::classic());
    summary << u8"识别完成\n"
            << u8"面数量：" << result.face_count << '\n'
            << u8"特征实例：" << result.instances.size() << '\n'
            << u8"BIN 缓存：" << (result.cache_hit ? u8"命中" : u8"未命中")
            << '\n'
            << u8"用时：" << std::fixed << std::setprecision(1)
            << result.total_seconds << u8" 秒";
    display.summary_utf8 = summary.str();

    display.listing_lines_utf8 = {
        "BrepMFR recognition result",
        "STEP: " + result.step_path.generic_u8string(),
        "Result: " + result_path.generic_u8string(),
        "Face numbering: OCC/AAG face_index (not NX Face Tag)",
        ""};

    auto instances = result.instances;
    std::sort(
        instances.begin(),
        instances.end(),
        [](const InstanceResult& left, const InstanceResult& right) {
            return left.instance_id < right.instance_id;
        });
    if (instances.empty())
    {
        display.listing_lines_utf8.push_back(
            "No non-stock feature instances were recognized.");
    }
    else
    {
        for (const auto& instance : instances)
        {
            display.listing_lines_utf8.push_back(FormatInstance(instance));
        }
    }
    return display;
}

std::string FormatErrorDisplay(
    const ErrorResult& error,
    const std::filesystem::path& result_path,
    const std::filesystem::path& log_path)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << u8"识别失败\n"
           << u8"阶段：" << error.stage << '\n'
           << u8"错误：" << error.error_type << ": " << error.message << '\n'
           << "STEP: " << error.step_path.generic_u8string() << '\n'
           << "Result: " << result_path.generic_u8string() << '\n'
           << "Log: " << log_path.generic_u8string();
    return stream.str();
}
}
}
