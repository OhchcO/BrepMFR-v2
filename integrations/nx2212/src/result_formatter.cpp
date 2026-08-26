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
void AppendTags(
    std::ostringstream& stream,
    const std::vector<FaceKey>& tags)
{
    for (std::size_t index = 0; index < tags.size(); ++index)
    {
        if (index != 0)
        {
            stream << ", ";
        }
        stream << tags[index];
    }
}

std::string FormatInstance(
    const InstanceResult& instance,
    const InstanceColorRecord* color)
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
    stream << ']';
    if (color != nullptr)
    {
        stream << ", NX Face Tags [";
        AppendTags(stream, color->nx_face_tags);
        stream << "], color " << color->color_index;
    }
    stream << ", confidence " << std::fixed << std::setprecision(4)
           << instance.confidence;
    return stream.str();
}

std::string MappingDiagnostic(
    const MappingDisplay& mapping,
    const char* fallback)
{
    return mapping.diagnostic.empty() ? fallback : mapping.diagnostic;
}
}

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed)
{
    MappingDisplay mapping;
    mapping.status = MappingDisplayStatus::Unsupported;
    mapping.diagnostic = "JSON 1.0";
    return FormatSuccessDisplay(
        result, result_path, process_elapsed, mapping);
}

DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed,
    const MappingDisplay& mapping)
{
    DisplayText display;
    const double process_seconds =
        static_cast<double>(process_elapsed.count()) / 1000.0;
    std::ostringstream summary;
    summary.imbue(std::locale::classic());
    summary << u8"识别完成\n"
            << u8"面数量：" << result.face_count << '\n'
            << u8"特征实例：" << result.instances.size() << '\n'
            << u8"BIN 缓存：" << (result.cache_hit ? u8"命中" : u8"未命中")
            << '\n'
            << u8"用时：" << std::fixed << std::setprecision(1)
            << result.total_seconds << u8" 秒" << '\n'
            << u8"Python 进程总用时：" << process_seconds << u8" 秒";
    switch (mapping.status)
    {
    case MappingDisplayStatus::Unsupported:
        summary << '\n' << u8"映射不可用："
                << MappingDiagnostic(mapping, "JSON 1.0");
        break;
    case MappingDisplayStatus::Unavailable:
    case MappingDisplayStatus::Failed:
        summary << '\n' << u8"识别成功，但面映射失败" << '\n'
                << u8"面映射：" << MappingDiagnostic(mapping, "unknown reason");
        break;
    case MappingDisplayStatus::Colored:
        summary << '\n' << u8"已映射面：" << mapping.mapped_faces
                << '/' << result.face_count << '\n'
                << u8"已着色实例：" << mapping.colored_instances << '\n'
                << u8"映射用时：" << std::fixed << std::setprecision(1)
                << static_cast<double>(mapping.elapsed.count()) / 1000.0
                << u8" 秒";
        break;
    }
    display.summary_utf8 = summary.str();

    std::ostringstream timing_line;
    timing_line.imbue(std::locale::classic());
    timing_line << "Process total: " << std::fixed << std::setprecision(1)
                << process_seconds << " s; pipeline: "
                << result.total_seconds << " s";

    display.listing_lines_utf8 = {
        "BrepMFR recognition result",
        "STEP: " + result.step_path.generic_u8string(),
        "Result: " + result_path.generic_u8string(),
        "Face numbering: OCC/AAG face_index (not NX Face Tag)",
        timing_line.str(),
        ""};

    std::ostringstream mapping_line;
    mapping_line.imbue(std::locale::classic());
    switch (mapping.status)
    {
    case MappingDisplayStatus::Unsupported:
        mapping_line << "Face mapping unavailable: "
                     << MappingDiagnostic(mapping, "JSON 1.0");
        break;
    case MappingDisplayStatus::Unavailable:
    case MappingDisplayStatus::Failed:
        mapping_line << "Face mapping failed: "
                     << MappingDiagnostic(mapping, "unknown reason");
        break;
    case MappingDisplayStatus::Colored:
        mapping_line << "Mapped faces: " << mapping.mapped_faces << '/'
                     << result.face_count << "; max error: "
                     << std::fixed << std::setprecision(6) << mapping.max_error
                     << "; mapping: " << std::setprecision(1)
                     << static_cast<double>(mapping.elapsed.count()) / 1000.0
                     << " s";
        break;
    }
    display.listing_lines_utf8.insert(
        display.listing_lines_utf8.end() - 1, mapping_line.str());

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
            const InstanceColorRecord* color = nullptr;
            if (mapping.status == MappingDisplayStatus::Colored)
            {
                const auto found = std::find_if(
                    mapping.instances.begin(), mapping.instances.end(),
                    [&instance](const InstanceColorRecord& record) {
                        return record.instance_id == instance.instance_id;
                    });
                if (found != mapping.instances.end())
                {
                    color = &*found;
                }
            }
            display.listing_lines_utf8.push_back(
                FormatInstance(instance, color));
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
