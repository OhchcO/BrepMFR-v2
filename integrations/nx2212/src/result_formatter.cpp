#include "result_formatter.hpp"

#include <algorithm>
#include <array>
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
           << LocalizedFeatureNameUtf8(instance.feature_name)
           << u8"：OCC 面索引 [";
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
        stream << u8"，NX 面标签 [";
        AppendTags(stream, color->nx_face_tags);
        stream << u8"]，颜色编号 " << color->color_index;
    }
    stream << u8"，置信度 " << std::fixed << std::setprecision(4)
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

std::string LocalizedFeatureNameUtf8(const std::string& feature_name)
{
    struct FeatureName
    {
        const char* english;
        const char* chinese;
    };
    static const std::array<FeatureName, 25> names{{
        {"chamfer", u8"倒角"},
        {"through_hole", u8"通孔"},
        {"triangular_passage", u8"三角形通孔"},
        {"rectangular_passage", u8"矩形通孔"},
        {"6sides_passage", u8"六边形通孔"},
        {"triangular_through_slot", u8"三角形通槽"},
        {"rectangular_through_slot", u8"矩形通槽"},
        {"circular_through_slot", u8"圆形通槽"},
        {"rectangular_through_step", u8"矩形台阶"},
        {"2sides_through_step", u8"两侧贯通台阶"},
        {"slanted_through_step", u8"斜向贯通台阶"},
        {"Oring", u8"O型环槽"},
        {"blind_hole", u8"盲孔"},
        {"triangular_pocket", u8"三角形腔"},
        {"rectangular_pocket", u8"矩形腔"},
        {"6sides_pocket", u8"六边形腔"},
        {"circular_end_pocket", u8"圆形腔"},
        {"rectangular_blind_slot", u8"矩形盲槽"},
        {"v_circular_end_blind_slot", u8"V形圆盲槽"},
        {"h_circular_end_blind_slot", u8"半圆形盲槽"},
        {"triangular_blind_step", u8"三角形盲台阶"},
        {"circular_blind_step", u8"圆形盲台阶"},
        {"rectangular_blind_step", u8"矩形盲台阶"},
        {"round", u8"圆角"},
        {"stock", u8"毛坯"},
    }};
    const auto found = std::find_if(
        names.begin(), names.end(),
        [&feature_name](const FeatureName& item) {
            return feature_name == item.english;
        });
    return found == names.end() ? feature_name : found->chinese;
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
                << u8"面映射：" << MappingDiagnostic(mapping, u8"未知原因");
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
    timing_line << u8"进程总用时：" << std::fixed << std::setprecision(1)
                << process_seconds << u8" 秒；识别流水线："
                << result.total_seconds << u8" 秒";

    display.listing_lines_utf8 = {
        u8"BrepMFR 特征识别结果",
        u8"STEP 文件：" + result.step_path.generic_u8string(),
        u8"结果文件：" + result_path.generic_u8string(),
        u8"面编号说明：OCC/AAG 面索引（不是 NX 面标签）",
        timing_line.str(),
        ""};

    std::ostringstream mapping_line;
    mapping_line.imbue(std::locale::classic());
    switch (mapping.status)
    {
    case MappingDisplayStatus::Unsupported:
        mapping_line << u8"面映射不可用："
                     << MappingDiagnostic(mapping, "JSON 1.0");
        break;
    case MappingDisplayStatus::Unavailable:
    case MappingDisplayStatus::Failed:
        mapping_line << u8"面映射失败："
                     << MappingDiagnostic(mapping, u8"未知原因");
        break;
    case MappingDisplayStatus::Colored:
        mapping_line << u8"已映射面：" << mapping.mapped_faces << '/'
                     << result.face_count << u8"；最大误差："
                     << std::fixed << std::setprecision(6) << mapping.max_error
                     << u8"；映射用时：" << std::setprecision(1)
                     << static_cast<double>(mapping.elapsed.count()) / 1000.0
                     << u8" 秒";
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
            u8"未识别到非毛坯特征实例。");
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
           << u8"STEP 文件：" << error.step_path.generic_u8string() << '\n'
           << u8"结果文件：" << result_path.generic_u8string() << '\n'
           << u8"日志文件：" << log_path.generic_u8string();
    return stream.str();
}
}
}
