#include "result_formatter.hpp"

#include "test_support.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <string>

namespace
{
brepmfr::nx2212::SuccessResult SampleSuccess()
{
    brepmfr::nx2212::SuccessResult result;
    result.step_path = "D:/parts/sample.step";
    result.face_count = 2;
    result.cache_hit = true;
    result.total_seconds = 6.0;
    result.instances.push_back({
        0, 12, "blind_hole", {1}, 0.95});
    return result;
}

std::string JoinLines(const brepmfr::nx2212::DisplayText& display)
{
    std::string joined;
    for (const auto& line : display.listing_lines_utf8)
    {
        joined += line;
        joined.push_back('\n');
    }
    return joined;
}
}

BREPMFR_TEST(FormatSuccessDisplayEmitsSummaryAndOccFaceWarning)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto display = FormatSuccessDisplay(
        SampleSuccess(),
        "D:/output/result.json",
        std::chrono::milliseconds(7500));
    Expect(
        display.summary_utf8.find("面数量：2") != std::string::npos,
        "summary omitted face count");
    Expect(
        display.summary_utf8.find("特征实例：1") != std::string::npos,
        "summary omitted instance count");
    Expect(
        display.summary_utf8.find("BIN 缓存：命中") != std::string::npos,
        "summary omitted cache state");
    Expect(
        display.summary_utf8.find("用时：6.0 秒") != std::string::npos,
        "summary omitted total duration");

    const std::string all_lines = [&display]() {
        std::string joined;
        for (const auto& line : display.listing_lines_utf8)
        {
            joined += line;
            joined.push_back('\n');
        }
        return joined;
    }();
    Expect(
        all_lines.find(u8"面编号说明：OCC/AAG 面索引（不是 NX 面标签）") !=
            std::string::npos,
        "listing omitted OCC face numbering warning");
    Expect(
        all_lines.find(u8"进程总用时：7.5 秒；识别流水线：6.0 秒") !=
            std::string::npos,
        "listing omitted actual process and pipeline durations");
    Expect(
        all_lines.find(
            u8"[0] 盲孔：OCC 面索引 [1]，置信度 0.9500") !=
            std::string::npos,
        "listing instance line mismatch");
}

BREPMFR_TEST(FormatSuccessDisplayUsesChineseNamesForEveryModelClass)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    struct NameCase
    {
        const char* english;
        const char* chinese;
    };
    const std::array<NameCase, 25> cases{{
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

    SuccessResult result;
    result.step_path = "D:/parts/all_classes.step";
    result.face_count = static_cast<int>(cases.size());
    result.total_seconds = 1.0;
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        result.instances.push_back({
            static_cast<int>(index),
            static_cast<int>(index),
            cases[index].english,
            {static_cast<int>(index)},
            0.9});
    }

    const std::string listing = JoinLines(FormatSuccessDisplay(
        result, "D:/output/result.json"));
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const std::string expected =
            "[" + std::to_string(index) + "] " + cases[index].chinese +
            u8"：OCC 面索引 [" + std::to_string(index) + u8"]，置信度 0.9000";
        ExpectContains(
            listing,
            expected,
            "Chinese feature-name mapping mismatch for " +
                std::string(cases[index].english));
    }
}

BREPMFR_TEST(FormatSuccessDisplayKeepsUnknownFeatureNameReadable)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    auto result = SampleSuccess();
    result.instances.front().feature_name = "future_feature";
    const std::string listing = JoinLines(FormatSuccessDisplay(
        result, "D:/output/result.json"));

    ExpectContains(
        listing,
        u8"[0] future_feature：OCC 面索引 [1]，置信度 0.9500",
        "unknown feature name did not fall back to its original value");
}

BREPMFR_TEST(FormatSuccessDisplayExplainsEmptyInstanceResult)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    auto success = SampleSuccess();
    success.instances.clear();
    const auto display = FormatSuccessDisplay(
        success, "D:/output/result.json");

    bool found = false;
    for (const auto& line : display.listing_lines_utf8)
    {
        found = found ||
            line == u8"未识别到非毛坯特征实例。";
    }
    Expect(found, "empty success did not explain that no feature was recognized");
}

BREPMFR_TEST(FormatSuccessDisplayExplainsJson10MappingIsUnsupported)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    MappingDisplay mapping{
        MappingDisplayStatus::Unsupported,
        "JSON 1.0",
        0,
        0,
        0.0,
        std::chrono::milliseconds{0},
        {}};
    const auto display = FormatSuccessDisplay(
        SampleSuccess(),
        "D:/output/result.json",
        std::chrono::milliseconds{7500},
        mapping);

    ExpectContains(
        display.summary_utf8,
        u8"映射不可用：JSON 1.0",
        "JSON 1.0 mapping status missing");
    ExpectContains(
        JoinLines(display),
        u8"[0] 盲孔：OCC 面索引 [1]，置信度 0.9500",
        "unsupported mapping removed OCC result");
}

BREPMFR_TEST(FormatSuccessDisplayKeepsOccResultsWhenMappingFails)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    for (const auto status : {
             MappingDisplayStatus::Unavailable,
             MappingDisplayStatus::Failed})
    {
        MappingDisplay mapping{
            status,
            "face signatures unavailable",
            0,
            0,
            0.0,
            std::chrono::milliseconds{250},
            {}};
        const auto display = FormatSuccessDisplay(
            SampleSuccess(),
            "D:/output/result.json",
            std::chrono::milliseconds{7500},
            mapping);
        ExpectContains(
            display.summary_utf8,
            u8"识别成功，但面映射失败",
            "mapping failure summary missing");
        ExpectContains(
            display.summary_utf8,
            "face signatures unavailable",
            "mapping diagnostic missing");
        ExpectContains(
            JoinLines(display),
            u8"[0] 盲孔：OCC 面索引 [1]，置信度 0.9500",
            "mapping failure removed OCC result");
    }
}

BREPMFR_TEST(FormatSuccessDisplayShowsColoredNxFaceMapping)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    MappingDisplay mapping{
        MappingDisplayStatus::Colored,
        "",
        2,
        1,
        0.125,
        std::chrono::milliseconds{400},
        {{0, 36, {501}}}};
    const auto display = FormatSuccessDisplay(
        SampleSuccess(),
        "D:/output/result.json",
        std::chrono::milliseconds{7500},
        mapping);

    ExpectContains(display.summary_utf8, u8"已映射面：2/2",
                   "mapped face summary missing");
    ExpectContains(display.summary_utf8, u8"已着色实例：1",
                   "colored instance summary missing");
    ExpectContains(
        JoinLines(display),
        u8"已映射面：2/2；最大误差：0.125000；映射用时：0.4 秒",
        "mapping metrics line mismatch");
    ExpectContains(
        JoinLines(display),
        u8"[0] 盲孔：OCC 面索引 [1]，NX 面标签 [501]，颜色编号 36，置信度 0.9500",
        "colored instance line mismatch");
}

BREPMFR_TEST(FormatErrorDisplayIncludesDiagnosticsPaths)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    ErrorResult error;
    error.step_path = "D:/parts/sample.step";
    error.stage = "conversion";
    error.error_type = "ValueError";
    error.message = "shape mismatch";
    const auto text = FormatErrorDisplay(
        error,
        "D:/output/result.json",
        "D:/output/run.log");

    for (const std::string expected : {
             std::string(u8"阶段：conversion"),
             std::string(u8"错误：ValueError: shape mismatch"),
             std::string(u8"STEP 文件：D:/parts/sample.step"),
             std::string(u8"结果文件：D:/output/result.json"),
             std::string(u8"日志文件：D:/output/run.log")})
    {
        Expect(
            text.find(expected) != std::string::npos,
            "error display omitted " + expected);
    }
}
