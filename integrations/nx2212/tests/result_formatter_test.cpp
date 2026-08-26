#include "result_formatter.hpp"

#include "test_support.hpp"

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
        all_lines.find("Face numbering: OCC/AAG face_index (not NX Face Tag)") !=
            std::string::npos,
        "listing omitted OCC face numbering warning");
    Expect(
        all_lines.find("Process total: 7.5 s; pipeline: 6.0 s") !=
            std::string::npos,
        "listing omitted actual process and pipeline durations");
    Expect(
        all_lines.find(
            "[0] blind_hole: OCC faces [1], confidence 0.9500") !=
            std::string::npos,
        "listing instance line mismatch");
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
            line == "No non-stock feature instances were recognized.";
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
        "[0] blind_hole: OCC faces [1], confidence 0.9500",
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
            "[0] blind_hole: OCC faces [1], confidence 0.9500",
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
        "Mapped faces: 2/2; max error: 0.125000; mapping: 0.4 s",
        "mapping metrics line mismatch");
    ExpectContains(
        JoinLines(display),
        "[0] blind_hole: OCC faces [1], NX Face Tags [501], color 36, confidence 0.9500",
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
             "conversion",
             "ValueError",
             "shape mismatch",
             "D:/output/result.json",
             "D:/output/run.log"})
    {
        Expect(
            text.find(expected) != std::string::npos,
            "error display omitted " + expected);
    }
}
