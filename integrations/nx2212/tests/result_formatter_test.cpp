#include "result_formatter.hpp"

#include "test_support.hpp"

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
}

BREPMFR_TEST(FormatSuccessDisplayEmitsSummaryAndOccFaceWarning)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto display = FormatSuccessDisplay(
        SampleSuccess(), "D:/output/result.json");
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
