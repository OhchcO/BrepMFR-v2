#include "progress_layout.hpp"

#include "test_support.hpp"

namespace
{
int Bottom(const brepmfr::nx2212::ProgressRect& rectangle)
{
    return rectangle.y + rectangle.height;
}

void ExpectInside(
    const brepmfr::nx2212::ProgressWindowLayout& layout,
    const brepmfr::nx2212::ProgressRect& rectangle,
    const char* name)
{
    using brepmfr::nx2212::test::Expect;
    Expect(rectangle.x >= 0 && rectangle.y >= 0, std::string(name) + " is negative");
    Expect(
        rectangle.x + rectangle.width <= layout.client_width,
        std::string(name) + " exceeds client width");
    Expect(
        Bottom(rectangle) <= layout.client_height,
        std::string(name) + " is clipped by client height");
}
}

BREPMFR_TEST(ProgressLayoutKeepsEveryControlVisibleAndSeparated)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto layout = CalculateProgressWindowLayout(96);
    ExpectInside(layout, layout.status, "status");
    ExpectInside(layout, layout.input, "input");
    ExpectInside(layout, layout.stage, "stage");
    ExpectInside(layout, layout.progress, "progress");
    ExpectInside(layout, layout.elapsed, "elapsed");
    Expect(Bottom(layout.status) <= layout.input.y, "status overlaps input");
    Expect(Bottom(layout.input) <= layout.stage.y, "input overlaps stage");
    Expect(Bottom(layout.stage) <= layout.progress.y, "stage overlaps progress");
    Expect(Bottom(layout.progress) <= layout.elapsed.y, "progress overlaps elapsed");
}

BREPMFR_TEST(ProgressLayoutScalesForOneHundredFiftyPercentDpi)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto standard = CalculateProgressWindowLayout(96);
    const auto scaled = CalculateProgressWindowLayout(144);
    ExpectEqual(
        scaled.client_width,
        standard.client_width * 3 / 2,
        "client width did not scale with DPI");
    ExpectEqual(
        scaled.client_height,
        standard.client_height * 3 / 2,
        "client height did not scale with DPI");
    ExpectEqual(
        scaled.elapsed.y,
        standard.elapsed.y * 3 / 2,
        "elapsed label position did not scale with DPI");
    ExpectEqual(
        scaled.progress.height,
        standard.progress.height * 3 / 2,
        "progress height did not scale with DPI");
}

BREPMFR_TEST(ProgressStageTextExplainsConversionInferenceAndUnknownStages)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto conversion = ProgressStageText("conversion");
    Expect(
        conversion.find(L"STEP") != std::wstring::npos &&
            conversion.find(L"BIN") != std::wstring::npos,
        "conversion stage did not explain STEP to BIN work");
    Expect(
        ProgressStageText("inference").find(L"GPU") != std::wstring::npos,
        "inference stage did not identify GPU inference");
    Expect(
        !ProgressStageText("future-stage").empty(),
        "unknown progress stage produced an empty label");
}
