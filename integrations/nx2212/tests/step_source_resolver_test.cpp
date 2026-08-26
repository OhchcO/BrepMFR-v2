#include "step_source_resolver.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace
{
std::string ToSystemCodePage(
    const std::filesystem::path& path,
    bool* used_default_character)
{
    const std::wstring value = path.wstring();
    const int required = WideCharToMultiByte(
        CP_ACP, WC_NO_BEST_FIT_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0)
    {
        throw std::runtime_error("could not encode test path using CP_ACP");
    }
    BOOL used_default = FALSE;
    std::string bytes(static_cast<std::size_t>(required), '\0');
    if (WideCharToMultiByte(
            CP_ACP, WC_NO_BEST_FIT_CHARS, value.data(),
            static_cast<int>(value.size()), bytes.data(), required,
            nullptr, &used_default) != required)
    {
        throw std::runtime_error("could not encode test path using CP_ACP");
    }
    *used_default_character = used_default != FALSE;
    return bytes;
}

void ExpectResolutionRejected(
    const std::filesystem::path& work_part_path,
    const std::string& expected_error,
    const std::string& message)
{
    std::string error;
    try
    {
        (void)brepmfr::nx2212::ResolveOpenedStepSource(work_part_path);
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
    }
    brepmfr::nx2212::test::Expect(!error.empty(), message);
    brepmfr::nx2212::test::ExpectContains(
        error, expected_error,
        message + " (wrong diagnostic: " + error + ")");
}
}

BREPMFR_TEST(ResolveOpenedStepSourceKeepsAnExistingStepPath)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");

    ExpectEqual(
        ResolveOpenedStepSource(step_path),
        std::filesystem::weakly_canonical(step_path),
        "an existing STEP Work Part path was not preserved");
}

BREPMFR_TEST(ResolveOpenedStepSourceUsesTheTranslatorLogForGeneratedPrt)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        std::string("STEP to UG\n") +
        "! STATUS-   Converting Part: " + step_path.generic_string() + "\n" +
        "! SUMMARY-  Description Open CASCADE Model\n" +
        "! SUMMARY-  advanced_face 41\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectEqual(
        ResolveOpenedStepSource(prt_path),
        std::filesystem::weakly_canonical(step_path),
        "the NX-generated PRT was not traced back to its source STEP");
}

BREPMFR_TEST(ResolveOpenedStepSourceIgnoresMarkersInsideSummaryText)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");
    const auto other_path = temporary.WriteFile("other.step", "other STEP");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + step_path.generic_string() + "\n" +
        "! SUMMARY-  Description Converting Part: " +
            other_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectEqual(
        ResolveOpenedStepSource(prt_path),
        std::filesystem::weakly_canonical(step_path),
        "summary text overrode the exact NX converting record");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsAValidAndInvalidAssociationMix)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");
    const auto missing_path = temporary.Path() / "missing.step";
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + step_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n" +
        "! STATUS-   Converting Part: " + missing_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectResolutionRejected(
        prt_path,
        "could not uniquely trace",
        "a missing conflicting STEP association was silently discarded");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsAnAssociationInAnUnrelatedLog)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + step_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("unrelated.log", log);

    ExpectResolutionRejected(
        prt_path,
        "could not read the NX STEP translator log",
        "an unrelated log was accepted as the NX translator provenance");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsAMismatchedFiledPart)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto step_path = temporary.WriteFile("part.step", "STEP fixture");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const auto other_prt = temporary.WriteFile(
        "other_step.prt", "other PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + step_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + other_prt.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectResolutionRejected(
        prt_path,
        "could not uniquely trace",
        "a translator log for a different PRT was accepted");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsANonStepSourceAssociation)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto source_path = temporary.WriteFile("part.iges", "IGES fixture");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + source_path.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectResolutionRejected(
        prt_path,
        "translator log contains an invalid source",
        "a non-STEP translator source was accepted");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsTwoDistinctValidSources)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto first_step = temporary.WriteFile("first.step", "first STEP");
    const auto second_step = temporary.WriteFile("second.stp", "second STEP");
    const auto prt_path = temporary.WriteFile("part_step.prt", "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + first_step.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n" +
        "! STATUS-   Converting Part: " + second_step.generic_string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_string() + "\n";
    temporary.WriteFile("part.log", log);

    ExpectResolutionRejected(
        prt_path,
        "could not uniquely trace",
        "two distinct source STEP associations were accepted");
}

BREPMFR_TEST(ResolveOpenedStepSourceRejectsAnUnsupportedWorkPart)
{
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto text_path = temporary.WriteFile("part.txt", "not a CAD Part");

    ExpectResolutionRejected(
        text_path,
        "must originate from an opened STEP file",
        "an unsupported Work Part extension was accepted");
}

BREPMFR_TEST(ResolveOpenedStepSourceReadsUtf8TranslatorPaths)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto directory = std::filesystem::u8path(u8"UTF8零件");
    const auto step_path = temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型.step"), "STEP fixture");
    const auto prt_path = temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型_step.prt"), "PRT fixture");
    const std::string log =
        "! STATUS-   Converting Part: " + step_path.generic_u8string() + "\n" +
        "! INFO-     Filing Part ..: " + prt_path.generic_u8string() + "\n";
    temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型.log"), log);

    ExpectEqual(
        ResolveOpenedStepSource(prt_path),
        std::filesystem::weakly_canonical(step_path),
        "the UTF-8 NX log did not preserve a Unicode STEP path");
}

BREPMFR_TEST(ResolveOpenedStepSourceReadsSystemCodePageTranslatorPaths)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto directory = std::filesystem::u8path(u8"零件");
    const auto step_path = temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型.step"), "STEP fixture");
    const auto prt_path = temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型_step.prt"), "PRT fixture");
    bool step_used_default = false;
    bool prt_used_default = false;
    const std::string encoded_step = ToSystemCodePage(
        step_path, &step_used_default);
    const std::string encoded_prt = ToSystemCodePage(
        prt_path, &prt_used_default);
    if (step_used_default || prt_used_default)
    {
        return;
    }
    const std::string log =
        "! STATUS-   Converting Part: " + encoded_step + "\n" +
        "! INFO-     Filing Part ..: " + encoded_prt + "\n";
    temporary.WriteFile(
        directory / std::filesystem::u8path(u8"模型.log"), log);

    ExpectEqual(
        ResolveOpenedStepSource(prt_path),
        std::filesystem::weakly_canonical(step_path),
        "the system-code-page NX log did not preserve a Unicode STEP path");
}
