#include "output_paths.hpp"
#include "test_support.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace
{
using brepmfr::nx2212::BuildOutputPaths;
using brepmfr::nx2212::SanitizeStepStem;
using brepmfr::nx2212::StablePathHash;
using brepmfr::nx2212::test::Expect;
using brepmfr::nx2212::test::ExpectEqual;
}

BREPMFR_TEST(stable_path_hash_uses_normalized_lowercase_generic_path)
{
    ExpectEqual(
        StablePathHash(L"C:\\Parts\\Widget.step"),
        std::uint64_t{0xefa652266344ab46ULL},
        "normalized path hash changed");
}

BREPMFR_TEST(step_stem_replaces_windows_invalid_filename_characters)
{
    ExpectEqual(
        SanitizeStepStem(L"a<b>:c"),
        std::wstring(L"a_b__c"),
        "invalid filename characters were not replaced");
    ExpectEqual(
        SanitizeStepStem(L"part. "),
        std::wstring(L"part__"),
        "trailing dot or space was not replaced");
    ExpectEqual(
        SanitizeStepStem(L""),
        std::wstring(L"step"),
        "empty stem did not receive a safe name");
}

BREPMFR_TEST(output_paths_separate_same_stem_from_different_directories)
{
    const auto left = BuildOutputPaths(
        L"C:\\out", L"C:\\a\\part.step");
    const auto right = BuildOutputPaths(
        L"C:\\out", L"D:\\b\\part.step");

    Expect(left.directory != right.directory, "same-name STEP paths collided");
    ExpectEqual(
        left.result_path.filename(),
        std::filesystem::path(L"result.json"),
        "result filename changed");
    ExpectEqual(
        left.bin_path.parent_path(),
        left.directory,
        "BIN directory changed");
    ExpectEqual(
        left.metadata_path.parent_path(),
        left.directory,
        "metadata directory changed");
    ExpectEqual(
        left.result_path.parent_path(),
        left.directory,
        "result directory changed");
    ExpectEqual(
        left.log_path.parent_path(),
        left.directory,
        "log directory changed");
}
