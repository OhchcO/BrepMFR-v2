#include "instance_browser.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace brepmfr::nx2212;
using namespace brepmfr::nx2212::test;

SuccessResult BrowserRecognition()
{
    SuccessResult recognition;
    recognition.face_count = 5;
    recognition.cache_hit = true;
    recognition.total_seconds = 3.4;
    recognition.instances = {
        {7, 3, "rectangular_pocket", {3, 4}, 0.875},
        {2, 12, "blind_hole", {1}, 0.95}};
    return recognition;
}

MappingDisplay BrowserMapping()
{
    MappingDisplay mapping;
    mapping.status = MappingDisplayStatus::Colored;
    mapping.mapped_faces = 5;
    mapping.colored_instances = 2;
    mapping.elapsed = std::chrono::milliseconds(300);
    mapping.instances = {
        {7, 2, {507, 508}},
        {2, 1, {501}}};
    return mapping;
}

void ExpectBuildFailure(
    const SuccessResult& recognition,
    const MappingDisplay& mapping,
    const std::string& expected)
{
    try
    {
        (void)BuildInstanceBrowserModel(
            recognition,
            std::chrono::milliseconds(9100),
            mapping,
            DisplayText{"summary", {"line one", "line two"}});
        throw std::runtime_error("browser model unexpectedly succeeded");
    }
    catch (const std::invalid_argument& error)
    {
        ExpectContains(
            error.what(),
            expected,
            "wrong model diagnostic; expected '" + expected +
                "' in '" + error.what() + "'");
    }
}
}

BREPMFR_TEST(InstanceBrowserModelJoinsAndSortsRecognitionRows)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto model = BuildInstanceBrowserModel(
        BrowserRecognition(),
        std::chrono::milliseconds(9100),
        BrowserMapping(),
        DisplayText{"summary", {"line one", "line two"}});

    ExpectEqual(model.rows.size(), std::size_t{2}, "row count mismatch");
    ExpectEqual(model.rows[0].instance_id, 2, "rows were not ID sorted");
    ExpectEqual(model.rows[0].feature_name_utf8, std::string("blind_hole"),
                "feature name mismatch");
    ExpectEqual(model.rows[0].occ_face_indices, std::vector<int>{1},
                "OCC face indices mismatch");
    ExpectEqual(model.rows[0].nx_face_tags, std::vector<FaceKey>{501},
                "row tags mismatch");
    ExpectEqual(model.rows[1].instance_id, 7, "second row mismatch");
    ExpectEqual(model.rows[1].nx_face_tags, std::vector<FaceKey>({507, 508}),
                "second row tags mismatch");
    ExpectEqual(model.summary.face_count, 5, "summary face count mismatch");
    ExpectEqual(model.summary.instance_count, std::size_t{2},
                "summary instance count mismatch");
    Expect(model.summary.cache_hit, "summary cache state mismatch");
    ExpectNear(model.summary.pipeline_seconds, 3.4, 1e-12,
               "pipeline timing mismatch");
    ExpectEqual(model.summary.process_elapsed,
                std::chrono::milliseconds(9100), "process timing mismatch");
    ExpectEqual(model.summary.mapped_faces, std::size_t{5},
                "mapped face count mismatch");
    ExpectEqual(model.summary.colored_instances, std::size_t{2},
                "colored instance count mismatch");
    ExpectEqual(model.summary.mapping_elapsed,
                std::chrono::milliseconds(300), "mapping timing mismatch");
    ExpectEqual(model.complete_text.listing_lines_utf8.size(), std::size_t{2},
                "complete text was not retained");
}

BREPMFR_TEST(InstanceBrowserModelRejectsInconsistentMappingRecords)
{
    using namespace brepmfr::nx2212;

    {
        auto mapping = BrowserMapping();
        mapping.status = MappingDisplayStatus::Failed;
        ExpectBuildFailure(BrowserRecognition(), mapping, "colored mapping");
    }
    {
        auto recognition = BrowserRecognition();
        recognition.instances.push_back(recognition.instances.front());
        ExpectBuildFailure(recognition, BrowserMapping(),
                           "duplicate recognition instance");
    }
    {
        auto mapping = BrowserMapping();
        mapping.instances.pop_back();
        ExpectBuildFailure(BrowserRecognition(), mapping,
                           "missing color record");
    }
    {
        auto mapping = BrowserMapping();
        mapping.instances.push_back({99, 4, {599}});
        ExpectBuildFailure(BrowserRecognition(), mapping,
                           "unknown recognition instance");
    }
    {
        auto mapping = BrowserMapping();
        mapping.instances.front().nx_face_tags.clear();
        ExpectBuildFailure(BrowserRecognition(), mapping, "has no NX faces");
    }
    {
        auto mapping = BrowserMapping();
        mapping.instances.front().nx_face_tags = {507, 507};
        ExpectBuildFailure(BrowserRecognition(), mapping,
                           "duplicate NX face tag");
    }
    {
        auto mapping = BrowserMapping();
        mapping.instances.front().nx_face_tags = {507, 501};
        ExpectBuildFailure(BrowserRecognition(), mapping,
                           "duplicate NX face tag");
    }
}

BREPMFR_TEST(InstanceBrowserPaginationHandlesExactBoundaries)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    struct PageCase
    {
        std::size_t rows;
        std::size_t pages;
        bool visible;
    };
    const std::vector<PageCase> cases = {
        {0, 1, false}, {1, 1, false}, {50, 1, false},
        {51, 2, true}, {100, 2, true}, {101, 3, true}};

    for (const auto& item : cases)
    {
        InstanceBrowserState state(item.rows);
        ExpectEqual(state.CurrentPage(), std::size_t{0},
                    "initial page mismatch");
        ExpectEqual(state.PageCount(), item.pages, "page count mismatch");
        ExpectEqual(state.PaginationVisible(), item.visible,
                    "pagination visibility mismatch");
        ExpectEqual(state.PageBegin(), std::size_t{0},
                    "first page begin mismatch");
        ExpectEqual(state.PageEnd(),
                    (std::min)(item.rows, kInstanceBrowserPageSize),
                    "first page end mismatch");
        Expect(!state.CanGoPrevious(),
               "previous was enabled on first page");
        ExpectEqual(state.CanGoNext(), item.pages > 1,
                    "next enablement mismatch");
    }
}

BREPMFR_TEST(InstanceBrowserPaginationNavigatesAndClearsSelection)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    InstanceBrowserState state(101);
    state.SelectGlobalIndex(4);
    Expect(state.SelectedGlobalIndex().has_value(),
           "selection was not recorded");
    Expect(state.GoNext(), "first next navigation failed");
    ExpectEqual(state.PageBegin(), std::size_t{50},
                "second page begin mismatch");
    ExpectEqual(state.PageEnd(), std::size_t{100},
                "second page end mismatch");
    Expect(!state.SelectedGlobalIndex().has_value(),
           "navigation did not clear selection");

    ExpectEqual(state.GlobalIndexForVisibleRow(0), std::size_t{50},
                "visible row mapping mismatch");
    state.SelectGlobalIndex(99);
    Expect(state.GoNext(), "second next navigation failed");
    ExpectEqual(state.PageBegin(), std::size_t{100},
                "third page begin mismatch");
    ExpectEqual(state.PageEnd(), std::size_t{101},
                "third page end mismatch");
    Expect(!state.CanGoNext(), "next enabled on last page");
    Expect(!state.GoNext(), "navigation moved beyond last page");

    Expect(state.GoPrevious(), "previous navigation failed");
    ExpectEqual(state.PageBegin(), std::size_t{50},
                "previous page begin mismatch");
    Expect(state.GoPrevious(), "return to first page failed");
    ExpectEqual(state.PageBegin(), std::size_t{0},
                "first page begin mismatch after return");
    Expect(!state.CanGoPrevious(), "previous enabled on first page");
    Expect(!state.GoPrevious(), "navigation moved before first page");

    try
    {
        (void)state.GlobalIndexForVisibleRow(50);
        throw std::runtime_error("out-of-range visible row was accepted");
    }
    catch (const std::out_of_range&)
    {
    }
    try
    {
        state.SelectGlobalIndex(101);
        throw std::runtime_error("out-of-range global row was accepted");
    }
    catch (const std::out_of_range&)
    {
    }
}
