#include "color_plan.hpp"

#include "test_support.hpp"

#include <sstream>
#include <string>

namespace
{
using brepmfr::nx2212::ColorPlan;
using brepmfr::nx2212::FaceMappingResult;
using brepmfr::nx2212::SuccessResult;

SuccessResult SampleRecognition()
{
    SuccessResult result;
    result.face_count = 4;
    result.faces = {
        {0, 12, "blind_hole", 0.95, 0},
        {1, 2, "triangular_passage", 0.90, 1},
        {2, 12, "blind_hole", 0.93, 0},
        {3, 24, "stock", 0.99, std::nullopt}};
    result.instances = {
        {1, 2, "triangular_passage", {1}, 0.90},
        {0, 12, "blind_hole", {0, 2}, 0.94}};
    return result;
}

FaceMappingResult SampleMapping()
{
    FaceMappingResult result;
    result.success = true;
    result.matches = {
        {2, 502, 0.0, 0.0},
        {0, 503, 0.0, 0.0},
        {3, 504, 0.0, 0.0},
        {1, 501, 0.0, 0.0}};
    return result;
}

std::string SerializePlan(const ColorPlan& plan)
{
    std::ostringstream stream;
    for (const auto& assignment : plan.assignments)
    {
        stream << assignment.nx_face_tag << ':' << assignment.instance_id
               << ':' << assignment.color_index << ';';
    }
    stream << '|';
    for (const auto& instance : plan.instances)
    {
        stream << instance.instance_id << ':' << instance.color_index << ':';
        for (const auto tag : instance.nx_face_tags)
        {
            stream << tag << ',';
        }
        stream << ';';
    }
    return stream.str();
}

void ExpectClosedFailure(const ColorPlan& plan, const std::string& message)
{
    using namespace brepmfr::nx2212::test;
    Expect(!plan.success, message + " was accepted");
    Expect(plan.assignments.empty(), message + " leaked assignments");
    Expect(plan.instances.empty(), message + " leaked instance colors");
    Expect(!plan.diagnostic.empty(), message + " has no diagnostic");
}
}

BREPMFR_TEST(ColorPlanAssignsOneDeterministicDistinctColorPerInstance)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto first = BuildColorPlan(SampleRecognition(), SampleMapping());
    const auto second = BuildColorPlan(SampleRecognition(), SampleMapping());

    Expect(first.success, first.diagnostic);
    ExpectEqual(SerializePlan(first), SerializePlan(second),
                "repeated color plans differ");
    ExpectEqual(first.instances.size(), static_cast<std::size_t>(2),
                "instance color count mismatch");
    ExpectEqual(first.instances[0].instance_id, 0, "instances were not sorted");
    ExpectEqual(first.instances[0].color_index, 1, "first color mismatch");
    ExpectEqual(first.instances[0].nx_face_tags,
                std::vector<FaceKey>{502, 503}, "first tags were not sorted");
    ExpectEqual(first.instances[1].instance_id, 1, "instances were not sorted");
    ExpectEqual(first.instances[1].color_index, 2, "second color mismatch");
    ExpectEqual(first.instances[1].nx_face_tags,
                std::vector<FaceKey>{501}, "second tags mismatch");
    ExpectEqual(first.assignments.size(), static_cast<std::size_t>(3),
                "stock face was assigned a feature color");
    ExpectEqual(first.assignments[0].nx_face_tag, FaceKey{502},
                "assignments were not sorted by instance and tag");
    ExpectEqual(first.assignments[1].nx_face_tag, FaceKey{503},
                "assignments were not sorted by instance and tag");
    ExpectEqual(first.assignments[2].nx_face_tag, FaceKey{501},
                "second instance assignment mismatch");
}

BREPMFR_TEST(ColorPlanRejectsFailedOrIncompleteFaceMappings)
{
    using namespace brepmfr::nx2212;

    auto failed = SampleMapping();
    failed.success = false;
    failed.diagnostic = "ambiguous mapping";
    ExpectClosedFailure(
        BuildColorPlan(SampleRecognition(), failed), "failed face mapping");

    auto incomplete = SampleMapping();
    incomplete.matches.erase(incomplete.matches.begin());
    ExpectClosedFailure(
        BuildColorPlan(SampleRecognition(), incomplete), "missing mapped face");
}

BREPMFR_TEST(ColorPlanRejectsMoreInstancesThanDistinctNxColors)
{
    using namespace brepmfr::nx2212;

    SuccessResult recognition;
    recognition.face_count = 217;
    FaceMappingResult mapping;
    mapping.success = true;
    for (int index = 0; index < 217; ++index)
    {
        recognition.instances.push_back(
            {index, 12, "feature", {index}, 1.0});
        mapping.matches.push_back(
            {index, static_cast<FaceKey>(1000 + index), 0.0, 0.0});
    }
    ExpectClosedFailure(
        BuildColorPlan(recognition, mapping), "217 feature instances");
}
