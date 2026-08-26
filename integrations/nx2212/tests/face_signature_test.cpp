#include "face_signature.hpp"

#include "test_support.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
using brepmfr::nx2212::FaceGraph;
using brepmfr::nx2212::FaceKey;
using brepmfr::nx2212::FaceSignature;
using brepmfr::nx2212::SurfaceType;
using brepmfr::nx2212::Vec3;

FaceSignature MakeFace(FaceKey key, int neighbor_count)
{
    FaceSignature face{};
    face.key = key;
    face.surface_type = SurfaceType::Plane;
    face.area = 1.0;
    face.perimeter = 4.0;
    face.centroid = {static_cast<double>(key), 0.0, 0.0};
    face.bounds = {{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
    face.loop_count = 1;
    face.edge_count = 4;
    face.neighbor_count = neighbor_count;
    face.surface.origin = Vec3{0.0, 0.0, 0.0};
    face.surface.axis = Vec3{0.0, 0.0, 1.0};
    face.sample_points = {{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
    return face;
}

FaceGraph MakeValidGraph()
{
    FaceGraph graph;
    graph.faces = {MakeFace(0, 1), MakeFace(1, 1)};
    graph.adjacency_edges = {{0, 1}};
    return graph;
}

void ExpectInvalidArgument(
    const std::function<void()>& operation,
    const std::string& message)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    brepmfr::nx2212::test::Expect(rejected, message);
}
}

BREPMFR_TEST(UnitBoxTransformUsesBodyCenterAndLongestExtent)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const Bounds3 bounds{{10.0, 20.0, 30.0}, {14.0, 22.0, 31.0}};
    const auto transform = MakeUnitBoxTransform(bounds);
    ExpectEqual(transform.center, Vec3{12.0, 21.0, 30.5}, "center mismatch");
    ExpectNear(transform.scale, 0.5, 1e-12, "scale mismatch");
    const auto minimum = NormalizePoint(transform, bounds.minimum);
    ExpectNear(minimum.x, -1.0, 1e-12, "normalized x mismatch");
    ExpectNear(minimum.y, -0.5, 1e-12, "normalized y mismatch");
    ExpectNear(minimum.z, -0.25, 1e-12, "normalized z mismatch");
    ExpectNear(NormalizeLength(transform, 4.0), 2.0, 1e-12, "length mismatch");
    ExpectNear(NormalizeArea(transform, 4.0), 1.0, 1e-12, "area mismatch");
}

BREPMFR_TEST(UnitBoxTransformRejectsDegenerateOrNonFiniteBounds)
{
    using namespace brepmfr::nx2212;

    ExpectInvalidArgument(
        [] { (void)MakeUnitBoxTransform({{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}); },
        "zero-extent bounds were accepted");
    ExpectInvalidArgument(
        [] {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            (void)MakeUnitBoxTransform({{nan, 0.0, 0.0}, {1.0, 1.0, 1.0}});
        },
        "NaN bounds were accepted");
}

BREPMFR_TEST(CanonicalAxisNormalizesDirectionAndSign)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto z = CanonicalAxis({0.0, 0.0, -2.0});
    ExpectNear(z.x, 0.0, 1e-12, "z axis x mismatch");
    ExpectNear(z.y, 0.0, 1e-12, "z axis y mismatch");
    ExpectNear(z.z, 1.0, 1e-12, "z axis sign mismatch");
    const auto x = CanonicalAxis({-3.0, 0.0, 0.0});
    ExpectNear(x.x, 1.0, 1e-12, "x axis sign mismatch");
    ExpectInvalidArgument(
        [] { (void)CanonicalAxis({0.0, 0.0, 0.0}); },
        "zero axis was accepted");
}

BREPMFR_TEST(PlaneAndAxisOriginsUseDifferentCanonicalProjections)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const Vec3 axis{0.0, 0.0, 1.0};
    ExpectEqual(
        CanonicalPlaneOrigin({3.0, 4.0, 5.0}, axis),
        Vec3{0.0, 0.0, 5.0},
        "plane origin is not the closest point on the plane");
    ExpectEqual(
        CanonicalPlaneOrigin({8.0, -2.0, 5.0}, axis),
        Vec3{0.0, 0.0, 5.0},
        "plane origin depends on the kernel reference point");
    ExpectEqual(
        CanonicalAxisOrigin({3.0, 4.0, 5.0}, axis),
        Vec3{3.0, 4.0, 0.0},
        "axis origin is not the closest point on the axis");
}

BREPMFR_TEST(FaceGridSamplingIncludesOnlyStrictlyInteriorUvPoints)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    Expect(
        IncludeFaceGridSample(1),
        "strictly interior UV point was excluded");
    Expect(
        !IncludeFaceGridSample(2),
        "outside UV point was included");
    Expect(
        !IncludeFaceGridSample(3),
        "boundary UV point was included");
}

BREPMFR_TEST(ValidateFaceGraphAcceptsConsistentGraph)
{
    brepmfr::nx2212::ValidateFaceGraph(MakeValidGraph());
}

BREPMFR_TEST(ValidateFaceGraphRejectsStructuralCorruption)
{
    using namespace brepmfr::nx2212;

    auto duplicate_key = MakeValidGraph();
    duplicate_key.faces[1].key = 0;
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(duplicate_key); },
        "duplicate face key was accepted");

    auto self_loop = MakeValidGraph();
    self_loop.adjacency_edges = {{0, 0}};
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(self_loop); },
        "self-loop was accepted");

    auto duplicate_edge = MakeValidGraph();
    duplicate_edge.adjacency_edges.push_back({0, 1});
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(duplicate_edge); },
        "duplicate adjacency edge was accepted");

    auto unknown_endpoint = MakeValidGraph();
    unknown_endpoint.adjacency_edges = {{0, 2}};
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(unknown_endpoint); },
        "unknown adjacency endpoint was accepted");

    auto empty_samples = MakeValidGraph();
    empty_samples.faces[0].sample_points.clear();
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(empty_samples); },
        "empty sample points were accepted");

    auto wrong_degree = MakeValidGraph();
    wrong_degree.faces[0].neighbor_count = 0;
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(wrong_degree); },
        "neighbor count mismatch was accepted");

    auto non_finite = MakeValidGraph();
    non_finite.faces[0].area = std::numeric_limits<double>::quiet_NaN();
    ExpectInvalidArgument(
        [&] { ValidateFaceGraph(non_finite); },
        "non-finite signature value was accepted");
}
