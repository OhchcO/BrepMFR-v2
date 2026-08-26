#include "face_matcher.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace
{
using brepmfr::nx2212::AdjacencyEdge;
using brepmfr::nx2212::FaceGraph;
using brepmfr::nx2212::FaceKey;
using brepmfr::nx2212::FaceMappingResult;
using brepmfr::nx2212::FaceSignature;
using brepmfr::nx2212::SurfaceType;
using brepmfr::nx2212::Vec3;

FaceSignature MakeFace(FaceKey key, double marker, int neighbor_count)
{
    FaceSignature face{};
    face.key = key;
    face.surface_type = SurfaceType::Plane;
    face.area = 1.0 + marker;
    face.perimeter = 4.0 + marker;
    face.centroid = {marker * 0.1, marker * 0.02, 0.0};
    face.bounds = {
        {face.centroid.x - 0.01, face.centroid.y - 0.01, 0.0},
        {face.centroid.x + 0.01, face.centroid.y + 0.01, 0.0}};
    face.loop_count = 1;
    face.edge_count = 4;
    face.neighbor_count = neighbor_count;
    face.surface.origin = face.centroid;
    face.surface.axis = Vec3{0.0, 0.0, 1.0};
    face.sample_points = {
        face.centroid,
        {face.centroid.x + 0.005, face.centroid.y, 0.0}};
    return face;
}

FaceGraph MakeCycleGraph(const std::vector<FaceKey>& keys)
{
    FaceGraph graph;
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        graph.faces.push_back(MakeFace(
            keys[index], static_cast<double>(index), 2));
    }
    const auto add_edge = [&](std::size_t left, std::size_t right) {
        auto edge = AdjacencyEdge{keys[left], keys[right]};
        if (edge.first > edge.second)
        {
            std::swap(edge.first, edge.second);
        }
        graph.adjacency_edges.push_back(edge);
    };
    add_edge(0, 1);
    add_edge(1, 2);
    add_edge(2, 3);
    add_edge(3, 0);
    std::sort(graph.adjacency_edges.begin(), graph.adjacency_edges.end());
    return graph;
}

FaceGraph MakeNxEquivalent(const FaceGraph& occ)
{
    FaceGraph nx = occ;
    const std::vector<FaceKey> tags{101, 102, 103, 104};
    for (std::size_t index = 0; index < nx.faces.size(); ++index)
    {
        nx.faces[index].key = tags[index];
    }
    nx.adjacency_edges = {
        {101, 102}, {101, 104}, {102, 103}, {103, 104}};
    return nx;
}

FaceGraph PermuteFaces(const FaceGraph& graph, const std::vector<std::size_t>& order)
{
    FaceGraph result = graph;
    result.faces.clear();
    for (const std::size_t index : order)
    {
        result.faces.push_back(graph.faces.at(index));
    }
    return result;
}

FaceKey TagFor(const FaceMappingResult& result, int occ_face_index)
{
    const auto found = std::find_if(
        result.matches.begin(), result.matches.end(),
        [occ_face_index](const auto& match) {
            return match.occ_face_index == occ_face_index;
        });
    if (found == result.matches.end())
    {
        throw std::runtime_error("OCC face was not matched");
    }
    return found->nx_face_tag;
}

FaceGraph MakeIsolatedGraph(
    const std::vector<FaceKey>& keys,
    const std::vector<double>& markers)
{
    FaceGraph graph;
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        graph.faces.push_back(MakeFace(keys[index], markers.at(index), 0));
    }
    return graph;
}

void ExpectClosedFailure(
    const FaceMappingResult& result,
    const std::string& message)
{
    using namespace brepmfr::nx2212::test;
    Expect(!result.success, message + " was accepted");
    Expect(result.matches.empty(), message + " leaked partial matches");
    Expect(!result.diagnostic.empty(), message + " has no diagnostic");
}
}

BREPMFR_TEST(FaceMatcherMapsPermutedFacesBySignatureNotVectorPosition)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto occ = MakeCycleGraph({0, 1, 2, 3});
    const auto nx = PermuteFaces(MakeNxEquivalent(occ), {2, 0, 3, 1});
    const auto result = MatchFaces(occ, nx);

    Expect(result.success, result.diagnostic);
    ExpectEqual(result.matches.size(), static_cast<std::size_t>(4),
                "match count mismatch");
    ExpectEqual(TagFor(result, 0), FaceKey{101}, "face 0 tag mismatch");
    ExpectEqual(TagFor(result, 1), FaceKey{102}, "face 1 tag mismatch");
    ExpectEqual(TagFor(result, 2), FaceKey{103}, "face 2 tag mismatch");
    ExpectEqual(TagFor(result, 3), FaceKey{104}, "face 3 tag mismatch");
    ExpectNear(result.total_cost, 0.0, 1e-12, "exact mapping cost mismatch");
    ExpectNear(result.max_error, 0.0, 1e-12, "exact mapping error mismatch");
}

BREPMFR_TEST(FaceMatcherAcceptsSmallNormalizedGeometryPerturbations)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto occ = MakeCycleGraph({0, 1, 2, 3});
    auto nx = PermuteFaces(MakeNxEquivalent(occ), {3, 1, 0, 2});
    for (auto& face : nx.faces)
    {
        face.area *= 1.0 + 1e-7;
        face.perimeter *= 1.0 + 1e-7;
        face.centroid.x += 1e-7;
        face.bounds.minimum.x += 1e-7;
        face.bounds.maximum.x += 1e-7;
        face.surface.origin->x += 1e-7;
        for (auto& sample : face.sample_points)
        {
            sample.x += 1e-7;
        }
    }

    const auto result = MatchFaces(occ, nx);
    Expect(result.success, result.diagnostic);
    ExpectEqual(TagFor(result, 0), FaceKey{101}, "perturbed face 0 mismatch");
    ExpectEqual(TagFor(result, 3), FaceKey{104}, "perturbed face 3 mismatch");
    Expect(result.max_error > 0.0, "perturbation error was not recorded");
    Expect(result.max_error < 0.02, "small perturbation exceeded normalized error");
}

BREPMFR_TEST(FaceMatcherRejectsDifferentFaceCountsWithoutPartialMatches)
{
    using namespace brepmfr::nx2212;

    const auto occ = MakeIsolatedGraph({0}, {0.0});
    const auto extra_nx = MakeIsolatedGraph({101, 102}, {0.0, 1.0});
    ExpectClosedFailure(MatchFaces(occ, extra_nx), "extra NX face");
    ExpectClosedFailure(MatchFaces(extra_nx, occ), "missing NX face");
}

BREPMFR_TEST(FaceMatcherRejectsIncompatibleSurfaceTypeWithoutPartialMatches)
{
    using namespace brepmfr::nx2212;

    const auto occ = MakeIsolatedGraph({0}, {0.0});
    auto nx = MakeIsolatedGraph({101}, {0.0});
    nx.faces[0].surface_type = SurfaceType::Cylinder;
    nx.faces[0].surface.radius = 1.0;
    ExpectClosedFailure(MatchFaces(occ, nx), "incompatible surface type");
}

BREPMFR_TEST(FaceMatcherRejectsGeometryOutsideToleranceWithoutPartialMatches)
{
    using namespace brepmfr::nx2212;

    const auto occ = MakeIsolatedGraph({0}, {0.0});
    auto nx = MakeIsolatedGraph({101}, {0.0});
    nx.faces[0].centroid.x += 1e-3;
    ExpectClosedFailure(MatchFaces(occ, nx), "out-of-tolerance centroid");
}

BREPMFR_TEST(FaceMatcherRejectsAssignmentThatBreaksAdjacency)
{
    using namespace brepmfr::nx2212;

    const auto occ = MakeCycleGraph({0, 1, 2, 3});
    auto nx = MakeNxEquivalent(occ);
    nx.adjacency_edges = {
        {101, 102}, {101, 103}, {102, 104}, {103, 104}};
    ExpectClosedFailure(MatchFaces(occ, nx), "broken NX adjacency");
}

BREPMFR_TEST(FaceMatcherUsesTopologyToRejectInvalidGeometricTie)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    FaceGraph occ;
    occ.faces = {
        MakeFace(0, 0.0, 1),
        MakeFace(1, 0.0, 1),
        MakeFace(2, 2.0, 1),
        MakeFace(3, 3.0, 1)};
    occ.adjacency_edges = {{0, 2}, {1, 3}};

    FaceGraph nx = occ;
    nx.faces[0].key = 101;
    nx.faces[1].key = 102;
    nx.faces[2].key = 103;
    nx.faces[3].key = 104;
    nx.adjacency_edges = {{101, 103}, {102, 104}};

    const auto result = MatchFaces(occ, nx);
    Expect(result.success, result.diagnostic);
    ExpectEqual(TagFor(result, 0), FaceKey{101}, "topology-resolved face 0 mismatch");
    ExpectEqual(TagFor(result, 1), FaceKey{102}, "topology-resolved face 1 mismatch");
}

BREPMFR_TEST(FaceMatcherRejectsTopologyValidAmbiguity)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const auto occ = MakeIsolatedGraph({0, 1}, {0.0, 0.0});
    const auto nx = MakeIsolatedGraph({101, 102}, {0.0, 0.0});
    const auto result = MatchFaces(occ, nx);

    ExpectClosedFailure(result, "ambiguous mapping");
    ExpectContains(result.diagnostic, "ambiguous", "ambiguity reason missing");
}
