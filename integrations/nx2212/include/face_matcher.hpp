#pragma once

#include "face_signature.hpp"

#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
struct FaceMatchOptions
{
    double relative_scalar_tolerance = 1e-4;
    double normalized_position_tolerance = 1e-5;
    double axis_angle_tolerance = 1e-6;
    double sample_distance_tolerance = 2e-3;
    double ambiguity_cost_margin = 1e-8;
};

struct FaceMatch
{
    int occ_face_index;
    FaceKey nx_face_tag;
    double cost;
    double max_normalized_error;
};

struct FaceMappingResult
{
    bool success = false;
    std::vector<FaceMatch> matches;
    double total_cost = 0.0;
    double max_error = 0.0;
    std::string diagnostic;
};

FaceMappingResult MatchFaces(
    const FaceGraph& occ_graph,
    const FaceGraph& nx_graph,
    const FaceMatchOptions& options = {});
}
}
