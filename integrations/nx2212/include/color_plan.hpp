#pragma once

#include "face_matcher.hpp"
#include "result_reader.hpp"

#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
struct FaceColorAssignment
{
    FaceKey nx_face_tag;
    int instance_id;
    int color_index;
};

struct InstanceColorRecord
{
    int instance_id;
    int color_index;
    std::vector<FaceKey> nx_face_tags;
};

struct ColorPlan
{
    bool success = false;
    std::vector<FaceColorAssignment> assignments;
    std::vector<InstanceColorRecord> instances;
    std::string diagnostic;
};

ColorPlan BuildColorPlan(
    const SuccessResult& recognition,
    const FaceMappingResult& mapping);
}
}
