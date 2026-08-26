#include "color_plan.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
ColorPlan Failure(const std::string& diagnostic)
{
    ColorPlan result;
    result.diagnostic = diagnostic;
    return result;
}
}

ColorPlan BuildColorPlan(
    const SuccessResult& recognition,
    const FaceMappingResult& mapping)
{
    if (!mapping.success)
    {
        return Failure(
            mapping.diagnostic.empty()
                ? "face mapping was not successful"
                : "face mapping was not successful: " + mapping.diagnostic);
    }
    if (recognition.face_count < 0)
    {
        return Failure("recognition face count is negative");
    }
    if (recognition.instances.size() > 216)
    {
        return Failure("feature instance count exceeds 216 distinct NX colors");
    }
    if (mapping.matches.size() !=
        static_cast<std::size_t>(recognition.face_count))
    {
        return Failure("face mapping does not cover every recognition face");
    }

    std::map<int, FaceKey> tag_for_face;
    std::set<FaceKey> unique_tags;
    for (const auto& match : mapping.matches)
    {
        if (match.occ_face_index < 0 ||
            match.occ_face_index >= recognition.face_count)
        {
            return Failure("face mapping contains an out-of-range OCC face index");
        }
        if (!tag_for_face.emplace(
                match.occ_face_index, match.nx_face_tag).second)
        {
            return Failure("face mapping contains duplicate OCC face indices");
        }
        if (!unique_tags.insert(match.nx_face_tag).second)
        {
            return Failure("face mapping contains duplicate NX Face Tags");
        }
    }
    for (int face_index = 0; face_index < recognition.face_count; ++face_index)
    {
        if (tag_for_face.find(face_index) == tag_for_face.end())
        {
            return Failure("face mapping is missing a recognition face");
        }
    }

    auto instances = recognition.instances;
    std::sort(
        instances.begin(), instances.end(),
        [](const InstanceResult& left, const InstanceResult& right) {
            return left.instance_id < right.instance_id;
        });
    std::set<int> unique_instance_ids;
    std::set<int> assigned_occ_faces;
    std::vector<InstanceColorRecord> pending_instances;
    std::vector<FaceColorAssignment> pending_assignments;
    for (std::size_t ordinal = 0; ordinal < instances.size(); ++ordinal)
    {
        const auto& instance = instances[ordinal];
        if (instance.instance_id < 0 ||
            !unique_instance_ids.insert(instance.instance_id).second)
        {
            return Failure("feature instance IDs must be unique and nonnegative");
        }
        if (instance.face_indices.empty())
        {
            return Failure("feature instance does not contain any faces");
        }
        InstanceColorRecord record;
        record.instance_id = instance.instance_id;
        record.color_index = static_cast<int>(ordinal + 1);
        std::set<int> instance_faces;
        for (const int face_index : instance.face_indices)
        {
            if (face_index < 0 || face_index >= recognition.face_count ||
                !instance_faces.insert(face_index).second ||
                !assigned_occ_faces.insert(face_index).second)
            {
                return Failure(
                    "feature instances contain invalid or duplicate face references");
            }
            record.nx_face_tags.push_back(tag_for_face.at(face_index));
        }
        std::sort(record.nx_face_tags.begin(), record.nx_face_tags.end());
        for (const FaceKey tag : record.nx_face_tags)
        {
            pending_assignments.push_back(
                {tag, record.instance_id, record.color_index});
        }
        pending_instances.push_back(std::move(record));
    }

    ColorPlan result;
    result.success = true;
    result.assignments = std::move(pending_assignments);
    result.instances = std::move(pending_instances);
    return result;
}
}
}
