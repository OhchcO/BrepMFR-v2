#include "result_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace brepmfr
{
namespace nx2212
{
namespace
{
using Json = nlohmann::json;

[[noreturn]] void InvalidResult(const std::string& message)
{
    throw std::runtime_error("invalid inference result: " + message);
}

const Json& RequireField(
    const Json& object,
    const char* field,
    const std::string& context)
{
    if (!object.is_object())
    {
        InvalidResult(context + " must be an object");
    }
    const auto found = object.find(field);
    if (found == object.end())
    {
        InvalidResult(context + "." + field + " is required");
    }
    return *found;
}

const Json& RequireObjectField(
    const Json& object,
    const char* field,
    const std::string& context)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_object())
    {
        InvalidResult(context + "." + field + " must be an object");
    }
    return value;
}

const Json& RequireArrayField(
    const Json& object,
    const char* field,
    const std::string& context)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_array())
    {
        InvalidResult(context + "." + field + " must be an array");
    }
    return value;
}

std::string RequireString(
    const Json& object,
    const char* field,
    const std::string& context,
    bool require_nonempty = false)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_string())
    {
        InvalidResult(context + "." + field + " must be a string");
    }
    const auto text = value.get<std::string>();
    if (require_nonempty && text.empty())
    {
        InvalidResult(context + "." + field + " must not be empty");
    }
    return text;
}

int RequireInteger(
    const Json& object,
    const char* field,
    const std::string& context)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_number_integer())
    {
        InvalidResult(context + "." + field + " must be an integer");
    }

    if (value.is_number_unsigned())
    {
        const auto unsigned_value = value.get<std::uint64_t>();
        if (unsigned_value >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        {
            InvalidResult(context + "." + field + " is outside int range");
        }
        return static_cast<int>(unsigned_value);
    }

    const auto signed_value = value.get<std::int64_t>();
    if (signed_value < std::numeric_limits<int>::min() ||
        signed_value > std::numeric_limits<int>::max())
    {
        InvalidResult(context + "." + field + " is outside int range");
    }
    return static_cast<int>(signed_value);
}

double RequireFiniteNumber(
    const Json& object,
    const char* field,
    const std::string& context)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_number())
    {
        InvalidResult(context + "." + field + " must be a number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number))
    {
        InvalidResult(context + "." + field + " must be finite");
    }
    return number;
}

bool RequireBoolean(
    const Json& object,
    const char* field,
    const std::string& context)
{
    const auto& value = RequireField(object, field, context);
    if (!value.is_boolean())
    {
        InvalidResult(context + "." + field + " must be a boolean");
    }
    return value.get<bool>();
}

void RequireExactFields(
    const Json& object,
    std::initializer_list<const char*> expected,
    const std::string& context)
{
    if (!object.is_object())
    {
        InvalidResult(context + " must be an object");
    }
    std::set<std::string> fields;
    for (const auto& item : object.items())
    {
        fields.insert(item.key());
    }
    std::set<std::string> expected_fields;
    for (const char* field : expected)
    {
        expected_fields.insert(field);
    }
    if (fields != expected_fields)
    {
        InvalidResult(context + " fields do not match the contract");
    }
}

double RequireFiniteValue(const Json& value, const std::string& context)
{
    if (!value.is_number())
    {
        InvalidResult(context + " must be a number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number))
    {
        InvalidResult(context + " must be finite");
    }
    return number;
}

int RequireNonnegativeIntegerValue(
    const Json& value,
    const std::string& context)
{
    Json wrapper = {{"value", value}};
    const int result = RequireInteger(wrapper, "value", context);
    if (result < 0)
    {
        InvalidResult(context + " must be nonnegative");
    }
    return result;
}

Vec3 ReadVec3(const Json& value, const std::string& context)
{
    if (!value.is_array() || value.size() != 3)
    {
        InvalidResult(context + " must contain exactly three numbers");
    }
    return {
        RequireFiniteValue(value[0], context + "[0]"),
        RequireFiniteValue(value[1], context + "[1]"),
        RequireFiniteValue(value[2], context + "[2]")};
}

SurfaceType ReadSurfaceType(const Json& face, const std::string& context)
{
    const auto value = RequireString(face, "surface_type", context, true);
    if (value == "plane") return SurfaceType::Plane;
    if (value == "cylinder") return SurfaceType::Cylinder;
    if (value == "cone") return SurfaceType::Cone;
    if (value == "sphere") return SurfaceType::Sphere;
    if (value == "torus") return SurfaceType::Torus;
    if (value == "revolution") return SurfaceType::Revolution;
    if (value == "bspline") return SurfaceType::BSpline;
    if (value == "other") return SurfaceType::Other;
    InvalidResult(context + ".surface_type is unsupported");
}

SurfaceParameters ReadSurfaceParameters(
    const Json& face,
    SurfaceType type,
    const std::string& context)
{
    const auto& parameters =
        RequireObjectField(face, "surface_parameters", context);
    const auto parameter_context = context + ".surface_parameters";
    SurfaceParameters result;
    switch (type)
    {
    case SurfaceType::Plane:
        RequireExactFields(parameters, {"origin", "axis"}, parameter_context);
        result.origin = ReadVec3(parameters.at("origin"), parameter_context + ".origin");
        result.axis = ReadVec3(parameters.at("axis"), parameter_context + ".axis");
        break;
    case SurfaceType::Cylinder:
        RequireExactFields(
            parameters, {"origin", "axis", "radius"}, parameter_context);
        result.origin = ReadVec3(parameters.at("origin"), parameter_context + ".origin");
        result.axis = ReadVec3(parameters.at("axis"), parameter_context + ".axis");
        result.radius = RequireFiniteValue(
            parameters.at("radius"), parameter_context + ".radius");
        break;
    case SurfaceType::Cone:
        RequireExactFields(
            parameters, {"origin", "axis", "half_angle"}, parameter_context);
        result.origin = ReadVec3(parameters.at("origin"), parameter_context + ".origin");
        result.axis = ReadVec3(parameters.at("axis"), parameter_context + ".axis");
        result.half_angle = RequireFiniteValue(
            parameters.at("half_angle"), parameter_context + ".half_angle");
        break;
    case SurfaceType::Sphere:
        RequireExactFields(parameters, {"origin", "radius"}, parameter_context);
        result.origin = ReadVec3(parameters.at("origin"), parameter_context + ".origin");
        result.radius = RequireFiniteValue(
            parameters.at("radius"), parameter_context + ".radius");
        break;
    case SurfaceType::Torus:
        RequireExactFields(
            parameters,
            {"origin", "axis", "radius", "secondary_radius"},
            parameter_context);
        result.origin = ReadVec3(parameters.at("origin"), parameter_context + ".origin");
        result.axis = ReadVec3(parameters.at("axis"), parameter_context + ".axis");
        result.radius = RequireFiniteValue(
            parameters.at("radius"), parameter_context + ".radius");
        result.secondary_radius = RequireFiniteValue(
            parameters.at("secondary_radius"),
            parameter_context + ".secondary_radius");
        break;
    case SurfaceType::Revolution:
    case SurfaceType::BSpline:
    case SurfaceType::Other:
        RequireExactFields(parameters, {}, parameter_context);
        break;
    }

    const auto require_positive = [&](const std::optional<double>& value,
                                      const char* field) {
        if (value && *value <= 0.0)
        {
            InvalidResult(parameter_context + "." + field + " must be positive");
        }
    };
    require_positive(result.radius, "radius");
    require_positive(result.secondary_radius, "secondary_radius");
    require_positive(result.half_angle, "half_angle");
    if (result.axis)
    {
        const auto canonical = CanonicalAxis(*result.axis);
        const double dx = canonical.x - result.axis->x;
        const double dy = canonical.y - result.axis->y;
        const double dz = canonical.z - result.axis->z;
        const double error = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (error > 1e-6)
        {
            InvalidResult(parameter_context + ".axis must be canonical unit length");
        }
    }
    return result;
}

FaceSignature ReadFaceSignature(const Json& value, std::size_t position)
{
    const auto context =
        "face_mapping.faces[" + std::to_string(position) + "]";
    RequireExactFields(
        value,
        {"face_index", "surface_type", "area", "perimeter", "centroid",
         "bounding_box", "loop_count", "edge_count", "neighbor_count",
         "surface_parameters", "sample_points"},
        context);

    FaceSignature face{};
    face.key = static_cast<FaceKey>(
        RequireNonnegativeIntegerValue(value.at("face_index"), context + ".face_index"));
    face.surface_type = ReadSurfaceType(value, context);
    face.area = RequireFiniteNumber(value, "area", context);
    face.perimeter = RequireFiniteNumber(value, "perimeter", context);
    face.centroid = ReadVec3(value.at("centroid"), context + ".centroid");
    const auto& bounds = value.at("bounding_box");
    if (!bounds.is_array() || bounds.size() != 6)
    {
        InvalidResult(context + ".bounding_box must contain exactly six numbers");
    }
    face.bounds = {
        {RequireFiniteValue(bounds[0], context + ".bounding_box[0]"),
         RequireFiniteValue(bounds[1], context + ".bounding_box[1]"),
         RequireFiniteValue(bounds[2], context + ".bounding_box[2]")},
        {RequireFiniteValue(bounds[3], context + ".bounding_box[3]"),
         RequireFiniteValue(bounds[4], context + ".bounding_box[4]"),
         RequireFiniteValue(bounds[5], context + ".bounding_box[5]")}};
    face.loop_count = RequireInteger(value, "loop_count", context);
    face.edge_count = RequireInteger(value, "edge_count", context);
    face.neighbor_count = RequireInteger(value, "neighbor_count", context);
    face.surface = ReadSurfaceParameters(value, face.surface_type, context);
    const auto& samples = RequireArrayField(value, "sample_points", context);
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        face.sample_points.push_back(ReadVec3(
            samples[index],
            context + ".sample_points[" + std::to_string(index) + "]"));
    }
    return face;
}

FaceMappingPayload ReadFaceMapping(
    const Json& mapping,
    int expected_face_count,
    const std::vector<FaceResult>& core_faces)
{
    const auto contract =
        RequireString(mapping, "contract_version", "face_mapping", true);
    if (contract != "1.0")
    {
        InvalidResult("face_mapping.contract_version is unsupported");
    }
    const auto status = RequireString(mapping, "status", "face_mapping", true);
    if (status == "unavailable")
    {
        RequireExactFields(
            mapping, {"contract_version", "status", "reason"}, "face_mapping");
        FaceMappingPayload result;
        result.status = MappingPayloadStatus::Unavailable;
        result.diagnostic = RequireString(mapping, "reason", "face_mapping", true);
        if (result.diagnostic.find_first_not_of(" \t\r\n") == std::string::npos)
        {
            InvalidResult("face_mapping.reason must not be blank");
        }
        return result;
    }
    if (status != "ready")
    {
        InvalidResult("face_mapping.status must be ready or unavailable");
    }
    RequireExactFields(
        mapping,
        {"contract_version", "status", "normalization", "faces",
         "adjacency_edges"},
        "face_mapping");

    const auto& normalization =
        RequireObjectField(mapping, "normalization", "face_mapping");
    RequireExactFields(
        normalization, {"kind", "box_side", "scale_rule"},
        "face_mapping.normalization");
    if (RequireString(normalization, "kind", "face_mapping.normalization") !=
            "aag_unit_box" ||
        RequireFiniteNumber(normalization, "box_side", "face_mapping.normalization") !=
            1.0 ||
        RequireString(normalization, "scale_rule", "face_mapping.normalization") !=
            "2_over_longest_bbox_extent")
    {
        InvalidResult("face_mapping normalization does not match the contract");
    }

    FaceMappingPayload result;
    result.status = MappingPayloadStatus::Ready;
    const auto& faces = RequireArrayField(mapping, "faces", "face_mapping");
    if (faces.size() != static_cast<std::size_t>(expected_face_count))
    {
        InvalidResult("face_mapping face count does not match input.face_count");
    }
    std::set<FaceKey> core_keys;
    for (const auto& face : core_faces)
    {
        core_keys.insert(static_cast<FaceKey>(face.face_index));
    }
    std::set<FaceKey> mapping_keys;
    for (std::size_t position = 0; position < faces.size(); ++position)
    {
        auto face = ReadFaceSignature(faces[position], position);
        if (!mapping_keys.insert(face.key).second)
        {
            InvalidResult("face_mapping face keys are not unique");
        }
        result.graph.faces.push_back(std::move(face));
    }
    if (mapping_keys != core_keys)
    {
        InvalidResult("face_mapping keys do not match core face_index values");
    }
    std::sort(
        result.graph.faces.begin(), result.graph.faces.end(),
        [](const FaceSignature& left, const FaceSignature& right) {
            return left.key < right.key;
        });

    const auto& edges =
        RequireArrayField(mapping, "adjacency_edges", "face_mapping");
    AdjacencyEdge previous{};
    bool has_previous = false;
    for (std::size_t position = 0; position < edges.size(); ++position)
    {
        const auto& edge = edges[position];
        const auto context =
            "face_mapping.adjacency_edges[" + std::to_string(position) + "]";
        if (!edge.is_array() || edge.size() != 2)
        {
            InvalidResult(context + " must contain exactly two endpoints");
        }
        const AdjacencyEdge parsed{
            static_cast<FaceKey>(RequireNonnegativeIntegerValue(edge[0], context)),
            static_cast<FaceKey>(RequireNonnegativeIntegerValue(edge[1], context))};
        if (has_previous && !(previous < parsed))
        {
            InvalidResult("face_mapping adjacency_edges must be sorted and unique");
        }
        result.graph.adjacency_edges.push_back(parsed);
        previous = parsed;
        has_previous = true;
    }
    ValidateFaceGraph(result.graph);
    return result;
}

std::wstring WindowsPathKey(const std::filesystem::path& path)
{
    auto normalized = std::filesystem::absolute(path).lexically_normal();
    normalized.make_preferred();
    auto text = normalized.wstring();
    std::transform(text.begin(), text.end(), text.begin(), [](wchar_t value) {
        if (value >= L'A' && value <= L'Z')
        {
            return static_cast<wchar_t>(value - L'A' + L'a');
        }
        return value;
    });
    return text;
}

std::filesystem::path ReadAndCheckStepPath(
    const Json& document,
    const std::filesystem::path& selected_step)
{
    const auto& input = RequireObjectField(document, "input", "root");
    const auto text = RequireString(input, "step_path", "input", true);
    const auto document_path = std::filesystem::u8path(text);
    if (WindowsPathKey(document_path) != WindowsPathKey(selected_step))
    {
        InvalidResult("input.step_path does not match the selected STEP");
    }
    return document_path;
}

FaceResult ReadFace(const Json& item, std::size_t position)
{
    const auto context = "faces[" + std::to_string(position) + "]";
    if (!item.is_object())
    {
        InvalidResult(context + " must be an object");
    }

    FaceResult face;
    face.face_index = RequireInteger(item, "face_index", context);
    face.class_id = RequireInteger(item, "class_id", context);
    face.feature_name = RequireString(item, "feature_name", context);
    face.confidence = RequireFiniteNumber(item, "confidence", context);
    if (face.face_index < 0)
    {
        InvalidResult(context + ".face_index must be nonnegative");
    }
    if (face.class_id < 0)
    {
        InvalidResult(context + ".class_id must be nonnegative");
    }

    const auto& instance_id = RequireField(item, "instance_id", context);
    if (!instance_id.is_null())
    {
        face.instance_id = RequireInteger(item, "instance_id", context);
        if (*face.instance_id < 0)
        {
            InvalidResult(context + ".instance_id must be nonnegative or null");
        }
    }
    return face;
}

InstanceResult ReadInstance(const Json& item, std::size_t position)
{
    const auto context = "instances[" + std::to_string(position) + "]";
    if (!item.is_object())
    {
        InvalidResult(context + " must be an object");
    }

    InstanceResult instance;
    instance.instance_id = RequireInteger(item, "instance_id", context);
    instance.class_id = RequireInteger(item, "class_id", context);
    instance.feature_name = RequireString(item, "feature_name", context);
    instance.confidence = RequireFiniteNumber(item, "confidence", context);
    if (instance.instance_id < 0)
    {
        InvalidResult(context + ".instance_id must be nonnegative");
    }
    if (instance.class_id < 0)
    {
        InvalidResult(context + ".class_id must be nonnegative");
    }

    const auto& face_indices =
        RequireArrayField(item, "face_indices", context);
    int previous = -1;
    for (std::size_t index = 0; index < face_indices.size(); ++index)
    {
        Json wrapper = {{"value", face_indices[index]}};
        const int face_index = RequireInteger(
            wrapper,
            "value",
            context + ".face_indices[" + std::to_string(index) + "]");
        if (face_index < 0)
        {
            InvalidResult(context + ".face_indices must be nonnegative");
        }
        if (index != 0 && face_index <= previous)
        {
            InvalidResult(
                context + ".face_indices must be strictly increasing");
        }
        instance.face_indices.push_back(face_index);
        previous = face_index;
    }
    return instance;
}

SuccessResult ReadSuccess(
    const Json& document,
    const std::filesystem::path& selected_step,
    const std::string& version)
{
    SuccessResult result;
    result.schema_version = version;
    result.step_path = ReadAndCheckStepPath(document, selected_step);
    const auto& input = RequireObjectField(document, "input", "root");
    result.face_count = RequireInteger(input, "face_count", "input");
    if (result.face_count < 0)
    {
        InvalidResult("input.face_count must be nonnegative");
    }

    const auto& cache = RequireObjectField(document, "cache", "root");
    result.cache_hit = RequireBoolean(cache, "hit", "cache");
    const auto& timings =
        RequireObjectField(document, "timings_seconds", "root");
    result.total_seconds = RequireFiniteNumber(timings, "total", "timings_seconds");
    if (result.total_seconds < 0.0)
    {
        InvalidResult("timings_seconds.total must be nonnegative");
    }

    const auto& face_items = RequireArrayField(document, "faces", "root");
    if (face_items.size() != static_cast<std::size_t>(result.face_count))
    {
        InvalidResult("faces length does not equal input.face_count");
    }

    std::map<int, std::size_t> face_positions;
    for (std::size_t position = 0; position < face_items.size(); ++position)
    {
        auto face = ReadFace(face_items[position], position);
        if (!face_positions.emplace(face.face_index, result.faces.size()).second)
        {
            InvalidResult("face_index values must be unique");
        }
        result.faces.push_back(std::move(face));
    }
    for (int expected = 0; expected < result.face_count; ++expected)
    {
        if (face_positions.find(expected) == face_positions.end())
        {
            InvalidResult("face_index values must be contiguous and 0-based");
        }
    }

    const auto& instance_items =
        RequireArrayField(document, "instances", "root");
    std::map<int, std::size_t> instance_positions;
    for (std::size_t position = 0; position < instance_items.size(); ++position)
    {
        auto instance = ReadInstance(instance_items[position], position);
        if (!instance_positions.emplace(
                instance.instance_id, result.instances.size()).second)
        {
            InvalidResult("instance_id values must be unique");
        }
        result.instances.push_back(std::move(instance));
    }

    for (const auto& instance : result.instances)
    {
        for (const int face_index : instance.face_indices)
        {
            const auto face_position = face_positions.find(face_index);
            if (face_position == face_positions.end())
            {
                InvalidResult("instance references a nonexistent face_index");
            }
            const auto& face = result.faces[face_position->second];
            if (!face.instance_id || *face.instance_id != instance.instance_id)
            {
                InvalidResult("instance and per-face instance_id disagree");
            }
            if (face.class_id != instance.class_id ||
                face.feature_name != instance.feature_name)
            {
                InvalidResult("instance class or feature name disagrees with its face");
            }
        }
    }

    for (const auto& face : result.faces)
    {
        if (!face.instance_id)
        {
            continue;
        }
        const auto instance_position = instance_positions.find(*face.instance_id);
        if (instance_position == instance_positions.end())
        {
            InvalidResult("per-face instance_id references a nonexistent instance");
        }
        const auto& instance = result.instances[instance_position->second];
        if (!std::binary_search(
                instance.face_indices.begin(),
                instance.face_indices.end(),
                face.face_index))
        {
            InvalidResult("per-face instance_id is missing from instance face_indices");
        }
        if (face.class_id != instance.class_id ||
            face.feature_name != instance.feature_name)
        {
            InvalidResult("face class or feature name disagrees with its instance");
        }
    }

    std::sort(
        result.instances.begin(),
        result.instances.end(),
        [](const InstanceResult& left, const InstanceResult& right) {
            return left.instance_id < right.instance_id;
        });
    if (version == "1.0")
    {
        result.mapping.status = MappingPayloadStatus::Unsupported;
    }
    else
    {
        try
        {
            result.mapping = ReadFaceMapping(
                RequireObjectField(document, "face_mapping", "root"),
                result.face_count,
                result.faces);
        }
        catch (const std::exception& error)
        {
            result.mapping.status = MappingPayloadStatus::Invalid;
            result.mapping.diagnostic = error.what();
            result.mapping.graph = {};
        }
    }
    return result;
}

ErrorResult ReadError(
    const Json& document,
    const std::filesystem::path& selected_step)
{
    ErrorResult result;
    result.step_path = ReadAndCheckStepPath(document, selected_step);
    result.stage = RequireString(document, "stage", "root", true);
    const auto& error = RequireObjectField(document, "error", "root");
    result.error_type = RequireString(error, "type", "error", true);
    result.message = RequireString(error, "message", "error", true);
    return result;
}
}

InferenceResult ReadInferenceResult(
    const std::filesystem::path& result_path,
    const std::filesystem::path& selected_step)
{
    std::ifstream stream(result_path, std::ios::binary);
    if (!stream)
    {
        InvalidResult("could not open result.json");
    }

    Json document;
    try
    {
        stream >> document;
    }
    catch (const std::exception& error)
    {
        InvalidResult(std::string("JSON parse failed: ") + error.what());
    }
    if (!document.is_object())
    {
        InvalidResult("root must be an object");
    }

    const auto version =
        RequireString(document, "schema_version", "root");
    if (version != "1.0" && version != "1.1")
    {
        InvalidResult("unsupported schema_version");
    }

    const auto status = RequireString(document, "status", "root");
    if (status == "success")
    {
        return ReadSuccess(document, selected_step, version);
    }
    if (status == "error")
    {
        return ReadError(document, selected_step);
    }
    InvalidResult("status must be success or error");
}
}
}
