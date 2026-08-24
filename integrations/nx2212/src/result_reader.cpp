#include "result_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
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
    const std::filesystem::path& selected_step)
{
    SuccessResult result;
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
    if (version != "1.0")
    {
        InvalidResult("unsupported schema_version");
    }

    const auto status = RequireString(document, "status", "root");
    if (status == "success")
    {
        return ReadSuccess(document, selected_step);
    }
    if (status == "error")
    {
        return ReadError(document, selected_step);
    }
    InvalidResult("status must be success or error");
}
}
}
