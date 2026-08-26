#pragma once

#include "face_signature.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
struct FaceResult
{
    int face_index = 0;
    int class_id = 0;
    std::string feature_name;
    double confidence = 0.0;
    std::optional<int> instance_id;
};

struct InstanceResult
{
    int instance_id = 0;
    int class_id = 0;
    std::string feature_name;
    std::vector<int> face_indices;
    double confidence = 0.0;
};

enum class MappingPayloadStatus
{
    Unsupported,
    Ready,
    Unavailable,
    Invalid
};

struct FaceMappingPayload
{
    MappingPayloadStatus status = MappingPayloadStatus::Unsupported;
    std::string diagnostic;
    FaceGraph graph;
};

struct SuccessResult
{
    std::string schema_version;
    std::filesystem::path step_path;
    int face_count = 0;
    bool cache_hit = false;
    double total_seconds = 0.0;
    std::vector<FaceResult> faces;
    std::vector<InstanceResult> instances;
    FaceMappingPayload mapping;
};

struct ErrorResult
{
    std::filesystem::path step_path;
    std::string stage;
    std::string error_type;
    std::string message;
};

using InferenceResult = std::variant<SuccessResult, ErrorResult>;

InferenceResult ReadInferenceResult(
    const std::filesystem::path& result_path,
    const std::filesystem::path& selected_step);
}
}
