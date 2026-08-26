#include "face_signature.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr double AxisTolerance = 1e-12;

bool IsFinite(const Vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

void RequireFinite(double value, const char* message)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(message);
    }
}

void ValidateTransform(const UnitBoxTransform& transform)
{
    if (!IsFinite(transform.center) || !std::isfinite(transform.scale) ||
        transform.scale <= 0.0)
    {
        throw std::invalid_argument("unit-box transform is invalid");
    }
}

void ValidateOptionalPoint(const std::optional<Vec3>& value)
{
    if (value && !IsFinite(*value))
    {
        throw std::invalid_argument("surface point is not finite");
    }
}

void ValidateOptionalScalar(
    const std::optional<double>& value,
    bool nonnegative,
    const char* message)
{
    if (value && (!std::isfinite(*value) || (nonnegative && *value < 0.0)))
    {
        throw std::invalid_argument(message);
    }
}
}

bool operator==(const Vec3& left, const Vec3& right) noexcept
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool IncludeFaceGridSample(int containment_status) noexcept
{
    return containment_status == 1;
}

UnitBoxTransform MakeUnitBoxTransform(const Bounds3& body_bounds)
{
    if (!IsFinite(body_bounds.minimum) || !IsFinite(body_bounds.maximum))
    {
        throw std::invalid_argument("body bounds are not finite");
    }
    const Vec3 extent{
        body_bounds.maximum.x - body_bounds.minimum.x,
        body_bounds.maximum.y - body_bounds.minimum.y,
        body_bounds.maximum.z - body_bounds.minimum.z};
    if (extent.x < 0.0 || extent.y < 0.0 || extent.z < 0.0)
    {
        throw std::invalid_argument("body bounds are reversed");
    }
    const double longest = std::max({extent.x, extent.y, extent.z});
    if (!std::isfinite(longest) || longest <= 0.0)
    {
        throw std::invalid_argument("body bounds have no positive extent");
    }
    return {
        {(body_bounds.minimum.x + body_bounds.maximum.x) * 0.5,
         (body_bounds.minimum.y + body_bounds.maximum.y) * 0.5,
         (body_bounds.minimum.z + body_bounds.maximum.z) * 0.5},
        2.0 / longest};
}

Vec3 NormalizePoint(const UnitBoxTransform& transform, const Vec3& point)
{
    ValidateTransform(transform);
    if (!IsFinite(point))
    {
        throw std::invalid_argument("point is not finite");
    }
    return {
        (point.x - transform.center.x) * transform.scale,
        (point.y - transform.center.y) * transform.scale,
        (point.z - transform.center.z) * transform.scale};
}

double NormalizeLength(const UnitBoxTransform& transform, double length)
{
    ValidateTransform(transform);
    RequireFinite(length, "length is not finite");
    if (length < 0.0)
    {
        throw std::invalid_argument("length is negative");
    }
    return length * transform.scale;
}

double NormalizeArea(const UnitBoxTransform& transform, double area)
{
    ValidateTransform(transform);
    RequireFinite(area, "area is not finite");
    if (area < 0.0)
    {
        throw std::invalid_argument("area is negative");
    }
    return area * transform.scale * transform.scale;
}

Vec3 CanonicalAxis(const Vec3& axis)
{
    if (!IsFinite(axis))
    {
        throw std::invalid_argument("axis is not finite");
    }
    const double norm = std::sqrt(
        axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (!std::isfinite(norm) || norm <= AxisTolerance)
    {
        throw std::invalid_argument("axis has zero length");
    }
    Vec3 result{axis.x / norm, axis.y / norm, axis.z / norm};
    const double components[] = {result.x, result.y, result.z};
    for (const double component : components)
    {
        if (std::abs(component) > AxisTolerance)
        {
            if (component < 0.0)
            {
                result = {-result.x, -result.y, -result.z};
            }
            break;
        }
    }
    return result;
}

Vec3 CanonicalPlaneOrigin(const Vec3& point, const Vec3& axis)
{
    if (!IsFinite(point))
    {
        throw std::invalid_argument("plane point is not finite");
    }
    const Vec3 unit_axis = CanonicalAxis(axis);
    const double projection = point.x * unit_axis.x +
        point.y * unit_axis.y + point.z * unit_axis.z;
    return {
        unit_axis.x * projection,
        unit_axis.y * projection,
        unit_axis.z * projection};
}

Vec3 CanonicalAxisOrigin(const Vec3& point, const Vec3& axis)
{
    if (!IsFinite(point))
    {
        throw std::invalid_argument("axis point is not finite");
    }
    const Vec3 unit_axis = CanonicalAxis(axis);
    const double projection = point.x * unit_axis.x +
        point.y * unit_axis.y + point.z * unit_axis.z;
    return {
        point.x - unit_axis.x * projection,
        point.y - unit_axis.y * projection,
        point.z - unit_axis.z * projection};
}

void ValidateFaceGraph(const FaceGraph& graph)
{
    std::unordered_map<FaceKey, int> degrees;
    degrees.reserve(graph.faces.size());
    for (const auto& face : graph.faces)
    {
        if (!degrees.emplace(face.key, 0).second)
        {
            throw std::invalid_argument("face keys are not unique");
        }
        if (!std::isfinite(face.area) || face.area < 0.0 ||
            !std::isfinite(face.perimeter) || face.perimeter < 0.0 ||
            !IsFinite(face.centroid) || !IsFinite(face.bounds.minimum) ||
            !IsFinite(face.bounds.maximum))
        {
            throw std::invalid_argument("face signature contains invalid geometry");
        }
        if (face.bounds.minimum.x > face.bounds.maximum.x ||
            face.bounds.minimum.y > face.bounds.maximum.y ||
            face.bounds.minimum.z > face.bounds.maximum.z)
        {
            throw std::invalid_argument("face bounds are reversed");
        }
        if (face.loop_count <= 0 || face.edge_count <= 0 ||
            face.neighbor_count < 0)
        {
            throw std::invalid_argument("face topology counts are invalid");
        }
        ValidateOptionalPoint(face.surface.origin);
        if (face.surface.axis)
        {
            (void)CanonicalAxis(*face.surface.axis);
        }
        ValidateOptionalScalar(face.surface.radius, true, "radius is invalid");
        ValidateOptionalScalar(
            face.surface.secondary_radius,
            true,
            "secondary radius is invalid");
        ValidateOptionalScalar(
            face.surface.half_angle,
            false,
            "half angle is invalid");
        if (face.sample_points.empty())
        {
            throw std::invalid_argument("face sample points are empty");
        }
        for (const auto& point : face.sample_points)
        {
            if (!IsFinite(point))
            {
                throw std::invalid_argument("face sample point is not finite");
            }
        }
    }

    std::set<AdjacencyEdge> unique_edges;
    for (const auto& edge : graph.adjacency_edges)
    {
        if (edge.first == edge.second)
        {
            throw std::invalid_argument("adjacency self-loop is invalid");
        }
        if (edge.first > edge.second)
        {
            throw std::invalid_argument("adjacency edge is not canonical");
        }
        const auto left = degrees.find(edge.first);
        const auto right = degrees.find(edge.second);
        if (left == degrees.end() || right == degrees.end())
        {
            throw std::invalid_argument("adjacency endpoint is unknown");
        }
        if (!unique_edges.insert(edge).second)
        {
            throw std::invalid_argument("adjacency edge is duplicated");
        }
        ++left->second;
        ++right->second;
    }

    for (const auto& face : graph.faces)
    {
        if (degrees.at(face.key) != face.neighbor_count)
        {
            throw std::invalid_argument("face neighbor count does not match adjacency");
        }
    }
}
}
}
