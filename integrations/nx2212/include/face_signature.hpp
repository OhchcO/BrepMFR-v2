#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
using FaceKey = std::uint64_t;

struct Vec3
{
    double x;
    double y;
    double z;
};

struct Bounds3
{
    Vec3 minimum;
    Vec3 maximum;
};

bool operator==(const Vec3& left, const Vec3& right) noexcept;
bool IncludeFaceGridSample(int containment_status) noexcept;

enum class SurfaceType
{
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    Revolution,
    BSpline,
    Other
};

struct SurfaceParameters
{
    std::optional<Vec3> origin;
    std::optional<Vec3> axis;
    std::optional<double> radius;
    std::optional<double> secondary_radius;
    std::optional<double> half_angle;
};

struct FaceSignature
{
    FaceKey key;
    SurfaceType surface_type;
    double area;
    double perimeter;
    Vec3 centroid;
    Bounds3 bounds;
    int loop_count;
    int edge_count;
    int neighbor_count;
    SurfaceParameters surface;
    std::vector<Vec3> sample_points;
};

using AdjacencyEdge = std::pair<FaceKey, FaceKey>;

struct FaceGraph
{
    std::vector<FaceSignature> faces;
    std::vector<AdjacencyEdge> adjacency_edges;
};

struct UnitBoxTransform
{
    Vec3 center;
    double scale;
};

UnitBoxTransform MakeUnitBoxTransform(const Bounds3& body_bounds);
Vec3 NormalizePoint(const UnitBoxTransform& transform, const Vec3& point);
double NormalizeLength(const UnitBoxTransform& transform, double length);
double NormalizeArea(const UnitBoxTransform& transform, double area);
Vec3 CanonicalAxis(const Vec3& axis);
Vec3 CanonicalPlaneOrigin(const Vec3& point, const Vec3& axis);
Vec3 CanonicalAxisOrigin(const Vec3& point, const Vec3& axis);
void ValidateFaceGraph(const FaceGraph& graph);
}
}
