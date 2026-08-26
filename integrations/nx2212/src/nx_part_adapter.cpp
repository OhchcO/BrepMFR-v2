#include "nx_part_adapter.hpp"

#include "face_signature.hpp"
#include "step_source_resolver.hpp"

#include <NXOpen/Body.hxx>
#include <NXOpen/BodyCollection.hxx>
#include <NXOpen/Edge.hxx>
#include <NXOpen/Face.hxx>
#include <NXOpen/Measurement.hxx>
#include <NXOpen/NXString.hxx>
#include <NXOpen/Part.hxx>
#include <NXOpen/PartCollection.hxx>
#include <NXOpen/Session.hxx>

#include <uf.h>
#include <uf_modl.h>
#include <uf_modl_utilities.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr int kUvGridSize = 5;
constexpr double kGeometryAccuracy = 0.999;
constexpr double kZeroTolerance = 1e-12;

struct CurrentPartData
{
    PartSnapshot snapshot;
    NXOpen::Part* part = nullptr;
    NXOpen::Body* body = nullptr;
    std::vector<NXOpen::Face*> faces;
};

std::string UfMessage(int code)
{
    char message[133] = {};
    if (UF_get_fail_message(code, message) == 0 && message[0] != '\0')
    {
        return message;
    }
    return "UF error " + std::to_string(code);
}

void CheckUf(int code, const char* operation)
{
    if (code != 0)
    {
        throw std::runtime_error(
            std::string(operation) + " failed: " + UfMessage(code));
    }
}

std::wstring LowerAscii(std::wstring value)
{
    for (wchar_t& character : value)
    {
        if (character >= L'A' && character <= L'Z')
        {
            character = static_cast<wchar_t>(character - L'A' + L'a');
        }
    }
    return value;
}

std::filesystem::path ValidateStepPath(NXOpen::Part* part)
{
    const NXOpen::NXString full_path = part->FullPath();
    const char* utf8 = full_path.GetUTF8Text();
    if (utf8 == nullptr || utf8[0] == '\0')
    {
        throw std::runtime_error("current Work Part has no file path");
    }
    return ResolveOpenedStepSource(std::filesystem::u8path(utf8));
}

CurrentPartData CaptureCurrentPartData()
{
    NXOpen::Session* session = NXOpen::Session::GetSession();
    if (session == nullptr || session->Parts() == nullptr)
    {
        throw std::runtime_error("NX session or Part collection is unavailable");
    }
    NXOpen::Part* part = session->Parts()->Work();
    if (part == nullptr || part->Bodies() == nullptr)
    {
        throw std::runtime_error("NX has no current Work Part");
    }

    std::vector<NXOpen::Body*> bodies;
    for (NXOpen::Body* body : *part->Bodies())
    {
        if (body != nullptr)
        {
            bodies.push_back(body);
        }
    }
    if (bodies.size() != 1)
    {
        throw std::runtime_error(
            "current Work Part must contain exactly one body");
    }
    NXOpen::Body* body = bodies.front();
    if (!body->IsSolidBody() || body->IsSheetBody() ||
        body->IsConvergentBody())
    {
        throw std::runtime_error(
            "current Work Part body must be one exact solid body");
    }
    auto faces = body->GetFaces();
    faces.erase(
        std::remove(faces.begin(), faces.end(), nullptr), faces.end());
    if (faces.empty())
    {
        throw std::runtime_error("current Work Part body has no faces");
    }

    CurrentPartData data;
    data.part = part;
    data.body = body;
    data.faces = std::move(faces);
    data.snapshot.step_path = ValidateStepPath(part);
    data.snapshot.part_tag = static_cast<FaceKey>(part->Tag());
    data.snapshot.body_tag = static_cast<FaceKey>(body->Tag());
    data.snapshot.face_count = data.faces.size();
    return data;
}

std::wstring ComparablePath(const std::filesystem::path& path)
{
    return LowerAscii(std::filesystem::weakly_canonical(path).wstring());
}

void ValidateSnapshot(
    const PartSnapshot& expected,
    const PartSnapshot& current)
{
    if (ComparablePath(expected.step_path) != ComparablePath(current.step_path))
    {
        throw std::runtime_error(
            "current Work Part path changed while recognition was running");
    }
    if (expected.part_tag != current.part_tag)
    {
        throw std::runtime_error(
            "current Work Part changed while recognition was running");
    }
    if (expected.body_tag != current.body_tag)
    {
        throw std::runtime_error(
            "current body changed while recognition was running");
    }
    if (expected.face_count != current.face_count)
    {
        throw std::runtime_error(
            "current body face count changed while recognition was running");
    }
}

Vec3 ToVec3(const double value[3])
{
    return {value[0], value[1], value[2]};
}

Vec3 ToVec3(const NXOpen::Point3d& value)
{
    return {value.X, value.Y, value.Z};
}

Vec3 Subtract(const Vec3& left, const Vec3& right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 Scale(const Vec3& value, double scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 ExtremePoint(
    tag_t object,
    const std::array<double, 3>& direction,
    const std::array<double, 3>& tie_direction_1,
    const std::array<double, 3>& tie_direction_2)
{
    double primary[3] = {
        direction[0], direction[1], direction[2]};
    double secondary[3] = {
        tie_direction_1[0], tie_direction_1[1], tie_direction_1[2]};
    double tertiary[3] = {
        tie_direction_2[0], tie_direction_2[1], tie_direction_2[2]};
    tag_t subentity = NULL_TAG;
    double point[3] = {};
    CheckUf(
        UF_MODL_ask_extreme(
            object, primary, secondary, tertiary, &subentity, point),
        "UF_MODL_ask_extreme");
    return ToVec3(point);
}

Bounds3 ExactFaceBounds(tag_t face)
{
    const std::array<double, 3> x{1.0, 0.0, 0.0};
    const std::array<double, 3> y{0.0, 1.0, 0.0};
    const std::array<double, 3> z{0.0, 0.0, 1.0};
    const std::array<double, 3> negative_x{-1.0, 0.0, 0.0};
    const std::array<double, 3> negative_y{0.0, -1.0, 0.0};
    const std::array<double, 3> negative_z{0.0, 0.0, -1.0};
    const Vec3 minimum_x = ExtremePoint(face, negative_x, y, z);
    const Vec3 maximum_x = ExtremePoint(face, x, y, z);
    const Vec3 minimum_y = ExtremePoint(face, negative_y, z, x);
    const Vec3 maximum_y = ExtremePoint(face, y, z, x);
    const Vec3 minimum_z = ExtremePoint(face, negative_z, x, y);
    const Vec3 maximum_z = ExtremePoint(face, z, x, y);
    return {
        {minimum_x.x, minimum_y.y, minimum_z.z},
        {maximum_x.x, maximum_y.y, maximum_z.z}};
}

int FaceLoopCount(tag_t face)
{
    uf_loop_p_t loops = nullptr;
    CheckUf(UF_MODL_ask_face_loops(face, &loops),
            "UF_MODL_ask_face_loops");
    struct LoopGuard
    {
        uf_loop_p_t* loops;
        ~LoopGuard()
        {
            if (loops != nullptr && *loops != nullptr)
            {
                (void)UF_MODL_delete_loop_list(loops);
            }
        }
    } guard{&loops};
    int count = 0;
    CheckUf(UF_MODL_ask_loop_list_count(loops, &count),
            "UF_MODL_ask_loop_list_count");
    return count;
}

SurfaceType SurfaceTypeFromUf(int type)
{
    switch (type)
    {
    case 16: return SurfaceType::Cylinder;
    case 17: return SurfaceType::Cone;
    case 18: return SurfaceType::Sphere;
    case 19: return SurfaceType::Torus;
    case 22: return SurfaceType::Plane;
    case 43: return SurfaceType::BSpline;
    default: return SurfaceType::Other;
    }
}

SurfaceParameters ReadSurfaceParameters(
    tag_t face,
    SurfaceType* surface_type)
{
    int type = 0;
    double point[3] = {};
    double direction[3] = {};
    double box[6] = {};
    double radius = 0.0;
    double secondary = 0.0;
    int normal_direction = 0;
    CheckUf(
        UF_MODL_ask_face_data(
            face, &type, point, direction, box, &radius,
            &secondary, &normal_direction),
        "UF_MODL_ask_face_data");
    (void)box;
    (void)normal_direction;

    *surface_type = SurfaceTypeFromUf(type);
    SurfaceParameters parameters;
    const Vec3 location = ToVec3(point);
    const Vec3 axis = ToVec3(direction);
    switch (*surface_type)
    {
    case SurfaceType::Plane:
        parameters.origin = location;
        parameters.axis = axis;
        break;
    case SurfaceType::Cylinder:
        parameters.origin = location;
        parameters.axis = axis;
        parameters.radius = radius;
        break;
    case SurfaceType::Cone:
        if (std::abs(std::tan(secondary)) <= kZeroTolerance)
        {
            throw std::runtime_error("cone half angle is degenerate");
        }
        parameters.origin = Subtract(
            location, Scale(axis, radius / std::tan(secondary)));
        parameters.axis = axis;
        parameters.half_angle = secondary;
        break;
    case SurfaceType::Sphere:
        parameters.origin = location;
        parameters.radius = radius;
        break;
    case SurfaceType::Torus:
        parameters.origin = location;
        parameters.axis = axis;
        parameters.radius = radius;
        parameters.secondary_radius = secondary;
        break;
    case SurfaceType::Revolution:
    case SurfaceType::BSpline:
    case SurfaceType::Other:
        break;
    }
    return parameters;
}

std::vector<Vec3> SampleFace(tag_t face)
{
    double limits[4] = {};
    CheckUf(UF_MODL_ask_face_uv_minmax(face, limits),
            "UF_MODL_ask_face_uv_minmax");

    std::vector<double> u_values;
    std::vector<double> v_values;
    u_values.reserve(kUvGridSize * kUvGridSize);
    v_values.reserve(kUvGridSize * kUvGridSize);
    for (int u_index = 0; u_index < kUvGridSize; ++u_index)
    {
        const double u_fraction = static_cast<double>(u_index) /
            static_cast<double>(kUvGridSize - 1);
        for (int v_index = 0; v_index < kUvGridSize; ++v_index)
        {
            const double v_fraction = static_cast<double>(v_index) /
                static_cast<double>(kUvGridSize - 1);
            u_values.push_back(limits[0] + (limits[1] - limits[0]) * u_fraction);
            v_values.push_back(limits[2] + (limits[3] - limits[2]) * v_fraction);
        }
    }
    std::vector<int> containment(u_values.size(), 0);
    CheckUf(
        UF_MODL_ask_uv_points_containment(
            static_cast<int>(u_values.size()),
            u_values.data(), v_values.data(), face, containment.data()),
        "UF_MODL_ask_uv_points_containment");

    std::vector<Vec3> samples;
    for (std::size_t index = 0; index < u_values.size(); ++index)
    {
        if (!IncludeFaceGridSample(containment[index]))
        {
            continue;
        }
        double parameter[2] = {u_values[index], v_values[index]};
        double point[3] = {};
        double u_first[3] = {};
        double v_first[3] = {};
        double u_second[3] = {};
        double v_second[3] = {};
        double normal[3] = {};
        double radii[2] = {};
        CheckUf(
            UF_MODL_ask_face_props(
                face, parameter, point, u_first, v_first,
                u_second, v_second, normal, radii),
            "UF_MODL_ask_face_props");
        samples.push_back(ToVec3(point));
    }
    if (samples.empty())
    {
        throw std::runtime_error("face UV grid produced no contained samples");
    }
    return samples;
}

FaceSignature ReadRawFace(
    NXOpen::Session* session,
    NXOpen::Face* face)
{
    NXOpen::Measurement* measurement = session->Measurement();
    if (measurement == nullptr)
    {
        throw std::runtime_error("NX measurement service is unavailable");
    }
    double area = 0.0;
    double perimeter = 0.0;
    double radius_diameter = 0.0;
    NXOpen::Point3d centroid{};
    double minimum_radius = 0.0;
    double area_error = 0.0;
    NXOpen::Point3d anchor{};
    bool approximate = false;
    measurement->GetFaceProperties(
        std::vector<NXOpen::Face*>{face},
        kGeometryAccuracy,
        NXOpen::Measurement::AlternateFaceRadius,
        true,
        &area,
        &perimeter,
        &radius_diameter,
        &centroid,
        &minimum_radius,
        &area_error,
        &anchor,
        &approximate);
    (void)radius_diameter;
    (void)minimum_radius;
    (void)area_error;
    (void)anchor;
    if (approximate)
    {
        throw std::runtime_error("NX returned approximate face properties");
    }

    FaceSignature signature{};
    signature.key = static_cast<FaceKey>(face->Tag());
    signature.area = area;
    signature.perimeter = perimeter;
    signature.centroid = ToVec3(centroid);
    signature.bounds = ExactFaceBounds(face->Tag());
    signature.loop_count = FaceLoopCount(face->Tag());
    signature.edge_count = static_cast<int>(face->GetUnsortedEdges().size());
    signature.neighbor_count = 0;
    signature.surface = ReadSurfaceParameters(
        face->Tag(), &signature.surface_type);
    signature.sample_points = SampleFace(face->Tag());
    return signature;
}

Bounds3 BodyBounds(const std::vector<FaceSignature>& faces)
{
    if (faces.empty())
    {
        throw std::runtime_error("cannot compute bounds for an empty body");
    }
    Bounds3 bounds = faces.front().bounds;
    for (const auto& face : faces)
    {
        bounds.minimum.x = std::min(bounds.minimum.x, face.bounds.minimum.x);
        bounds.minimum.y = std::min(bounds.minimum.y, face.bounds.minimum.y);
        bounds.minimum.z = std::min(bounds.minimum.z, face.bounds.minimum.z);
        bounds.maximum.x = std::max(bounds.maximum.x, face.bounds.maximum.x);
        bounds.maximum.y = std::max(bounds.maximum.y, face.bounds.maximum.y);
        bounds.maximum.z = std::max(bounds.maximum.z, face.bounds.maximum.z);
    }
    return bounds;
}

void NormalizeSurface(
    SurfaceType type,
    const UnitBoxTransform& transform,
    SurfaceParameters* surface)
{
    if (surface->origin)
    {
        surface->origin = NormalizePoint(transform, *surface->origin);
    }
    if (surface->axis)
    {
        surface->axis = CanonicalAxis(*surface->axis);
    }
    if (surface->radius)
    {
        surface->radius = NormalizeLength(transform, *surface->radius);
    }
    if (surface->secondary_radius)
    {
        surface->secondary_radius = NormalizeLength(
            transform, *surface->secondary_radius);
    }
    if (surface->origin && surface->axis && type == SurfaceType::Plane)
    {
        surface->origin = CanonicalPlaneOrigin(
            *surface->origin, *surface->axis);
    }
    else if (surface->origin && surface->axis &&
             type == SurfaceType::Cylinder)
    {
        surface->origin = CanonicalAxisOrigin(
            *surface->origin, *surface->axis);
    }
}

void NormalizeFace(
    const UnitBoxTransform& transform,
    FaceSignature* face)
{
    face->area = NormalizeArea(transform, face->area);
    face->perimeter = NormalizeLength(transform, face->perimeter);
    face->centroid = NormalizePoint(transform, face->centroid);
    face->bounds.minimum = NormalizePoint(transform, face->bounds.minimum);
    face->bounds.maximum = NormalizePoint(transform, face->bounds.maximum);
    NormalizeSurface(face->surface_type, transform, &face->surface);
    for (Vec3& sample : face->sample_points)
    {
        sample = NormalizePoint(transform, sample);
    }
    std::sort(
        face->sample_points.begin(), face->sample_points.end(),
        [](const Vec3& left, const Vec3& right) {
            if (left.x != right.x) return left.x < right.x;
            if (left.y != right.y) return left.y < right.y;
            return left.z < right.z;
        });
}

std::vector<AdjacencyEdge> ReadAdjacency(
    NXOpen::Body* body,
    const std::unordered_map<FaceKey, std::size_t>& face_positions,
    std::vector<FaceSignature>* faces)
{
    std::set<AdjacencyEdge> edges;
    for (NXOpen::Edge* edge : body->GetEdges())
    {
        if (edge == nullptr)
        {
            continue;
        }
        const auto connected = edge->GetUnsortedFaces();
        if (connected.size() != 2 || connected[0] == nullptr ||
            connected[1] == nullptr)
        {
            continue;
        }
        FaceKey left = static_cast<FaceKey>(connected[0]->Tag());
        FaceKey right = static_cast<FaceKey>(connected[1]->Tag());
        if (left == right || face_positions.find(left) == face_positions.end() ||
            face_positions.find(right) == face_positions.end())
        {
            continue;
        }
        if (left > right)
        {
            std::swap(left, right);
        }
        edges.insert({left, right});
    }
    for (const auto& edge : edges)
    {
        ++faces->at(face_positions.at(edge.first)).neighbor_count;
        ++faces->at(face_positions.at(edge.second)).neighbor_count;
    }
    return {edges.begin(), edges.end()};
}

template <typename Function>
auto WithStage(const char* stage, Function function) -> decltype(function())
{
    try
    {
        return function();
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            std::string(stage) + " failed: " + error.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            std::string(stage) + " failed with an unknown NX error");
    }
}
}

PartSnapshot CaptureNxCurrentPart()
{
    return WithStage("CaptureNxCurrentPart", []() {
        return CaptureCurrentPartData().snapshot;
    });
}

void ValidateNxCurrentPart(const PartSnapshot& expected)
{
    WithStage("ValidateNxCurrentPart", [&expected]() {
        ValidateSnapshot(expected, CaptureCurrentPartData().snapshot);
        return 0;
    });
}

FaceGraph ReadNxFaceGraph(const PartSnapshot& expected)
{
    return WithStage("ReadNxFaceGraph", [&expected]() {
        const CurrentPartData current = CaptureCurrentPartData();
        ValidateSnapshot(expected, current.snapshot);

        NXOpen::Session* session = NXOpen::Session::GetSession();
        std::vector<FaceSignature> faces;
        faces.reserve(current.faces.size());
        std::unordered_map<FaceKey, std::size_t> face_positions;
        for (NXOpen::Face* face : current.faces)
        {
            FaceSignature signature = ReadRawFace(session, face);
            if (!face_positions.emplace(signature.key, faces.size()).second)
            {
                throw std::runtime_error("NX body contains duplicate face tags");
            }
            faces.push_back(std::move(signature));
        }

        const UnitBoxTransform transform = MakeUnitBoxTransform(
            BodyBounds(faces));
        for (auto& face : faces)
        {
            NormalizeFace(transform, &face);
        }
        FaceGraph graph;
        graph.adjacency_edges = ReadAdjacency(
            current.body, face_positions, &faces);
        graph.faces = std::move(faces);
        ValidateFaceGraph(graph);
        return graph;
    });
}
}
}
