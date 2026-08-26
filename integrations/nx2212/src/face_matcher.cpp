#include "face_matcher.hpp"

#include "assignment_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
struct CandidateScore
{
    double cost = 0.0;
    double max_normalized_error = 0.0;
};

FaceMappingResult Failure(const std::string& diagnostic)
{
    FaceMappingResult result;
    result.diagnostic = diagnostic;
    return result;
}

double Distance(const Vec3& left, const Vec3& right)
{
    const double dx = left.x - right.x;
    const double dy = left.y - right.y;
    const double dz = left.z - right.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double RelativeError(double left, double right)
{
    const double denominator = std::max({std::abs(left), std::abs(right), 1e-12});
    return std::abs(left - right) / denominator;
}

double DirectedHausdorff(
    const std::vector<Vec3>& source,
    const std::vector<Vec3>& target)
{
    double maximum = 0.0;
    for (const auto& source_point : source)
    {
        double minimum = std::numeric_limits<double>::infinity();
        for (const auto& target_point : target)
        {
            minimum = std::min(minimum, Distance(source_point, target_point));
        }
        maximum = std::max(maximum, minimum);
    }
    return maximum;
}

double SymmetricHausdorff(
    const std::vector<Vec3>& left,
    const std::vector<Vec3>& right)
{
    return std::max(
        DirectedHausdorff(left, right),
        DirectedHausdorff(right, left));
}

bool AddComponent(
    CandidateScore& score,
    double error,
    double tolerance)
{
    if (!std::isfinite(error) || error > tolerance)
    {
        return false;
    }
    const double normalized = error / tolerance;
    score.cost += normalized;
    score.max_normalized_error =
        std::max(score.max_normalized_error, normalized);
    return true;
}

bool SameParameterPresence(
    const SurfaceParameters& left,
    const SurfaceParameters& right)
{
    return left.origin.has_value() == right.origin.has_value() &&
        left.axis.has_value() == right.axis.has_value() &&
        left.radius.has_value() == right.radius.has_value() &&
        left.secondary_radius.has_value() == right.secondary_radius.has_value() &&
        left.half_angle.has_value() == right.half_angle.has_value();
}

std::optional<CandidateScore> ScoreCandidate(
    const FaceSignature& occ,
    const FaceSignature& nx,
    const FaceMatchOptions& options)
{
    if (occ.surface_type != nx.surface_type ||
        occ.edge_count != nx.edge_count ||
        occ.loop_count != nx.loop_count ||
        occ.neighbor_count != nx.neighbor_count ||
        !SameParameterPresence(occ.surface, nx.surface))
    {
        return std::nullopt;
    }

    CandidateScore score;
    if (!AddComponent(score, RelativeError(occ.area, nx.area),
                      options.relative_scalar_tolerance) ||
        !AddComponent(score, RelativeError(occ.perimeter, nx.perimeter),
                      options.relative_scalar_tolerance) ||
        !AddComponent(score, Distance(occ.centroid, nx.centroid),
                      options.normalized_position_tolerance) ||
        !AddComponent(score, Distance(occ.bounds.minimum, nx.bounds.minimum),
                      options.normalized_position_tolerance) ||
        !AddComponent(score, Distance(occ.bounds.maximum, nx.bounds.maximum),
                      options.normalized_position_tolerance) ||
        !AddComponent(score,
                      SymmetricHausdorff(occ.sample_points, nx.sample_points),
                      options.sample_distance_tolerance))
    {
        return std::nullopt;
    }

    if (occ.surface.origin &&
        !AddComponent(score, Distance(*occ.surface.origin, *nx.surface.origin),
                      options.normalized_position_tolerance))
    {
        return std::nullopt;
    }
    if (occ.surface.axis)
    {
        const auto left = CanonicalAxis(*occ.surface.axis);
        const auto right = CanonicalAxis(*nx.surface.axis);
        const double dot = std::clamp(
            std::abs(left.x * right.x + left.y * right.y + left.z * right.z),
            0.0,
            1.0);
        if (!AddComponent(score, std::acos(dot), options.axis_angle_tolerance))
        {
            return std::nullopt;
        }
    }
    const auto add_optional_scalar = [&](const std::optional<double>& left,
                                         const std::optional<double>& right) {
        return !left || AddComponent(
            score, RelativeError(*left, *right),
            options.relative_scalar_tolerance);
    };
    if (!add_optional_scalar(occ.surface.radius, nx.surface.radius) ||
        !add_optional_scalar(
            occ.surface.secondary_radius, nx.surface.secondary_radius) ||
        !add_optional_scalar(occ.surface.half_angle, nx.surface.half_angle))
    {
        return std::nullopt;
    }
    return score;
}

std::set<AdjacencyEdge> CanonicalEdges(const FaceGraph& graph)
{
    return {graph.adjacency_edges.begin(), graph.adjacency_edges.end()};
}

bool AdjacencyMatches(
    const FaceGraph& occ_graph,
    const FaceGraph& nx_graph,
    const std::vector<std::size_t>& assignment)
{
    std::unordered_map<FaceKey, FaceKey> mapped_tags;
    for (std::size_t row = 0; row < assignment.size(); ++row)
    {
        mapped_tags.emplace(
            occ_graph.faces[row].key,
            nx_graph.faces[assignment[row]].key);
    }
    std::set<AdjacencyEdge> mapped_edges;
    for (const auto& edge : occ_graph.adjacency_edges)
    {
        AdjacencyEdge mapped{mapped_tags.at(edge.first), mapped_tags.at(edge.second)};
        if (mapped.first > mapped.second)
        {
            std::swap(mapped.first, mapped.second);
        }
        mapped_edges.insert(mapped);
    }
    return mapped_edges == CanonicalEdges(nx_graph);
}

using ForbiddenEdge = std::pair<std::size_t, std::size_t>;

struct MurtyNode
{
    std::vector<std::size_t> fixed_prefix;
    std::set<ForbiddenEdge> forbidden_edges;
    AssignmentSolution solution;
};

struct HigherMurtyCost
{
    bool operator()(const MurtyNode& left, const MurtyNode& right) const
    {
        return left.solution.total_cost > right.solution.total_cost;
    }
};

using MurtyQueue =
    std::priority_queue<MurtyNode, std::vector<MurtyNode>, HigherMurtyCost>;

std::optional<AssignmentSolution> SolveMurtyNode(
    const std::vector<std::vector<double>>& base_costs,
    const std::vector<std::size_t>& fixed_prefix,
    const std::set<ForbiddenEdge>& forbidden_edges)
{
    const std::size_t size = base_costs.size();
    const double infinity = std::numeric_limits<double>::infinity();
    auto constrained = base_costs;
    for (const auto& edge : forbidden_edges)
    {
        if (edge.first >= size || edge.second >= size)
        {
            return std::nullopt;
        }
        constrained[edge.first][edge.second] = infinity;
    }

    std::set<std::size_t> fixed_columns;
    for (std::size_t row = 0; row < fixed_prefix.size(); ++row)
    {
        const std::size_t column = fixed_prefix[row];
        if (column >= size || !fixed_columns.insert(column).second)
        {
            return std::nullopt;
        }
        for (std::size_t candidate = 0; candidate < size; ++candidate)
        {
            if (candidate != column)
            {
                constrained[row][candidate] = infinity;
            }
        }
        for (std::size_t other_row = 0; other_row < size; ++other_row)
        {
            if (other_row != row)
            {
                constrained[other_row][column] = infinity;
            }
        }
    }
    return SolveMinimumCostAssignment(constrained);
}

void PushMurtyChildren(
    const std::vector<std::vector<double>>& costs,
    const MurtyNode& parent,
    MurtyQueue& queue)
{
    const std::size_t size = parent.solution.column_for_row.size();
    const std::size_t first_free_row = parent.fixed_prefix.size();
    for (std::size_t split_row = first_free_row;
         split_row < size;
         ++split_row)
    {
        MurtyNode child;
        child.fixed_prefix.assign(
            parent.solution.column_for_row.begin(),
            parent.solution.column_for_row.begin() + split_row);
        child.forbidden_edges = parent.forbidden_edges;
        child.forbidden_edges.insert(
            {split_row, parent.solution.column_for_row[split_row]});
        const auto solution = SolveMurtyNode(
            costs, child.fixed_prefix, child.forbidden_edges);
        if (solution)
        {
            child.solution = *solution;
            queue.push(std::move(child));
        }
    }
}

bool HasTopologyValidAlternative(
    const FaceGraph& occ_graph,
    const FaceGraph& nx_graph,
    const std::vector<std::vector<double>>& costs,
    const AssignmentSolution& best,
    double margin)
{
    MurtyNode root;
    root.solution = best;
    MurtyQueue queue;
    PushMurtyChildren(costs, root, queue);
    const double limit = best.total_cost + margin;
    while (!queue.empty() && queue.top().solution.total_cost <= limit)
    {
        MurtyNode node = queue.top();
        queue.pop();
        if (node.solution.column_for_row != best.column_for_row &&
            AdjacencyMatches(
                occ_graph, nx_graph, node.solution.column_for_row))
        {
            return true;
        }
        PushMurtyChildren(costs, node, queue);
    }
    return false;
}

bool EveryRowHasOnlyOneCandidate(
    const std::vector<std::vector<double>>& costs)
{
    for (const auto& row : costs)
    {
        const auto candidate_count = std::count_if(
            row.begin(), row.end(), [](double cost) {
                return std::isfinite(cost);
            });
        if (candidate_count != 1)
        {
            return false;
        }
    }
    return true;
}

void ValidateOptions(const FaceMatchOptions& options)
{
    const double positive[] = {
        options.relative_scalar_tolerance,
        options.normalized_position_tolerance,
        options.axis_angle_tolerance,
        options.sample_distance_tolerance};
    for (const double value : positive)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("face match tolerances must be positive");
        }
    }
    if (!std::isfinite(options.ambiguity_cost_margin) ||
        options.ambiguity_cost_margin < 0.0)
    {
        throw std::invalid_argument(
            "face match ambiguity margin must be nonnegative");
    }
}
}

FaceMappingResult MatchFaces(
    const FaceGraph& occ_graph,
    const FaceGraph& nx_graph,
    const FaceMatchOptions& options)
{
    try
    {
        ValidateOptions(options);
        ValidateFaceGraph(occ_graph);
        ValidateFaceGraph(nx_graph);
    }
    catch (const std::exception& error)
    {
        return Failure(std::string("invalid face graph: ") + error.what());
    }
    if (occ_graph.faces.empty())
    {
        return Failure("face graphs must not be empty");
    }
    if (occ_graph.faces.size() != nx_graph.faces.size())
    {
        return Failure("OCC and NX face counts differ");
    }
    for (const auto& face : occ_graph.faces)
    {
        if (face.key > static_cast<FaceKey>(std::numeric_limits<int>::max()))
        {
            return Failure("OCC face index is outside int range");
        }
    }

    const std::size_t size = occ_graph.faces.size();
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> costs(
        size, std::vector<double>(size, infinity));
    std::vector<std::vector<double>> maximum_errors(
        size, std::vector<double>(size, infinity));
    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t column = 0; column < size; ++column)
        {
            const auto score = ScoreCandidate(
                occ_graph.faces[row], nx_graph.faces[column], options);
            if (score)
            {
                costs[row][column] = score->cost;
                maximum_errors[row][column] = score->max_normalized_error;
            }
        }
    }

    const auto solution = SolveMinimumCostAssignment(costs);
    if (!solution)
    {
        return Failure("no complete OCC-to-NX face assignment exists");
    }
    if (!AdjacencyMatches(
            occ_graph, nx_graph, solution->column_for_row))
    {
        return Failure("best face assignment does not preserve adjacency");
    }
    if (!EveryRowHasOnlyOneCandidate(costs) &&
        HasTopologyValidAlternative(
            occ_graph,
            nx_graph,
            costs,
            *solution,
            options.ambiguity_cost_margin))
    {
        return Failure("ambiguous topology-valid face assignment");
    }

    FaceMappingResult result;
    result.success = true;
    result.total_cost = solution->total_cost;
    for (std::size_t row = 0; row < size; ++row)
    {
        const std::size_t column = solution->column_for_row[row];
        const double maximum_error = maximum_errors[row][column];
        result.matches.push_back({
            static_cast<int>(occ_graph.faces[row].key),
            nx_graph.faces[column].key,
            costs[row][column],
            maximum_error});
        result.max_error = std::max(result.max_error, maximum_error);
    }
    std::sort(
        result.matches.begin(), result.matches.end(),
        [](const FaceMatch& left, const FaceMatch& right) {
            return left.occ_face_index < right.occ_face_index;
        });
    return result;
}
}
}
