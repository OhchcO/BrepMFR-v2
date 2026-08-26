#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
struct AssignmentSolution
{
    std::vector<std::size_t> column_for_row;
    double total_cost;
};

std::optional<AssignmentSolution> SolveMinimumCostAssignment(
    const std::vector<std::vector<double>>& costs);
}
}
