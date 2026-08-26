#include "assignment_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
std::optional<AssignmentSolution> SolveMinimumCostAssignment(
    const std::vector<std::vector<double>>& costs)
{
    const std::size_t size = costs.size();
    for (const auto& row : costs)
    {
        if (row.size() != size)
        {
            throw std::invalid_argument("assignment cost matrix must be square");
        }
        for (const double cost : row)
        {
            if (std::isnan(cost) || cost < 0.0 ||
                (std::isinf(cost) && cost < 0.0))
            {
                throw std::invalid_argument(
                    "assignment costs must be nonnegative finite values or +infinity");
            }
        }
    }
    if (size == 0)
    {
        return AssignmentSolution{{}, 0.0};
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> row_potential(size + 1, 0.0);
    std::vector<double> column_potential(size + 1, 0.0);
    std::vector<std::size_t> matched_row(size + 1, 0);
    std::vector<std::size_t> predecessor(size + 1, 0);

    for (std::size_t row = 1; row <= size; ++row)
    {
        matched_row[0] = row;
        std::vector<double> minimum_slack(size + 1, infinity);
        std::vector<bool> used(size + 1, false);
        std::size_t column = 0;

        do
        {
            used[column] = true;
            const std::size_t current_row = matched_row[column];
            double delta = infinity;
            std::size_t next_column = 0;
            for (std::size_t candidate = 1; candidate <= size; ++candidate)
            {
                if (used[candidate])
                {
                    continue;
                }
                const double edge_cost = costs[current_row - 1][candidate - 1];
                if (std::isfinite(edge_cost))
                {
                    const double reduced_cost = edge_cost -
                        row_potential[current_row] -
                        column_potential[candidate];
                    if (reduced_cost < minimum_slack[candidate])
                    {
                        minimum_slack[candidate] = reduced_cost;
                        predecessor[candidate] = column;
                    }
                }
                if (minimum_slack[candidate] < delta)
                {
                    delta = minimum_slack[candidate];
                    next_column = candidate;
                }
            }
            if (!std::isfinite(delta))
            {
                return std::nullopt;
            }
            for (std::size_t candidate = 0; candidate <= size; ++candidate)
            {
                if (used[candidate])
                {
                    row_potential[matched_row[candidate]] += delta;
                    column_potential[candidate] -= delta;
                }
                else if (candidate != 0 &&
                         std::isfinite(minimum_slack[candidate]))
                {
                    minimum_slack[candidate] -= delta;
                }
            }
            column = next_column;
        } while (matched_row[column] != 0);

        do
        {
            const std::size_t previous = predecessor[column];
            matched_row[column] = matched_row[previous];
            column = previous;
        } while (column != 0);
    }

    const std::size_t unassigned = std::numeric_limits<std::size_t>::max();
    AssignmentSolution result;
    result.column_for_row.assign(size, unassigned);
    result.total_cost = 0.0;
    for (std::size_t column = 1; column <= size; ++column)
    {
        const std::size_t row = matched_row[column];
        if (row == 0 || result.column_for_row[row - 1] != unassigned)
        {
            return std::nullopt;
        }
        const double cost = costs[row - 1][column - 1];
        if (!std::isfinite(cost))
        {
            return std::nullopt;
        }
        result.column_for_row[row - 1] = column - 1;
        result.total_cost += cost;
    }
    if (!std::isfinite(result.total_cost))
    {
        throw std::overflow_error("assignment total cost overflowed");
    }
    for (const std::size_t column : result.column_for_row)
    {
        if (column == unassigned)
        {
            return std::nullopt;
        }
    }
    return result;
}
}
}
