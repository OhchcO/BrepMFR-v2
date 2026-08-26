#include "assignment_solver.hpp"

#include "test_support.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void ExpectInvalidArgument(
    const std::function<void()>& operation,
    const std::string& message)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    brepmfr::nx2212::test::Expect(rejected, message);
}
}

BREPMFR_TEST(AssignmentSolverFindsGlobalNotGreedyMinimum)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const std::vector<std::vector<double>> costs{
        {1.0, 2.0, 100.0},
        {1.1, 100.0, 100.0},
        {100.0, 1.0, 1.0}};
    const auto solution = SolveMinimumCostAssignment(costs);
    Expect(solution.has_value(), "perfect assignment missing");
    ExpectEqual(
        solution->column_for_row,
        std::vector<std::size_t>{1, 0, 2},
        "greedy result returned");
    ExpectNear(solution->total_cost, 4.1, 1e-12, "cost mismatch");
}

BREPMFR_TEST(AssignmentSolverHandlesPermutationAndEmptyMatrix)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const double infinity = std::numeric_limits<double>::infinity();
    const auto permutation = SolveMinimumCostAssignment({
        {infinity, 2.0, infinity},
        {infinity, infinity, 3.0},
        {4.0, infinity, infinity}});
    Expect(permutation.has_value(), "permutation assignment missing");
    ExpectEqual(
        permutation->column_for_row,
        std::vector<std::size_t>{1, 2, 0},
        "permutation assignment mismatch");
    ExpectNear(permutation->total_cost, 9.0, 1e-12, "permutation cost mismatch");

    const auto empty = SolveMinimumCostAssignment({});
    Expect(empty.has_value(), "0x0 assignment missing");
    Expect(empty->column_for_row.empty(), "0x0 assignment was not empty");
    ExpectNear(empty->total_cost, 0.0, 0.0, "0x0 cost mismatch");
}

BREPMFR_TEST(AssignmentSolverRejectsInvalidMatrices)
{
    using brepmfr::nx2212::SolveMinimumCostAssignment;

    ExpectInvalidArgument(
        [] { (void)SolveMinimumCostAssignment({{1.0, 2.0}, {3.0}}); },
        "non-square matrix was accepted");
    ExpectInvalidArgument(
        [] {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            (void)SolveMinimumCostAssignment({{nan}});
        },
        "NaN cost was accepted");
    ExpectInvalidArgument(
        [] { (void)SolveMinimumCostAssignment({{-1.0}}); },
        "negative cost was accepted");
    ExpectInvalidArgument(
        [] {
            const double negative_infinity =
                -std::numeric_limits<double>::infinity();
            (void)SolveMinimumCostAssignment({{negative_infinity}});
        },
        "negative infinity was accepted");
}

BREPMFR_TEST(AssignmentSolverReportsMissingPerfectAssignments)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    const double infinity = std::numeric_limits<double>::infinity();
    const auto empty_row = SolveMinimumCostAssignment({
        {0.0, 1.0},
        {infinity, infinity}});
    Expect(!empty_row.has_value(), "row without candidates was accepted");

    const auto hall_violation = SolveMinimumCostAssignment({
        {0.0, infinity, infinity},
        {1.0, infinity, infinity},
        {infinity, 0.0, 0.0}});
    Expect(
        !hall_violation.has_value(),
        "matrix without a perfect matching was accepted");
}
