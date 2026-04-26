#include "placement/generation.h"
#include "placement/losses.h"
#include "placement/metrics.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectNear(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void deterministicMetricsMatchPythonReference() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
            {1.0F, 1.0F, 10.0F, 10.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 2LL}}, long_options);

    const placement::OverlapMetrics overlap =
        placement::calculateOverlapMetrics(cell_features);
    expect(overlap.overlap_count == 1, "expected one overlapping pair");
    expectNear(overlap.total_overlap_area, 2.0, 1e-5, "total overlap area");
    expectNear(overlap.max_overlap_area, 2.0, 1e-5, "max overlap area");
    expectNear(overlap.overlap_percentage, 100.0 / 3.0, 1e-5, "overlap percentage");
    expect(overlap.cells_with_overlap == 2, "expected two cells with overlap");
    expect(!overlap.has_zero_overlap, "expected nonzero overlap flag");

    const placement::Metrics metrics =
        placement::calculateNormalizedMetrics(cell_features, pin_features, edge_list);
    expect(metrics.total_cells == 3, "expected three total cells");
    expect(metrics.num_nets == 1, "expected one net");
    expect(metrics.num_cells_with_overlaps == 2, "expected two overlapping cells");
    expectNear(metrics.overlap_ratio, 2.0 / 3.0, 1e-6, "overlap ratio");

    const double smooth_wirelength = 10.0 + 0.1 * std::log(2.0);
    expectNear(
        metrics.normalized_wl,
        smooth_wirelength / 3.0,
        1e-5,
        "normalized wirelength");

    const auto no_edges = torch::zeros({0, 2}, long_options);
    const placement::Metrics no_edge_metrics =
        placement::calculateNormalizedMetrics(cell_features, pin_features, no_edges);
    expectNear(no_edge_metrics.normalized_wl, 0.0, 1e-12, "zero-edge wirelength");
}

void generatedProblemCanBeMeasured() {
    torch::manual_seed(66);
    placement::PlacementProblem problem =
        placement::generatePlacementInput(2, 5, torch::kCPU, false);

    expect(problem.cell_features.size(0) == 7, "generated cell count");
    expect(problem.cell_features.size(1) == 6, "generated cell feature width");
    expect(problem.pin_features.size(1) == 7, "generated pin feature width");
    expect(problem.edge_list.size(1) == 2, "generated edge width");

    placement::initializeCellPositions(problem.cell_features);
    const placement::Metrics metrics = placement::calculateNormalizedMetrics(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list);

    expect(metrics.total_cells == 7, "metrics total cell count");
    expect(metrics.num_nets == problem.edge_list.size(0), "metrics net count");
    expect(
        metrics.num_cells_with_overlaps >= 0 &&
            metrics.num_cells_with_overlaps <= metrics.total_cells,
        "overlapping cell count range");
    expect(
        metrics.overlap_ratio >= 0.0 && metrics.overlap_ratio <= 1.0,
        "overlap ratio range");
    expect(std::isfinite(metrics.normalized_wl), "finite normalized wirelength");
    expect(metrics.normalized_wl >= 0.0, "nonnegative normalized wirelength");
}

void deterministicLossesMatchPythonReference() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
            {1.0F, 1.0F, 10.0F, 10.0F, 1.0F, 1.0F},
        },
        float_options);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 2LL}}, long_options);

    const auto pairwise_overlap =
        placement::computePairwiseOverlapAreas(cell_features);
    const auto expected_pairwise_overlap = torch::tensor(
        {
            {4.0F, 2.0F, 0.0F},
            {2.0F, 4.0F, 0.0F},
            {0.0F, 0.0F, 1.0F},
        },
        float_options);
    expect(
        torch::allclose(pairwise_overlap, expected_pairwise_overlap),
        "pairwise overlap areas");

    const double smooth_wirelength = 10.0 + 0.1 * std::log(2.0);
    expectNear(
        placement::wirelengthAttractionLoss(cell_features, pin_features, edge_list)
            .item<double>(),
        smooth_wirelength,
        1e-5,
        "wirelength attraction loss");

    expectNear(
        placement::overlapRepulsionLoss(cell_features, pin_features, edge_list)
            .item<double>(),
        std::log1p(2.0) * 200.0,
        1e-4,
        "overlap repulsion loss");
}

void lossEdgeCasesStayFinite() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto single_cell = torch::tensor(
        {{4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F}},
        float_options);
    const auto single_pin = torch::tensor(
        {{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F}},
        float_options);
    const auto no_edges = torch::zeros({0, 2}, long_options);

    const auto single_overlap =
        placement::computePairwiseOverlapAreas(single_cell);
    expect(single_overlap.size(0) == 1, "single-cell overlap row count");
    expect(single_overlap.size(1) == 1, "single-cell overlap column count");
    expectNear(
        placement::wirelengthAttractionLoss(single_cell, single_pin, no_edges)
            .item<double>(),
        0.0,
        1e-12,
        "zero-edge wirelength loss");
    expectNear(
        placement::overlapRepulsionLoss(single_cell, single_pin, no_edges)
            .item<double>(),
        0.0,
        1e-12,
        "single-cell overlap loss");
}

void lossesBackpropagateThroughCellPositions() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64);

    const auto cell_static_prefix = torch::tensor(
        {
            {4.0F, 1.0F},
            {4.0F, 1.0F},
        },
        float_options);
    auto positions = torch::tensor(
        {
            {0.0F, 0.0F},
            {1.0F, 0.0F},
        },
        float_options.requires_grad(true));
    const auto cell_static_suffix = torch::tensor(
        {
            {2.0F, 2.0F},
            {2.0F, 2.0F},
        },
        float_options);
    const auto cell_features =
        torch::cat({cell_static_prefix, positions, cell_static_suffix}, 1);
    const auto pin_features = torch::tensor(
        {
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
            {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F, 0.1F},
        },
        float_options);
    const auto edge_list = torch::tensor({{0LL, 1LL}}, long_options);

    const auto loss =
        placement::wirelengthAttractionLoss(cell_features, pin_features, edge_list) +
        placement::overlapRepulsionLoss(cell_features, pin_features, edge_list);
    loss.backward();

    expect(positions.grad().defined(), "positions gradient is defined");
    expect(torch::all(torch::isfinite(positions.grad())).item<bool>(), "finite gradients");
    expect(positions.grad().abs().sum().item<double>() > 0.0, "nonzero gradients");
}

}  // namespace

int main() {
    try {
        deterministicMetricsMatchPythonReference();
        generatedProblemCanBeMeasured();
        deterministicLossesMatchPythonReference();
        lossEdgeCasesStayFinite();
        lossesBackpropagateThroughCellPositions();
    } catch (const std::exception& error) {
        std::cerr << "placement_unit_tests failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "placement_unit_tests passed\n";
    return 0;
}
