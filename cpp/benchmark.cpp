#include "placement/benchmark.h"

#include "placement/generation.h"
#include "placement/metrics.h"
#include "placement/training.h"

#include <chrono>
#include <utility>

namespace placement {

const std::vector<BenchmarkCase>& activeBenchmarkCases() {
    static const std::vector<BenchmarkCase> cases = {
        {1, 2, 20, 1001},
        {2, 3, 25, 1002},
        {3, 2, 30, 1003},
        {4, 3, 50, 1004},
        {5, 4, 75, 1005},
        {6, 5, 100, 1006},
        {7, 5, 150, 1007},
        {8, 7, 150, 1008},
        {9, 8, 200, 1009},
        {10, 10, 2000, 1010},
    };
    return cases;
}

BenchmarkResult runBenchmarkCase(
    const BenchmarkCase& test_case,
    const TrainingConfig& config) {
    if (test_case.seed != 0) {
        torch::manual_seed(test_case.seed);
    }

    TrainingConfig benchmark_config = config;
    benchmark_config.verbose = false;
    const torch::Device device(benchmark_config.device);

    PlacementProblem problem = generatePlacementInput(
        test_case.num_macros,
        test_case.num_std_cells,
        device,
        false);
    initializeCellPositions(problem.cell_features);

    const auto start_time = std::chrono::steady_clock::now();
    const TrainingResult training_result = trainPlacement(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list,
        benchmark_config);
    const auto elapsed = std::chrono::steady_clock::now() - start_time;

    const Metrics metrics = calculateNormalizedMetrics(
        training_result.final_cell_features,
        problem.pin_features,
        problem.edge_list);

    BenchmarkResult result;
    result.test_id = test_case.test_id;
    result.num_macros = test_case.num_macros;
    result.num_std_cells = test_case.num_std_cells;
    result.total_cells = metrics.total_cells;
    result.num_nets = metrics.num_nets;
    result.seed = test_case.seed;
    result.device = benchmark_config.device;
    result.elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    result.num_cells_with_overlaps = metrics.num_cells_with_overlaps;
    result.overlap_ratio = metrics.overlap_ratio;
    result.normalized_wl = metrics.normalized_wl;
    result.passed = result.num_cells_with_overlaps == 0;
    return result;
}

BenchmarkSummary runBenchmarkCases(
    const std::vector<BenchmarkCase>& test_cases,
    const TrainingConfig& config) {
    BenchmarkSummary summary;
    if (test_cases.empty()) {
        return summary;
    }

    summary.results.reserve(test_cases.size());
    double overlap_sum = 0.0;
    double wirelength_sum = 0.0;

    const auto start_time = std::chrono::steady_clock::now();
    for (const BenchmarkCase& test_case : test_cases) {
        BenchmarkResult result = runBenchmarkCase(test_case, config);

        overlap_sum += result.overlap_ratio;
        wirelength_sum += result.normalized_wl;
        if (result.passed) {
            ++summary.passed_count;
        } else {
            ++summary.failed_count;
        }

        summary.results.push_back(std::move(result));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start_time;

    const double case_count = static_cast<double>(test_cases.size());
    summary.average_overlap = overlap_sum / case_count;
    summary.average_wirelength = wirelength_sum / case_count;
    summary.total_elapsed_seconds = std::chrono::duration<double>(elapsed).count();

    return summary;
}

BenchmarkSummary runActiveBenchmarkCases(const TrainingConfig& config) {
    return runBenchmarkCases(activeBenchmarkCases(), config);
}

}  // namespace placement
