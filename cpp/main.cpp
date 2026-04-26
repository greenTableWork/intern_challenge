#include "placement/benchmark.h"
#include "placement/generation.h"
#include "placement/metrics.h"
#include "placement/training.h"
#include "placement/types.h"

#include <CLI/CLI.hpp>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/torch.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    bool run_benchmark = false;
    std::string device = "auto";
    int test_case_id = 0;
    int num_macros = 3;
    int num_std_cells = 10;
    int seed = 42;
};

std::string deviceTypeName(c10::DeviceType device) {
    switch (device) {
        case c10::DeviceType::CPU:
            return "cpu";
        case c10::DeviceType::CUDA:
            return "cuda";
        case c10::DeviceType::MPS:
            return "mps";
        default:
            return "unknown";
    }
}

c10::DeviceType resolveDeviceType(const std::string& device) {
    if (device == "auto") {
        if (torch::cuda::is_available()) {
            return torch::kCUDA;
        }
        if (torch::mps::is_available()) {
            return torch::kMPS;
        }
        return torch::kCPU;
    }
    if (device == "cpu") {
        return torch::kCPU;
    }
    if (device == "cuda") {
        if (!torch::cuda::is_available()) {
            throw std::invalid_argument("CUDA device requested but unavailable");
        }
        return torch::kCUDA;
    }
    if (device == "mps") {
        if (!torch::mps::is_available()) {
            throw std::invalid_argument("MPS device requested but unavailable");
        }
        return torch::kMPS;
    }
    throw std::invalid_argument("Unsupported device: " + device);
}

void seedTorch(int seed) {
    torch::manual_seed(static_cast<uint64_t>(seed));
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(static_cast<uint64_t>(seed));
    }
    if (torch::mps::is_available()) {
        torch::mps::manual_seed(static_cast<uint64_t>(seed));
    }
}

std::vector<placement::BenchmarkCase> allKnownBenchmarkCases() {
    std::vector<placement::BenchmarkCase> cases(
        placement::activeBenchmarkCases().begin(),
        placement::activeBenchmarkCases().end());
    cases.push_back({11, 10, 10000, 1011});
    cases.push_back({12, 10, 100000, 1012});
    return cases;
}

std::optional<placement::BenchmarkCase> findBenchmarkCase(int test_case_id) {
    for (const placement::BenchmarkCase& test_case : allKnownBenchmarkCases()) {
        if (test_case.test_id == test_case_id) {
            return test_case;
        }
    }
    return std::nullopt;
}

void printRule(char c = '=') {
    std::cout << std::string(70, c) << "\n";
}

void printTrainingConfig(const placement::TrainingConfig& config) {
    std::cout << "Using hyperparameters:\n";
    std::cout << "  num_epochs: " << config.num_epochs << "\n";
    std::cout << "  lr: " << config.lr << "\n";
    std::cout << "  lambda_wirelength: " << config.lambda_wirelength << "\n";
    std::cout << "  lambda_overlap: " << config.lambda_overlap << "\n";
    std::cout << "  scheduler: " << config.scheduler_name << "\n";
    std::cout << "  scheduler_patience: " << config.scheduler_patience << "\n";
    std::cout << "  scheduler_factor: " << config.scheduler_factor << "\n";
    std::cout << "  scheduler_eta_min: " << config.scheduler_eta_min << "\n";
    std::cout << "  scheduler_step_size: " << config.scheduler_step_size << "\n";
    std::cout << "  scheduler_gamma: " << config.scheduler_gamma << "\n";
    std::cout << "  track_overlap_metrics: "
              << (config.track_overlap_metrics ? "true" : "false") << "\n";
    std::cout << "  early_stop_enabled: "
              << (config.early_stop_enabled ? "true" : "false") << "\n";
    std::cout << "  early_stop_patience: " << config.early_stop_patience << "\n";
    std::cout << "  early_stop_min_delta: " << config.early_stop_min_delta << "\n";
    std::cout << "  early_stop_overlap_threshold: "
              << config.early_stop_overlap_threshold << "\n";
    std::cout << "  early_stop_zero_overlap_patience: "
              << config.early_stop_zero_overlap_patience << "\n";
}

void printBenchmarkResult(const placement::BenchmarkResult& result) {
    const char* status = result.passed ? "PASS" : "FAIL";
    std::cout << "Completed test " << result.test_id << ":\n";
    std::cout << "  Device: " << deviceTypeName(result.device) << "\n";
    std::cout << "  Overlap Ratio: " << std::fixed << std::setprecision(4)
              << result.overlap_ratio << " (" << result.num_cells_with_overlaps
              << "/" << result.total_cells << " cells)\n";
    std::cout << "  Normalized WL: " << std::fixed << std::setprecision(4)
              << result.normalized_wl << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << result.elapsed_seconds << "s\n";
    std::cout << "  Status: " << status << "\n\n";
}

int runBenchmark(const placement::TrainingConfig& config) {
    printRule();
    std::cout << "PLACEMENT CHALLENGE TEST SUITE\n";
    printRule();
    std::cout << "\nRunning " << placement::activeBenchmarkCases().size()
              << " active test cases serially.\n";
    printTrainingConfig(config);
    std::cout << "\n";

    int case_index = 1;
    for (const placement::BenchmarkCase& test_case :
         placement::activeBenchmarkCases()) {
        const char* size_category =
            test_case.num_std_cells <= 30
                ? "Small"
                : test_case.num_std_cells <= 100 ? "Medium" : "Large";
        std::cout << "Test " << case_index++ << "/"
                  << placement::activeBenchmarkCases().size() << ": "
                  << size_category << " (" << test_case.num_macros
                  << " macros, " << test_case.num_std_cells << " std cells)\n";
        std::cout << "  Seed: " << test_case.seed << "\n";
    }
    std::cout << "\n";

    const placement::BenchmarkSummary summary =
        placement::runActiveBenchmarkCases(config);
    for (const placement::BenchmarkResult& result : summary.results) {
        printBenchmarkResult(result);
    }

    printRule();
    std::cout << "FINAL RESULTS\n";
    printRule();
    std::cout << "Average Overlap: " << std::fixed << std::setprecision(4)
              << summary.average_overlap << "\n";
    std::cout << "Average Wirelength: " << std::fixed << std::setprecision(4)
              << summary.average_wirelength << "\n";
    std::cout << "Total Runtime: " << std::fixed << std::setprecision(2)
              << summary.total_elapsed_seconds << "s\n";
    std::cout << "Passed: " << summary.passed_count << "\n";
    std::cout << "Failed: " << summary.failed_count << "\n";
    return 0;
}

int runSinglePlacement(
    const CliOptions& options,
    const placement::TrainingConfig& config) {
    placement::BenchmarkCase selected_case{
        0,
        options.num_macros,
        options.num_std_cells,
        options.seed,
    };
    if (options.test_case_id != 0) {
        const std::optional<placement::BenchmarkCase> test_case =
            findBenchmarkCase(options.test_case_id);
        if (!test_case.has_value()) {
            throw std::invalid_argument(
                "Unknown benchmark test case id: " +
                std::to_string(options.test_case_id));
        }
        selected_case = *test_case;
    }

    printRule();
    std::cout << "VLSI CELL PLACEMENT OPTIMIZATION\n";
    printRule();
    std::cout << "\nGenerating placement problem:\n";
    if (selected_case.test_id != 0) {
        std::cout << "  - benchmark test case: " << selected_case.test_id << "\n";
    }
    std::cout << "  - " << selected_case.num_macros << " macros\n";
    std::cout << "  - " << selected_case.num_std_cells << " standard cells\n";
    std::cout << "  - seed: " << selected_case.seed << "\n";
    std::cout << "  - device: " << deviceTypeName(config.device) << "\n";

    seedTorch(selected_case.seed);
    const torch::Device device(config.device);
    placement::PlacementProblem problem = placement::generatePlacementInput(
        selected_case.num_macros,
        selected_case.num_std_cells,
        device);
    placement::initializeCellPositions(problem.cell_features);

    std::cout << "\n";
    printRule();
    std::cout << "INITIAL STATE\n";
    printRule();
    const placement::OverlapMetrics initial_metrics =
        placement::calculateOverlapMetrics(problem.cell_features);
    std::cout << "Overlap count: " << initial_metrics.overlap_count << "\n";
    std::cout << "Total overlap area: " << std::fixed << std::setprecision(2)
              << initial_metrics.total_overlap_area << "\n";
    std::cout << "Max overlap area: " << std::fixed << std::setprecision(2)
              << initial_metrics.max_overlap_area << "\n";
    std::cout << "Overlap percentage: " << std::fixed << std::setprecision(2)
              << initial_metrics.overlap_percentage << "%\n";

    std::cout << "\n";
    printRule();
    std::cout << "RUNNING OPTIMIZATION\n";
    printRule();
    placement::TrainingResult training_result = placement::trainPlacement(
        problem.cell_features,
        problem.pin_features,
        problem.edge_list,
        config);

    std::cout << "\n";
    printRule();
    std::cout << "FINAL RESULTS\n";
    printRule();
    const placement::OverlapMetrics final_overlap_metrics =
        placement::calculateOverlapMetrics(training_result.final_cell_features);
    std::cout << "Overlap count (pairs): "
              << final_overlap_metrics.overlap_count << "\n";
    std::cout << "Total overlap area: " << std::fixed << std::setprecision(2)
              << final_overlap_metrics.total_overlap_area << "\n";
    std::cout << "Max overlap area: " << std::fixed << std::setprecision(2)
              << final_overlap_metrics.max_overlap_area << "\n";

    std::cout << "\n";
    printRule('-');
    std::cout << "TEST SUITE METRICS\n";
    printRule('-');
    const placement::Metrics normalized_metrics =
        placement::calculateNormalizedMetrics(
            training_result.final_cell_features,
            problem.pin_features,
            problem.edge_list);
    std::cout << "Overlap Ratio: " << std::fixed << std::setprecision(4)
              << normalized_metrics.overlap_ratio << " ("
              << normalized_metrics.num_cells_with_overlaps << "/"
              << normalized_metrics.total_cells << " cells)\n";
    std::cout << "Normalized Wirelength: " << std::fixed << std::setprecision(4)
              << normalized_metrics.normalized_wl << "\n";
    if (training_result.stopped_early) {
        std::cout << "Stopped Early: " << training_result.stop_reason
                  << " at best epoch " << training_result.best_epoch << "\n";
    }

    std::cout << "\n";
    printRule();
    std::cout << "SUCCESS CRITERIA\n";
    printRule();
    if (normalized_metrics.num_cells_with_overlaps == 0) {
        std::cout << "PASS: No overlapping cells.\n";
        return 0;
    }

    std::cout << "FAIL: Overlaps remain in "
              << normalized_metrics.num_cells_with_overlaps << " cells.\n";
    return 0;
}

void configureCli(
    CLI::App& app,
    CliOptions& options,
    placement::TrainingConfig& config) {
    app.add_flag(
        "--benchmark",
        options.run_benchmark,
        "Run the active benchmark suite instead of a single placement.");
    app.add_option(
           "--device",
           options.device,
           "Device to run on: auto, cpu, cuda, or mps.")
        ->check(CLI::IsMember({"auto", "cpu", "cuda", "mps"}));
    app.add_option(
        "--test-case-id",
        options.test_case_id,
        "Optional benchmark test case id for a single placement run.");
    app.add_option(
        "--num-macros",
        options.num_macros,
        "Number of macro cells for a single placement run.");
    app.add_option(
        "--num-std-cells",
        options.num_std_cells,
        "Number of standard cells for a single placement run.");
    app.add_option("--seed", options.seed, "Random seed for a single placement run.");

    app.add_option(
        "--num-epochs",
        config.num_epochs,
        "Number of optimization epochs.");
    app.add_option("--lr", config.lr, "Learning rate for Adam.");
    app.add_option(
        "--lambda-wirelength",
        config.lambda_wirelength,
        "Weight applied to the wirelength loss.");
    app.add_option(
        "--lambda-overlap",
        config.lambda_overlap,
        "Weight applied to the overlap loss.");
    app.add_option("--scheduler", config.scheduler_name, "Learning-rate scheduler.")
        ->check(CLI::IsMember(
            {"plateau", "cosine", "step", "exponential", "none"}));
    app.add_option(
        "--scheduler-patience",
        config.scheduler_patience,
        "Patience for ReduceLROnPlateau.");
    app.add_option(
        "--scheduler-factor",
        config.scheduler_factor,
        "Decay factor for ReduceLROnPlateau.");
    app.add_option(
        "--scheduler-eta-min",
        config.scheduler_eta_min,
        "Minimum learning rate for cosine annealing.");
    app.add_option(
        "--scheduler-step-size",
        config.scheduler_step_size,
        "Step size in epochs for StepLR.");
    app.add_option(
        "--scheduler-gamma",
        config.scheduler_gamma,
        "Gamma decay for step and exponential schedulers.");
    app.add_flag(
        "--track-overlap-metrics",
        config.track_overlap_metrics,
        "Compute overlap metrics every epoch.");
    app.add_flag(
        "--no-early-stop",
        [&config](int64_t count) {
            if (count > 0) {
                config.early_stop_enabled = false;
            }
        },
        "Disable overlap-first early stopping.");
    app.add_option(
        "--early-stop-patience",
        config.early_stop_patience,
        "Patience before stopping when overlap stops improving.");
    app.add_option(
        "--early-stop-min-delta",
        config.early_stop_min_delta,
        "Minimum improvement required to reset early-stop patience.");
    app.add_option(
        "--early-stop-overlap-threshold",
        config.early_stop_overlap_threshold,
        "Overlap threshold treated as effectively zero.");
    app.add_option(
        "--early-stop-zero-overlap-patience",
        config.early_stop_zero_overlap_patience,
        "Extra patience after zero overlap is reached.");
    app.add_flag("--quiet", [&config](int64_t count) {
        if (count > 0) {
            config.verbose = false;
        }
    }, "Suppress per-epoch output for a single placement run.");
    app.add_option(
        "--log-interval",
        config.log_interval,
        "Epoch interval for verbose training logs.");
}

void validateOptions(
    const CliOptions& options,
    const placement::TrainingConfig& config) {
    if (options.num_macros < 0 || options.num_std_cells < 0) {
        throw std::invalid_argument("Cell counts must be nonnegative");
    }
    if (options.num_macros + options.num_std_cells < 0) {
        throw std::invalid_argument("Cell counts overflowed");
    }
    if (config.num_epochs < 0) {
        throw std::invalid_argument("Number of epochs must be nonnegative");
    }
    if (config.lr <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (config.scheduler_patience < 0) {
        throw std::invalid_argument("Scheduler patience must be nonnegative");
    }
    if (config.scheduler_factor <= 0.0) {
        throw std::invalid_argument("Scheduler factor must be positive");
    }
    if (config.scheduler_step_size <= 0) {
        throw std::invalid_argument("Scheduler step size must be positive");
    }
    if (config.early_stop_patience <= 0 ||
        config.early_stop_zero_overlap_patience <= 0) {
        throw std::invalid_argument("Early-stop patience values must be positive");
    }
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions options;
    placement::TrainingConfig config;
    config.log_interval = 200;

    CLI::App app{"Placement C++ runner"};
    configureCli(app, options, config);
    CLI11_PARSE(app, argc, argv);

    try {
        validateOptions(options, config);
        config.device = resolveDeviceType(options.device);

        if (options.run_benchmark) {
            config.verbose = false;
            return runBenchmark(config);
        }
        return runSinglePlacement(options, config);
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what_without_backtrace() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
    }
    return 1;
}
