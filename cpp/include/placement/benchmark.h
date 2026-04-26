#pragma once

#include "placement/types.h"

#include <vector>

namespace placement {

const std::vector<BenchmarkCase>& activeBenchmarkCases();

BenchmarkResult runBenchmarkCase(
    const BenchmarkCase& test_case,
    const TrainingConfig& config = {});

BenchmarkSummary runBenchmarkCases(
    const std::vector<BenchmarkCase>& test_cases,
    const TrainingConfig& config = {});

BenchmarkSummary runActiveBenchmarkCases(const TrainingConfig& config = {});

}  // namespace placement
