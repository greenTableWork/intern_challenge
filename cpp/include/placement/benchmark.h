#pragma once

#include "placement/types.h"

#include <vector>

namespace placement {

const std::vector<BenchmarkCase>& activeBenchmarkCases();

BenchmarkResult runBenchmarkCase(
    const BenchmarkCase& test_case,
    const TrainingConfig& config = {});

}  // namespace placement
