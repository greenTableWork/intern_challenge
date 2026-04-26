#include "placement/benchmark.h"

#include <stdexcept>

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
    (void)test_case;
    (void)config;
    throw std::logic_error("runBenchmarkCase is implemented in Step 6");
}

}  // namespace placement
