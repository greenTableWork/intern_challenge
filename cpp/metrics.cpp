#include "placement/metrics.h"

namespace placement {

OverlapMetrics calculateOverlapMetrics(const torch::Tensor& cell_features) {
    OverlapMetrics metrics;
    metrics.has_zero_overlap = cell_features.size(0) <= 1;
    return metrics;
}

Metrics calculateNormalizedMetrics(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    (void)pin_features;
    Metrics metrics;
    metrics.total_cells = cell_features.size(0);
    metrics.num_nets = edge_list.size(0);
    return metrics;
}

}  // namespace placement
