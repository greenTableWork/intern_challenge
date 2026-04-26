#include "placement/losses.h"

#include <stdexcept>

namespace placement {

torch::Tensor computePairwiseOverlapAreas(const torch::Tensor& cell_features) {
    (void)cell_features;
    throw std::logic_error("computePairwiseOverlapAreas is implemented in Step 4");
}

torch::Tensor wirelengthAttractionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    (void)cell_features;
    (void)pin_features;
    (void)edge_list;
    return torch::zeros({}, torch::kFloat64);
}

torch::Tensor overlapRepulsionLoss(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list) {
    (void)cell_features;
    (void)pin_features;
    (void)edge_list;
    return torch::zeros({}, torch::kFloat64);
}

}  // namespace placement
