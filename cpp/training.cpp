#include "placement/training.h"

namespace placement {

TrainingResult trainPlacement(
    const torch::Tensor& cell_features,
    const torch::Tensor& pin_features,
    const torch::Tensor& edge_list,
    const TrainingConfig& config) {
    (void)pin_features;
    (void)edge_list;
    (void)config;
    TrainingResult result;
    result.initial_cell_features = cell_features.clone();
    result.final_cell_features = cell_features.clone();
    return result;
}

}  // namespace placement
