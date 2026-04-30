#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <string>

namespace placement {

void plotPlacement(
    const torch::Tensor& initial_cell_features,
    const torch::Tensor& final_cell_features,
    const std::filesystem::path& output_path);

void plotPlacementState(
    const torch::Tensor& cell_features,
    const std::filesystem::path& output_path,
    const std::string& title);

}  // namespace placement
