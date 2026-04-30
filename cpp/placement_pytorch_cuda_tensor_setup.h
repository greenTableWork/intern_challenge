#pragma once

#include <ATen/Tensor.h>

#include <cstdint>

namespace placement_cuda {

enum class CellFeatureIdx : int64_t {
    Area = 0,
    NumPins = 1,
    X = 2,
    Y = 3,
    Width = 4,
    Height = 5,
    Count = 6,
};

}  // namespace placement_cuda

void fillPlacementTensorSetupCuda(
    const at::Tensor& macro_areas,
    const at::Tensor& std_area_indices,
    const at::Tensor& std_cell_areas,
    const at::Tensor& areas,
    const at::Tensor& cell_widths,
    const at::Tensor& cell_heights,
    const at::Tensor& cell_features,
    int64_t macro_count,
    int64_t std_cell_count,
    double min_macro_area,
    double max_macro_area,
    double standard_cell_height,
    uint64_t seed);
