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

enum class PinFeatureIdx : int64_t {
    CellIdx = 0,
    PinX = 1,
    PinY = 2,
    X = 3,
    Y = 4,
    Width = 5,
    Height = 6,
    Count = 7,
};

void fillPlacementTensorSetupCuda(
    const at::Tensor& macro_areas,
    const at::Tensor& std_area_indices,
    const at::Tensor& std_cell_areas,
    const at::Tensor& areas,
    const at::Tensor& num_pins_per_cell,
    const at::Tensor& cell_widths,
    const at::Tensor& cell_heights,
    const at::Tensor& cell_features,
    int64_t macro_count,
    int64_t std_cell_count,
    double min_macro_area,
    double max_macro_area,
    double standard_cell_height,
    uint64_t seed);

void initializeCellPositionsCuda(
    const at::Tensor& cell_features,
    double spread_scale,
    uint64_t seed);

void computePinOffsetsCuda(
    const at::Tensor& num_pins_per_cell,
    const at::Tensor& pin_offsets);

void fillPinFeaturesCuda(
    const at::Tensor& cell_features,
    const at::Tensor& pin_offsets,
    const at::Tensor& pin_features,
    uint64_t seed);

void fillEdgeListCuda(
    const at::Tensor& edge_list_capacity,
    const at::Tensor& edge_count,
    const at::Tensor& pin_offsets,
    int64_t total_cells,
    int64_t max_pin_capacity,
    uint64_t seed);

}  // namespace placement_cuda
