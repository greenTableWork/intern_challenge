#pragma once

#include <ATen/Tensor.h>

#include <cstdint>

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
