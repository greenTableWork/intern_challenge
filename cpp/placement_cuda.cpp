#include "placement_cuda_tensor_setup.h"

#include "placement/visualization.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {

#ifndef PLACEMENT_REPO_ROOT
#define PLACEMENT_REPO_ROOT "."
#endif

constexpr double kMinMacroArea = 100.0;
constexpr double kMaxMacroArea = 10000.0;
constexpr double kStandardCellHeight = 1.0;

struct PlacementTensorSetup {
    torch::Tensor macro_areas;
    torch::Tensor std_area_indices;
    torch::Tensor std_cell_areas;
    torch::Tensor areas;
    torch::Tensor num_pins_per_cell;
    torch::Tensor cell_widths;
    torch::Tensor cell_heights;
    torch::Tensor cell_features;
};

void checkTensor(
    const torch::Tensor& tensor,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> sizes) {
    TORCH_CHECK(tensor.is_cuda(), "expected CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, "unexpected tensor dtype");
    TORCH_CHECK(tensor.sizes() == sizes, "unexpected tensor shape");
}

PlacementTensorSetup buildPlacementTensorSetupCuda(
    int64_t macro_count,
    int64_t std_cell_count,
    uint64_t seed) {
    const int64_t total_cells = macro_count + std_cell_count;
    const torch::Device device(torch::kCUDA);
    const auto float_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto long_options =
        torch::TensorOptions().dtype(torch::kInt64).device(device);

    PlacementTensorSetup setup{
        torch::empty({macro_count}, float_options),
        torch::empty({std_cell_count}, long_options),
        torch::empty({std_cell_count}, float_options),
        torch::empty({total_cells}, float_options),
        torch::empty({total_cells}, long_options),
        torch::empty({total_cells}, float_options),
        torch::empty({total_cells}, float_options),
        torch::empty({total_cells, 6}, float_options),
    };

    fillPlacementTensorSetupCuda(
        setup.macro_areas,
        setup.std_area_indices,
        setup.std_cell_areas,
        setup.areas,
        setup.num_pins_per_cell,
        setup.cell_widths,
        setup.cell_heights,
        setup.cell_features,
        macro_count,
        std_cell_count,
        kMinMacroArea,
        kMaxMacroArea,
        kStandardCellHeight,
        seed);

    checkTensor(setup.macro_areas, torch::kFloat32, {macro_count});
    checkTensor(setup.std_area_indices, torch::kInt64, {std_cell_count});
    checkTensor(setup.std_cell_areas, torch::kFloat32, {std_cell_count});
    checkTensor(setup.areas, torch::kFloat32, {total_cells});
    checkTensor(setup.num_pins_per_cell, torch::kInt64, {total_cells});
    checkTensor(setup.cell_widths, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_heights, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_features, torch::kFloat32, {total_cells, 6});

    return setup;
}

#ifdef PLACEMENT_CUDA_ENABLE_DEBUG_RENDER
std::filesystem::path debugRenderOutputPath() {
    return std::filesystem::path(PLACEMENT_REPO_ROOT) /
           "placement_cuda_tensor_setup_debug.png";
}

void renderCudaTensorSetupDebug(const PlacementTensorSetup& setup) {
    const std::filesystem::path output_path = debugRenderOutputPath();
    placement::plotPlacement(
        setup.cell_features,
        setup.cell_features,
        output_path);
    std::cout << "Rendered CUDA tensor setup to: " << output_path << "\n";
}
#endif

void runCudaTensorSetupCheck(
    int64_t macro_count,
    int64_t std_cell_count,
    uint64_t seed) {
    PlacementTensorSetup setup =
        buildPlacementTensorSetupCuda(macro_count, std_cell_count, seed);
    torch::cuda::synchronize();
#ifdef PLACEMENT_CUDA_ENABLE_DEBUG_RENDER
    renderCudaTensorSetupDebug(setup);
#endif
}

}  // namespace

int main() {
    constexpr int64_t num_macros = 3;
    constexpr int64_t num_std_cells = 10;
    constexpr uint64_t seed = 42;

    torch::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    runCudaTensorSetupCheck(num_macros, num_std_cells, seed);

    return 0;
}
