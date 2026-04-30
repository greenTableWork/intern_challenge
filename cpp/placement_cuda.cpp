#include "placement_cuda_tensor_setup.h"

#include "placement/visualization.h"

#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace {

#ifndef PLACEMENT_REPO_ROOT
#define PLACEMENT_REPO_ROOT "."
#endif

constexpr double kMinMacroArea = 100.0;
constexpr double kMaxMacroArea = 10000.0;
constexpr double kStandardCellHeight = 1.0;
constexpr double kInitialSpreadScale = 0.6;

int64_t featureIndex(placement_cuda::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

struct PlacementTensorSetup {
    torch::Tensor macro_areas;
    torch::Tensor std_area_indices;
    torch::Tensor std_cell_areas;
    torch::Tensor areas;
    torch::Tensor num_pins_per_cell;
    torch::Tensor pin_offsets;
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
        torch::empty({total_cells + 1}, long_options),
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
    computePinOffsetsCuda(setup.num_pins_per_cell, setup.pin_offsets);

    checkTensor(setup.macro_areas, torch::kFloat32, {macro_count});
    checkTensor(setup.std_area_indices, torch::kInt64, {std_cell_count});
    checkTensor(setup.std_cell_areas, torch::kFloat32, {std_cell_count});
    checkTensor(setup.areas, torch::kFloat32, {total_cells});
    checkTensor(setup.num_pins_per_cell, torch::kInt64, {total_cells});
    checkTensor(setup.pin_offsets, torch::kInt64, {total_cells + 1});
    checkTensor(setup.cell_widths, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_heights, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_features, torch::kFloat32, {total_cells, 6});

    return setup;
}

#ifdef PLACEMENT_CUDA_DEBUG_BUILD
void printCudaTensorSetupDebug(
    const PlacementTensorSetup& setup,
    int64_t macro_count,
    int64_t std_cell_count,
    uint64_t seed) {
    const int64_t total_cells = macro_count + std_cell_count;
    const auto macro_areas = setup.macro_areas.detach().cpu().contiguous();
    const auto std_area_indices =
        setup.std_area_indices.detach().cpu().contiguous();
    const auto std_cell_areas =
        setup.std_cell_areas.detach().cpu().contiguous();
    const auto num_pins_per_cell =
        setup.num_pins_per_cell.detach().cpu().contiguous();
    const auto pin_offsets = setup.pin_offsets.detach().cpu().contiguous();
    const auto cell_features = setup.cell_features.detach().cpu().contiguous();

    const auto macro_areas_a = macro_areas.accessor<float, 1>();
    const auto std_area_indices_a = std_area_indices.accessor<int64_t, 1>();
    const auto std_cell_areas_a = std_cell_areas.accessor<float, 1>();
    const auto num_pins_a = num_pins_per_cell.accessor<int64_t, 1>();
    const auto pin_offsets_a = pin_offsets.accessor<int64_t, 1>();
    const auto cells_a = cell_features.accessor<float, 2>();

    std::cout << "\nplacement_cuda debug dump\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  macros: " << macro_count << "\n";
    std::cout << "  standard cells: " << std_cell_count << "\n";
    std::cout << "  total cells: " << total_cells << "\n";
    std::cout << "  total pins: " << pin_offsets_a[total_cells] << "\n\n";

    std::cout << "macro_areas:";
    for (int64_t index = 0; index < macro_count; ++index) {
        std::cout << " " << macro_areas_a[index];
    }
    std::cout << "\n";

    std::cout << "std_area_indices:";
    for (int64_t index = 0; index < std_cell_count; ++index) {
        std::cout << " " << std_area_indices_a[index];
    }
    std::cout << "\n";

    std::cout << "std_cell_areas:";
    for (int64_t index = 0; index < std_cell_count; ++index) {
        std::cout << " " << std_cell_areas_a[index];
    }
    std::cout << "\n";

    std::cout << "num_pins_per_cell:";
    for (int64_t index = 0; index < total_cells; ++index) {
        std::cout << " " << num_pins_a[index];
    }
    std::cout << "\n";

    std::cout << "pin_offsets:";
    for (int64_t index = 0; index <= total_cells; ++index) {
        std::cout << " " << pin_offsets_a[index];
    }
    std::cout << "\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "cells:\n";
    std::cout << "  " << std::setw(4) << "idx" << "  " << std::setw(5) << "type"
              << "  " << std::setw(10) << "area" << "  " << std::setw(8)
              << "width" << "  " << std::setw(8) << "height" << "  "
              << std::setw(8) << "x" << "  " << std::setw(8) << "y" << "  "
              << std::setw(4) << "pins" << "  "
              << "pin_range\n";
    for (int64_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        const char* cell_type = cell_idx < macro_count ? "macro" : "std";
        const int64_t pin_begin = pin_offsets_a[cell_idx];
        const int64_t pin_end = pin_offsets_a[cell_idx + 1];
        std::cout << "  " << std::setw(4) << cell_idx << "  " << std::setw(5)
                  << cell_type << "  " << std::setw(10)
                  << cells_a[cell_idx][featureIndex(placement_cuda::CellFeatureIdx::Area)]
                  << "  " << std::setw(8)
                  << cells_a[cell_idx][featureIndex(placement_cuda::CellFeatureIdx::Width)]
                  << "  " << std::setw(8)
                  << cells_a[cell_idx][featureIndex(placement_cuda::CellFeatureIdx::Height)]
                  << "  " << std::setw(8)
                  << cells_a[cell_idx][featureIndex(placement_cuda::CellFeatureIdx::X)]
                  << "  " << std::setw(8)
                  << cells_a[cell_idx][featureIndex(placement_cuda::CellFeatureIdx::Y)]
                  << "  " << std::setw(4) << num_pins_a[cell_idx] << "  ["
                  << pin_begin << ", " << pin_end << ")\n";
    }
    std::cout << std::defaultfloat << "\n";
}
#endif

#ifdef PLACEMENT_CUDA_ENABLE_DEBUG_RENDER
std::filesystem::path debugRenderOutputPath() {
    return std::filesystem::path(PLACEMENT_REPO_ROOT) /
           "placement_cuda_tensor_setup_debug.png";
}

void renderCudaTensorSetupDebug(const torch::Tensor& initialized_cell_features) {
    const std::filesystem::path output_path = debugRenderOutputPath();
    placement::plotPlacementState(
        initialized_cell_features,
        output_path,
        "Initial Placement");
    std::cout << "Rendered CUDA tensor setup to: " << output_path << "\n";
}
#endif

void runCudaTensorSetupCheck(
    int64_t macro_count,
    int64_t std_cell_count,
    uint64_t seed) {
    PlacementTensorSetup setup =
        buildPlacementTensorSetupCuda(macro_count, std_cell_count, seed);
    initializeCellPositionsCuda(setup.cell_features, kInitialSpreadScale, seed);
    torch::cuda::synchronize();
#ifdef PLACEMENT_CUDA_DEBUG_BUILD
    printCudaTensorSetupDebug(setup, macro_count, std_cell_count, seed);
#endif
#ifdef PLACEMENT_CUDA_ENABLE_DEBUG_RENDER
    renderCudaTensorSetupDebug(setup.cell_features);
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
