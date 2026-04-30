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
constexpr int64_t kMaxConnectionsPerPin = 3;
constexpr int64_t kCellFeatureCount =
    static_cast<int64_t>(placement_cuda::CellFeatureIdx::Count);
constexpr int64_t kPinFeatureCount =
    static_cast<int64_t>(placement_cuda::PinFeatureIdx::Count);

int64_t featureIndex(placement_cuda::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int64_t featureIndex(placement_cuda::PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

struct PlacementTensorSetup {
    torch::Tensor macro_areas;
    torch::Tensor std_area_indices;
    torch::Tensor std_cell_areas;
    torch::Tensor areas;
    torch::Tensor num_pins_per_cell;
    torch::Tensor pin_offsets;
    torch::Tensor pin_features;
    torch::Tensor edge_list;
    torch::Tensor edge_count;
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

int64_t readTotalPinsHost(const torch::Tensor& pin_offsets, int64_t total_cells) {
    // Tensor allocation still needs host shape metadata; later CUDA milestones
    // can replace this with a capacity-based allocation strategy.
    return pin_offsets.index({total_cells}).item<int64_t>();
}

int64_t readEdgeCountHost(const torch::Tensor& edge_count) {
    // The first CUDA edge-list pass uses a bounded capacity tensor and exposes
    // the valid prefix to match the existing [E, 2] edge_list schema.
    return edge_count.item<int64_t>();
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
        torch::Tensor(),
        torch::Tensor(),
        torch::empty({1}, long_options),
        torch::empty({total_cells}, float_options),
        torch::empty({total_cells}, float_options),
        torch::empty({total_cells, kCellFeatureCount}, float_options),
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
    const int64_t total_pins = readTotalPinsHost(setup.pin_offsets, total_cells);
    setup.pin_features = torch::empty({total_pins, kPinFeatureCount}, float_options);
    fillPinFeaturesCuda(setup.cell_features, setup.pin_offsets, setup.pin_features, seed);
    const int64_t max_edges = total_pins * kMaxConnectionsPerPin;
    auto edge_list_capacity = torch::empty({max_edges, 2}, long_options);
    fillEdgeListCuda(edge_list_capacity, setup.edge_count, total_pins, seed);
    const int64_t total_edges = readEdgeCountHost(setup.edge_count);
    TORCH_CHECK(total_edges <= max_edges, "CUDA edge count exceeded capacity");
    setup.edge_list = edge_list_capacity.narrow(0, 0, total_edges);

    checkTensor(setup.macro_areas, torch::kFloat32, {macro_count});
    checkTensor(setup.std_area_indices, torch::kInt64, {std_cell_count});
    checkTensor(setup.std_cell_areas, torch::kFloat32, {std_cell_count});
    checkTensor(setup.areas, torch::kFloat32, {total_cells});
    checkTensor(setup.num_pins_per_cell, torch::kInt64, {total_cells});
    checkTensor(setup.pin_offsets, torch::kInt64, {total_cells + 1});
    checkTensor(setup.pin_features, torch::kFloat32, {total_pins, kPinFeatureCount});
    checkTensor(setup.edge_list, torch::kInt64, {total_edges, 2});
    checkTensor(setup.edge_count, torch::kInt64, {1});
    checkTensor(setup.cell_widths, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_heights, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_features, torch::kFloat32, {total_cells, kCellFeatureCount});

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
    const auto pin_features = setup.pin_features.detach().cpu().contiguous();
    const auto edge_list = setup.edge_list.detach().cpu().contiguous();
    const auto edge_count = setup.edge_count.detach().cpu().contiguous();
    const auto cell_features = setup.cell_features.detach().cpu().contiguous();

    const auto macro_areas_a = macro_areas.accessor<float, 1>();
    const auto std_area_indices_a = std_area_indices.accessor<int64_t, 1>();
    const auto std_cell_areas_a = std_cell_areas.accessor<float, 1>();
    const auto num_pins_a = num_pins_per_cell.accessor<int64_t, 1>();
    const auto pin_offsets_a = pin_offsets.accessor<int64_t, 1>();
    const auto pins_a = pin_features.accessor<float, 2>();
    const auto edges_a = edge_list.accessor<int64_t, 2>();
    const auto edge_count_a = edge_count.accessor<int64_t, 1>();
    const auto cells_a = cell_features.accessor<float, 2>();
    const int64_t total_pins = pin_features.size(0);
    const int64_t total_edges = edge_list.size(0);

    std::cout << "\nplacement_cuda debug dump\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  macros: " << macro_count << "\n";
    std::cout << "  standard cells: " << std_cell_count << "\n";
    std::cout << "  total cells: " << total_cells << "\n";
    std::cout << "  total pins: " << pin_offsets_a[total_cells] << "\n";
    std::cout << "  total edges: " << total_edges << "\n";
    std::cout << "  edge_count tensor: " << edge_count_a[0] << "\n";
    std::cout << "  pin_features shape: [" << total_pins << ", "
              << kPinFeatureCount << "]\n\n";
    std::cout << "  edge_list shape: [" << total_edges << ", 2]\n\n";

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

    std::cout << "\npins:\n";
    std::cout << "  " << std::setw(4) << "idx" << "  " << std::setw(4)
              << "cell" << "  " << std::setw(8) << "pin_x" << "  "
              << std::setw(8) << "pin_y" << "  " << std::setw(8) << "x"
              << "  " << std::setw(8) << "y" << "  " << std::setw(6)
              << "width" << "  " << std::setw(6) << "height" << "\n";
    for (int64_t pin_idx = 0; pin_idx < total_pins; ++pin_idx) {
        std::cout << "  " << std::setw(4) << pin_idx << "  " << std::setw(4)
                  << static_cast<int64_t>(
                         pins_a[pin_idx][featureIndex(
                             placement_cuda::PinFeatureIdx::CellIdx)])
                  << "  " << std::setw(8)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::PinX)]
                  << "  " << std::setw(8)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::PinY)]
                  << "  " << std::setw(8)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::X)]
                  << "  " << std::setw(8)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::Y)]
                  << "  " << std::setw(6)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::Width)]
                  << "  " << std::setw(6)
                  << pins_a[pin_idx][featureIndex(placement_cuda::PinFeatureIdx::Height)]
                  << "\n";
    }

    std::cout << "\nedges:\n";
    std::cout << "  " << std::setw(4) << "idx" << "  " << std::setw(6)
              << "src" << "  " << std::setw(6) << "tgt" << "\n";
    for (int64_t edge_idx = 0; edge_idx < total_edges; ++edge_idx) {
        std::cout << "  " << std::setw(4) << edge_idx << "  " << std::setw(6)
                  << edges_a[edge_idx][0] << "  " << std::setw(6)
                  << edges_a[edge_idx][1] << "\n";
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
