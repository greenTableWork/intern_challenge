#include "placement_pytorch_cuda_tensor_setup.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

namespace {

namespace cuda_setup = placement_cuda;

constexpr int64_t kCellFeatureCount =
    static_cast<int64_t>(cuda_setup::CellFeatureIdx::Count);

__device__ __forceinline__ int64_t featureIndex(cuda_setup::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

__global__ void placementTensorSetupKernel(
    float* macro_areas,
    int64_t* std_area_indices,
    float* std_cell_areas,
    float* areas,
    float* cell_widths,
    float* cell_heights,
    float* cell_features,
    int64_t macro_count,
    int64_t std_cell_count,
    float min_macro_area,
    float max_macro_area,
    float standard_cell_height,
    uint64_t seed) {
    const int64_t total_cells = macro_count + std_cell_count;
    for (int64_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
         cell_idx < total_cells;
         cell_idx += blockDim.x * gridDim.x) {
        float area = 0.0F;
        float width = 0.0F;
        float height = 0.0F;
        // rng used by pytorch
        curandStatePhilox4_32_10_t rng;
        curand_init(
            static_cast<unsigned long long>(seed),
            static_cast<unsigned long long>(cell_idx),
            0,
            &rng);

        if (cell_idx < macro_count) {
            area =
                curand_uniform(&rng) * (max_macro_area - min_macro_area) +
                min_macro_area;
            width = sqrtf(area);
            height = width;
            macro_areas[cell_idx] = area;
        } else {
            const int64_t std_idx = cell_idx - macro_count;
            const int64_t area_idx = static_cast<int64_t>(curand(&rng) % 3U);
            area = static_cast<float>(area_idx + 1);
            width = area / standard_cell_height;
            height = standard_cell_height;
            std_area_indices[std_idx] = area_idx;
            std_cell_areas[std_idx] = area;
        }

        areas[cell_idx] = area;
        cell_widths[cell_idx] = width;
        cell_heights[cell_idx] = height;

        const int64_t feature_offset = cell_idx * kCellFeatureCount;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Area)] =
            area;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::NumPins)] =
            0.0F;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::X)] =
            0.0F;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Y)] =
            0.0F;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Width)] =
            width;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Height)] =
            height;
    }
}

void checkCudaTensor(
    const at::Tensor& tensor,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> sizes) {
    TORCH_CHECK(tensor.is_cuda(), "expected CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), "expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, "unexpected tensor dtype");
    TORCH_CHECK(tensor.sizes() == sizes, "unexpected tensor shape");
}

}  // namespace

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
    uint64_t seed) {
    const int64_t total_cells = macro_count + std_cell_count;
    checkCudaTensor(macro_areas, at::kFloat, {macro_count});
    checkCudaTensor(std_area_indices, at::kLong, {std_cell_count});
    checkCudaTensor(std_cell_areas, at::kFloat, {std_cell_count});
    checkCudaTensor(areas, at::kFloat, {total_cells});
    checkCudaTensor(cell_widths, at::kFloat, {total_cells});
    checkCudaTensor(cell_heights, at::kFloat, {total_cells});
    checkCudaTensor(cell_features, at::kFloat, {total_cells, kCellFeatureCount});

    if (total_cells == 0) {
        return;
    }

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    placementTensorSetupKernel<<<
        blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        macro_areas.data_ptr<float>(),
        std_area_indices.data_ptr<int64_t>(),
        std_cell_areas.data_ptr<float>(),
        areas.data_ptr<float>(),
        cell_widths.data_ptr<float>(),
        cell_heights.data_ptr<float>(),
        cell_features.data_ptr<float>(),
        macro_count,
        std_cell_count,
        static_cast<float>(min_macro_area),
        static_cast<float>(max_macro_area),
        static_cast<float>(standard_cell_height),
        seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
