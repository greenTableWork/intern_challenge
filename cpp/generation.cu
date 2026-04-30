#include "placement/generation_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstdint>

namespace {

namespace cuda_setup = placement_cuda;

constexpr int64_t kCellFeatureCount =
    static_cast<int64_t>(cuda_setup::CellFeatureIdx::Count);
constexpr int64_t kPinFeatureCount =
    static_cast<int64_t>(cuda_setup::PinFeatureIdx::Count);
constexpr float kTwoPi = 6.2831853071795864769F;
constexpr float kPinSize = 0.1F;
constexpr int64_t kMaxConnectionsPerPin = 3;
constexpr uint64_t kPositionSeedOffset = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kPinSeedOffset = 0xD1B54A32D192ED03ULL;
constexpr uint64_t kEdgeSeedOffset = 0x94D049BB133111EBULL;

__device__ __forceinline__ int64_t featureIndex(cuda_setup::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

__device__ __forceinline__ int64_t featureIndex(cuda_setup::PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

__global__ void placementTensorSetupKernel(
    float* macro_areas,
    int64_t* std_area_indices,
    float* std_cell_areas,
    float* areas,
    int64_t* num_pins_per_cell,
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
        int64_t num_pins = 0;
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
            const auto sqrt_area = static_cast<int64_t>(sqrtf(area));
            num_pins =
                sqrt_area +
                static_cast<int64_t>(
                    curand(&rng) % static_cast<unsigned int>(sqrt_area + 1));
        } else {
            const int64_t std_idx = cell_idx - macro_count;
            const int64_t area_idx = static_cast<int64_t>(curand(&rng) % 3U);
            area = static_cast<float>(area_idx + 1);
            width = area / standard_cell_height;
            height = standard_cell_height;
            std_area_indices[std_idx] = area_idx;
            std_cell_areas[std_idx] = area;
            num_pins = 3 + static_cast<int64_t>(curand(&rng) % 4U);
        }

        areas[cell_idx] = area;
        num_pins_per_cell[cell_idx] = num_pins;
        cell_widths[cell_idx] = width;
        cell_heights[cell_idx] = height;

        const int64_t feature_offset = cell_idx * kCellFeatureCount;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Area)] =
            area;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::NumPins)] =
            static_cast<float>(num_pins);
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

__global__ void sumCellAreasKernel(
    const float* cell_features,
    float* total_area,
    int64_t total_cells) {
    extern __shared__ float block_area_sums[];

    float sum = 0.0F;
    for (int64_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
         cell_idx < total_cells;
         cell_idx += blockDim.x * gridDim.x) {
        const int64_t feature_offset = cell_idx * kCellFeatureCount;
        sum += cell_features[
            feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Area)];
    }

    block_area_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            block_area_sums[threadIdx.x] +=
                block_area_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total_area, block_area_sums[0]);
    }
}

__global__ void initializeCellPositionsKernel(
    float* cell_features,
    const float* total_area,
    int64_t total_cells,
    float spread_scale,
    uint64_t seed) {
    const float spread_radius = fmaxf(sqrtf(total_area[0]) * spread_scale, 1.0F);
    for (int64_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
         cell_idx < total_cells;
         cell_idx += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t rng;
        curand_init(
            static_cast<unsigned long long>(seed + kPositionSeedOffset),
            static_cast<unsigned long long>(cell_idx),
            0,
            &rng);

        const float angle = curand_uniform(&rng) * kTwoPi;
        const float radius = curand_uniform(&rng) * spread_radius;
        const int64_t feature_offset = cell_idx * kCellFeatureCount;
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::X)] =
            radius * cosf(angle);
        cell_features[feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Y)] =
            radius * sinf(angle);
    }
}

__global__ void appendTotalPinCountKernel(
    const int64_t* num_pins_per_cell,
    int64_t* pin_offsets,
    int64_t total_cells) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    pin_offsets[total_cells] = total_cells == 0
        ? 0
        : pin_offsets[total_cells - 1] + num_pins_per_cell[total_cells - 1];
}

__global__ void fillPinFeaturesKernel(
    const float* cell_features,
    const int64_t* pin_offsets,
    float* pin_features,
    int64_t total_cells,
    uint64_t seed) {
    constexpr float margin = kPinSize / 2.0F;
    for (int64_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
         cell_idx < total_cells;
         cell_idx += blockDim.x * gridDim.x) {
        const int64_t pin_begin = pin_offsets[cell_idx];
        const int64_t pin_end = pin_offsets[cell_idx + 1];
        const int64_t cell_feature_offset = cell_idx * kCellFeatureCount;
        const float cell_width =
            cell_features[
                cell_feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Width)];
        const float cell_height =
            cell_features[
                cell_feature_offset + featureIndex(cuda_setup::CellFeatureIdx::Height)];
        const bool can_place_randomly =
            cell_width > 2.0F * margin && cell_height > 2.0F * margin;

        for (int64_t pin_idx = pin_begin; pin_idx < pin_end; ++pin_idx) {
            float pin_x = cell_width / 2.0F;
            float pin_y = cell_height / 2.0F;
            if (can_place_randomly) {
                curandStatePhilox4_32_10_t rng;
                curand_init(
                    static_cast<unsigned long long>(seed + kPinSeedOffset),
                    static_cast<unsigned long long>(pin_idx),
                    0,
                    &rng);
                pin_x = curand_uniform(&rng) * (cell_width - 2.0F * margin) + margin;
                pin_y =
                    curand_uniform(&rng) * (cell_height - 2.0F * margin) + margin;
            }

            const int64_t pin_feature_offset = pin_idx * kPinFeatureCount;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::CellIdx)] =
                static_cast<float>(cell_idx);
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::PinX)] =
                pin_x;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::PinY)] =
                pin_y;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::X)] =
                pin_x;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::Y)] =
                pin_y;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::Width)] =
                kPinSize;
            pin_features[
                pin_feature_offset + featureIndex(cuda_setup::PinFeatureIdx::Height)] =
                kPinSize;
        }
    }
}

__global__ void fillEdgeListKernel(
    int64_t* edge_list_capacity,
    int64_t* edge_count,
    const int64_t* pin_offsets,
    int64_t total_cells,
    int64_t max_edges,
    uint64_t seed) {
    const int64_t total_pins = pin_offsets[total_cells];
    if (total_pins < 2) {
        return;
    }

    for (int64_t pin_idx = blockIdx.x * blockDim.x + threadIdx.x;
         pin_idx < total_pins;
         pin_idx += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t rng;
        curand_init(
            static_cast<unsigned long long>(seed + kEdgeSeedOffset),
            static_cast<unsigned long long>(pin_idx),
            0,
            &rng);

        const int64_t num_connections =
            1 + static_cast<int64_t>(curand(&rng) % kMaxConnectionsPerPin);
        for (int64_t connection = 0; connection < num_connections; ++connection) {
            const int64_t other_pin =
                static_cast<int64_t>(
                    static_cast<uint64_t>(curand(&rng)) %
                    static_cast<uint64_t>(total_pins));
            if (other_pin == pin_idx) {
                continue;
            }

            const unsigned long long edge_idx = atomicAdd(
                reinterpret_cast<unsigned long long*>(edge_count),
                1ULL);
            if (edge_idx >= static_cast<unsigned long long>(max_edges)) {
                continue;
            }

            const int64_t src_pin = pin_idx < other_pin ? pin_idx : other_pin;
            const int64_t tgt_pin = pin_idx < other_pin ? other_pin : pin_idx;
            const int64_t edge_offset = static_cast<int64_t>(edge_idx) * 2;
            edge_list_capacity[edge_offset] = src_pin;
            edge_list_capacity[edge_offset + 1] = tgt_pin;
        }
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

namespace placement_cuda {

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
    uint64_t seed) {
    const int64_t total_cells = macro_count + std_cell_count;
    checkCudaTensor(macro_areas, at::kFloat, {macro_count});
    checkCudaTensor(std_area_indices, at::kLong, {std_cell_count});
    checkCudaTensor(std_cell_areas, at::kFloat, {std_cell_count});
    checkCudaTensor(areas, at::kFloat, {total_cells});
    checkCudaTensor(num_pins_per_cell, at::kLong, {total_cells});
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
        num_pins_per_cell.data_ptr<int64_t>(),
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

void computePinOffsetsCuda(
    const at::Tensor& num_pins_per_cell,
    const at::Tensor& pin_offsets) {
    const int64_t total_cells = num_pins_per_cell.size(0);
    checkCudaTensor(num_pins_per_cell, at::kLong, {total_cells});
    checkCudaTensor(pin_offsets, at::kLong, {total_cells + 1});

    auto stream = at::cuda::getCurrentCUDAStream();
    if (total_cells > 0) {
        auto counts_begin =
            thrust::device_pointer_cast(num_pins_per_cell.data_ptr<int64_t>());
        auto offsets_begin = thrust::device_pointer_cast(pin_offsets.data_ptr<int64_t>());
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream.stream()),
            counts_begin,
            counts_begin + total_cells,
            offsets_begin);
    }

    appendTotalPinCountKernel<<<1, 1, 0, stream>>>(
        num_pins_per_cell.data_ptr<int64_t>(),
        pin_offsets.data_ptr<int64_t>(),
        total_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fillPinFeaturesCuda(
    const at::Tensor& cell_features,
    const at::Tensor& pin_offsets,
    const at::Tensor& pin_features,
    uint64_t seed) {
    const int64_t total_cells = cell_features.size(0);
    const int64_t total_pins = pin_features.size(0);
    checkCudaTensor(cell_features, at::kFloat, {total_cells, kCellFeatureCount});
    checkCudaTensor(pin_offsets, at::kLong, {total_cells + 1});
    checkCudaTensor(pin_features, at::kFloat, {total_pins, kPinFeatureCount});

    if (total_cells == 0 || total_pins == 0) {
        return;
    }

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    fillPinFeaturesKernel<<<
        blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        cell_features.data_ptr<float>(),
        pin_offsets.data_ptr<int64_t>(),
        pin_features.data_ptr<float>(),
        total_cells,
        seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fillEdgeListCuda(
    const at::Tensor& edge_list_capacity,
    const at::Tensor& edge_count,
    const at::Tensor& pin_offsets,
    int64_t total_cells,
    int64_t max_pin_capacity,
    uint64_t seed) {
    TORCH_CHECK(total_cells >= 0, "total cells must be non-negative");
    TORCH_CHECK(max_pin_capacity >= 0, "max pin capacity must be non-negative");
    const int64_t max_edges = edge_list_capacity.size(0);
    checkCudaTensor(edge_list_capacity, at::kLong, {max_edges, 2});
    checkCudaTensor(edge_count, at::kLong, {1});
    checkCudaTensor(pin_offsets, at::kLong, {total_cells + 1});
    TORCH_CHECK(
        max_edges == max_pin_capacity * kMaxConnectionsPerPin,
        "edge capacity must be max_pin_capacity * max connections per pin");

    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        edge_count.data_ptr<int64_t>(),
        0,
        sizeof(int64_t),
        stream));
    if (max_pin_capacity < 2 || max_edges == 0) {
        return;
    }

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>(
            (max_pin_capacity + threads_per_block - 1) / threads_per_block);
    fillEdgeListKernel<<<
        blocks,
        threads_per_block,
        0,
        stream>>>(
        edge_list_capacity.data_ptr<int64_t>(),
        edge_count.data_ptr<int64_t>(),
        pin_offsets.data_ptr<int64_t>(),
        total_cells,
        max_edges,
        seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void initializeCellPositionsCuda(
    const at::Tensor& cell_features,
    double spread_scale,
    uint64_t seed) {
    TORCH_CHECK(spread_scale >= 0.0, "spread scale must be non-negative");
    checkCudaTensor(
        cell_features,
        at::kFloat,
        {cell_features.size(0), kCellFeatureCount});

    const int64_t total_cells = cell_features.size(0);
    if (total_cells == 0) {
        return;
    }

    const auto total_area = at::empty({1}, cell_features.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(total_area.data_ptr<float>(), 0, sizeof(float), stream));

    constexpr int threads_per_block = 256;
    const int blocks =
        static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    sumCellAreasKernel<<<
        blocks,
        threads_per_block,
        threads_per_block * sizeof(float),
        stream>>>(
        cell_features.data_ptr<float>(),
        total_area.data_ptr<float>(),
        total_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    initializeCellPositionsKernel<<<
        blocks,
        threads_per_block,
        0,
        stream>>>(
        cell_features.data_ptr<float>(),
        total_area.data_ptr<float>(),
        total_cells,
        static_cast<float>(spread_scale),
        seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace placement_cuda
