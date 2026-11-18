#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

// R, points, plane offset, node word offset, cell bitmap offset, Fxy, output offset, direct.
constexpr int COLS = 8;

__device__ inline void geometry(float x, float y, int r, int& cell, float* weights) {
    const float px = fmaf(static_cast<float>(r - 1), x, 0.5f);
    const float py = fmaf(static_cast<float>(r - 1), y, 0.5f);
    const int ix = static_cast<int>(floorf(px)), iy = static_cast<int>(floorf(py));
    const float fx = px - static_cast<float>(ix), fy = py - static_cast<float>(iy);
    cell = ix + iy * r;
    weights[0] = (1.0f - fx) * (1.0f - fy);
    weights[1] = fx * (1.0f - fy);
    weights[2] = (1.0f - fx) * fy;
    weights[3] = fx * fy;
}

__device__ inline bool visible(const uint8_t* bitmap, const int* descriptor, int cell) {
    return (bitmap[descriptor[4] + (cell >> 3)] >> (cell & 7)) & 1;
}

__device__ inline int packed_row(const uint64_t* node_bitmap, const int* prefix,
                                 const int* d, int node) {
    if (d[7]) return node;
    const int word = d[3] + (node >> 6);
    const int bit = node & 63;
    const uint64_t mask = (uint64_t{1} << bit) - uint64_t{1};
    // Visible cells always have their complete four-corner node closure.
    return prefix[word] + __popcll(node_bitmap[word] & mask);
}

__global__ void query_kernel(const float* coordinates, const float* grid, const int* descriptors,
                             const uint64_t* node_bitmap, const int* prefix, const uint8_t* bitmap, __half* output,
                             bool* visibility, int n, int width, int levels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int level = blockIdx.y;
    const int* d = descriptors + level * COLS;
    int cell;
    float weights[4];
    geometry(coordinates[2 * i], coordinates[2 * i + 1], d[0], cell, weights);
    const bool on = visible(bitmap, d, cell);
    visibility[i * levels + level] = on;
    __half* result = output + i * width + d[6];
    for (int channel = 0; channel < d[5]; ++channel) {
        __half value = __float2half_rn(0.0f);
        if (on) {
            #pragma unroll
            for (int corner = 0; corner < 4; ++corner) {
                const int node = (cell + (corner & 1) + (corner >> 1) * d[0]) % d[1];
                const int row = packed_row(node_bitmap, prefix, d, node);
                value = __hfma(__float2half_rn(weights[corner]),
                              __float2half_rn(grid[d[2] + row * d[5] + channel]), value);
            }
        }
        result[channel] = value;
    }
}

__global__ void gradient_kernel(const float* coordinates, const __half* upstream,
                                const int* descriptors, const uint64_t* node_bitmap, const int* prefix,
                                const uint8_t* bitmap,
                                float* gradient, int n, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int* d = descriptors + blockIdx.y * COLS;
    int cell;
    float weights[4];
    geometry(coordinates[2 * i], coordinates[2 * i + 1], d[0], cell, weights);
    if (!visible(bitmap, d, cell)) return;
    const __half* incoming = upstream + i * width + d[6];
    for (int channel = 0; channel < d[5]; ++channel) {
        const float value = __half2float(incoming[channel]);
        if (value == 0.0f) continue;
        #pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const int node = (cell + (corner & 1) + (corner >> 1) * d[0]) % d[1];
            const int row = packed_row(node_bitmap, prefix, d, node);
            atomicAdd(gradient + d[2] + row * d[5] + channel, weights[corner] * value);
        }
    }
}

void check_inputs(const at::Tensor& coordinates, const at::Tensor& descriptors,
                  const at::Tensor& node_bitmap, const at::Tensor& prefix, const at::Tensor& bitmap) {
    TORCH_CHECK(coordinates.is_cuda() && coordinates.is_contiguous() && coordinates.scalar_type() == at::kFloat &&
                coordinates.dim() == 2 && coordinates.size(1) == 2, "coordinates must be CUDA contiguous FP32 Nx2");
    TORCH_CHECK(descriptors.is_cuda() && descriptors.is_contiguous() && descriptors.scalar_type() == at::kInt &&
                descriptors.dim() == 2 && (descriptors.size(0) >= 2 && descriptors.size(0) <= 32) && descriptors.size(1) == COLS,
                "descriptor must be CUDA contiguous int32 Lx8");
    TORCH_CHECK(node_bitmap.is_cuda() && node_bitmap.is_contiguous() && node_bitmap.scalar_type() == at::kUInt64 &&
                node_bitmap.dim() == 1, "invalid uint64 node bitmap");
    TORCH_CHECK(prefix.is_cuda() && prefix.is_contiguous() && prefix.scalar_type() == at::kInt && prefix.dim() == 1 &&
                prefix.numel() == node_bitmap.numel(), "invalid int32 node prefix");
    TORCH_CHECK(bitmap.is_cuda() && bitmap.is_contiguous() && bitmap.scalar_type() == at::kByte && bitmap.dim() == 1, "invalid bitmap");
    TORCH_CHECK(coordinates.device() == descriptors.device() && coordinates.device() == node_bitmap.device() &&
                coordinates.device() == prefix.device() &&
                coordinates.device() == bitmap.device(), "all tensors must share a CUDA device");
}

std::vector<at::Tensor> packed_forward_cuda(at::Tensor coordinates, at::Tensor grid,
    at::Tensor descriptors, at::Tensor node_bitmap, at::Tensor prefix, at::Tensor bitmap, int64_t width) {
    check_inputs(coordinates, descriptors, node_bitmap, prefix, bitmap);
    TORCH_CHECK((width >= 2 && width <= 128), "feature width must be within 2..128");
    TORCH_CHECK(grid.is_cuda() && grid.device() == coordinates.device() && grid.is_contiguous() &&
                grid.scalar_type() == at::kFloat && grid.dim() == 1, "grid must be flat CUDA FP32");
    const c10::cuda::CUDAGuard guard(coordinates.device());
    const int n = coordinates.size(0), levels = descriptors.size(0);
    auto output = at::empty({n, width}, coordinates.options().dtype(at::kHalf));
    auto visibility = at::empty({n, levels}, coordinates.options().dtype(at::kBool));
    if (n) {
        query_kernel<<<dim3((n + 127) / 128, levels), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            coordinates.data_ptr<float>(), grid.data_ptr<float>(), descriptors.data_ptr<int>(),
            node_bitmap.data_ptr<uint64_t>(), prefix.data_ptr<int>(), bitmap.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), visibility.data_ptr<bool>(), n, width, levels);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, visibility};
}

at::Tensor packed_backward_cuda(at::Tensor coordinates, at::Tensor upstream,
    at::Tensor descriptors, at::Tensor node_bitmap, at::Tensor prefix, at::Tensor bitmap, int64_t parameter_count) {
    check_inputs(coordinates, descriptors, node_bitmap, prefix, bitmap);
    TORCH_CHECK(upstream.is_cuda() && upstream.device() == coordinates.device() && upstream.is_contiguous() &&
                upstream.scalar_type() == at::kHalf && upstream.dim() == 2 && upstream.size(0) == coordinates.size(0) &&
                (upstream.size(1) >= 2 && upstream.size(1) <= 128), "upstream must be CUDA contiguous FP16 NxF");
    TORCH_CHECK(parameter_count >= 0, "parameter count must be nonnegative");
    const c10::cuda::CUDAGuard guard(coordinates.device());
    const int n = coordinates.size(0), levels = descriptors.size(0), width = upstream.size(1);
    auto gradient = at::zeros({parameter_count}, coordinates.options());
    if (n) {
        gradient_kernel<<<dim3((n + 127) / 128, levels), 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            coordinates.data_ptr<float>(), reinterpret_cast<const __half*>(upstream.data_ptr<at::Half>()),
            descriptors.data_ptr<int>(), node_bitmap.data_ptr<uint64_t>(), prefix.data_ptr<int>(), bitmap.data_ptr<uint8_t>(),
            gradient.data_ptr<float>(), n, width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return gradient;
}
