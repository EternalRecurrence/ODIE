#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace odie {
struct Level {
    int resolution, points, capacity, dim, feature_offset;
    int grid_offset, map_offset, bitmap_offset, full, enabled;
};
__device__ inline void geometry(float x, float y, int r, int& cell, float* weights) {
    const float px = fmaf(float(r - 1), x, 0.5f), py = fmaf(float(r - 1), y, 0.5f);
    const int ix = int(floorf(px)), iy = int(floorf(py));
    const float fx = px - float(ix), fy = py - float(iy);
    cell = ix + iy * r;
    weights[0] = (1 - fx) * (1 - fy); weights[1] = fx * (1 - fy);
    weights[2] = (1 - fx) * fy; weights[3] = fx * fy;
}
__device__ inline bool visible(const Level& d, int cell, const uint8_t* bits) {
    return d.full || ((bits[d.bitmap_offset + (cell >> 3)] >> (cell & 7)) & 1);
}
__global__ void prepare_batch(float* xy, const uint32_t* permutation, uint32_t begin, int real, int padded,
                              int width, int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded) return;
    const uint32_t id = i < real ? (permutation ? permutation[begin + i] : begin + i) : 0;
    xy[2 * i] = __fdiv_rn(float(id % width), float(width - 1));
    xy[2 * i + 1] = __fdiv_rn(float(id / width), float(height - 1));
}
__global__ void query_grid(const float* xy, const float* grid, const Level* levels, const int* map,
                           const uint8_t* bitmap, __half* raw, uint8_t* visibility,
                           int real, int padded, int logical_width, int input_width, int n_levels, int active) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (i >= padded) return;
    const Level d = levels[l];
    int cell = 0; float w[4];
    if (i < real && d.enabled) geometry(xy[2 * i], xy[2 * i + 1], d.resolution, cell, w);
    const bool on = i < real && d.enabled && visible(d, cell, bitmap);
    visibility[i * n_levels + l] = on;
    for (int c = 0; c < d.dim; ++c) {
        __half value = __float2half_rn(0);
        if (on) for (int k = 0; k < 4; ++k) {
            const int node = (cell + (k & 1) + (k >> 1) * d.resolution) % d.points;
            const int row = d.full ? node : map[d.map_offset + node];
            value = __hfma(__float2half_rn(w[k]), __float2half_rn(grid[d.grid_offset + row * d.dim + c]), value);
        }
        raw[i * input_width + d.feature_offset + c] = value;
    }
    if (l == 0) for (int c = logical_width; c < input_width; ++c) raw[i * input_width + c] = __float2half_rn(1);
}
__global__ void gate_weight(const __half* raw, const __half* logits, const uint8_t* visibility,
                            const Level* levels, __half* weighted, int padded, int logical_width,
                            int input_width, int gate_width, int n_levels, int k_fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded) return;
    for (int l = 0; l < n_levels; ++l) {
        const Level d = levels[l];
        const float s = l < k_fixed ? 1.0f : 1.0f / (1.0f + expf(-__half2float(logits[i * gate_width + l])));
        const __half weight = __float2half_rn(visibility[i * n_levels + l] ? s : 0);
        for (int c = 0; c < d.dim; ++c) {
            const int j = i * input_width + d.feature_offset + c;
            weighted[j] = __hmul(raw[j], weight);
        }
    }
    for (int c = logical_width; c < input_width; ++c) weighted[i * input_width + c] = __float2half_rn(1);
}
__global__ void weight_backward(const __half* raw, const __half* logits, const __half* decoder_input_grad,
                                const uint8_t* visibility, const Level* levels, __half* direct_grad,
                                __half* gate_output_grad, int padded, int logical_width, int input_width,
                                int gate_width, int n_levels, int k_fixed, float local_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded) return;
    for (int l = 0; l < n_levels; ++l) {
        const Level d = levels[l];
        const float s = l < k_fixed ? 1.0f : 1.0f / (1.0f + expf(-__half2float(logits[i * gate_width + l])));
        const bool on = visibility[i * n_levels + l];
        const __half weight = __float2half_rn(on ? s : 0);
        __half dw = __float2half_rn(0);
        for (int c = 0; c < d.dim; ++c) {
            const int j = i * input_width + d.feature_offset + c;
            const __half dz = __float2half_rn(__half2float(decoder_input_grad[j]) / local_scale);
            direct_grad[j] = __hmul(dz, weight);
            dw = __hadd(dw, __hmul(dz, raw[j]));
        }
        const float da = on && l >= k_fixed ? __half2float(dw) * (1.0f - s) * s : 0;
        // Match cast-to-half and local binding scale as two rounding boundaries.
        gate_output_grad[i * gate_width + l] = __hmul(__float2half_rn(da), __float2half_rn(local_scale));
    }
    for (int l = n_levels; l < gate_width; ++l) gate_output_grad[i * gate_width + l] = __float2half_rn(0);
    for (int c = logical_width; c < input_width; ++c) direct_grad[i * input_width + c] = __float2half_rn(0);
}
__global__ void grid_backward(const float* xy, const __half* direct, const __half* indirect,
                              const Level* levels, const int* map, const uint8_t* bitmap, float* grad,
                              int real, int input_width, int active, float local_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    if (i >= real) return;
    const Level d = levels[l];
    if (!d.enabled) return;
    int cell; float w[4]; geometry(xy[2 * i], xy[2 * i + 1], d.resolution, cell, w);
    if (!visible(d, cell, bitmap)) return;
    for (int c = 0; c < d.dim; ++c) {
        const int j = i * input_width + d.feature_offset + c;
        const __half incoming = __hadd(direct[j], __float2half_rn(__half2float(indirect[j]) / local_scale));
        const float v = __half2float(incoming);
        if (v == 0) continue;
        for (int k = 0; k < 4; ++k) {
            const int node = (cell + (k & 1) + (k >> 1) * d.resolution) % d.points;
            const int row = d.full ? node : map[d.map_offset + node];
            atomicAdd(grad + d.grid_offset + row * d.dim + c, w[k] * v);
        }
    }
}
__global__ void loss_and_scores(const __half* prediction, const uint8_t* target, const uint32_t* permutation,
                                const float* xy, __half* output_grad, double* total_sse, double* scores,
                                uint32_t* counts, int score_resolution, int real, int padded,
                                int output_width, uint32_t begin, float external_scale, float local_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double error = 0;
    if (i < real) {
        const uint32_t id = permutation ? permutation[begin + i] : begin + i;
        for (int c = 0; c < 3; ++c) {
            const float delta = __half2float(prediction[i * output_width + c]) - __fdiv_rn(float(target[3 * id + c]), 255.0f);
            const float square = delta * delta;
            error += double(square);
            if (output_grad) output_grad[i * output_width + c] = __hmul(
                __float2half_rn((2.0f * delta / float(real * 3)) * external_scale), __float2half_rn(local_scale));
        }
        if (scores) {
            int cell; float w[4]; geometry(xy[2 * i], xy[2 * i + 1], score_resolution, cell, w);
            atomicAdd(scores + cell, error / 3.0);
            if (counts) atomicAdd(counts + cell, 1u);
        }
    }
    if (output_grad && i < padded) for (int c = i < real ? 3 : 0; c < output_width; ++c) output_grad[i * output_width + c] = __float2half_rn(0);
    __shared__ double sums[128];
    sums[threadIdx.x] = error; __syncthreads();
    for (int stride = 64; stride; stride >>= 1) { if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(total_sse, sums[0]);
}
__global__ void half_unscale(__half* values, uint32_t n, float scale) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = __float2half_rn(__half2float(values[i]) / scale);
}
__global__ void cast_float_half(const float* input, __half* output, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half_rn(input[i]);
}
} // namespace odie
