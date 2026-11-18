// Deterministic GPU residual scoring with CPU raster-order accumulation.
// Each thread owns one cell, so no atomics or inter-cell synchronization.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

__global__ void accumulate(const float* prediction, const void* target,
    int64_t begin, int64_t end, int width, int resolution, int first_y,
    int active_cells, const int* xb, const int* yb, double* totals, bool rgb8) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= active_cells) return;
    int cx = t % resolution, cy = first_y + t / resolution;
    int64_t cell = int64_t(cy) * resolution + cx;
    int y0 = max(yb[cy], int(begin / width));
    int y1 = min(yb[cy + 1], int((end - 1) / width) + 1);
    double sum = totals[cell];
    for (int y = y0; y < y1; ++y) {
        int64_t row = int64_t(y) * width;
        int x0 = int(max(int64_t(xb[cx]), begin - row));
        int x1 = int(min(int64_t(xb[cx + 1]), end - row));
        for (int x = x0; x < x1; ++x) {
            int64_t p = (row + x - begin) * 3;
            // Match the CPU reference's FP32 division, not CUDA's
            // scalar-reciprocal shortcut. Float32 target inputs are unchanged.
            const auto* bytes = static_cast<const uint8_t*>(target);
            const auto* fp32 = static_cast<const float*>(target);
            float t0 = rgb8 ? __fdiv_rn(float(bytes[p]), 255.f) : fp32[p];
            float t1 = rgb8 ? __fdiv_rn(float(bytes[p + 1]), 255.f) : fp32[p + 1];
            float t2 = rgb8 ? __fdiv_rn(float(bytes[p + 2]), 255.f) : fp32[p + 2];
            float d0 = __fsub_rn(prediction[p], t0);
            float d1 = __fsub_rn(prediction[p + 1], t1);
            float d2 = __fsub_rn(prediction[p + 2], t2);
            double rgb = __dadd_rn(double(__fmul_rn(d0, d0)), double(__fmul_rn(d1, d1)));
            rgb = __dadd_rn(rgb, double(__fmul_rn(d2, d2)));
            sum = __dadd_rn(sum, __ddiv_rn(rgb, 3.0));
        }
    }
    totals[cell] = sum;
}

API int score_chunk(const float* prediction, const void* target,
    int64_t begin, int64_t end, int width, int resolution, int first_y,
    int active_cells, const int* xb, const int* yb, double* totals, bool rgb8, uint64_t stream) {
    accumulate<<<(active_cells + 255) / 256, 256, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(prediction, target, begin, end,
        width, resolution, first_y, active_cells, xb, yb, totals, rgb8);
    return int(cudaGetLastError());
}
