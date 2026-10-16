#include "arena/measurement/launch_floor.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace {
__global__ void arena_empty_kernel() {}
__global__ void arena_roundtrip_kernel(const float* in, float* out) { out[0] = in[0]; }
}

namespace arena {

float measure_launch_floor_ms() {
    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) != cudaSuccess) return 0.0f;
    if (cudaEventCreate(&stop) != cudaSuccess) { cudaEventDestroy(start); return 0.0f; }

    for (int i = 0; i < 200; i++) arena_empty_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    // Bracketed exactly like op_ms: event, launch, event, synchronize. The
    // minimum over several batch medians, because a single median of 100 moved
    // by 30% between processes (2.2 to 2.9 us) as clocks and the host
    // scheduler varied, and every small-shape score moved with it.
    float best = 0.0f;
    for (int batch = 0; batch < 8; batch++) {
        std::vector<float> t;
        for (int i = 0; i < 64; i++) {
            cudaEventRecord(start, nullptr);
            arena_empty_kernel<<<1, 32>>>();
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        const float med = t[t.size() / 2];
        if (batch == 0 || med < best) best = med;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (cudaGetLastError() != cudaSuccess) return 0.0f;
    return best;
}

float measure_roundtrip_floor_ms() {
    int l2 = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    const size_t scrub_bytes = std::max<size_t>(2 * (size_t)l2, 64u << 20);
    float *scrub = nullptr, *out = nullptr;
    if (cudaMalloc(&scrub, scrub_bytes) != cudaSuccess) return 0.0f;
    if (cudaMalloc(&out, 256) != cudaSuccess) { cudaFree(scrub); return 0.0f; }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // The input lives at the start of the scrub buffer and the memset that
    // evicts L2 runs before the start event, so each timed launch reads its
    // one value from DRAM, as a kernel on cold input would.
    float best = 0.0f;
    for (int batch = 0; batch < 4; batch++) {
        std::vector<float> t;
        for (int i = 0; i < 32; i++) {
            cudaMemsetAsync(scrub, i & 0xff, scrub_bytes, nullptr);
            cudaEventRecord(start, nullptr);
            arena_roundtrip_kernel<<<1, 1>>>(scrub + (scrub_bytes / sizeof(float)) / 2, out);
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        const float med = t[t.size() / 2];
        if (batch == 0 || med < best) best = med;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(scrub);
    cudaFree(out);
    if (cudaGetLastError() != cudaSuccess) return 0.0f;
    return best;
}

}
