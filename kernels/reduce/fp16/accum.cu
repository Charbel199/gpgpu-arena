// Two reductions over identical fp16 input, differing only in the type of the
// per-thread accumulator. The grid is deliberately small so each thread sums a
// long run of values, which is where the accumulator type starts to matter.
//
// Half has 11 mantissa bits, so its integers stop being exact above 2048: add
// 1.0 to 2048.0 in half and nothing happens. fp32 has 24 bits and keeps going.
#include <cuda_fp16.h>
#include <cuda/atomic>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, off);
    return val;
}

// Accumulates each thread's chunk in half, then widens for the block reduction.
extern "C" __global__ void reduce_sum_fp16_accum(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_sums[33];

    __half acc = __float2half(0.0f);
    const int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        acc = __hadd(acc, input[i]);
    }

    float sum = warp_reduce_sum(__half2float(acc));
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            cuda::atomic_ref<float, cuda::thread_scope_device> ref(*output);
            ref.fetch_add(sum);
        }
    }
}

// Identical, except the per-thread accumulator is float.
extern "C" __global__ void reduce_sum_fp32_accum(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_sums[33];

    float acc = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        acc += __half2float(input[i]);
    }

    float sum = warp_reduce_sum(acc);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            cuda::atomic_ref<float, cuda::thread_scope_device> ref(*output);
            ref.fetch_add(sum);
        }
    }
}

// Same clean float accumulation as reduce_sum_fp32_accum, but the result has
// to land in a half. That is not just a narrower store: with one output slot
// and many blocks, the cross-block combine becomes a half atomic, so the
// global accumulation inherits half's precision no matter how carefully each
// block summed its own chunk.
extern "C" __global__ void reduce_sum_fp16_out(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int n
) {
    __shared__ float warp_sums[33];

    float acc = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        acc += __half2float(input[i]);
    }

    float sum = warp_reduce_sum(acc);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(output, __float2half(sum));
    }
}
