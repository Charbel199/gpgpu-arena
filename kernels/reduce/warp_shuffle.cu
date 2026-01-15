// SOL2: Warp Shuffle Reduction
// Uses __shfl_down_sync for intra-warp reduction, then block atomic
#include <cuda/atomic>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float block_sum;
    
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads();
    
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % 32;
    
    float val = (id < n) ? input[id] : 0.0f;
    float warp_sum = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_block> block_sum_ref(block_sum);
        block_sum_ref.fetch_add(warp_sum);
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
        output_ref.fetch_add(block_sum);
    }
}
