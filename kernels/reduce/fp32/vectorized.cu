// SOL5: Vectorized 128-bit Loads + Grid-Stride + Warp Shuffle
// Maximizes memory bandwidth with float4 loads
#include <cuda/atomic>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

extern "C" __global__ void reduce_sum_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_sums[32];
    
    float thread_sum = 0.0f;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    
    // load 4 elements at a time (128-bit loads)
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    int num_v4_chunks = n / 4;
    
    // add 4 elements at a time to the thread_sum
    for (int i = tid; i < num_v4_chunks; i += num_threads) {
        float4 chunk = input_v4[i];
        thread_sum += chunk.x + chunk.y + chunk.z + chunk.w;
    } 
    
    // handle tail elements (not divisible by 4)
    for (int i = num_v4_chunks * 4 + tid; i < n; i += num_threads) {
        thread_sum += input[i];
    }
    
    float sum = warp_reduce_sum(thread_sum);
    
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
            output_ref.fetch_add(sum);
        }
    }
}
