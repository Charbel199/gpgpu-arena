extern "C" __global__ void reduce_sum_baseline(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < n) {
        atomicAdd(&output[0], input[id]); // every thread does an atomicAdd to global memory (Not even worth using the newer atomicAdds syntax)
    }
}
