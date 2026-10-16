// naive row-wise layer norm: one block per row
// each block sums the row for the mean, re-reads it for the variance, then
// re-reads it a third time to normalize, scale and shift
#define LN_EPS 1e-5f

extern "C" __global__ void layernorm_naive(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int cols
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const float* row_in = input + (size_t)row * cols;
    float* row_out = output + (size_t)row * cols;

    // pass 1: mean
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += row_in[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    const float mean = smem[0] / cols;
    __syncthreads();

    // pass 2: variance around the mean (two-pass, so a large mean does not
    // cancel away the variance)
    float local_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float d = row_in[i] - mean;
        local_sq += d * d;
    }
    smem[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    const float rstd = rsqrtf(smem[0] / cols + LN_EPS);

    // pass 3: normalize, scale, shift
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * rstd * gamma[i] + beta[i];
    }
}
