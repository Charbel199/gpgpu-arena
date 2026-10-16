// Naive attention: one block per (query row, head). The block computes the
// row's S scores into shared memory, takes their max and sum with simple
// tree reductions, then each thread produces some of the D outputs by
// walking all S values again. Every score is recomputed from global memory
// and K and V are re-read for every query: a starting point, not a design.
extern "C" __global__ void attention_naive(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int BH, int S, int D
) {
    extern __shared__ float smem[];           // S scores, then blockDim.x scratch
    float* scores = smem;
    float* red = smem + S;

    const int row = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const float* q = Q + ((size_t)bh * S + row) * D;
    const float* k = K + (size_t)bh * S * D;
    const float* v = V + (size_t)bh * S * D;
    const float scale = rsqrtf((float)D);

    float local_max = -3.0e38f;
    for (int j = tid; j < S; j += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) s += q[d] * k[(size_t)j * D + d];
        s *= scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < S; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];

    for (int d = tid; d < D; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < S; j++) o += scores[j] * v[(size_t)j * D + d];
        O[((size_t)bh * S + row) * D + d] = o * inv;
    }
}
