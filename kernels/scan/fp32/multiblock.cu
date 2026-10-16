// Single-pass exclusive scan across many blocks, written as plainly as
// possible. Each block takes the next tile from a counter, scans it in shared
// memory, waits for the previous tile to publish its inclusive prefix, then
// publishes its own. Correct for any n, but tiles finish strictly one after
// another, so it is slow: a starting point, not a reference.
//
// Workspace (zeroed before every launch, at least 64 + 16 * ceil(n / 32)
// bytes): word 0 is the tile counter, word 1 + t holds tile t's inclusive
// prefix in the low 32 bits and a ready flag in the high 32.
//
// arena: THREADS=256 GRID_PER_SM=4

extern "C" __global__ void exclusive_scan_multiblock(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    unsigned long long* workspace
) {
    __shared__ float s[1024];
    __shared__ unsigned tile_s;
    __shared__ float carry_s;

    const int T = blockDim.x;
    const unsigned num_tiles = (n + T - 1) / T;

    while (true) {
        if (threadIdx.x == 0) tile_s = atomicAdd(reinterpret_cast<unsigned*>(workspace), 1u);
        __syncthreads();
        const unsigned tile = tile_s;
        if (tile >= num_tiles) return;

        const int i = tile * T + threadIdx.x;
        const float x = i < n ? input[i] : 0.0f;
        s[threadIdx.x] = x;
        __syncthreads();

        // inclusive scan of the tile, one step per doubling distance
        for (int off = 1; off < T; off <<= 1) {
            const float v = threadIdx.x >= off ? s[threadIdx.x - off] : 0.0f;
            __syncthreads();
            s[threadIdx.x] += v;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float carry = 0.0f;
            if (tile > 0) {
                volatile unsigned long long* prev = workspace + tile;   // tile - 1 lives at 1 + (tile - 1)
                unsigned long long w;
                do { w = *prev; } while ((w >> 32) == 0);
                carry = __uint_as_float(static_cast<unsigned>(w));
            }
            const float inclusive = carry + s[T - 1];
            __threadfence();
            atomicExch(workspace + 1 + tile,
                       (1ull << 32) | static_cast<unsigned long long>(__float_as_uint(inclusive)));
            carry_s = carry;
        }
        __syncthreads();

        if (i < n) output[i] = carry_s + (threadIdx.x > 0 ? s[threadIdx.x - 1] : 0.0f);
        __syncthreads();
    }
}
