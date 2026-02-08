
extern "C" __global__ void exclusive_scan(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {

    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockSize = blockDim.x;
    

    // phase 1: chunk scan per thread

    const int elements_per_thread = (n+blockDim.x-1)/blockDim.x; // ceiling division to cover all elements    

    
    int thread_start = gid*elements_per_thread; // with 1 block it's just threadIdx.x*elements_per_thread
    float local_chunk_scan = 0; // exclusive scan starts with 0
    for (int i=1; i<elements_per_thread; i++){
        if (thread_start + i < n){
            local_chunk_scan = input[thread_start + i - 1] + local_chunk_scan; // cumulative sum within the chunk
        }
    }
    if (thread_start + elements_per_thread - 1 < n) {
        smem[tid] = local_chunk_scan + input[thread_start + elements_per_thread - 1];
    } else {
        smem[tid] = local_chunk_scan; 
    }
    __syncthreads(); 


     // phase 2: scan using the last element of every thread chunk
    if (tid == 0){
        for (int i=1; i<blockSize; i++){
            smem[i] += smem[i-1]; // cumulative sum in shared memory, in this case an inclusive scan since we will later access smem[tid-1]
        }
    }
    __syncthreads();   

     // phase 3: add offsets to all chunks
    float offset = 0;
    if (tid > 0){
        offset = smem[tid-1];
    }

    float local_chunk_scan2 = 0; // we need to recompute the local chunk scan for each thread since we need to add the offset to each element of the chunk 
    // can't create a dynamic array at the start so opted to recompute
    if (thread_start < n) {  
        output[thread_start] = local_chunk_scan2 + offset;
    }

    for (int i=1; i<elements_per_thread; i++){
        if (thread_start + i < n) {
            local_chunk_scan2 = input[thread_start + i - 1] + local_chunk_scan2;
            output[thread_start+i] = local_chunk_scan2 + offset;
        }
    }
  


    

}
