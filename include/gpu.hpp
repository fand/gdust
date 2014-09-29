template<unsigned int blockSize>
__device__ void
reduceBlock(float *sdata1, float *sdata2, float *sdata3) {
  // make sure all threads are ready
  __syncthreads();

  unsigned int tid = threadIdx.x;
  float mySum1 = sdata1[tid];
  float mySum2 = sdata2[tid];
  float mySum3 = sdata3[tid];

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 256];
      sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 256];
      sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 256];
    }
    __syncthreads();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 128];
      sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 128];
      sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 128];
    }
    __syncthreads();
  }

  if (blockSize >= 128) {
    if (tid <  64) {
      sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 64];
      sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 64];
      sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile float *smem1 = sdata1;
    volatile float *smem2 = sdata2;
    volatile float *smem3 = sdata3;

    if (blockSize >=  64) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 32];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 32];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 32];
    }

    if (blockSize >=  32) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 16];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 16];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 16];
    }

    if (blockSize >=  16) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 8];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 8];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 8];
    }

    if (blockSize >=   8) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 4];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 4];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 4];
    }

    if (blockSize >=   4) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 2];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 2];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 2];
    }

    if (blockSize >=   2) {
      smem1[tid] = mySum1 = mySum1 + smem1[tid + 1];
      smem2[tid] = mySum2 = mySum2 + smem2[tid + 1];
      smem3[tid] = mySum3 = mySum3 + smem3[tid + 1];
    }
  }
}
